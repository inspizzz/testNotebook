import numpy as np
import pandas as pd
import json
import time
import logging
import math
import csv
import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from pathlib import Path

from neuroplatform import (
    IntanSofware,
    TriggerController,
    Database,
    StimParam,
    StimShape,
    StimPolarity,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StimulationRecord:
    electrode_idx: int
    amplitude_ua: float
    duration_us: float
    timestamp_utc: str
    trigger_key: int = 0
    phase: str = ""
    condition: str = ""
    trial_index: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialResult:
    phase: str
    condition: str
    trial_index: int
    stim_electrode: int
    resp_electrode: int
    amplitude_ua: float
    duration_us: float
    polarity: str
    timestamp_utc: str
    spike_count: int
    spike_electrodes: List[int] = field(default_factory=list)
    latencies_ms: List[float] = field(default_factory=list)
    median_latency_ms: float = 0.0


class DataSaver:
    def __init__(self, output_dir: Path, fs_name: str) -> None:
        self._dir = output_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._prefix = self._dir / f"{fs_name}_{timestamp}"

    def save_stimulation_log(self, stimulations: List[StimulationRecord]) -> Path:
        path = Path(f"{self._prefix}_stimulations.json")
        records = [asdict(s) for s in stimulations]
        path.write_text(json.dumps(records, indent=2, default=str))
        logger.info("Saved stimulation log -> %s  (%d records)", path, len(records))
        return path

    def save_spike_events(self, df: pd.DataFrame) -> Path:
        path = Path(f"{self._prefix}_spike_events.csv")
        df.to_csv(path, index=False)
        logger.info("Saved spike events -> %s  (%d rows)", path, len(df))
        return path

    def save_triggers(self, df: pd.DataFrame) -> Path:
        path = Path(f"{self._prefix}_triggers.csv")
        df.to_csv(path, index=False)
        logger.info("Saved triggers -> %s  (%d rows)", path, len(df))
        return path

    def save_summary(self, summary: Dict[str, Any]) -> Path:
        path = Path(f"{self._prefix}_summary.json")
        path.write_text(json.dumps(summary, indent=2, default=str))
        logger.info("Saved summary -> %s", path)
        return path

    def save_spike_waveforms(self, waveform_records: list) -> Path:
        path = Path(f"{self._prefix}_spike_waveforms.json")
        path.write_text(json.dumps(waveform_records, indent=2, default=str))
        logger.info("Saved spike waveforms -> %s  (%d spike(s))", path, len(waveform_records))
        return path

    def save_trial_results(self, trials: List[TrialResult]) -> Path:
        path = Path(f"{self._prefix}_trial_results.json")
        records = [asdict(t) for t in trials]
        path.write_text(json.dumps(records, indent=2, default=str))
        logger.info("Saved trial results -> %s  (%d records)", path, len(records))
        return path


class Experiment:
    """
    Activity Modulation Experiment
    ==============================
    Uses scan-identified active electrode pairs to attempt increasing and
    decreasing neural response activity via tetanic high-frequency
    stimulation (potentiation) and low-frequency matched-pulse control.

    Protocol:
      1. BASELINE: Deliver test pulses at 0.1 Hz on selected pairs, record
         evoked spike counts for 20 trials per pair.
      2. TETANUS (increase condition): Deliver 5 trains of high-frequency
         (100 Hz) biphasic pulses (1 s each, 10 s inter-train interval)
         on the stimulation electrode.
      3. POST-TETANUS: Repeat baseline test pulses (20 trials) to measure
         potentiation.
      4. REST: 60 s recovery.
      5. LOW-FREQ CONTROL (decrease/control condition): Deliver same total
         pulse count as tetanus but at 1 Hz (slow, non-potentiating).
      6. POST-CONTROL: Repeat test pulses (20 trials) to measure any change.

    Selected pairs (from deep scan, response_rate >= 0.93):
      - Pair A: stim=5 -> resp=7, amp=3.0, dur=400, NegativeFirst (93%)
      - Pair B: stim=9 -> resp=10, amp=3.0, dur=200, NegativeFirst (100%)
      - Pair C: stim=10 -> resp=12, amp=3.0, dur=300, NegativeFirst (100%)
      - Pair D: stim=17 -> resp=18, amp=3.0, dur=400, NegativeFirst (95%)
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        baseline_trials: int = 20,
        post_trials: int = 20,
        test_isi_s: float = 10.0,
        tetanus_trains: int = 5,
        tetanus_train_duration_s: float = 1.0,
        tetanus_frequency_hz: float = 100.0,
        inter_train_interval_s: float = 10.0,
        tetanus_amplitude_ua: float = 3.0,
        tetanus_duration_us: float = 200.0,
        rest_between_conditions_s: float = 60.0,
        post_stim_wait_s: float = 0.5,
        spike_window_ms: float = 100.0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.baseline_trials = baseline_trials
        self.post_trials = post_trials
        self.test_isi_s = test_isi_s
        self.tetanus_trains = tetanus_trains
        self.tetanus_train_duration_s = tetanus_train_duration_s
        self.tetanus_frequency_hz = tetanus_frequency_hz
        self.inter_train_interval_s = inter_train_interval_s
        self.tetanus_amplitude_ua = min(tetanus_amplitude_ua, 4.0)
        self.tetanus_duration_us = min(tetanus_duration_us, 400.0)
        self.rest_between_conditions_s = rest_between_conditions_s
        self.post_stim_wait_s = post_stim_wait_s
        self.spike_window_ms = spike_window_ms

        self.target_pairs = [
            {"stim": 5, "resp": 7, "amp": 3.0, "dur": 400.0, "pol": "NegativeFirst"},
            {"stim": 9, "resp": 10, "amp": 3.0, "dur": 200.0, "pol": "NegativeFirst"},
            {"stim": 10, "resp": 12, "amp": 3.0, "dur": 300.0, "pol": "NegativeFirst"},
            {"stim": 17, "resp": 18, "amp": 3.0, "dur": 400.0, "pol": "NegativeFirst"},
        ]

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._trial_results: List[TrialResult] = []

    def run(self) -> Dict[str, Any]:
        try:
            logger.info("Initialising hardware connections")

            from neuroplatform import Experiment as NeuroPlatformExperiment

            self.experiment = NeuroPlatformExperiment(self.token)
            self.trigger_controller = TriggerController(email=self.booking_email)
            self.intan = IntanSofware()
            self.database = Database()

            logger.info("Experiment: %s", self.experiment.exp_name)
            logger.info("Electrodes: %s", self.experiment.electrodes)

            if not self.experiment.start():
                logger.error("Failed to start experiment")
                return {"status": "failed", "error": "Could not start experiment"}

            recording_start = datetime.now(timezone.utc)

            for pair_idx, pair in enumerate(self.target_pairs):
                logger.info(
                    "=== Processing pair %d/%d: stim=%d -> resp=%d ===",
                    pair_idx + 1, len(self.target_pairs),
                    pair["stim"], pair["resp"],
                )

                pol = (
                    StimPolarity.NegativeFirst
                    if pair["pol"] == "NegativeFirst"
                    else StimPolarity.PositiveFirst
                )

                logger.info("Phase 1: BASELINE test pulses")
                self._phase_test_pulses(
                    pair, pol, phase="baseline",
                    num_trials=self.baseline_trials,
                )

                logger.info("Phase 2: TETANIC stimulation (increase attempt)")
                self._phase_tetanus(pair, pol)

                logger.info("Phase 3: POST-TETANUS test pulses")
                self._phase_test_pulses(
                    pair, pol, phase="post_tetanus",
                    num_trials=self.post_trials,
                )

                logger.info("Phase 4: REST period (%d s)", self.rest_between_conditions_s)
                time.sleep(self.rest_between_conditions_s)

                logger.info("Phase 5: LOW-FREQ control stimulation (decrease/control)")
                self._phase_low_freq_control(pair, pol)

                logger.info("Phase 6: POST-CONTROL test pulses")
                self._phase_test_pulses(
                    pair, pol, phase="post_control",
                    num_trials=self.post_trials,
                )

                if pair_idx < len(self.target_pairs) - 1:
                    logger.info("Inter-pair rest (30 s)")
                    time.sleep(30.0)

            recording_stop = datetime.now(timezone.utc)

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _phase_test_pulses(
        self,
        pair: Dict,
        polarity: StimPolarity,
        phase: str,
        num_trials: int,
    ) -> None:
        stim_elec = pair["stim"]
        resp_elec = pair["resp"]
        amp = min(pair["amp"], 4.0)
        dur = min(pair["dur"], 400.0)

        for trial_i in range(num_trials):
            spike_df = self._stimulate_and_record(
                electrode_idx=stim_elec,
                amplitude_ua=amp,
                duration_us=dur,
                polarity=polarity,
                trigger_key=0,
                post_stim_wait_s=self.post_stim_wait_s,
                recording_window_s=self.spike_window_ms / 1000.0 + self.post_stim_wait_s,
                phase=phase,
                condition="test_pulse",
                trial_index=trial_i,
            )

            resp_spikes = pd.DataFrame()
            if not spike_df.empty:
                ch_col = "channel" if "channel" in spike_df.columns else None
                if ch_col is None:
                    for c in spike_df.columns:
                        if "channel" in c.lower() or "electrode" in c.lower():
                            ch_col = c
                            break
                if ch_col is not None:
                    resp_spikes = spike_df[spike_df[ch_col] == resp_elec]

            latencies = []
            if not resp_spikes.empty and "Time" in resp_spikes.columns:
                stim_time = datetime.now(timezone.utc) - timedelta(
                    seconds=self.post_stim_wait_s
                )
                for _, row in resp_spikes.iterrows():
                    t = row["Time"]
                    if hasattr(t, "timestamp"):
                        lat_ms = (t.timestamp() - stim_time.timestamp()) * 1000.0
                    else:
                        lat_ms = 0.0
                    if 0 < lat_ms < self.spike_window_ms:
                        latencies.append(lat_ms)

            trial_result = TrialResult(
                phase=phase,
                condition="test_pulse",
                trial_index=trial_i,
                stim_electrode=stim_elec,
                resp_electrode=resp_elec,
                amplitude_ua=amp,
                duration_us=dur,
                polarity=pair["pol"],
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                spike_count=len(resp_spikes),
                latencies_ms=latencies,
                median_latency_ms=float(np.median(latencies)) if latencies else 0.0,
            )
            self._trial_results.append(trial_result)

            time.sleep(self.test_isi_s)

    def _phase_tetanus(
        self,
        pair: Dict,
        polarity: StimPolarity,
    ) -> None:
        stim_elec = pair["stim"]
        amp = min(self.tetanus_amplitude_ua, 4.0)
        dur = min(self.tetanus_duration_us, 400.0)

        pulses_per_train = int(self.tetanus_frequency_hz * self.tetanus_train_duration_s)
        pulse_period_us = int(1e6 / self.tetanus_frequency_hz)

        if pulse_period_us < 1:
            pulse_period_us = 10000

        ramp_steps = 5
        ramp_amplitudes = [
            amp * (i + 1) / ramp_steps for i in range(ramp_steps)
        ]

        for train_i in range(self.tetanus_trains):
            logger.info(
                "  Tetanus train %d/%d on electrode %d (%d pulses at %.0f Hz)",
                train_i + 1, self.tetanus_trains, stim_elec,
                pulses_per_train, self.tetanus_frequency_hz,
            )

            for ramp_i, ramp_amp in enumerate(ramp_amplitudes):
                ramp_amp = min(ramp_amp, 4.0)
                self._send_single_pulse(
                    electrode_idx=stim_elec,
                    amplitude_ua=ramp_amp,
                    duration_us=dur,
                    polarity=polarity,
                    trigger_key=1,
                    phase="tetanus_ramp",
                    condition="tetanus",
                    trial_index=train_i * 1000 + ramp_i,
                )
                time.sleep(1.0 / self.tetanus_frequency_hz)

            remaining_pulses = pulses_per_train - ramp_steps
            if remaining_pulses < 0:
                remaining_pulses = 0

            stim = StimParam()
            stim.index = stim_elec
            stim.enable = True
            stim.trigger_key = 1
            stim.trigger_delay = 0
            stim.nb_pulse = min(remaining_pulses, 255)
            stim.pulse_train_period = pulse_period_us
            stim.post_stim_ref_period = 1000.0
            stim.stim_shape = StimShape.Biphasic
            stim.polarity = polarity
            stim.phase_amplitude1 = amp
            stim.phase_duration1 = dur
            stim.phase_amplitude2 = amp
            stim.phase_duration2 = dur
            stim.enable_amp_settle = True
            stim.pre_stim_amp_settle = 0.0
            stim.post_stim_amp_settle = 1000.0
            stim.enable_charge_recovery = True
            stim.post_charge_recovery_on = 0.0
            stim.post_charge_recovery_off = 100.0

            self.intan.send_stimparam([stim])

            pattern = np.zeros(16, dtype=np.uint8)
            pattern[1] = 1
            self.trigger_controller.send(pattern)
            time.sleep(0.05)
            pattern[1] = 0
            self.trigger_controller.send(pattern)

            self._stimulation_log.append(StimulationRecord(
                electrode_idx=stim_elec,
                amplitude_ua=amp,
                duration_us=dur,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                trigger_key=1,
                phase="tetanus_train",
                condition="tetanus",
                trial_index=train_i,
                extra={
                    "nb_pulse": stim.nb_pulse,
                    "pulse_train_period_us": pulse_period_us,
                    "train_index": train_i,
                },
            ))

            train_time_s = remaining_pulses / self.tetanus_frequency_hz if self.tetanus_frequency_hz > 0 else 1.0
            time.sleep(train_time_s + 0.5)

            if train_i < self.tetanus_trains - 1:
                time.sleep(self.inter_train_interval_s)

    def _phase_low_freq_control(
        self,
        pair: Dict,
        polarity: StimPolarity,
    ) -> None:
        stim_elec = pair["stim"]
        amp = min(self.tetanus_amplitude_ua, 4.0)
        dur = min(self.tetanus_duration_us, 400.0)

        total_tetanus_pulses = int(
            self.tetanus_frequency_hz
            * self.tetanus_train_duration_s
            * self.tetanus_trains
        )

        control_freq_hz = 1.0
        control_pulses = min(total_tetanus_pulses, 50)

        logger.info(
            "  Low-freq control: %d pulses at %.1f Hz on electrode %d",
            control_pulses, control_freq_hz, stim_elec,
        )

        for pulse_i in range(control_pulses):
            self._send_single_pulse(
                electrode_idx=stim_elec,
                amplitude_ua=amp,
                duration_us=dur,
                polarity=polarity,
                trigger_key=2,
                phase="low_freq_control",
                condition="control",
                trial_index=pulse_i,
            )
            time.sleep(1.0 / control_freq_hz)

    def _send_single_pulse(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity,
        trigger_key: int,
        phase: str = "",
        condition: str = "",
        trial_index: int = 0,
    ) -> None:
        amplitude_ua = min(abs(amplitude_ua), 4.0)
        duration_us = min(abs(duration_us), 400.0)

        stim = StimParam()
        stim.index = electrode_idx
        stim.enable = True
        stim.trigger_key = trigger_key
        stim.trigger_delay = 0
        stim.nb_pulse = 0
        stim.pulse_train_period = 10000
        stim.post_stim_ref_period = 1000.0
        stim.stim_shape = StimShape.Biphasic
        stim.polarity = polarity
        stim.phase_amplitude1 = amplitude_ua
        stim.phase_duration1 = duration_us
        stim.phase_amplitude2 = amplitude_ua
        stim.phase_duration2 = duration_us
        stim.enable_amp_settle = True
        stim.pre_stim_amp_settle = 0.0
        stim.post_stim_amp_settle = 1000.0
        stim.enable_charge_recovery = True
        stim.post_charge_recovery_on = 0.0
        stim.post_charge_recovery_off = 100.0

        self.intan.send_stimparam([stim])

        pattern = np.zeros(16, dtype=np.uint8)
        pattern[trigger_key] = 1
        self.trigger_controller.send(pattern)
        time.sleep(0.05)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            trigger_key=trigger_key,
            phase=phase,
            condition=condition,
            trial_index=trial_index,
        ))

    def _stimulate_and_record(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity = StimPolarity.NegativeFirst,
        trigger_key: int = 0,
        post_stim_wait_s: float = 0.5,
        recording_window_s: float = 0.6,
        phase: str = "",
        condition: str = "",
        trial_index: int = 0,
    ) -> pd.DataFrame:
        amplitude_ua = min(abs(amplitude_ua), 4.0)
        duration_us = min(abs(duration_us), 400.0)

        stim = StimParam()
        stim.index = electrode_idx
        stim.enable = True
        stim.trigger_key = trigger_key
        stim.trigger_delay = 0
        stim.nb_pulse = 0
        stim.pulse_train_period = 10000
        stim.post_stim_ref_period = 1000.0
        stim.stim_shape = StimShape.Biphasic
        stim.polarity = polarity
        stim.phase_amplitude1 = amplitude_ua
        stim.phase_duration1 = duration_us
        stim.phase_amplitude2 = amplitude_ua
        stim.phase_duration2 = duration_us
        stim.enable_amp_settle = True
        stim.pre_stim_amp_settle = 0.0
        stim.post_stim_amp_settle = 1000.0
        stim.enable_charge_recovery = True
        stim.post_charge_recovery_on = 0.0
        stim.post_charge_recovery_off = 100.0

        self.intan.send_stimparam([stim])

        pattern = np.zeros(16, dtype=np.uint8)
        pattern[trigger_key] = 1
        self.trigger_controller.send(pattern)
        time.sleep(0.05)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            trigger_key=trigger_key,
            phase=phase,
            condition=condition,
            trial_index=trial_index,
        ))

        time.sleep(post_stim_wait_s)

        query_start = datetime.now(timezone.utc) - timedelta(seconds=recording_window_s)
        query_stop = datetime.now(timezone.utc)

        fs_name = getattr(self.experiment, "exp_name", "unknown")
        spike_df = self.database.get_spike_event(query_start, query_stop, fs_name)
        return spike_df

    def _save_all(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> None:
        fs_name = getattr(self.experiment, "exp_name", "unknown")
        saver = DataSaver(self._output_dir, fs_name)

        saver.save_stimulation_log(self._stimulation_log)

        spike_df = self.database.get_spike_event(
            recording_start, recording_stop, fs_name
        )
        saver.save_spike_events(spike_df)

        trigger_df = self.database.get_all_triggers(
            recording_start, recording_stop
        )
        saver.save_triggers(trigger_df)

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "total_trial_results": len(self._trial_results),
            "target_pairs": self.target_pairs,
            "tetanus_params": {
                "trains": self.tetanus_trains,
                "frequency_hz": self.tetanus_frequency_hz,
                "train_duration_s": self.tetanus_train_duration_s,
                "amplitude_ua": self.tetanus_amplitude_ua,
                "duration_us": self.tetanus_duration_us,
                "inter_train_interval_s": self.inter_train_interval_s,
            },
            "test_params": {
                "baseline_trials": self.baseline_trials,
                "post_trials": self.post_trials,
                "test_isi_s": self.test_isi_s,
                "spike_window_ms": self.spike_window_ms,
            },
        }
        saver.save_summary(summary)

        saver.save_trial_results(self._trial_results)

        waveform_records = self._fetch_spike_waveforms(
            fs_name, spike_df, trigger_df, recording_start, recording_stop
        )
        saver.save_spike_waveforms(waveform_records)

    def _fetch_spike_waveforms(
        self,
        fs_name: str,
        spike_df: pd.DataFrame,
        trigger_df: pd.DataFrame,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> list:
        waveform_records = []
        if spike_df.empty:
            return waveform_records

        electrode_col = None
        for col in spike_df.columns:
            if col in ("channel", "index"):
                electrode_col = col
                break
            if "electrode" in col.lower() or "idx" in col.lower() or "channel" in col.lower():
                electrode_col = col
                break

        if electrode_col is None:
            logger.warning("Cannot determine electrode column for waveform fetch")
            return waveform_records

        unique_electrodes = spike_df[electrode_col].unique()

        resp_electrodes = set()
        for pair in self.target_pairs:
            resp_electrodes.add(pair["resp"])
            resp_electrodes.add(pair["stim"])

        fetch_electrodes = [e for e in unique_electrodes if int(e) in resp_electrodes]

        for electrode_idx in fetch_electrodes:
            try:
                raw_df = self.database.get_raw_spike(
                    recording_start, recording_stop, int(electrode_idx)
                )
                if not raw_df.empty:
                    waveform_records.append({
                        "electrode_idx": int(electrode_idx),
                        "num_waveforms": len(raw_df),
                        "waveform_samples": raw_df.values.tolist(),
                    })
            except Exception as exc:
                logger.warning(
                    "Failed to fetch waveforms for electrode %s: %s",
                    electrode_idx, exc,
                )

        return waveform_records

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        logger.info("Compiling results")

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "total_stimulations": len(self._stimulation_log),
            "total_trial_results": len(self._trial_results),
            "pair_analyses": [],
        }

        for pair in self.target_pairs:
            stim_e = pair["stim"]
            resp_e = pair["resp"]

            pair_trials = [
                t for t in self._trial_results
                if t.stim_electrode == stim_e and t.resp_electrode == resp_e
            ]

            baseline_trials = [t for t in pair_trials if t.phase == "baseline"]
            post_tet_trials = [t for t in pair_trials if t.phase == "post_tetanus"]
            post_ctrl_trials = [t for t in pair_trials if t.phase == "post_control"]

            def _mean_count(trials):
                if not trials:
                    return 0.0
                return sum(t.spike_count for t in trials) / len(trials)

            def _mean_latency(trials):
                all_lat = []
                for t in trials:
                    all_lat.extend(t.latencies_ms)
                if not all_lat:
                    return 0.0
                return float(np.mean(all_lat))

            baseline_mean = _mean_count(baseline_trials)
            post_tet_mean = _mean_count(post_tet_trials)
            post_ctrl_mean = _mean_count(post_ctrl_trials)

            baseline_lat = _mean_latency(baseline_trials)
            post_tet_lat = _mean_latency(post_tet_trials)
            post_ctrl_lat = _mean_latency(post_ctrl_trials)

            norm_efficacy_tet = (
                post_tet_mean / baseline_mean if baseline_mean > 0 else 0.0
            )
            norm_efficacy_ctrl = (
                post_ctrl_mean / baseline_mean if baseline_mean > 0 else 0.0
            )

            pair_analysis = {
                "stim_electrode": stim_e,
                "resp_electrode": resp_e,
                "amplitude_ua": pair["amp"],
                "duration_us": pair["dur"],
                "polarity": pair["pol"],
                "baseline_mean_spike_count": round(baseline_mean, 3),
                "post_tetanus_mean_spike_count": round(post_tet_mean, 3),
                "post_control_mean_spike_count": round(post_ctrl_mean, 3),
                "normalized_efficacy_tetanus": round(norm_efficacy_tet, 3),
                "normalized_efficacy_control": round(norm_efficacy_ctrl, 3),
                "baseline_mean_latency_ms": round(baseline_lat, 3),
                "post_tetanus_mean_latency_ms": round(post_tet_lat, 3),
                "post_control_mean_latency_ms": round(post_ctrl_lat, 3),
                "potentiated": norm_efficacy_tet > 1.5,
                "depressed_by_control": norm_efficacy_ctrl < 0.7,
                "num_baseline_trials": len(baseline_trials),
                "num_post_tetanus_trials": len(post_tet_trials),
                "num_post_control_trials": len(post_ctrl_trials),
            }
            summary["pair_analyses"].append(pair_analysis)

            logger.info(
                "Pair stim=%d->resp=%d: baseline=%.2f, post_tet=%.2f (%.2fx), "
                "post_ctrl=%.2f (%.2fx)",
                stim_e, resp_e,
                baseline_mean, post_tet_mean, norm_efficacy_tet,
                post_ctrl_mean, norm_efficacy_ctrl,
            )

        potentiated_count = sum(
            1 for p in summary["pair_analyses"] if p["potentiated"]
        )
        summary["potentiated_pairs"] = potentiated_count
        summary["total_pairs"] = len(self.target_pairs)

        return summary

    def _cleanup(self) -> None:
        logger.info("Cleaning up resources")

        if self.experiment is not None:
            try:
                self.experiment.stop()
            except Exception as exc:
                logger.error("Error stopping experiment: %s", exc)

        if self.intan is not None:
            try:
                self.intan.close()
            except Exception as exc:
                logger.error("Error closing IntanSofware: %s", exc)

        if self.trigger_controller is not None:
            try:
                self.trigger_controller.close()
            except Exception as exc:
                logger.error("Error closing TriggerController: %s", exc)
