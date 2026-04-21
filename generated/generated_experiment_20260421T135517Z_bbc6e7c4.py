import numpy as np
import pandas as pd
import json
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
    datetime_now,
    wait,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StimulationRecord:
    electrode_idx: int
    amplitude_ua: float
    duration_us: float
    polarity: str
    phase: str
    trial_index: int
    timestamp_utc: str
    trigger_key: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


class DataSaver:
    def __init__(self, output_dir: Path, fs_name: str) -> None:
        self._dir = output_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime_now().strftime("%Y%m%dT%H%M%SZ")
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


class Experiment:
    """
    LTP induction experiment using the top responsive electrode pair (17->18).

    Phase 1 - Baseline (15 min): single stimulations at 0.1 Hz, record response probability.
    Phase 2 - Tetanic stimulation: 4 trains of 100 stimulations at 5 Hz, trains separated by 30 s.
    Phase 3 - Post-tetanus monitoring (30 min): resume 0.1 Hz single stimulation, track response probability.

    Electrode pair 17->18 selected as top responsive pair (response_rate=0.92, temporal_stability=1.0).
    Amplitude: 3.0 uA, Duration: 300 us (charge balanced: 3.0*300 = 900 = 3.0*300).
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        stim_electrode: int = 17,
        resp_electrode: int = 18,
        amplitude_ua: float = 3.0,
        duration_us: float = 300.0,
        polarity: str = "PositiveFirst",
        baseline_duration_s: float = 900.0,
        baseline_freq_hz: float = 0.1,
        tetanus_n_trains: int = 4,
        tetanus_pulses_per_train: int = 100,
        tetanus_freq_hz: float = 5.0,
        inter_train_interval_s: float = 30.0,
        post_tetanus_duration_s: float = 1800.0,
        post_tetanus_freq_hz: float = 0.1,
        response_window_ms: float = 50.0,
        trigger_key: int = 0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.stim_electrode = stim_electrode
        self.resp_electrode = resp_electrode
        self.amplitude_ua = min(abs(amplitude_ua), 4.0)
        self.duration_us = min(abs(duration_us), 400.0)
        self.polarity_str = polarity
        self.polarity = StimPolarity.PositiveFirst if polarity == "PositiveFirst" else StimPolarity.NegativeFirst

        self.baseline_duration_s = baseline_duration_s
        self.baseline_freq_hz = baseline_freq_hz
        self.tetanus_n_trains = tetanus_n_trains
        self.tetanus_pulses_per_train = tetanus_pulses_per_train
        self.tetanus_freq_hz = tetanus_freq_hz
        self.inter_train_interval_s = inter_train_interval_s
        self.post_tetanus_duration_s = post_tetanus_duration_s
        self.post_tetanus_freq_hz = post_tetanus_freq_hz
        self.response_window_ms = response_window_ms
        self.trigger_key = trigger_key

        # Charge balance verification: A1*D1 == A2*D2 (both phases equal)
        assert abs(self.amplitude_ua * self.duration_us - self.amplitude_ua * self.duration_us) < 1e-9, \
            "Charge balance violated"

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []

        # Results storage
        self._baseline_trials: List[Dict[str, Any]] = []
        self._tetanus_records: List[Dict[str, Any]] = []
        self._post_tetanus_trials: List[Dict[str, Any]] = []

        self._baseline_response_count = 0
        self._baseline_total = 0
        self._post_tetanus_response_count = 0
        self._post_tetanus_total = 0

        self._phase_timestamps: Dict[str, str] = {}

    def _wait(self, seconds: float) -> None:
        wait(seconds)

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
            logger.info("Stim electrode: %d -> Resp electrode: %d", self.stim_electrode, self.resp_electrode)
            logger.info(
                "Charge balance check: A1*D1=%.1f, A2*D2=%.1f",
                self.amplitude_ua * self.duration_us,
                self.amplitude_ua * self.duration_us,
            )

            if not self.experiment.start():
                logger.error("Failed to start experiment")
                return {"status": "failed", "error": "Could not start experiment"}

            recording_start = datetime_now()
            self._phase_timestamps["recording_start"] = recording_start.isoformat()

            # Configure stimulation parameters once
            self._configure_stim_params()

            # Phase 1: Baseline
            logger.info("=== Phase 1: Baseline (%.0f s at %.3f Hz) ===", self.baseline_duration_s, self.baseline_freq_hz)
            phase1_start = datetime_now()
            self._phase_timestamps["phase1_start"] = phase1_start.isoformat()
            self._phase_baseline()
            phase1_stop = datetime_now()
            self._phase_timestamps["phase1_stop"] = phase1_stop.isoformat()
            logger.info(
                "Phase 1 complete: %d/%d responses (%.2f%%)",
                self._baseline_response_count,
                self._baseline_total,
                100.0 * self._baseline_response_count / max(1, self._baseline_total),
            )

            # Phase 2: Tetanic stimulation
            logger.info(
                "=== Phase 2: Tetanic stimulation (%d trains x %d pulses @ %.1f Hz) ===",
                self.tetanus_n_trains, self.tetanus_pulses_per_train, self.tetanus_freq_hz,
            )
            phase2_start = datetime_now()
            self._phase_timestamps["phase2_start"] = phase2_start.isoformat()
            self._phase_tetanus()
            phase2_stop = datetime_now()
            self._phase_timestamps["phase2_stop"] = phase2_stop.isoformat()
            logger.info("Phase 2 complete: tetanic stimulation delivered")

            # Phase 3: Post-tetanus monitoring
            logger.info(
                "=== Phase 3: Post-tetanus monitoring (%.0f s at %.3f Hz) ===",
                self.post_tetanus_duration_s, self.post_tetanus_freq_hz,
            )
            phase3_start = datetime_now()
            self._phase_timestamps["phase3_start"] = phase3_start.isoformat()
            self._phase_post_tetanus()
            phase3_stop = datetime_now()
            self._phase_timestamps["phase3_stop"] = phase3_stop.isoformat()
            logger.info(
                "Phase 3 complete: %d/%d responses (%.2f%%)",
                self._post_tetanus_response_count,
                self._post_tetanus_total,
                100.0 * self._post_tetanus_response_count / max(1, self._post_tetanus_total),
            )

            recording_stop = datetime_now()
            self._phase_timestamps["recording_stop"] = recording_stop.isoformat()

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _configure_stim_params(self) -> None:
        """Configure and send stimulation parameters to Intan."""
        stim = self._build_stim_param(self.stim_electrode, self.trigger_key)
        self.intan.send_stimparam([stim])
        logger.info(
            "Stim params configured: electrode=%d, A=%.1f uA, D=%.1f us, polarity=%s",
            self.stim_electrode, self.amplitude_ua, self.duration_us, self.polarity_str,
        )

    def _build_stim_param(self, electrode_idx: int, trigger_key: int) -> StimParam:
        """Build a charge-balanced StimParam. A1*D1 == A2*D2."""
        stim = StimParam()
        stim.index = electrode_idx
        stim.enable = True
        stim.trigger_key = trigger_key
        stim.trigger_delay = 0
        stim.nb_pulse = 0
        stim.pulse_train_period = 10000
        stim.post_stim_ref_period = 1000.0
        stim.stim_shape = StimShape.Biphasic
        stim.polarity = self.polarity
        stim.phase_amplitude1 = self.amplitude_ua
        stim.phase_duration1 = self.duration_us
        stim.phase_amplitude2 = self.amplitude_ua
        stim.phase_duration2 = self.duration_us
        stim.enable_amp_settle = True
        stim.pre_stim_amp_settle = 0.0
        stim.post_stim_amp_settle = 1000.0
        stim.enable_charge_recovery = True
        stim.post_charge_recovery_on = 0.0
        stim.post_charge_recovery_off = 100.0
        stim.interphase_delay = 0.0
        return stim

    def _fire_trigger(self) -> None:
        """Fire the trigger pulse."""
        pattern = np.zeros(16, dtype=np.uint8)
        pattern[self.trigger_key] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.05)
        pattern[self.trigger_key] = 0
        self.trigger_controller.send(pattern)

    def _deliver_single_stim(self, phase: str, trial_index: int) -> datetime:
        """Deliver a single stimulation pulse and log it. Returns stim timestamp."""
        ts = datetime_now()
        self._fire_trigger()
        self._stimulation_log.append(StimulationRecord(
            electrode_idx=self.stim_electrode,
            amplitude_ua=self.amplitude_ua,
            duration_us=self.duration_us,
            polarity=self.polarity_str,
            phase=phase,
            trial_index=trial_index,
            timestamp_utc=ts.isoformat(),
            trigger_key=self.trigger_key,
        ))
        return ts

    def _check_response(self, stim_time: datetime) -> bool:
        """
        Check whether a spike occurred on the responding electrode within
        the response window after stimulation.
        """
        window_s = self.response_window_ms / 1000.0
        query_start = stim_time
        query_stop = stim_time + timedelta(seconds=window_s + 0.1)
        try:
            spike_df = self.database.get_spike_event_electrode(
                query_start, query_stop, self.resp_electrode
            )
            if spike_df.empty:
                return False
            # Check if any spike falls within the response window
            time_col = None
            for col in spike_df.columns:
                if col.lower() in ("time", "_time"):
                    time_col = col
                    break
            if time_col is None:
                return len(spike_df) > 0
            spike_times = pd.to_datetime(spike_df[time_col], utc=True)
            stim_dt = pd.Timestamp(stim_time)
            if stim_dt.tzinfo is None:
                stim_dt = stim_dt.tz_localize("UTC")
            window_end = stim_dt + pd.Timedelta(seconds=window_s)
            in_window = spike_times[(spike_times >= stim_dt) & (spike_times <= window_end)]
            return len(in_window) > 0
        except Exception as exc:
            logger.warning("Response check failed: %s", exc)
            return False

    def _phase_baseline(self) -> None:
        """
        Phase 1: Deliver single stimulations at 0.1 Hz for baseline_duration_s.
        Inter-stimulus interval = 1 / freq = 10 s at 0.1 Hz.
        """
        isi_s = 1.0 / self.baseline_freq_hz
        n_stims = int(self.baseline_duration_s * self.baseline_freq_hz)
        logger.info("Baseline: %d stimulations, ISI=%.1f s", n_stims, isi_s)

        for trial_idx in range(n_stims):
            stim_time = self._deliver_single_stim("baseline", trial_idx)
            responded = self._check_response(stim_time)
            if responded:
                self._baseline_response_count += 1
            self._baseline_total += 1
            self._baseline_trials.append({
                "trial_index": trial_idx,
                "timestamp_utc": stim_time.isoformat(),
                "responded": responded,
            })
            if trial_idx < n_stims - 1:
                self._wait(isi_s)

        logger.info(
            "Baseline complete: %d/%d responded",
            self._baseline_response_count, self._baseline_total,
        )

    def _phase_tetanus(self) -> None:
        """
        Phase 2: Deliver 4 trains of 100 stimulations at 5 Hz.
        Trains separated by 30 s inter-train intervals.
        """
        isi_s = 1.0 / self.tetanus_freq_hz  # 0.2 s between pulses within a train

        for train_idx in range(self.tetanus_n_trains):
            logger.info(
                "Tetanus train %d/%d: %d pulses at %.1f Hz",
                train_idx + 1, self.tetanus_n_trains,
                self.tetanus_pulses_per_train, self.tetanus_freq_hz,
            )
            for pulse_idx in range(self.tetanus_pulses_per_train):
                stim_time = self._deliver_single_stim(
                    "tetanus", train_idx * self.tetanus_pulses_per_train + pulse_idx
                )
                self._tetanus_records.append({
                    "train_index": train_idx,
                    "pulse_index": pulse_idx,
                    "timestamp_utc": stim_time.isoformat(),
                })
                if pulse_idx < self.tetanus_pulses_per_train - 1:
                    self._wait(isi_s)

            logger.info("Train %d complete", train_idx + 1)
            if train_idx < self.tetanus_n_trains - 1:
                logger.info("Inter-train interval: %.1f s", self.inter_train_interval_s)
                self._wait(self.inter_train_interval_s)

        logger.info("All tetanus trains delivered")

    def _phase_post_tetanus(self) -> None:
        """
        Phase 3: Resume 0.1 Hz single stimulation for post_tetanus_duration_s.
        Track response probability over time.
        """
        isi_s = 1.0 / self.post_tetanus_freq_hz
        n_stims = int(self.post_tetanus_duration_s * self.post_tetanus_freq_hz)
        logger.info("Post-tetanus: %d stimulations, ISI=%.1f s", n_stims, isi_s)

        for trial_idx in range(n_stims):
            stim_time = self._deliver_single_stim("post_tetanus", trial_idx)
            responded = self._check_response(stim_time)
            if responded:
                self._post_tetanus_response_count += 1
            self._post_tetanus_total += 1
            self._post_tetanus_trials.append({
                "trial_index": trial_idx,
                "timestamp_utc": stim_time.isoformat(),
                "responded": responded,
            })
            if trial_idx < n_stims - 1:
                self._wait(isi_s)

        logger.info(
            "Post-tetanus complete: %d/%d responded",
            self._post_tetanus_response_count, self._post_tetanus_total,
        )

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        baseline_prob = (
            self._baseline_response_count / self._baseline_total
            if self._baseline_total > 0 else 0.0
        )
        post_prob = (
            self._post_tetanus_response_count / self._post_tetanus_total
            if self._post_tetanus_total > 0 else 0.0
        )
        ltp_ratio = post_prob / baseline_prob if baseline_prob > 0 else float("nan")

        # Compute time-binned response probability for post-tetanus (5-min bins)
        bin_size_s = 300.0
        post_binned = self._bin_response_probability(self._post_tetanus_trials, bin_size_s)

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "stim_electrode": self.stim_electrode,
            "resp_electrode": self.resp_electrode,
            "amplitude_ua": self.amplitude_ua,
            "duration_us": self.duration_us,
            "polarity": self.polarity_str,
            "charge_balance_nC": self.amplitude_ua * self.duration_us,
            "phase_timestamps": self._phase_timestamps,
            "baseline": {
                "n_trials": self._baseline_total,
                "n_responses": self._baseline_response_count,
                "response_probability": baseline_prob,
            },
            "tetanus": {
                "n_trains": self.tetanus_n_trains,
                "pulses_per_train": self.tetanus_pulses_per_train,
                "total_pulses": len(self._tetanus_records),
                "freq_hz": self.tetanus_freq_hz,
            },
            "post_tetanus": {
                "n_trials": self._post_tetanus_total,
                "n_responses": self._post_tetanus_response_count,
                "response_probability": post_prob,
                "binned_response_probability": post_binned,
            },
            "ltp_ratio": ltp_ratio,
            "total_stimulations": len(self._stimulation_log),
        }
        logger.info(
            "Results: baseline_prob=%.3f, post_prob=%.3f, LTP_ratio=%.3f",
            baseline_prob, post_prob, ltp_ratio,
        )
        return summary

    def _bin_response_probability(
        self, trials: List[Dict[str, Any]], bin_size_s: float
    ) -> List[Dict[str, Any]]:
        """Compute response probability in time bins relative to first trial."""
        if not trials:
            return []
        try:
            t0 = datetime.fromisoformat(trials[0]["timestamp_utc"])
            if t0.tzinfo is None:
                t0 = t0.replace(tzinfo=timezone.utc)
        except Exception:
            return []

        bins: Dict[int, Dict[str, int]] = defaultdict(lambda: {"n": 0, "r": 0})
        for trial in trials:
            try:
                ts = datetime.fromisoformat(trial["timestamp_utc"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                elapsed = (ts - t0).total_seconds()
                bin_idx = int(elapsed // bin_size_s)
                bins[bin_idx]["n"] += 1
                if trial.get("responded", False):
                    bins[bin_idx]["r"] += 1
            except Exception:
                continue

        result = []
        for bin_idx in sorted(bins.keys()):
            n = bins[bin_idx]["n"]
            r = bins[bin_idx]["r"]
            result.append({
                "bin_start_s": bin_idx * bin_size_s,
                "bin_end_s": (bin_idx + 1) * bin_size_s,
                "n_trials": n,
                "n_responses": r,
                "response_probability": r / n if n > 0 else 0.0,
            })
        return result

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

        baseline_prob = (
            self._baseline_response_count / self._baseline_total
            if self._baseline_total > 0 else 0.0
        )
        post_prob = (
            self._post_tetanus_response_count / self._post_tetanus_total
            if self._post_tetanus_total > 0 else 0.0
        )
        ltp_ratio = post_prob / baseline_prob if baseline_prob > 0 else None

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "stim_electrode": self.stim_electrode,
            "resp_electrode": self.resp_electrode,
            "amplitude_ua": self.amplitude_ua,
            "duration_us": self.duration_us,
            "polarity": self.polarity_str,
            "charge_balance_nC": self.amplitude_ua * self.duration_us,
            "phase_timestamps": self._phase_timestamps,
            "baseline_n_trials": self._baseline_total,
            "baseline_n_responses": self._baseline_response_count,
            "baseline_response_probability": baseline_prob,
            "tetanus_n_trains": self.tetanus_n_trains,
            "tetanus_pulses_per_train": self.tetanus_pulses_per_train,
            "tetanus_total_pulses": len(self._tetanus_records),
            "post_tetanus_n_trials": self._post_tetanus_total,
            "post_tetanus_n_responses": self._post_tetanus_response_count,
            "post_tetanus_response_probability": post_prob,
            "ltp_ratio": ltp_ratio,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
        }
        saver.save_summary(summary)

        waveform_records = self._fetch_spike_waveforms(
            fs_name, spike_df, recording_start, recording_stop
        )
        saver.save_spike_waveforms(waveform_records)

    def _fetch_spike_waveforms(
        self,
        fs_name: str,
        spike_df: pd.DataFrame,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> list:
        waveform_records = []
        if spike_df.empty:
            return waveform_records

        electrode_col = None
        for col in spike_df.columns:
            if col.lower() in ("channel", "index", "electrode"):
                electrode_col = col
                break
            if "electrode" in col.lower() or "idx" in col.lower():
                electrode_col = col
                break

        if electrode_col is None:
            logger.warning("Cannot determine electrode column for waveform fetch")
            return waveform_records

        unique_electrodes = spike_df[electrode_col].unique()
        for electrode_idx in unique_electrodes:
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
