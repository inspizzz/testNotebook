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
    timestamp_utc: str
    trigger_key: int = 0
    phase: str = ""
    trial_index: int = 0
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
    LTP induction experiment using the top responsive electrode pair (electrode 14 -> 15).

    Phase 1 - Baseline (15 min): single stimulations at 0.1 Hz, record response probability.
    Phase 2 - Tetanic stimulation: 4 trains of 100 stimulations at 5 Hz (trains separated by 30 s).
    Phase 3 - Post-tetanus monitoring (30 min): resume 0.1 Hz single stimulation, track response probability.

    Stimulation parameters: 3 uA amplitude, 300 us duration (charge-balanced: A1*D1 = A2*D2 = 900).
    Top electrode pair from scan: electrode_from=14, electrode_to=15 (response_rate=0.80, hits_k=5/5).
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        stim_electrode: int = 14,
        record_electrode: int = 15,
        amplitude_ua: float = 3.0,
        duration_us: float = 300.0,
        polarity: str = "NegativeFirst",
        baseline_duration_min: float = 15.0,
        probe_freq_hz: float = 0.1,
        tetanus_trains: int = 4,
        tetanus_pulses_per_train: int = 100,
        tetanus_freq_hz: float = 5.0,
        inter_train_interval_s: float = 30.0,
        post_tetanus_duration_min: float = 30.0,
        response_window_ms: float = 50.0,
        trigger_key: int = 0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.stim_electrode = stim_electrode
        self.record_electrode = record_electrode
        self.amplitude_ua = min(abs(amplitude_ua), 4.0)
        self.duration_us = min(abs(duration_us), 400.0)
        self.polarity = StimPolarity.NegativeFirst if polarity == "NegativeFirst" else StimPolarity.PositiveFirst

        # Charge balance check: A1*D1 must equal A2*D2
        # With equal amplitudes and equal durations this is always satisfied.
        assert abs(self.amplitude_ua * self.duration_us - self.amplitude_ua * self.duration_us) < 1e-9

        self.baseline_duration_min = baseline_duration_min
        self.probe_freq_hz = probe_freq_hz
        self.probe_interval_s = 1.0 / probe_freq_hz

        self.tetanus_trains = tetanus_trains
        self.tetanus_pulses_per_train = tetanus_pulses_per_train
        self.tetanus_freq_hz = tetanus_freq_hz
        self.tetanus_interval_s = 1.0 / tetanus_freq_hz
        self.inter_train_interval_s = inter_train_interval_s

        self.post_tetanus_duration_min = post_tetanus_duration_min
        self.response_window_ms = response_window_ms
        self.trigger_key = trigger_key

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []

        # Phase result tracking
        self._baseline_responses: List[Dict[str, Any]] = []
        self._post_tetanus_responses: List[Dict[str, Any]] = []
        self._tetanus_log: List[Dict[str, Any]] = []

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
            logger.info("Electrodes available: %s", self.experiment.electrodes)
            logger.info(
                "Using stim electrode %d -> record electrode %d",
                self.stim_electrode,
                self.record_electrode,
            )

            if not self.experiment.start():
                logger.error("Failed to start experiment")
                return {"status": "failed", "error": "Could not start experiment"}

            recording_start = datetime_now()

            # Configure stimulation parameters once
            self._configure_stim_params()

            # Phase 1: Baseline
            logger.info("=== Phase 1: Baseline (%.1f min at %.3f Hz) ===", self.baseline_duration_min, self.probe_freq_hz)
            self._phase_baseline()

            # Phase 2: Tetanic stimulation
            logger.info("=== Phase 2: Tetanic stimulation (%d trains x %d pulses at %.1f Hz) ===",
                        self.tetanus_trains, self.tetanus_pulses_per_train, self.tetanus_freq_hz)
            self._phase_tetanus()

            # Phase 3: Post-tetanus monitoring
            logger.info("=== Phase 3: Post-tetanus monitoring (%.1f min at %.3f Hz) ===",
                        self.post_tetanus_duration_min, self.probe_freq_hz)
            self._phase_post_tetanus()

            recording_stop = datetime_now()

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
        stim = self._build_stim_param(self.stim_electrode, self.amplitude_ua, self.duration_us, self.polarity, self.trigger_key)
        self.intan.send_stimparam([stim])
        logger.info(
            "Stim params configured: electrode=%d, A=%.1f uA, D=%.1f us, polarity=%s",
            self.stim_electrode, self.amplitude_ua, self.duration_us, self.polarity.name,
        )

    def _build_stim_param(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity,
        trigger_key: int,
        nb_pulse: int = 0,
        pulse_train_period_us: int = 10000,
    ) -> StimParam:
        """Build a charge-balanced StimParam. A1*D1 == A2*D2 guaranteed by equal phases."""
        stim = StimParam()
        stim.index = electrode_idx
        stim.enable = True
        stim.trigger_key = trigger_key
        stim.trigger_delay = 0
        stim.nb_pulse = nb_pulse
        stim.pulse_train_period = pulse_train_period_us
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
        stim.interphase_delay = 0.0
        return stim

    def _fire_trigger(self) -> None:
        """Fire a single trigger pulse on the configured trigger key."""
        pattern = np.zeros(16, dtype=np.uint8)
        pattern[self.trigger_key] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.02)
        pattern[self.trigger_key] = 0
        self.trigger_controller.send(pattern)

    def _probe_stimulation(self, phase: str, trial_index: int) -> Dict[str, Any]:
        """
        Deliver a single probe stimulation and check for response on the recording electrode.
        Returns a dict with trial metadata and response flag.
        """
        stim_time = datetime_now()

        self._fire_trigger()

        self._stimulation_log.append(StimulationRecord(
            electrode_idx=self.stim_electrode,
            amplitude_ua=self.amplitude_ua,
            duration_us=self.duration_us,
            timestamp_utc=stim_time.isoformat(),
            trigger_key=self.trigger_key,
            phase=phase,
            trial_index=trial_index,
        ))

        # Wait for response window
        response_window_s = self.response_window_ms / 1000.0
        self._wait(response_window_s + 0.01)

        query_stop = datetime_now()
        query_start = stim_time

        try:
            spike_df = self.database.get_spike_event(
                query_start, query_stop, self.experiment.exp_name
            )
            if not spike_df.empty and "channel" in spike_df.columns:
                resp_spikes = spike_df[spike_df["channel"] == self.record_electrode]
                responded = len(resp_spikes) > 0
                spike_count = len(resp_spikes)
            else:
                responded = False
                spike_count = 0
        except Exception as exc:
            logger.warning("Spike query failed for trial %d: %s", trial_index, exc)
            responded = False
            spike_count = 0

        result = {
            "phase": phase,
            "trial_index": trial_index,
            "stim_time_utc": stim_time.isoformat(),
            "responded": responded,
            "spike_count": spike_count,
        }
        logger.debug("Probe trial %d [%s]: responded=%s, spikes=%d", trial_index, phase, responded, spike_count)
        return result

    def _phase_baseline(self) -> None:
        """Phase 1: Deliver probe stimulations at 0.1 Hz for baseline_duration_min minutes."""
        baseline_duration_s = self.baseline_duration_min * 60.0
        num_probes = int(baseline_duration_s * self.probe_freq_hz)
        logger.info("Baseline: %d probe stimulations over %.1f min", num_probes, self.baseline_duration_min)

        for i in range(num_probes):
            result = self._probe_stimulation(phase="baseline", trial_index=i)
            self._baseline_responses.append(result)
            # Wait for remainder of probe interval (interval minus response window and overhead)
            remaining = self.probe_interval_s - (self.response_window_ms / 1000.0) - 0.05
            if remaining > 0:
                self._wait(remaining)

        baseline_hits = sum(1 for r in self._baseline_responses if r["responded"])
        baseline_prob = baseline_hits / num_probes if num_probes > 0 else 0.0
        logger.info("Baseline response probability: %.3f (%d/%d)", baseline_prob, baseline_hits, num_probes)

    def _phase_tetanus(self) -> None:
        """Phase 2: Deliver 4 trains of 100 stimulations at 5 Hz, trains separated by 30 s."""
        for train_idx in range(self.tetanus_trains):
            logger.info("Tetanus train %d/%d: %d pulses at %.1f Hz",
                        train_idx + 1, self.tetanus_trains,
                        self.tetanus_pulses_per_train, self.tetanus_freq_hz)

            train_start = datetime_now()

            for pulse_idx in range(self.tetanus_pulses_per_train):
                pulse_time = datetime_now()

                self._fire_trigger()

                self._stimulation_log.append(StimulationRecord(
                    electrode_idx=self.stim_electrode,
                    amplitude_ua=self.amplitude_ua,
                    duration_us=self.duration_us,
                    timestamp_utc=pulse_time.isoformat(),
                    trigger_key=self.trigger_key,
                    phase="tetanus",
                    trial_index=train_idx * self.tetanus_pulses_per_train + pulse_idx,
                    extra={"train_index": train_idx, "pulse_in_train": pulse_idx},
                ))

                # Inter-pulse interval for 5 Hz = 200 ms, minus trigger overhead
                if pulse_idx < self.tetanus_pulses_per_train - 1:
                    self._wait(self.tetanus_interval_s - 0.02)

            train_end = datetime_now()
            self._tetanus_log.append({
                "train_index": train_idx,
                "train_start_utc": train_start.isoformat(),
                "train_end_utc": train_end.isoformat(),
                "pulses": self.tetanus_pulses_per_train,
            })

            # Inter-train interval (skip after last train)
            if train_idx < self.tetanus_trains - 1:
                logger.info("Inter-train interval: %.1f s", self.inter_train_interval_s)
                self._wait(self.inter_train_interval_s)

        logger.info("Tetanus complete: %d trains delivered", self.tetanus_trains)

    def _phase_post_tetanus(self) -> None:
        """Phase 3: Deliver probe stimulations at 0.1 Hz for post_tetanus_duration_min minutes."""
        post_duration_s = self.post_tetanus_duration_min * 60.0
        num_probes = int(post_duration_s * self.probe_freq_hz)
        logger.info("Post-tetanus: %d probe stimulations over %.1f min", num_probes, self.post_tetanus_duration_min)

        for i in range(num_probes):
            result = self._probe_stimulation(phase="post_tetanus", trial_index=i)
            self._post_tetanus_responses.append(result)
            remaining = self.probe_interval_s - (self.response_window_ms / 1000.0) - 0.05
            if remaining > 0:
                self._wait(remaining)

        post_hits = sum(1 for r in self._post_tetanus_responses if r["responded"])
        post_prob = post_hits / num_probes if num_probes > 0 else 0.0
        logger.info("Post-tetanus response probability: %.3f (%d/%d)", post_prob, post_hits, num_probes)

    def _save_all(self, recording_start: datetime, recording_stop: datetime) -> None:
        fs_name = getattr(self.experiment, "exp_name", "unknown")
        saver = DataSaver(self._output_dir, fs_name)

        saver.save_stimulation_log(self._stimulation_log)

        spike_df = self.database.get_spike_event(recording_start, recording_stop, fs_name)
        saver.save_spike_events(spike_df)

        trigger_df = self.database.get_all_triggers(recording_start, recording_stop)
        saver.save_triggers(trigger_df)

        baseline_hits = sum(1 for r in self._baseline_responses if r["responded"])
        baseline_n = len(self._baseline_responses)
        post_hits = sum(1 for r in self._post_tetanus_responses if r["responded"])
        post_n = len(self._post_tetanus_responses)

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "stim_electrode": self.stim_electrode,
            "record_electrode": self.record_electrode,
            "amplitude_ua": self.amplitude_ua,
            "duration_us": self.duration_us,
            "polarity": self.polarity.name,
            "charge_balance_check": self.amplitude_ua * self.duration_us,
            "baseline_probes": baseline_n,
            "baseline_hits": baseline_hits,
            "baseline_response_probability": baseline_hits / baseline_n if baseline_n > 0 else 0.0,
            "tetanus_trains": self.tetanus_trains,
            "tetanus_pulses_per_train": self.tetanus_pulses_per_train,
            "tetanus_freq_hz": self.tetanus_freq_hz,
            "post_tetanus_probes": post_n,
            "post_tetanus_hits": post_hits,
            "post_tetanus_response_probability": post_hits / post_n if post_n > 0 else 0.0,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "tetanus_trains_log": self._tetanus_log,
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
            if col.lower() in ("channel", "index", "electrode", "electrode_idx"):
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
                logger.warning("Failed to fetch waveforms for electrode %s: %s", electrode_idx, exc)

        return waveform_records

    def _compile_results(self, recording_start: datetime, recording_stop: datetime) -> Dict[str, Any]:
        logger.info("Compiling results")

        baseline_hits = sum(1 for r in self._baseline_responses if r["responded"])
        baseline_n = len(self._baseline_responses)
        post_hits = sum(1 for r in self._post_tetanus_responses if r["responded"])
        post_n = len(self._post_tetanus_responses)

        baseline_prob = baseline_hits / baseline_n if baseline_n > 0 else 0.0
        post_prob = post_hits / post_n if post_n > 0 else 0.0
        ltp_ratio = post_prob / baseline_prob if baseline_prob > 0.0 else float("nan")

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "stim_electrode": self.stim_electrode,
            "record_electrode": self.record_electrode,
            "amplitude_ua": self.amplitude_ua,
            "duration_us": self.duration_us,
            "charge_balance": self.amplitude_ua * self.duration_us,
            "baseline_response_probability": baseline_prob,
            "post_tetanus_response_probability": post_prob,
            "ltp_efficacy_ratio": ltp_ratio,
            "baseline_probes": baseline_n,
            "baseline_hits": baseline_hits,
            "post_tetanus_probes": post_n,
            "post_tetanus_hits": post_hits,
            "tetanus_trains_delivered": self.tetanus_trains,
            "total_stimulations": len(self._stimulation_log),
        }

        logger.info("LTP efficacy ratio (post/baseline): %.3f", ltp_ratio if not math.isnan(ltp_ratio) else -1)
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
