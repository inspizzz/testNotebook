import numpy as np
import pandas as pd
import json
import logging
import math
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
    timestamp_utc: str
    trigger_key: int = 0
    trial_index: int = 0
    phase: str = ""
    network_state: str = ""
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
    State-dependent gating experiment.

    Tests whether electrically evoked multi-unit spike counts differ
    significantly between spontaneous network burst states (B) and
    inter-burst quiescent states (Q), as predicted by the literature
    hypothesis.

    Phases:
      0 - Baseline recording (measure spontaneous activity to set thresholds)
      1 - Calibration block (fixed-timing stimulation, mixed states)
      2 - State-triggered stimulation (Q and B trials interleaved)
      3 - Recovery recording (no stimulation)
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        stim_electrode: int = 17,
        resp_electrode: int = 18,
        stim_amplitude_ua: float = 3.0,
        stim_duration_us: float = 200.0,
        stim_polarity: str = "PositiveFirst",
        trigger_key: int = 0,
        baseline_duration_s: float = 60.0,
        calibration_trials: int = 10,
        calibration_isi_s: float = 20.0,
        target_trials_per_condition: int = 30,
        max_wait_for_state_s: float = 60.0,
        min_isi_s: float = 10.0,
        state_window_s: float = 0.5,
        burst_multiplier: float = 2.0,
        response_window_ms: float = 100.0,
        recovery_duration_s: float = 60.0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.stim_electrode = stim_electrode
        self.resp_electrode = resp_electrode
        self.stim_amplitude_ua = min(abs(stim_amplitude_ua), 4.0)
        self.stim_duration_us = min(abs(stim_duration_us), 400.0)
        self.stim_polarity_str = stim_polarity
        self.trigger_key = trigger_key

        self.baseline_duration_s = baseline_duration_s
        self.calibration_trials = calibration_trials
        self.calibration_isi_s = calibration_isi_s
        self.target_trials_per_condition = target_trials_per_condition
        self.max_wait_for_state_s = max_wait_for_state_s
        self.min_isi_s = min_isi_s
        self.state_window_s = state_window_s
        self.burst_multiplier = burst_multiplier
        self.response_window_ms = response_window_ms
        self.recovery_duration_s = recovery_duration_s

        # Charge balance: A1*D1 = A2*D2 with equal amplitudes => equal durations
        self.phase_amplitude1 = self.stim_amplitude_ua
        self.phase_duration1 = self.stim_duration_us
        self.phase_amplitude2 = self.stim_amplitude_ua
        self.phase_duration2 = self.stim_duration_us

        # Hardware handles
        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        # Data storage
        self._stimulation_log: List[StimulationRecord] = []
        self._trial_results: List[Dict[str, Any]] = []
        self._baseline_spike_rate: float = 0.0
        self._quiescence_threshold: float = 0.0
        self._burst_threshold: float = 0.0

        # Phase boundary timestamps
        self._phase_times: Dict[str, datetime] = {}

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

            if not self.experiment.start():
                logger.error("Failed to start experiment")
                return {"status": "failed", "error": "Could not start experiment"}

            recording_start = datetime_now()
            self._phase_times["recording_start"] = recording_start

            self._configure_stimulation()

            self._phase_baseline()
            self._phase_calibration()
            self._phase_state_triggered()
            self._phase_recovery()

            recording_stop = datetime_now()
            self._phase_times["recording_stop"] = recording_stop

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _configure_stimulation(self) -> None:
        logger.info(
            "Configuring stimulation: electrode=%d amp=%.2f uA dur=%.1f us polarity=%s",
            self.stim_electrode,
            self.stim_amplitude_ua,
            self.stim_duration_us,
            self.stim_polarity_str,
        )
        polarity = (
            StimPolarity.PositiveFirst
            if self.stim_polarity_str == "PositiveFirst"
            else StimPolarity.NegativeFirst
        )

        stim = StimParam()
        stim.index = self.stim_electrode
        stim.enable = True
        stim.trigger_key = self.trigger_key
        stim.trigger_delay = 0
        stim.nb_pulse = 0
        stim.pulse_train_period = 10000
        stim.post_stim_ref_period = 1000.0
        stim.stim_shape = StimShape.Biphasic
        stim.polarity = polarity
        stim.phase_amplitude1 = self.phase_amplitude1
        stim.phase_duration1 = self.phase_duration1
        stim.phase_amplitude2 = self.phase_amplitude2
        stim.phase_duration2 = self.phase_duration2
        stim.enable_amp_settle = True
        stim.pre_stim_amp_settle = 0.0
        stim.post_stim_amp_settle = 1000.0
        stim.enable_charge_recovery = True
        stim.post_charge_recovery_on = 0.0
        stim.post_charge_recovery_off = 100.0

        self.intan.send_stimparam([stim])
        logger.info("Stimulation parameters sent")

    def _phase_baseline(self) -> None:
        logger.info("Phase 0: Baseline recording (%.1f s)", self.baseline_duration_s)
        self._phase_times["baseline_start"] = datetime_now()
        self._wait(self.baseline_duration_s)
        baseline_stop = datetime_now()
        self._phase_times["baseline_stop"] = baseline_stop

        baseline_start = self._phase_times["baseline_start"]
        try:
            spike_df = self.database.get_spike_event(
                baseline_start, baseline_stop, self.experiment.exp_name
            )
            if not spike_df.empty and "Time" in spike_df.columns:
                n_spikes = len(spike_df)
                duration = (baseline_stop - baseline_start).total_seconds()
                self._baseline_spike_rate = n_spikes / max(duration, 1.0)
                logger.info(
                    "Baseline spike rate: %.3f spikes/s (%d spikes in %.1f s)",
                    self._baseline_spike_rate,
                    n_spikes,
                    duration,
                )
            else:
                self._baseline_spike_rate = 1.0
                logger.warning("No spikes in baseline; using default rate=1.0")
        except Exception as exc:
            self._baseline_spike_rate = 1.0
            logger.warning("Baseline query failed: %s; using default rate=1.0", exc)

        window_spikes = self._baseline_spike_rate * self.state_window_s
        self._quiescence_threshold = max(window_spikes, 0.5)
        self._burst_threshold = self._quiescence_threshold * self.burst_multiplier
        logger.info(
            "State thresholds: Q < %.2f spikes/window, B > %.2f spikes/window",
            self._quiescence_threshold,
            self._burst_threshold,
        )

    def _phase_calibration(self) -> None:
        logger.info(
            "Phase 1: Calibration block (%d trials, ISI=%.1f s)",
            self.calibration_trials,
            self.calibration_isi_s,
        )
        self._phase_times["calibration_start"] = datetime_now()

        for trial_idx in range(self.calibration_trials):
            self._fire_stimulus(
                trial_index=trial_idx,
                phase="calibration",
                network_state="mixed",
            )
            self._wait(self.calibration_isi_s)

        self._phase_times["calibration_stop"] = datetime_now()
        logger.info("Calibration complete")

    def _phase_state_triggered(self) -> None:
        logger.info(
            "Phase 2: State-triggered stimulation (target %d Q + %d B trials)",
            self.target_trials_per_condition,
            self.target_trials_per_condition,
        )
        self._phase_times["state_triggered_start"] = datetime_now()

        q_count = 0
        b_count = 0
        total_attempts = 0
        max_attempts = (self.target_trials_per_condition * 2) * 10
        last_stim_time = datetime_now() - timedelta(seconds=self.min_isi_s + 1)

        while (
            q_count < self.target_trials_per_condition
            or b_count < self.target_trials_per_condition
        ) and total_attempts < max_attempts:

            total_attempts += 1

            elapsed_since_stim = (datetime_now() - last_stim_time).total_seconds()
            if elapsed_since_stim < self.min_isi_s:
                self._wait(self.min_isi_s - elapsed_since_stim)

            state = self._classify_network_state()

            if state == "quiescent" and q_count < self.target_trials_per_condition:
                self._fire_stimulus(
                    trial_index=q_count,
                    phase="state_triggered",
                    network_state="quiescent",
                )
                q_count += 1
                last_stim_time = datetime_now()
                logger.info("Q trial %d/%d fired", q_count, self.target_trials_per_condition)

            elif state == "burst" and b_count < self.target_trials_per_condition:
                self._fire_stimulus(
                    trial_index=b_count,
                    phase="state_triggered",
                    network_state="burst",
                )
                b_count += 1
                last_stim_time = datetime_now()
                logger.info("B trial %d/%d fired", b_count, self.target_trials_per_condition)

            else:
                self._wait(0.1)

        self._phase_times["state_triggered_stop"] = datetime_now()
        logger.info(
            "State-triggered phase complete: Q=%d B=%d (attempts=%d)",
            q_count,
            b_count,
            total_attempts,
        )

    def _phase_recovery(self) -> None:
        logger.info("Phase 3: Recovery recording (%.1f s)", self.recovery_duration_s)
        self._phase_times["recovery_start"] = datetime_now()
        self._wait(self.recovery_duration_s)
        self._phase_times["recovery_stop"] = datetime_now()
        logger.info("Recovery phase complete")

    def _classify_network_state(self) -> str:
        window_stop = datetime_now()
        window_start = window_stop - timedelta(seconds=self.state_window_s)
        try:
            spike_df = self.database.get_spike_event(
                window_start, window_stop, self.experiment.exp_name
            )
            if spike_df.empty or "Time" not in spike_df.columns:
                n_spikes = 0
            else:
                n_spikes = len(spike_df)
        except Exception as exc:
            logger.debug("State classification query failed: %s", exc)
            n_spikes = 0

        if n_spikes < self._quiescence_threshold:
            return "quiescent"
        elif n_spikes > self._burst_threshold:
            return "burst"
        else:
            return "intermediate"

    def _fire_stimulus(
        self,
        trial_index: int,
        phase: str,
        network_state: str,
    ) -> None:
        stim_time = datetime_now()

        pattern = np.zeros(16, dtype=np.uint8)
        pattern[self.trigger_key] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.05)
        pattern[self.trigger_key] = 0
        self.trigger_controller.send(pattern)

        self._stimulation_log.append(
            StimulationRecord(
                electrode_idx=self.stim_electrode,
                amplitude_ua=self.stim_amplitude_ua,
                duration_us=self.stim_duration_us,
                polarity=self.stim_polarity_str,
                timestamp_utc=stim_time.isoformat(),
                trigger_key=self.trigger_key,
                trial_index=trial_index,
                phase=phase,
                network_state=network_state,
            )
        )

        response_window_s = self.response_window_ms / 1000.0
        self._wait(response_window_s)

        query_start = stim_time
        query_stop = datetime_now()
        try:
            spike_df = self.database.get_spike_event(
                query_start, query_stop, self.experiment.exp_name
            )
            if spike_df.empty or "Time" not in spike_df.columns:
                evoked_spikes = 0
                resp_electrode_spikes = 0
            else:
                evoked_spikes = len(spike_df)
                if "channel" in spike_df.columns:
                    resp_electrode_spikes = int(
                        (spike_df["channel"] == self.resp_electrode).sum()
                    )
                else:
                    resp_electrode_spikes = 0
        except Exception as exc:
            logger.debug("Response query failed: %s", exc)
            evoked_spikes = 0
            resp_electrode_spikes = 0

        self._trial_results.append(
            {
                "trial_index": trial_index,
                "phase": phase,
                "network_state": network_state,
                "stim_electrode": self.stim_electrode,
                "resp_electrode": self.resp_electrode,
                "stim_time_utc": stim_time.isoformat(),
                "evoked_spikes_total": evoked_spikes,
                "evoked_spikes_resp_electrode": resp_electrode_spikes,
            }
        )

    def _save_all(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> None:
        fs_name = getattr(self.experiment, "exp_name", "unknown")
        saver = DataSaver(self._output_dir, fs_name)

        saver.save_stimulation_log(self._stimulation_log)

        try:
            spike_df = self.database.get_spike_event(
                recording_start, recording_stop, fs_name
            )
        except Exception as exc:
            logger.warning("Failed to fetch spike events: %s", exc)
            spike_df = pd.DataFrame()

        saver.save_spike_events(spike_df)

        try:
            trigger_df = self.database.get_all_triggers(
                recording_start, recording_stop
            )
        except Exception as exc:
            logger.warning("Failed to fetch triggers: %s", exc)
            trigger_df = pd.DataFrame()

        saver.save_triggers(trigger_df)

        q_trials = [t for t in self._trial_results if t["network_state"] == "quiescent"]
        b_trials = [t for t in self._trial_results if t["network_state"] == "burst"]

        q_spikes = [t["evoked_spikes_resp_electrode"] for t in q_trials]
        b_spikes = [t["evoked_spikes_resp_electrode"] for t in b_trials]

        q_mean = float(np.mean(q_spikes)) if q_spikes else 0.0
        b_mean = float(np.mean(b_spikes)) if b_spikes else 0.0
        q_std = float(np.std(q_spikes)) if q_spikes else 0.0
        b_std = float(np.std(b_spikes)) if b_spikes else 0.0

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "stim_electrode": self.stim_electrode,
            "resp_electrode": self.resp_electrode,
            "stim_amplitude_ua": self.stim_amplitude_ua,
            "stim_duration_us": self.stim_duration_us,
            "stim_polarity": self.stim_polarity_str,
            "baseline_spike_rate_hz": self._baseline_spike_rate,
            "quiescence_threshold_spikes_per_window": self._quiescence_threshold,
            "burst_threshold_spikes_per_window": self._burst_threshold,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "quiescent_trials_n": len(q_trials),
            "burst_trials_n": len(b_trials),
            "quiescent_mean_evoked_spikes": q_mean,
            "burst_mean_evoked_spikes": b_mean,
            "quiescent_std_evoked_spikes": q_std,
            "burst_std_evoked_spikes": b_std,
            "phase_times": {k: v.isoformat() for k, v in self._phase_times.items()},
            "trial_results": self._trial_results,
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
                    waveform_records.append(
                        {
                            "electrode_idx": int(electrode_idx),
                            "num_waveforms": len(raw_df),
                            "waveform_samples": raw_df.values.tolist(),
                        }
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to fetch waveforms for electrode %s: %s",
                    electrode_idx,
                    exc,
                )

        return waveform_records

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        logger.info("Compiling results")

        q_trials = [t for t in self._trial_results if t["network_state"] == "quiescent"]
        b_trials = [t for t in self._trial_results if t["network_state"] == "burst"]

        q_spikes = [t["evoked_spikes_resp_electrode"] for t in q_trials]
        b_spikes = [t["evoked_spikes_resp_electrode"] for t in b_trials]

        q_mean = float(np.mean(q_spikes)) if q_spikes else 0.0
        b_mean = float(np.mean(b_spikes)) if b_spikes else 0.0
        q_std = float(np.std(q_spikes)) if q_spikes else 0.0
        b_std = float(np.std(b_spikes)) if b_spikes else 0.0

        cohen_d = 0.0
        if q_spikes and b_spikes:
            pooled_std = math.sqrt(
                (
                    (len(q_spikes) - 1) * q_std ** 2
                    + (len(b_spikes) - 1) * b_std ** 2
                )
                / max(len(q_spikes) + len(b_spikes) - 2, 1)
            )
            if pooled_std > 0:
                cohen_d = (q_mean - b_mean) / pooled_std

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "stim_electrode": self.stim_electrode,
            "resp_electrode": self.resp_electrode,
            "stim_amplitude_ua": self.stim_amplitude_ua,
            "stim_duration_us": self.stim_duration_us,
            "baseline_spike_rate_hz": self._baseline_spike_rate,
            "quiescence_threshold": self._quiescence_threshold,
            "burst_threshold": self._burst_threshold,
            "total_stimulations": len(self._stimulation_log),
            "quiescent_trials_n": len(q_trials),
            "burst_trials_n": len(b_trials),
            "quiescent_mean_evoked_spikes": q_mean,
            "burst_mean_evoked_spikes": b_mean,
            "quiescent_std_evoked_spikes": q_std,
            "burst_std_evoked_spikes": b_std,
            "cohen_d_q_vs_b": cohen_d,
            "hypothesis_supported": cohen_d >= 0.5 and q_mean > b_mean,
        }

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
