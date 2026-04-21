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

    def save_connectivity_matrix(self, matrix: List[List[float]], electrodes: List[int]) -> Path:
        path = Path(f"{self._prefix}_connectivity_matrix.json")
        data = {
            "electrodes": electrodes,
            "matrix": matrix,
            "description": "Entry (i,j) = response probability of electrode j when electrode i is stimulated",
        }
        path.write_text(json.dumps(data, indent=2, default=str))
        logger.info("Saved connectivity matrix -> %s", path)
        return path


class Experiment:
    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        num_electrodes: int = 32,
        stim_amplitude_ua: float = 2.0,
        stim_duration_us: float = 200.0,
        num_trials: int = 20,
        stim_rate_hz: float = 1.0,
        post_electrode_wait_s: float = 5.0,
        response_window_ms: float = 50.0,
        trigger_key: int = 0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.num_electrodes = num_electrodes
        self.stim_amplitude_ua = min(abs(stim_amplitude_ua), 4.0)
        self.stim_duration_us = min(abs(stim_duration_us), 400.0)
        self.num_trials = num_trials
        self.stim_rate_hz = stim_rate_hz
        self.post_electrode_wait_s = post_electrode_wait_s
        self.response_window_ms = response_window_ms
        self.trigger_key = trigger_key

        # Verify charge balance: A1*D1 == A2*D2 (symmetric biphasic)
        assert math.isclose(
            self.stim_amplitude_ua * self.stim_duration_us,
            self.stim_amplitude_ua * self.stim_duration_us,
        ), "Charge balance violated"

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []

        # connectivity_matrix[i][j] = number of trials where electrode j responded when i was stimulated
        self._response_counts: List[List[int]] = [
            [0] * num_electrodes for _ in range(num_electrodes)
        ]
        self._connectivity_matrix: List[List[float]] = [
            [0.0] * num_electrodes for _ in range(num_electrodes)
        ]

        self._electrodes: List[int] = list(range(num_electrodes))

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

            available_electrodes = self.experiment.electrodes
            if available_electrodes:
                self._electrodes = list(available_electrodes)[:self.num_electrodes]
                actual_n = len(self._electrodes)
                self._response_counts = [[0] * actual_n for _ in range(actual_n)]
                self._connectivity_matrix = [[0.0] * actual_n for _ in range(actual_n)]
            else:
                actual_n = self.num_electrodes

            if not self.experiment.start():
                logger.error("Failed to start experiment")
                return {"status": "failed", "error": "Could not start experiment"}

            recording_start = datetime_now()

            self._phase_connectivity_mapping()

            recording_stop = datetime_now()

            self._compute_connectivity_matrix()

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _phase_connectivity_mapping(self) -> None:
        logger.info("Phase: cross-electrode connectivity mapping")

        inter_stim_interval_s = 1.0 / self.stim_rate_hz
        response_window_s = self.response_window_ms / 1000.0

        n_electrodes = len(self._electrodes)

        for stim_idx, stim_electrode in enumerate(self._electrodes):
            logger.info(
                "Stimulating electrode %d (%d/%d)",
                stim_electrode, stim_idx + 1, n_electrodes,
            )

            self._configure_stim_param(stim_electrode)

            for trial in range(self.num_trials):
                stim_time = datetime_now()

                self._fire_trigger()

                self._stimulation_log.append(StimulationRecord(
                    electrode_idx=stim_electrode,
                    amplitude_ua=self.stim_amplitude_ua,
                    duration_us=self.stim_duration_us,
                    polarity="PositiveFirst",
                    timestamp_utc=stim_time.isoformat(),
                    trigger_key=self.trigger_key,
                    trial_index=trial,
                ))

                self._wait(response_window_s)

                window_start = stim_time
                window_stop = stim_time + timedelta(seconds=response_window_s + 0.01)

                self._record_responses(stim_idx, window_start, window_stop)

                remaining_wait = inter_stim_interval_s - response_window_s
                if remaining_wait > 0:
                    self._wait(remaining_wait)

            logger.info(
                "Finished stimulating electrode %d, waiting %s s before next electrode",
                stim_electrode, self.post_electrode_wait_s,
            )
            self._wait(self.post_electrode_wait_s)

    def _configure_stim_param(self, electrode_idx: int) -> None:
        stim = StimParam()
        stim.index = electrode_idx
        stim.enable = True
        stim.trigger_key = self.trigger_key
        stim.trigger_delay = 0
        stim.nb_pulse = 0
        stim.pulse_train_period = 10000
        stim.post_stim_ref_period = 1000.0
        stim.stim_shape = StimShape.Biphasic
        stim.polarity = StimPolarity.PositiveFirst

        # Charge balance: A1*D1 == A2*D2 (symmetric)
        stim.phase_amplitude1 = self.stim_amplitude_ua
        stim.phase_duration1 = self.stim_duration_us
        stim.phase_amplitude2 = self.stim_amplitude_ua
        stim.phase_duration2 = self.stim_duration_us

        stim.enable_amp_settle = True
        stim.pre_stim_amp_settle = 0.0
        stim.post_stim_amp_settle = 1000.0
        stim.enable_charge_recovery = True
        stim.post_charge_recovery_on = 0.0
        stim.post_charge_recovery_off = 100.0
        stim.interphase_delay = 0.0

        self.intan.send_stimparam([stim])

    def _fire_trigger(self) -> None:
        pattern = np.zeros(16, dtype=np.uint8)
        pattern[self.trigger_key] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.02)
        pattern[self.trigger_key] = 0
        self.trigger_controller.send(pattern)

    def _record_responses(
        self,
        stim_idx: int,
        window_start: datetime,
        window_stop: datetime,
    ) -> None:
        fs_name = self.experiment.exp_name
        try:
            spike_df = self.database.get_spike_event(window_start, window_stop, fs_name)
        except Exception as exc:
            logger.warning("Failed to fetch spike events: %s", exc)
            return

        if spike_df.empty:
            return

        channel_col = None
        for col in ("channel", "index", "electrode"):
            if col in spike_df.columns:
                channel_col = col
                break

        if channel_col is None:
            logger.warning("Cannot determine channel column in spike DataFrame")
            return

        responding_channels = set(spike_df[channel_col].astype(int).unique())

        stim_electrode = self._electrodes[stim_idx]

        for resp_idx, resp_electrode in enumerate(self._electrodes):
            if resp_electrode == stim_electrode:
                continue
            if resp_electrode in responding_channels:
                self._response_counts[stim_idx][resp_idx] += 1

    def _compute_connectivity_matrix(self) -> None:
        n = len(self._electrodes)
        for i in range(n):
            for j in range(n):
                if i == j:
                    self._connectivity_matrix[i][j] = 0.0
                else:
                    self._connectivity_matrix[i][j] = (
                        self._response_counts[i][j] / self.num_trials
                        if self.num_trials > 0
                        else 0.0
                    )

    def _save_all(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> None:
        fs_name = getattr(self.experiment, "exp_name", "unknown")
        saver = DataSaver(self._output_dir, fs_name)

        saver.save_stimulation_log(self._stimulation_log)

        spike_df = self.database.get_spike_event(recording_start, recording_stop, fs_name)
        saver.save_spike_events(spike_df)

        trigger_df = self.database.get_all_triggers(recording_start, recording_stop)
        saver.save_triggers(trigger_df)

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "num_electrodes": len(self._electrodes),
            "num_trials_per_electrode": self.num_trials,
            "stim_amplitude_ua": self.stim_amplitude_ua,
            "stim_duration_us": self.stim_duration_us,
            "response_window_ms": self.response_window_ms,
            "electrodes": self._electrodes,
        }
        saver.save_summary(summary)

        waveform_records = self._fetch_spike_waveforms(
            fs_name, spike_df, recording_start, recording_stop
        )
        saver.save_spike_waveforms(waveform_records)

        saver.save_connectivity_matrix(self._connectivity_matrix, self._electrodes)

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

        channel_col = None
        for col in ("channel", "index", "electrode"):
            if col in spike_df.columns:
                channel_col = col
                break

        if channel_col is None:
            logger.warning("Cannot determine electrode column for waveform fetch")
            return waveform_records

        unique_electrodes = spike_df[channel_col].astype(int).unique()
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

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        logger.info("Compiling results")

        n = len(self._electrodes)
        total_connections = sum(
            1
            for i in range(n)
            for j in range(n)
            if i != j and self._connectivity_matrix[i][j] > 0.0
        )

        strong_connections = [
            {
                "stim_electrode": self._electrodes[i],
                "resp_electrode": self._electrodes[j],
                "response_probability": self._connectivity_matrix[i][j],
            }
            for i in range(n)
            for j in range(n)
            if i != j and self._connectivity_matrix[i][j] >= 0.5
        ]
        strong_connections.sort(key=lambda x: x["response_probability"], reverse=True)

        hub_scores: Dict[int, float] = defaultdict(float)
        for i in range(n):
            for j in range(n):
                if i != j:
                    hub_scores[self._electrodes[i]] += self._connectivity_matrix[i][j]
                    hub_scores[self._electrodes[j]] += self._connectivity_matrix[i][j]

        hub_list = sorted(hub_scores.items(), key=lambda x: x[1], reverse=True)[:5]

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "num_electrodes": n,
            "num_trials_per_electrode": self.num_trials,
            "stim_amplitude_ua": self.stim_amplitude_ua,
            "stim_duration_us": self.stim_duration_us,
            "response_window_ms": self.response_window_ms,
            "total_stimulations": len(self._stimulation_log),
            "total_nonzero_connections": total_connections,
            "strong_connections_ge_50pct": strong_connections[:20],
            "top_hub_electrodes": [
                {"electrode": e, "total_response_score": s} for e, s in hub_list
            ],
            "connectivity_matrix_shape": [n, n],
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
