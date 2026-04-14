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
    timestamp_utc: str
    trial_number: int
    trigger_key: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConnectivityEntry:
    stim_electrode: int
    response_electrode: int
    trial: int
    had_response: bool
    spike_count: int
    timestamp_utc: str


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

    def save_connectivity_matrix(self, matrix: np.ndarray, n_electrodes: int) -> Path:
        path = Path(f"{self._prefix}_connectivity_matrix.json")
        data = {
            "n_electrodes": n_electrodes,
            "matrix": matrix.tolist(),
            "description": "Entry (i,j) = response probability of electrode j when electrode i is stimulated",
        }
        path.write_text(json.dumps(data, indent=2, default=str))
        logger.info("Saved connectivity matrix -> %s", path)
        return path

    def save_connectivity_entries(self, entries: List[ConnectivityEntry]) -> Path:
        path = Path(f"{self._prefix}_connectivity_entries.json")
        records = [asdict(e) for e in entries]
        path.write_text(json.dumps(records, indent=2, default=str))
        logger.info("Saved connectivity entries -> %s  (%d records)", path, len(records))
        return path


class Experiment:
    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        n_electrodes: int = 32,
        num_trials: int = 20,
        stim_amplitude_ua: float = 2.0,
        stim_duration_us: float = 200.0,
        stim_polarity: str = "PositiveFirst",
        stim_rate_hz: float = 1.0,
        post_electrode_wait_s: float = 5.0,
        response_window_ms: float = 50.0,
        trigger_key: int = 0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.n_electrodes = n_electrodes
        self.num_trials = num_trials
        self.stim_amplitude_ua = min(abs(stim_amplitude_ua), 4.0)
        self.stim_duration_us = min(abs(stim_duration_us), 400.0)
        self.stim_polarity_str = stim_polarity
        self.stim_rate_hz = stim_rate_hz
        self.post_electrode_wait_s = post_electrode_wait_s
        self.response_window_ms = response_window_ms
        self.trigger_key = trigger_key

        self.inter_stim_interval_s = 1.0 / stim_rate_hz

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._connectivity_entries: List[ConnectivityEntry] = []

        self._connectivity_hits = np.zeros((n_electrodes, n_electrodes), dtype=int)
        self._connectivity_trials = np.zeros((n_electrodes, n_electrodes), dtype=int)

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

            self._phase_connectivity_mapping()

            recording_stop = datetime_now()

            connectivity_matrix = self._compute_connectivity_matrix()

            results = self._compile_results(recording_start, recording_stop, connectivity_matrix)

            self._save_all(recording_start, recording_stop, connectivity_matrix)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _phase_connectivity_mapping(self) -> None:
        logger.info("Phase: cross-electrode connectivity mapping")

        polarity = StimPolarity.PositiveFirst if self.stim_polarity_str == "PositiveFirst" else StimPolarity.NegativeFirst

        for stim_elec in range(self.n_electrodes):
            logger.info("Stimulating electrode %d / %d", stim_elec, self.n_electrodes - 1)

            self._configure_stim_params(stim_elec, polarity)

            for trial in range(self.num_trials):
                stim_time = datetime_now()

                self._fire_trigger()

                self._stimulation_log.append(StimulationRecord(
                    electrode_idx=stim_elec,
                    amplitude_ua=self.stim_amplitude_ua,
                    duration_us=self.stim_duration_us,
                    polarity=self.stim_polarity_str,
                    timestamp_utc=stim_time.isoformat(),
                    trial_number=trial,
                    trigger_key=self.trigger_key,
                ))

                self._wait(self.response_window_ms / 1000.0)

                window_start = stim_time
                window_stop = stim_time + timedelta(milliseconds=self.response_window_ms)

                fs_name = self.experiment.exp_name
                try:
                    spike_df = self.database.get_spike_event(
                        window_start, window_stop, fs_name
                    )
                except Exception as exc:
                    logger.warning("Failed to fetch spike events for stim_elec=%d trial=%d: %s", stim_elec, trial, exc)
                    spike_df = pd.DataFrame()

                self._process_trial_spikes(stim_elec, trial, spike_df, stim_time)

                remaining_wait = self.inter_stim_interval_s - (self.response_window_ms / 1000.0)
                if remaining_wait > 0:
                    self._wait(remaining_wait)

            logger.info("Post-electrode wait: %.1f s", self.post_electrode_wait_s)
            self._wait(self.post_electrode_wait_s)

        logger.info("Connectivity mapping complete")

    def _configure_stim_params(self, electrode_idx: int, polarity: StimPolarity) -> None:
        stim = StimParam()
        stim.index = electrode_idx
        stim.enable = True
        stim.trigger_key = self.trigger_key
        stim.trigger_delay = 0
        stim.nb_pulse = 0
        stim.pulse_train_period = 10000
        stim.post_stim_ref_period = 1000.0
        stim.stim_shape = StimShape.Biphasic
        stim.polarity = polarity

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

    def _process_trial_spikes(
        self,
        stim_elec: int,
        trial: int,
        spike_df: pd.DataFrame,
        stim_time: datetime,
    ) -> None:
        responding_electrodes = set()

        if not spike_df.empty:
            channel_col = None
            for col in ["channel", "index", "electrode"]:
                if col in spike_df.columns:
                    channel_col = col
                    break

            if channel_col is not None:
                for _, row in spike_df.iterrows():
                    resp_elec = int(row[channel_col])
                    if resp_elec != stim_elec and 0 <= resp_elec < self.n_electrodes:
                        responding_electrodes.add(resp_elec)

        for resp_elec in range(self.n_electrodes):
            if resp_elec == stim_elec:
                continue

            had_response = resp_elec in responding_electrodes
            spike_count = 1 if had_response else 0

            self._connectivity_trials[stim_elec, resp_elec] += 1
            if had_response:
                self._connectivity_hits[stim_elec, resp_elec] += 1

            self._connectivity_entries.append(ConnectivityEntry(
                stim_electrode=stim_elec,
                response_electrode=resp_elec,
                trial=trial,
                had_response=had_response,
                spike_count=spike_count,
                timestamp_utc=stim_time.isoformat(),
            ))

    def _compute_connectivity_matrix(self) -> np.ndarray:
        matrix = np.zeros((self.n_electrodes, self.n_electrodes), dtype=float)
        for i in range(self.n_electrodes):
            for j in range(self.n_electrodes):
                if i == j:
                    matrix[i, j] = 0.0
                elif self._connectivity_trials[i, j] > 0:
                    matrix[i, j] = self._connectivity_hits[i, j] / self._connectivity_trials[i, j]
                else:
                    matrix[i, j] = 0.0
        return matrix

    def _save_all(
        self,
        recording_start: datetime,
        recording_stop: datetime,
        connectivity_matrix: np.ndarray,
    ) -> None:
        fs_name = getattr(self.experiment, "exp_name", "unknown")
        saver = DataSaver(self._output_dir, fs_name)

        saver.save_stimulation_log(self._stimulation_log)

        saver.save_connectivity_entries(self._connectivity_entries)

        saver.save_connectivity_matrix(connectivity_matrix, self.n_electrodes)

        try:
            spike_df = self.database.get_spike_event(
                recording_start, recording_stop, fs_name
            )
        except Exception as exc:
            logger.warning("Failed to fetch full spike events: %s", exc)
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

        total_stims = len(self._stimulation_log)
        total_spikes = len(spike_df) if not spike_df.empty else 0
        total_triggers = len(trigger_df) if not trigger_df.empty else 0

        non_zero_connections = int(np.sum(connectivity_matrix > 0))
        mean_response_prob = float(np.mean(connectivity_matrix[connectivity_matrix > 0])) if non_zero_connections > 0 else 0.0

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "n_electrodes": self.n_electrodes,
            "num_trials_per_electrode": self.num_trials,
            "stim_amplitude_ua": self.stim_amplitude_ua,
            "stim_duration_us": self.stim_duration_us,
            "stim_polarity": self.stim_polarity_str,
            "response_window_ms": self.response_window_ms,
            "total_stimulations": total_stims,
            "total_spike_events": total_spikes,
            "total_triggers": total_triggers,
            "non_zero_connections": non_zero_connections,
            "mean_response_probability": mean_response_prob,
            "charge_balance_check": self.stim_amplitude_ua * self.stim_duration_us,
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

        channel_col = None
        for col in ["channel", "index", "electrode"]:
            if col in spike_df.columns:
                channel_col = col
                break

        if channel_col is None:
            logger.warning("Cannot determine electrode column for waveform fetch")
            return waveform_records

        unique_electrodes = spike_df[channel_col].unique()
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
                    electrode_idx, exc
                )

        return waveform_records

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
        connectivity_matrix: np.ndarray,
    ) -> Dict[str, Any]:
        logger.info("Compiling results")

        non_zero_connections = int(np.sum(connectivity_matrix > 0))
        mean_response_prob = float(np.mean(connectivity_matrix[connectivity_matrix > 0])) if non_zero_connections > 0 else 0.0

        top_connections = []
        indices = np.argwhere(connectivity_matrix > 0)
        probs = [(i, j, connectivity_matrix[i, j]) for i, j in indices]
        probs_sorted = sorted(probs, key=lambda x: x[2], reverse=True)
        for i, j, prob in probs_sorted[:20]:
            top_connections.append({
                "stim_electrode": int(i),
                "response_electrode": int(j),
                "response_probability": float(prob),
            })

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "n_electrodes": self.n_electrodes,
            "num_trials_per_electrode": self.num_trials,
            "total_stimulations": len(self._stimulation_log),
            "non_zero_connections": non_zero_connections,
            "mean_response_probability": mean_response_prob,
            "top_connections": top_connections,
            "connectivity_matrix_shape": list(connectivity_matrix.shape),
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
