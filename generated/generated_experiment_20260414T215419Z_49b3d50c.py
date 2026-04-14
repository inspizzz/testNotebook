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

    def save_connectivity_matrix(self, matrix: List[List[float]], n_electrodes: int) -> Path:
        path = Path(f"{self._prefix}_connectivity_matrix.json")
        data = {
            "n_electrodes": n_electrodes,
            "matrix": matrix,
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
        stim_amplitude_ua: float = 2.0,
        stim_duration_us: float = 200.0,
        stim_polarity: str = "PositiveFirst",
        num_trials_per_electrode: int = 20,
        inter_stim_interval_s: float = 1.0,
        post_electrode_wait_s: float = 5.0,
        response_window_ms: float = 50.0,
        trigger_key: int = 0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.n_electrodes = n_electrodes
        self.stim_amplitude_ua = stim_amplitude_ua
        self.stim_duration_us = stim_duration_us
        self.stim_polarity_str = stim_polarity
        self.num_trials_per_electrode = num_trials_per_electrode
        self.inter_stim_interval_s = inter_stim_interval_s
        self.post_electrode_wait_s = post_electrode_wait_s
        self.response_window_ms = response_window_ms
        self.trigger_key = trigger_key

        if stim_polarity == "PositiveFirst":
            self.stim_polarity = StimPolarity.PositiveFirst
        else:
            self.stim_polarity = StimPolarity.NegativeFirst

        a1 = self.stim_amplitude_ua
        d1 = self.stim_duration_us
        a2 = a1
        d2 = d1
        assert abs(a1 * d1 - a2 * d2) < 1e-9, "Charge balance violated"

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._connectivity_entries: List[ConnectivityEntry] = []

        self._hit_counts: List[List[int]] = [
            [0] * n_electrodes for _ in range(n_electrodes)
        ]
        self._trial_counts: List[int] = [0] * n_electrodes

        self._recording_start: Optional[datetime] = None
        self._recording_stop: Optional[datetime] = None

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

            self._recording_start = datetime_now()

            self._phase_connectivity_mapping()

            self._recording_stop = datetime_now()

            connectivity_matrix = self._build_connectivity_matrix()

            results = self._compile_results(
                self._recording_start,
                self._recording_stop,
                connectivity_matrix,
            )

            self._save_all(self._recording_start, self._recording_stop, connectivity_matrix)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _phase_connectivity_mapping(self) -> None:
        logger.info(
            "Phase: connectivity mapping — %d electrodes x %d trials",
            self.n_electrodes,
            self.num_trials_per_electrode,
        )

        a1 = self.stim_amplitude_ua
        d1 = self.stim_duration_us
        a2 = a1
        d2 = d1

        for stim_elec in range(self.n_electrodes):
            logger.info("Stimulating electrode %d", stim_elec)

            stim = StimParam()
            stim.index = stim_elec
            stim.enable = True
            stim.trigger_key = self.trigger_key
            stim.trigger_delay = 0
            stim.nb_pulse = 0
            stim.pulse_train_period = 10000
            stim.post_stim_ref_period = 1000.0
            stim.stim_shape = StimShape.Biphasic
            stim.polarity = self.stim_polarity
            stim.phase_amplitude1 = a1
            stim.phase_duration1 = d1
            stim.phase_amplitude2 = a2
            stim.phase_duration2 = d2
            stim.enable_amp_settle = True
            stim.pre_stim_amp_settle = 0.0
            stim.post_stim_amp_settle = 1000.0
            stim.enable_charge_recovery = True
            stim.post_charge_recovery_on = 0.0
            stim.post_charge_recovery_off = 100.0
            stim.interphase_delay = 0.0

            self.intan.send_stimparam([stim])

            for trial in range(self.num_trials_per_electrode):
                stim_time = datetime_now()

                pattern = np.zeros(16, dtype=np.uint8)
                pattern[self.trigger_key] = 1
                self.trigger_controller.send(pattern)
                self._wait(0.01)
                pattern[self.trigger_key] = 0
                self.trigger_controller.send(pattern)

                self._stimulation_log.append(StimulationRecord(
                    electrode_idx=stim_elec,
                    amplitude_ua=a1,
                    duration_us=d1,
                    polarity=self.stim_polarity_str,
                    timestamp_utc=stim_time.isoformat(),
                    trial_number=trial,
                    trigger_key=self.trigger_key,
                ))

                self._wait(self.response_window_ms / 1000.0)

                window_start = stim_time
                window_stop = stim_time + timedelta(milliseconds=self.response_window_ms + 20)

                try:
                    spike_df = self.database.get_spike_event(
                        window_start,
                        window_stop,
                        self.experiment.exp_name,
                    )
                except Exception as exc:
                    logger.warning("DB query failed for stim_elec=%d trial=%d: %s", stim_elec, trial, exc)
                    spike_df = pd.DataFrame()

                for resp_elec in range(self.n_electrodes):
                    if resp_elec == stim_elec:
                        continue

                    spike_count = 0
                    had_response = False

                    if not spike_df.empty:
                        channel_col = None
                        for col in spike_df.columns:
                            if col.lower() in ("channel", "index", "electrode"):
                                channel_col = col
                                break
                        if channel_col is not None:
                            elec_spikes = spike_df[spike_df[channel_col] == resp_elec]
                            if "Time" in elec_spikes.columns:
                                elec_spikes = elec_spikes[
                                    elec_spikes["Time"] >= window_start
                                ]
                                elec_spikes = elec_spikes[
                                    elec_spikes["Time"] <= window_stop
                                ]
                            spike_count = len(elec_spikes)
                            had_response = spike_count > 0

                    if had_response:
                        self._hit_counts[stim_elec][resp_elec] += 1

                    self._connectivity_entries.append(ConnectivityEntry(
                        stim_electrode=stim_elec,
                        response_electrode=resp_elec,
                        trial=trial,
                        had_response=had_response,
                        spike_count=spike_count,
                        timestamp_utc=stim_time.isoformat(),
                    ))

                self._trial_counts[stim_elec] += 1

                remaining = self.num_trials_per_electrode - trial - 1
                if remaining > 0:
                    wait_time = self.inter_stim_interval_s - (self.response_window_ms / 1000.0) - 0.01
                    if wait_time > 0:
                        self._wait(wait_time)

            logger.info(
                "Electrode %d done (%d trials). Waiting %0.1fs before next electrode.",
                stim_elec,
                self.num_trials_per_electrode,
                self.post_electrode_wait_s,
            )
            self._wait(self.post_electrode_wait_s)

    def _build_connectivity_matrix(self) -> List[List[float]]:
        matrix = []
        for i in range(self.n_electrodes):
            row = []
            n = self._trial_counts[i]
            for j in range(self.n_electrodes):
                if i == j:
                    row.append(0.0)
                elif n > 0:
                    row.append(self._hit_counts[i][j] / n)
                else:
                    row.append(0.0)
            matrix.append(row)
        return matrix

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
        connectivity_matrix: List[List[float]],
    ) -> Dict[str, Any]:
        logger.info("Compiling results")

        strong_connections = []
        for i in range(self.n_electrodes):
            for j in range(self.n_electrodes):
                if i != j and connectivity_matrix[i][j] > 0.5:
                    strong_connections.append({
                        "stim_electrode": i,
                        "response_electrode": j,
                        "response_probability": connectivity_matrix[i][j],
                    })

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "n_electrodes": self.n_electrodes,
            "num_trials_per_electrode": self.num_trials_per_electrode,
            "stim_amplitude_ua": self.stim_amplitude_ua,
            "stim_duration_us": self.stim_duration_us,
            "stim_polarity": self.stim_polarity_str,
            "response_window_ms": self.response_window_ms,
            "total_stimulations": len(self._stimulation_log),
            "strong_connections_count": len(strong_connections),
            "strong_connections": strong_connections[:50],
        }

        return summary

    def _save_all(
        self,
        recording_start: datetime,
        recording_stop: datetime,
        connectivity_matrix: List[List[float]],
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

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "n_electrodes": self.n_electrodes,
            "num_trials_per_electrode": self.num_trials_per_electrode,
            "stim_amplitude_ua": self.stim_amplitude_ua,
            "stim_duration_us": self.stim_duration_us,
            "stim_polarity": self.stim_polarity_str,
            "response_window_ms": self.response_window_ms,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "total_connectivity_entries": len(self._connectivity_entries),
        }
        saver.save_summary(summary)

        waveform_records = self._fetch_spike_waveforms(
            fs_name, spike_df, recording_start, recording_stop
        )
        saver.save_spike_waveforms(waveform_records)

        saver.save_connectivity_matrix(connectivity_matrix, self.n_electrodes)
        saver.save_connectivity_entries(self._connectivity_entries)

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
        for col in spike_df.columns:
            if col.lower() in ("channel", "index", "electrode"):
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
