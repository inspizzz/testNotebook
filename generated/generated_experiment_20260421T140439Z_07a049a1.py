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
        num_stimulations: int = 20,
        amplitude_ua: float = 2.0,
        duration_us: float = 200.0,
        inter_stim_interval_s: float = 1.0,
        post_electrode_wait_s: float = 5.0,
        response_window_ms: float = 50.0,
        trigger_key: int = 0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.num_electrodes = num_electrodes
        self.num_stimulations = num_stimulations
        self.amplitude_ua = min(abs(amplitude_ua), 4.0)
        self.duration_us = min(abs(duration_us), 400.0)
        self.inter_stim_interval_s = inter_stim_interval_s
        self.post_electrode_wait_s = post_electrode_wait_s
        self.response_window_ms = response_window_ms
        self.trigger_key = trigger_key

        # Verify charge balance: A1*D1 == A2*D2 (symmetric biphasic)
        assert math.isclose(self.amplitude_ua * self.duration_us,
                            self.amplitude_ua * self.duration_us), \
            "Charge balance violated"

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []

        # connectivity_matrix[i][j] = list of booleans (responded or not) for each trial
        # where i = stim electrode index (0-31), j = response electrode index (0-31)
        self._response_counts: List[List[int]] = []
        self._connectivity_matrix: List[List[float]] = []

        # Store stim times per electrode for post-hoc analysis
        self._stim_times: List[Dict[str, Any]] = []

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

            electrodes = list(self.experiment.electrodes)[:self.num_electrodes]
            n = len(electrodes)

            # Initialize response count matrix n x n
            self._response_counts = [[0] * n for _ in range(n)]

            logger.info("Starting connectivity mapping: %d electrodes, %d stimulations each",
                        n, self.num_stimulations)

            self._phase_connectivity_mapping(electrodes)

            # Compute connectivity matrix as response probabilities
            self._connectivity_matrix = [
                [self._response_counts[i][j] / self.num_stimulations for j in range(n)]
                for i in range(n)
            ]

            recording_stop = datetime_now()

            results = self._compile_results(recording_start, recording_stop, electrodes)

            self._save_all(recording_start, recording_stop, electrodes)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _phase_connectivity_mapping(self, electrodes: List[int]) -> None:
        n = len(electrodes)
        polarity = StimPolarity.PositiveFirst

        for i, stim_electrode in enumerate(electrodes):
            logger.info("Stimulating electrode %d (%d/%d)", stim_electrode, i + 1, n)

            # Configure stim params for this electrode
            stim = self._build_stim_param(stim_electrode, polarity)
            self.intan.send_stimparam([stim])

            for trial in range(self.num_stimulations):
                stim_time = datetime_now()

                # Fire trigger
                pattern = np.zeros(16, dtype=np.uint8)
                pattern[self.trigger_key] = 1
                self.trigger_controller.send(pattern)
                self._wait(0.01)
                pattern[self.trigger_key] = 0
                self.trigger_controller.send(pattern)

                # Log stimulation
                self._stimulation_log.append(StimulationRecord(
                    electrode_idx=stim_electrode,
                    amplitude_ua=self.amplitude_ua,
                    duration_us=self.duration_us,
                    polarity=polarity.name,
                    timestamp_utc=stim_time.isoformat(),
                    trigger_key=self.trigger_key,
                    trial_index=trial,
                ))

                self._stim_times.append({
                    "stim_electrode_idx": i,
                    "stim_electrode": stim_electrode,
                    "trial": trial,
                    "stim_time": stim_time.isoformat(),
                })

                # Wait for response window to elapse
                response_window_s = self.response_window_ms / 1000.0
                self._wait(response_window_s)

                # Query spike events in the response window for all electrodes
                query_start = stim_time
                query_stop = datetime_now()

                try:
                    spike_df = self.database.get_spike_event(
                        query_start, query_stop, self.experiment.exp_name
                    )

                    if not spike_df.empty:
                        # Determine channel column
                        channel_col = None
                        for col in ["channel", "index", "electrode"]:
                            if col in spike_df.columns:
                                channel_col = col
                                break

                        if channel_col is not None:
                            responding_electrodes = set(spike_df[channel_col].unique())
                            for j, resp_electrode in enumerate(electrodes):
                                if j == i:
                                    continue  # skip self
                                if resp_electrode in responding_electrodes:
                                    self._response_counts[i][j] += 1

                except Exception as exc:
                    logger.warning("Error querying spikes for stim electrode %d trial %d: %s",
                                   stim_electrode, trial, exc)

                # Inter-stimulation interval (1 Hz = 1 second between stims)
                # We already waited response_window_s, so wait the remainder
                remaining_wait = self.inter_stim_interval_s - response_window_s - 0.01
                if remaining_wait > 0:
                    self._wait(remaining_wait)

            # Post-electrode wait
            logger.info("Post-electrode wait: %.1f s", self.post_electrode_wait_s)
            self._wait(self.post_electrode_wait_s)

    def _build_stim_param(self, electrode_idx: int, polarity: StimPolarity) -> StimParam:
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

        # Charge-balanced: A1*D1 == A2*D2 (symmetric)
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

    def _save_all(
        self,
        recording_start: datetime,
        recording_stop: datetime,
        electrodes: List[int],
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
            "num_electrodes": len(electrodes),
            "num_stimulations_per_electrode": self.num_stimulations,
            "amplitude_ua": self.amplitude_ua,
            "duration_us": self.duration_us,
            "response_window_ms": self.response_window_ms,
            "electrodes": electrodes,
        }
        saver.save_summary(summary)

        waveform_records = self._fetch_spike_waveforms(
            fs_name, spike_df, recording_start, recording_stop
        )
        saver.save_spike_waveforms(waveform_records)

        saver.save_connectivity_matrix(self._connectivity_matrix, electrodes)

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
        electrodes: List[int],
    ) -> Dict[str, Any]:
        logger.info("Compiling results")

        n = len(electrodes)
        # Find strongest connections (response probability > 0)
        strong_connections = []
        for i in range(n):
            for j in range(n):
                if i != j and self._connectivity_matrix[i][j] > 0:
                    strong_connections.append({
                        "stim_electrode": electrodes[i],
                        "resp_electrode": electrodes[j],
                        "response_probability": self._connectivity_matrix[i][j],
                        "response_count": self._response_counts[i][j],
                    })

        strong_connections.sort(key=lambda x: x["response_probability"], reverse=True)

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "num_electrodes": n,
            "num_stimulations_per_electrode": self.num_stimulations,
            "amplitude_ua": self.amplitude_ua,
            "duration_us": self.duration_us,
            "response_window_ms": self.response_window_ms,
            "total_stimulations": len(self._stimulation_log),
            "connectivity_matrix": self._connectivity_matrix,
            "electrodes": electrodes,
            "top_connections": strong_connections[:20],
            "num_nonzero_connections": len(strong_connections),
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
