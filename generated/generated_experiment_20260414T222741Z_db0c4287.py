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
    Zero-amplitude stimulation experiment: sends 0 uA pulses (no actual
    stimulation) and records neural responses for 10 minutes.

    Since 0 uA amplitude is not a valid stimulation parameter for the
    hardware (it would produce no trigger event), we implement this by
    sending trigger events without configuring any stimulation electrode,
    effectively recording spontaneous activity during the trigger windows.
    The trigger fires at regular intervals and spike events are recorded
    around each trigger time, allowing comparison of evoked (post-trigger)
    vs baseline windows within the same recording.

    Charge balance is maintained trivially: 0 * d = 0 * d for any d.
    We use amplitude=0.0 conceptually; in practice we send triggers only
    (no stim params) so no hardware safety violation occurs.
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        experiment_duration_s: float = 600.0,
        inter_trial_interval_s: float = 2.0,
        response_window_s: float = 0.1,
        trigger_key: int = 0,
        fs_name: str = "fs_1",
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.experiment_duration_s = experiment_duration_s
        self.inter_trial_interval_s = inter_trial_interval_s
        self.response_window_s = response_window_s
        self.trigger_key = trigger_key
        self.fs_name = fs_name

        self.np_experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._trial_results: List[Dict[str, Any]] = []
        self._total_trials: int = 0

    def _wait(self, seconds: float) -> None:
        wait(seconds)

    def run(self) -> Dict[str, Any]:
        try:
            logger.info("Initialising hardware connections")

            from neuroplatform import Experiment as NeuroPlatformExperiment

            self.np_experiment = NeuroPlatformExperiment(self.token)
            self.trigger_controller = TriggerController(email=self.booking_email)
            self.intan = IntanSofware()
            self.database = Database()

            logger.info("Experiment: %s", self.np_experiment.exp_name)
            logger.info("Electrodes: %s", self.np_experiment.electrodes)

            self.fs_name = self.np_experiment.exp_name

            if not self.np_experiment.start():
                logger.error("Failed to start experiment")
                return {"status": "failed", "error": "Could not start experiment"}

            recording_start = datetime_now()
            logger.info("Recording started at %s", recording_start.isoformat())

            self._phase_zero_amplitude_stimulation(recording_start)

            recording_stop = datetime_now()
            logger.info("Recording stopped at %s", recording_stop.isoformat())

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _phase_zero_amplitude_stimulation(self, recording_start: datetime) -> None:
        """
        Send trigger pulses at regular intervals for 10 minutes.
        No actual electrical stimulation is applied (0 uA amplitude).
        Spike events are recorded around each trigger to capture any
        neural activity time-locked to the trigger event.
        """
        logger.info("Phase: zero-amplitude stimulation (10 minutes)")

        experiment_end = recording_start.timestamp() + self.experiment_duration_s
        trial_index = 0

        while datetime_now().timestamp() < experiment_end:
            trial_start_time = datetime_now()

            logger.debug("Trial %d: firing trigger (0 uA - no stimulation)", trial_index)

            pattern = np.zeros(16, dtype=np.uint8)
            pattern[self.trigger_key] = 1
            self.trigger_controller.send(pattern)

            self._wait(0.01)

            pattern_off = np.zeros(16, dtype=np.uint8)
            self.trigger_controller.send(pattern_off)

            self._stimulation_log.append(StimulationRecord(
                electrode_idx=-1,
                amplitude_ua=0.0,
                duration_us=0.0,
                timestamp_utc=trial_start_time.isoformat(),
                trigger_key=self.trigger_key,
                trial_index=trial_index,
                extra={
                    "note": "zero_amplitude_trigger_only",
                    "charge_balance": "0.0 * 0.0 == 0.0 * 0.0",
                }
            ))

            self._wait(self.response_window_s)

            query_stop = datetime_now()
            query_start = query_stop - timedelta(seconds=self.response_window_s + 0.05)

            try:
                spike_df = self.database.get_spike_event(
                    query_start, query_stop, self.fs_name
                )
                spike_count = len(spike_df) if not spike_df.empty else 0
            except Exception as exc:
                logger.warning("Failed to query spike events for trial %d: %s", trial_index, exc)
                spike_count = 0

            self._trial_results.append({
                "trial_index": trial_index,
                "trigger_time_utc": trial_start_time.isoformat(),
                "spike_count_in_window": spike_count,
                "response_window_s": self.response_window_s,
            })

            trial_index += 1
            self._total_trials = trial_index

            elapsed = datetime_now().timestamp() - recording_start.timestamp()
            remaining = self.experiment_duration_s - elapsed

            if remaining <= 0:
                break

            wait_time = min(
                self.inter_trial_interval_s - self.response_window_s - 0.01,
                remaining
            )
            if wait_time > 0:
                self._wait(wait_time)

        logger.info("Zero-amplitude stimulation phase complete. Trials: %d", self._total_trials)

    def _save_all(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> None:
        fs_name = self.fs_name
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

        total_spikes_in_windows = sum(
            r.get("spike_count_in_window", 0) for r in self._trial_results
        )

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "testing": self.testing,
            "total_trials": self._total_trials,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events_db": len(spike_df) if not spike_df.empty else 0,
            "total_triggers_db": len(trigger_df) if not trigger_df.empty else 0,
            "total_spikes_in_response_windows": total_spikes_in_windows,
            "inter_trial_interval_s": self.inter_trial_interval_s,
            "response_window_s": self.response_window_s,
            "amplitude_ua": 0.0,
            "note": "Zero-amplitude experiment: triggers fired without electrical stimulation",
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
        if spike_df is None or spike_df.empty:
            return waveform_records

        electrode_col = None
        for col in spike_df.columns:
            if col.lower() in ("channel", "index", "electrode", "electrode_idx"):
                electrode_col = col
                break
            if "electrode" in col.lower() or "channel" in col.lower():
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
                if raw_df is not None and not raw_df.empty:
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
    ) -> Dict[str, Any]:
        logger.info("Compiling results")

        total_spikes = sum(r.get("spike_count_in_window", 0) for r in self._trial_results)
        mean_spikes = total_spikes / self._total_trials if self._total_trials > 0 else 0.0

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.np_experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "total_trials": self._total_trials,
            "amplitude_ua": 0.0,
            "total_spikes_in_response_windows": total_spikes,
            "mean_spikes_per_trial": mean_spikes,
            "note": (
                "Zero-amplitude experiment: no electrical stimulation was applied. "
                "Triggers were fired at regular intervals and neural activity was "
                "recorded in post-trigger windows to characterise evoked-window "
                "activity in the absence of stimulation."
            ),
        }

        return summary

    def _cleanup(self) -> None:
        logger.info("Cleaning up resources")

        if self.np_experiment is not None:
            try:
                self.np_experiment.stop()
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
