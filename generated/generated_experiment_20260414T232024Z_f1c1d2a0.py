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
    Experiment that sends zero-amplitude (no-op) stimulation pulses and records
    neural responses for exactly 10 minutes. Since amplitude is 0 uA, no actual
    electrical stimulation is delivered, but triggers are fired and spike events
    are recorded to characterise evoked (trigger-aligned) activity.

    Charge balance is maintained: amplitude1 * duration1 == amplitude2 * duration2
    (0.0 * 200.0 == 0.0 * 200.0).

    The inter-trial interval is set so that approximately one trigger is fired
    every 5 seconds (0.2 Hz), yielding ~120 trials over 10 minutes.
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        stim_electrode: int = 0,
        stim_amplitude_ua: float = 0.0,
        stim_duration_us: float = 200.0,
        stim_polarity: str = "NegativeFirst",
        trigger_key: int = 0,
        inter_trial_interval_s: float = 5.0,
        experiment_duration_s: float = 600.0,
        post_stim_wait_s: float = 0.5,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.stim_electrode = stim_electrode
        self.stim_amplitude_ua = stim_amplitude_ua
        self.stim_duration_us = stim_duration_us
        self.stim_polarity = (
            StimPolarity.NegativeFirst
            if stim_polarity == "NegativeFirst"
            else StimPolarity.PositiveFirst
        )
        self.trigger_key = trigger_key
        self.inter_trial_interval_s = inter_trial_interval_s
        self.experiment_duration_s = experiment_duration_s
        self.post_stim_wait_s = post_stim_wait_s

        self.np_experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._trial_results: List[Dict[str, Any]] = []

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

            if not self.np_experiment.start():
                logger.error("Failed to start experiment")
                return {"status": "failed", "error": "Could not start experiment"}

            recording_start = datetime_now()
            logger.info("Recording started at %s", recording_start.isoformat())

            self._phase_null_stimulation(recording_start)

            recording_stop = datetime_now()
            logger.info("Recording stopped at %s", recording_stop.isoformat())

            self._save_all(recording_start, recording_stop)

            results = self._compile_results(recording_start, recording_stop)
            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _phase_null_stimulation(self, recording_start: datetime) -> None:
        """
        Fire zero-amplitude triggers at a fixed inter-trial interval for the
        full experiment duration (10 minutes). Because amplitude is 0 uA, no
        charge is injected. Charge balance holds trivially: 0*200 == 0*200.
        """
        logger.info("Phase: null stimulation (0 uA) for %.0f s", self.experiment_duration_s)

        amplitude_ua = self.stim_amplitude_ua
        duration_us = self.stim_duration_us

        stim = StimParam()
        stim.index = self.stim_electrode
        stim.enable = True
        stim.trigger_key = self.trigger_key
        stim.trigger_delay = 0
        stim.nb_pulse = 0
        stim.pulse_train_period = 10000
        stim.post_stim_ref_period = 1000.0
        stim.stim_shape = StimShape.Biphasic
        stim.polarity = self.stim_polarity
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
        logger.info(
            "StimParam configured: electrode=%d, A1=%.2f uA, D1=%.1f us, A2=%.2f uA, D2=%.1f us",
            stim.index, stim.phase_amplitude1, stim.phase_duration1,
            stim.phase_amplitude2, stim.phase_duration2,
        )

        trial_idx = 0
        while True:
            elapsed = (datetime_now() - recording_start).total_seconds()
            if elapsed >= self.experiment_duration_s:
                break

            trial_start = datetime_now()

            pattern = np.zeros(16, dtype=np.uint8)
            pattern[self.trigger_key] = 1
            self.trigger_controller.send(pattern)
            self._wait(0.05)
            pattern[self.trigger_key] = 0
            self.trigger_controller.send(pattern)

            stim_time = datetime_now()
            self._stimulation_log.append(StimulationRecord(
                electrode_idx=self.stim_electrode,
                amplitude_ua=amplitude_ua,
                duration_us=duration_us,
                timestamp_utc=stim_time.isoformat(),
                trigger_key=self.trigger_key,
                extra={"trial_idx": trial_idx},
            ))

            self._wait(self.post_stim_wait_s)

            query_stop = datetime_now()
            query_start = query_stop - timedelta(seconds=self.post_stim_wait_s + 0.1)
            try:
                spike_df = self.database.get_spike_event(
                    query_start, query_stop, self.np_experiment.exp_name
                )
                n_spikes = len(spike_df) if not spike_df.empty else 0
            except Exception as exc:
                logger.warning("Spike query failed on trial %d: %s", trial_idx, exc)
                n_spikes = 0

            self._trial_results.append({
                "trial_idx": trial_idx,
                "stim_time_utc": stim_time.isoformat(),
                "n_spikes_in_window": n_spikes,
            })

            logger.info(
                "Trial %d | elapsed=%.1f s | spikes_in_window=%d",
                trial_idx, elapsed, n_spikes,
            )

            trial_idx += 1

            elapsed_after = (datetime_now() - recording_start).total_seconds()
            remaining_in_session = self.experiment_duration_s - elapsed_after
            if remaining_in_session <= 0:
                break

            time_since_trial_start = (datetime_now() - trial_start).total_seconds()
            sleep_time = self.inter_trial_interval_s - time_since_trial_start
            if sleep_time > 0:
                self._wait(sleep_time)

        logger.info("Null stimulation phase complete. Total trials: %d", trial_idx)

    def _save_all(self, recording_start: datetime, recording_stop: datetime) -> None:
        fs_name = getattr(self.np_experiment, "exp_name", "unknown")
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
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df) if not spike_df.empty else 0,
            "total_triggers": len(trigger_df) if not trigger_df.empty else 0,
            "total_trials": len(self._trial_results),
            "stim_electrode": self.stim_electrode,
            "stim_amplitude_ua": self.stim_amplitude_ua,
            "stim_duration_us": self.stim_duration_us,
            "experiment_duration_s": self.experiment_duration_s,
            "inter_trial_interval_s": self.inter_trial_interval_s,
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
                    electrode_idx, exc,
                )

        return waveform_records

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        logger.info("Compiling results")
        total_spikes = sum(t.get("n_spikes_in_window", 0) for t in self._trial_results)
        n_trials = len(self._trial_results)
        mean_spikes_per_trial = total_spikes / n_trials if n_trials > 0 else 0.0

        return {
            "status": "completed",
            "experiment_name": self.np_experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "total_trials": n_trials,
            "total_stimulations": len(self._stimulation_log),
            "stim_amplitude_ua": self.stim_amplitude_ua,
            "stim_duration_us": self.stim_duration_us,
            "stim_electrode": self.stim_electrode,
            "total_spikes_in_windows": total_spikes,
            "mean_spikes_per_trial": mean_spikes_per_trial,
            "trial_results": self._trial_results,
        }

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
