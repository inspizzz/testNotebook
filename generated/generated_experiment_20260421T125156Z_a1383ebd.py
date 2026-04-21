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
    datetime_now,
    wait,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StimulationRecord:
    """A single stimulation event for the persistence log."""
    electrode_idx: int
    amplitude_ua: float
    duration_us: float
    polarity: str
    timestamp_utc: str
    trigger_key: int = 0
    trial_index: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


class DataSaver:
    """Handles persistence of stimulation records, spike events, and triggers."""

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
    Maximum-amplitude repeated stimulation experiment on the most responsive electrode.

    Based on scan results, electrode 17 stimulating electrode 18 shows the highest
    response rate (0.92) with the most reliable primary pathway (pair_01_mode_1,
    response_rate=0.88, temporal_stability=1.0). We stimulate electrode 17 at the
    maximum allowed amplitude (4.0 uA) and maximum duration (400 us), charge-balanced,
    repeated 100 times at 1 Hz.

    Charge balance: A1*D1 = A2*D2 => 4.0*400 = 4.0*400 (both phases equal).
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        stim_electrode: int = 17,
        resp_electrode: int = 18,
        amplitude_ua: float = 4.0,
        duration_us: float = 400.0,
        polarity: str = "PositiveFirst",
        num_trials: int = 100,
        isi_seconds: float = 1.0,
        trigger_key: int = 0,
        post_stim_wait_s: float = 0.5,
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
        self.num_trials = num_trials
        self.isi_seconds = isi_seconds
        self.trigger_key = trigger_key
        self.post_stim_wait_s = post_stim_wait_s

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._trial_spike_counts: List[int] = []

    def _wait(self, seconds: float) -> None:
        wait(seconds)

    def run(self) -> Dict[str, Any]:
        """Execute the full experiment and return a results dict."""
        recording_start = None
        recording_stop = None
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

            self._configure_stimulation()

            logger.info(
                "Starting %d trials: electrode %d -> %d, amplitude=%.1f uA, duration=%.0f us, polarity=%s, ISI=%.1f s",
                self.num_trials,
                self.stim_electrode,
                self.resp_electrode,
                self.amplitude_ua,
                self.duration_us,
                self.polarity_str,
                self.isi_seconds,
            )

            self._phase_repeated_stimulation()

            recording_stop = datetime_now()

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            if recording_start is None:
                recording_start = datetime_now()
            if recording_stop is None:
                recording_stop = datetime_now()
            try:
                self._save_all(recording_start, recording_stop)
            except Exception as save_exc:
                logger.error("Failed to save data after error: %s", save_exc)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _configure_stimulation(self) -> None:
        """Configure the stimulation parameters on the Intan hardware."""
        polarity_enum = (
            StimPolarity.PositiveFirst
            if self.polarity_str == "PositiveFirst"
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
        stim.polarity = polarity_enum

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

        logger.info(
            "Sending stim params: electrode=%d, A1=%.1f uA, D1=%.0f us, A2=%.1f uA, D2=%.0f us (charge balance: %.1f = %.1f)",
            stim.index,
            stim.phase_amplitude1,
            stim.phase_duration1,
            stim.phase_amplitude2,
            stim.phase_duration2,
            stim.phase_amplitude1 * stim.phase_duration1,
            stim.phase_amplitude2 * stim.phase_duration2,
        )

        self.intan.send_stimparam([stim])
        logger.info("Stimulation parameters configured")

    def _phase_repeated_stimulation(self) -> None:
        """Deliver num_trials stimulations at 1 Hz and record responses."""
        logger.info("Phase: repeated stimulation (%d trials)", self.num_trials)

        for trial_idx in range(1, self.num_trials + 1):
            trial_start = datetime_now()

            pattern = np.zeros(16, dtype=np.uint8)
            pattern[self.trigger_key] = 1
            self.trigger_controller.send(pattern)

            stim_time = datetime_now()

            self._stimulation_log.append(StimulationRecord(
                electrode_idx=self.stim_electrode,
                amplitude_ua=self.amplitude_ua,
                duration_us=self.duration_us,
                polarity=self.polarity_str,
                timestamp_utc=stim_time.isoformat(),
                trigger_key=self.trigger_key,
                trial_index=trial_idx,
            ))

            self._wait(0.05)

            pattern_off = np.zeros(16, dtype=np.uint8)
            self.trigger_controller.send(pattern_off)

            self._wait(self.post_stim_wait_s)

            query_stop = datetime_now()
            query_start = query_stop - timedelta(seconds=self.post_stim_wait_s + 0.1)

            try:
                spike_df = self.database.get_spike_event(
                    query_start,
                    query_stop,
                    self.experiment.exp_name,
                )
                if not spike_df.empty:
                    resp_spikes = spike_df[spike_df["channel"] == self.resp_electrode]
                    spike_count = len(resp_spikes)
                else:
                    spike_count = 0
            except Exception as exc:
                logger.warning("Trial %d: spike query failed: %s", trial_idx, exc)
                spike_count = 0

            self._trial_spike_counts.append(spike_count)

            elapsed = (datetime_now() - trial_start).total_seconds()
            remaining_wait = self.isi_seconds - elapsed
            if remaining_wait > 0:
                self._wait(remaining_wait)

            if trial_idx % 10 == 0:
                logger.info(
                    "Trial %d/%d complete. Spikes in last trial: %d. Cumulative response rate: %.2f",
                    trial_idx,
                    self.num_trials,
                    spike_count,
                    sum(1 for c in self._trial_spike_counts if c > 0) / len(self._trial_spike_counts),
                )

        logger.info("Repeated stimulation phase complete")

    def _save_all(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> None:
        """Persist all raw experiment data for downstream analysis."""
        fs_name = getattr(self.experiment, "exp_name", "unknown") if self.experiment else "unknown"
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

        responding_trials = sum(1 for c in self._trial_spike_counts if c > 0)
        response_rate = responding_trials / len(self._trial_spike_counts) if self._trial_spike_counts else 0.0
        total_spikes = sum(self._trial_spike_counts)

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
            "num_trials": self.num_trials,
            "isi_seconds": self.isi_seconds,
            "charge_balance_check": self.amplitude_ua * self.duration_us,
            "total_stimulations": len(self._stimulation_log),
            "responding_trials": responding_trials,
            "response_rate": response_rate,
            "total_spike_events_resp_electrode": total_spikes,
            "total_spike_events_db": len(spike_df),
            "total_triggers": len(trigger_df),
            "trial_spike_counts": self._trial_spike_counts,
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
        """Fetch raw spike waveform data for each electrode that had spikes."""
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
                    electrode_idx, exc
                )

        return waveform_records

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        """Assemble a summary dict to be returned from run()."""
        logger.info("Compiling results")

        responding_trials = sum(1 for c in self._trial_spike_counts if c > 0)
        response_rate = responding_trials / len(self._trial_spike_counts) if self._trial_spike_counts else 0.0
        total_spikes = sum(self._trial_spike_counts)
        mean_spikes = total_spikes / len(self._trial_spike_counts) if self._trial_spike_counts else 0.0

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
            "num_trials": self.num_trials,
            "isi_seconds": self.isi_seconds,
            "responding_trials": responding_trials,
            "response_rate": response_rate,
            "total_spikes_resp_electrode": total_spikes,
            "mean_spikes_per_trial": mean_spikes,
            "trial_spike_counts": self._trial_spike_counts,
        }

        return summary

    def _cleanup(self) -> None:
        """Release all hardware resources. Called from the finally block."""
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
