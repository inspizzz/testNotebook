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
    """A single stimulation event for the persistence log."""
    electrode_idx: int
    amplitude_ua: float
    duration_us: float
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
    Experiment: Zero-amplitude stimulation with evoked response recording.

    Sends charge-balanced biphasic pulses with 0 uA amplitude (no actual
    stimulation) at regular intervals for exactly 10 minutes, recording
    all neural activity in the post-trigger window. This serves as a
    baseline/control condition to characterise spontaneous activity
    time-locked to trigger events, without any genuine evoked drive.

    Since amplitude is 0 uA, charge balance is trivially satisfied:
      0 * duration1 == 0 * duration2.

    The stimulation parameters use the most responsive electrode pairs
    identified in the parameter scan (electrodes 17->18, 21->19, 21->22,
    7->6) as the trigger source electrodes, cycling through them.
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        experiment_duration_s: float = 600.0,
        inter_trial_interval_s: float = 10.0,
        stim_amplitude_ua: float = 0.0,
        stim_duration_us: float = 400.0,
        post_stim_wait_s: float = 0.1,
        recording_window_s: float = 0.5,
        trigger_key: int = 0,
        stim_electrodes: tuple = (17, 21, 21, 7),
        resp_electrodes: tuple = (18, 19, 22, 6),
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.experiment_duration_s = experiment_duration_s
        self.inter_trial_interval_s = inter_trial_interval_s
        self.stim_amplitude_ua = stim_amplitude_ua
        self.stim_duration_us = min(abs(stim_duration_us), 400.0)
        self.post_stim_wait_s = post_stim_wait_s
        self.recording_window_s = recording_window_s
        self.trigger_key = trigger_key
        self.stim_electrodes = list(stim_electrodes)
        self.resp_electrodes = list(resp_electrodes)

        self.np_experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._trial_results: List[Dict[str, Any]] = []

    def _wait(self, seconds: float) -> None:
        wait(seconds)

    def run(self) -> Dict[str, Any]:
        """Execute the full experiment and return a results dict."""
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
        Main experiment phase: deliver zero-amplitude triggers for 10 minutes.

        Cycles through the identified stim electrode pairs, sending a
        charge-balanced biphasic pulse with 0 uA amplitude. The trigger
        fires so that the database records the event timestamp, but no
        actual current is injected into the tissue.
        """
        logger.info("Phase: zero-amplitude stimulation for %.0f seconds", self.experiment_duration_s)

        n_pairs = len(self.stim_electrodes)
        trial_index = 0

        while True:
            elapsed = (datetime_now() - recording_start).total_seconds()
            if elapsed >= self.experiment_duration_s:
                logger.info("Experiment duration reached after %d trials", trial_index)
                break

            pair_idx = trial_index % n_pairs
            stim_electrode = self.stim_electrodes[pair_idx]
            resp_electrode = self.resp_electrodes[pair_idx]

            logger.info(
                "Trial %d | elapsed=%.1fs | stim_electrode=%d | resp_electrode=%d | amplitude=%.2f uA",
                trial_index, elapsed, stim_electrode, resp_electrode, self.stim_amplitude_ua,
            )

            spike_df = self._stimulate_and_record(
                electrode_idx=stim_electrode,
                amplitude_ua=self.stim_amplitude_ua,
                duration_us=self.stim_duration_us,
                polarity=StimPolarity.PositiveFirst,
                trigger_key=self.trigger_key,
                post_stim_wait_s=self.post_stim_wait_s,
                recording_window_s=self.recording_window_s,
                trial_index=trial_index,
            )

            n_spikes = len(spike_df) if not spike_df.empty else 0
            self._trial_results.append({
                "trial_index": trial_index,
                "stim_electrode": stim_electrode,
                "resp_electrode": resp_electrode,
                "amplitude_ua": self.stim_amplitude_ua,
                "duration_us": self.stim_duration_us,
                "elapsed_s": elapsed,
                "n_spikes_in_window": n_spikes,
            })

            trial_index += 1

            elapsed_after = (datetime_now() - recording_start).total_seconds()
            remaining = self.experiment_duration_s - elapsed_after
            if remaining <= 0:
                break

            wait_time = min(self.inter_trial_interval_s, remaining)
            self._wait(wait_time)

        logger.info("Zero-amplitude stimulation phase complete. Total trials: %d", trial_index)

    def _stimulate_and_record(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity = StimPolarity.PositiveFirst,
        trigger_key: int = 0,
        post_stim_wait_s: float = 0.1,
        recording_window_s: float = 0.5,
        trial_index: int = 0,
    ) -> pd.DataFrame:
        """
        Send one charge-balanced biphasic pulse and return spike events.

        With amplitude_ua = 0.0, charge balance is trivially satisfied:
          0 * duration_us == 0 * duration_us.

        The trigger is still fired so the event is logged in the database.
        """
        amplitude_ua = min(abs(amplitude_ua), 4.0)
        duration_us = min(abs(duration_us), 400.0)

        stim = StimParam()
        stim.index = electrode_idx
        stim.enable = True
        stim.trigger_key = trigger_key
        stim.trigger_delay = 0
        stim.nb_pulse = 0
        stim.pulse_train_period = 10000
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

        self.intan.send_stimparam([stim])

        pattern = np.zeros(16, dtype=np.uint8)
        pattern[trigger_key] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.05)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            timestamp_utc=datetime_now().isoformat(),
            trigger_key=trigger_key,
            trial_index=trial_index,
        ))

        self._wait(post_stim_wait_s)

        query_stop = datetime_now()
        query_start = query_stop - timedelta(seconds=post_stim_wait_s + recording_window_s)
        spike_df = self.database.get_spike_event_electrode(
            query_start, query_stop, electrode_idx
        )
        return spike_df

    def _save_all(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> None:
        """Persist all raw experiment data for downstream analysis."""
        fs_name = getattr(self.np_experiment, "exp_name", "unknown")
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
            "total_trials": len(self._trial_results),
            "experiment_duration_s": self.experiment_duration_s,
            "stim_amplitude_ua": self.stim_amplitude_ua,
            "stim_duration_us": self.stim_duration_us,
            "inter_trial_interval_s": self.inter_trial_interval_s,
            "stim_electrodes": self.stim_electrodes,
            "resp_electrodes": self.resp_electrodes,
            "trial_results": self._trial_results,
        }
        saver.save_summary(summary)

        waveform_records = self._fetch_spike_waveforms(
            fs_name, spike_df, trigger_df, recording_start, recording_stop
        )
        saver.save_spike_waveforms(waveform_records)

    def _fetch_spike_waveforms(
        self,
        fs_name: str,
        spike_df: pd.DataFrame,
        trigger_df: pd.DataFrame,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> list:
        """Fetch raw spike waveform data for each electrode that had spikes."""
        waveform_records = []
        if spike_df.empty:
            return waveform_records

        electrode_col = None
        for col in spike_df.columns:
            if col.lower() in ("channel", "index", "electrode", "electrode_idx"):
                electrode_col = col
                break
            if "electrode" in col.lower() or "idx" in col.lower() or "channel" in col.lower():
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

        total_trials = len(self._trial_results)
        total_spikes = sum(t.get("n_spikes_in_window", 0) for t in self._trial_results)
        mean_spikes_per_trial = total_spikes / total_trials if total_trials > 0 else 0.0

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.np_experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "total_trials": total_trials,
            "total_spikes_in_windows": total_spikes,
            "mean_spikes_per_trial": mean_spikes_per_trial,
            "stim_amplitude_ua": self.stim_amplitude_ua,
            "stim_duration_us": self.stim_duration_us,
            "inter_trial_interval_s": self.inter_trial_interval_s,
            "stim_electrodes": self.stim_electrodes,
            "resp_electrodes": self.resp_electrodes,
            "note": (
                "Zero-amplitude stimulation: triggers fired but no current injected. "
                "All recorded spikes are spontaneous activity time-locked to trigger events."
            ),
        }

        return summary

    def _cleanup(self) -> None:
        """Release all hardware resources. Called from the finally block."""
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
