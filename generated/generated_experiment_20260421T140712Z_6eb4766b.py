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
    Zero-amplitude stimulation experiment.

    Sends sham (0 uA) triggers at regular intervals for exactly 10 minutes
    and records all neural activity during that window. Because the hardware
    does not allow a true 0 uA pulse, we use the minimum representable
    charge-balanced biphasic waveform (amplitude=0.001 uA, duration=400 us
    on both phases) which delivers negligible charge and is effectively a
    no-stimulation control. All spike events are recorded and saved for
    downstream comparison against evoked responses from prior scans.
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        experiment_duration_s: float = 600.0,
        inter_trial_interval_s: float = 2.0,
        stim_electrode: int = 17,
        resp_electrode: int = 18,
        trigger_key: int = 0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.experiment_duration_s = experiment_duration_s
        self.inter_trial_interval_s = inter_trial_interval_s
        self.stim_electrode = stim_electrode
        self.resp_electrode = resp_electrode
        self.trigger_key = trigger_key

        self.experiment = None
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
            logger.info("Recording started at %s", recording_start.isoformat())

            self._phase_sham_stimulation(recording_start)

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

    def _phase_sham_stimulation(self, recording_start: datetime) -> None:
        """
        Send sham (effectively zero-amplitude) triggers for the full
        experiment duration, spaced by inter_trial_interval_s.

        Because the FinalSpark hardware enforces charge balance and will
        raise a StimulationSafetyError for amplitude=0, we use the smallest
        safe non-zero amplitude (0.001 uA) with duration=400 us on both
        phases, delivering ~0.4 pC — negligible compared to the ~800 pC
        used in the parameter scan. This is functionally a sham stimulation.

        Charge balance: 0.001 * 400 == 0.001 * 400  (satisfied).
        """
        logger.info("Phase: sham stimulation for %.0f s", self.experiment_duration_s)

        sham_amplitude = 0.001
        sham_duration = 400.0

        stim = StimParam()
        stim.index = self.stim_electrode
        stim.enable = True
        stim.trigger_key = self.trigger_key
        stim.trigger_delay = 0
        stim.nb_pulse = 0
        stim.pulse_train_period = 10000
        stim.post_stim_ref_period = 1000.0
        stim.stim_shape = StimShape.Biphasic
        stim.polarity = StimPolarity.PositiveFirst
        stim.phase_amplitude1 = sham_amplitude
        stim.phase_duration1 = sham_duration
        stim.phase_amplitude2 = sham_amplitude
        stim.phase_duration2 = sham_duration
        stim.enable_amp_settle = True
        stim.pre_stim_amp_settle = 0.0
        stim.post_stim_amp_settle = 1000.0
        stim.enable_charge_recovery = True
        stim.post_charge_recovery_on = 0.0
        stim.post_charge_recovery_off = 100.0

        self.intan.send_stimparam([stim])
        logger.info("Sham stim params sent to electrode %d", self.stim_electrode)

        trial_index = 0
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
                amplitude_ua=sham_amplitude,
                duration_us=sham_duration,
                timestamp_utc=stim_time.isoformat(),
                trigger_key=self.trigger_key,
                trial_index=trial_index,
                extra={
                    "polarity": "PositiveFirst",
                    "sham": True,
                    "elapsed_s": elapsed,
                },
            ))

            self._trial_results.append({
                "trial_index": trial_index,
                "stim_electrode": self.stim_electrode,
                "resp_electrode": self.resp_electrode,
                "stim_time_utc": stim_time.isoformat(),
                "amplitude_ua": sham_amplitude,
                "duration_us": sham_duration,
            })

            trial_index += 1
            logger.info(
                "Trial %d dispatched at %.1f s elapsed",
                trial_index, elapsed,
            )

            remaining = self.experiment_duration_s - (datetime_now() - recording_start).total_seconds()
            if remaining <= 0:
                break

            wait_time = min(self.inter_trial_interval_s, remaining)
            self._wait(wait_time)

        logger.info("Sham stimulation phase complete. Total trials: %d", trial_index)

    def _save_all(
        self,
        recording_start: datetime,
        recording_stop: datetime,
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
            "total_trials": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "experiment_duration_s": self.experiment_duration_s,
            "inter_trial_interval_s": self.inter_trial_interval_s,
            "stim_electrode": self.stim_electrode,
            "resp_electrode": self.resp_electrode,
            "sham_amplitude_ua": 0.001,
            "sham_duration_us": 400.0,
            "note": (
                "Zero-amplitude (sham) stimulation experiment. "
                "Amplitude set to 0.001 uA (hardware minimum) to satisfy "
                "charge-balance constraint while delivering negligible charge."
            ),
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
                    electrode_idx, exc,
                )

        return waveform_records

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        logger.info("Compiling results")
        duration_s = (recording_stop - recording_start).total_seconds()
        return {
            "status": "completed",
            "experiment_name": getattr(self.experiment, "exp_name", "unknown"),
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": duration_s,
            "total_trials": len(self._stimulation_log),
            "stim_electrode": self.stim_electrode,
            "resp_electrode": self.resp_electrode,
            "sham_amplitude_ua": 0.001,
            "sham_duration_us": 400.0,
            "trial_results": self._trial_results,
        }

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
