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
    amplitude_level_index: int
    trial_index: int
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
    Input-output curve experiment using escalating amplitude from 1.0 to 4.0 uA
    in 0.5 uA steps, with 400 us duration and PositiveFirst polarity.
    At each amplitude level, 20 stimulations are delivered at 1 Hz on the most
    responsive electrode (electrode 14, stimulating to electrode 15 based on
    deep scan results showing 94% response rate at amplitude 1.0 uA).
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        stim_electrode: int = 14,
        record_electrode: int = 15,
        duration_us: float = 400.0,
        polarity: str = "PositiveFirst",
        amplitude_min: float = 1.0,
        amplitude_max: float = 4.0,
        amplitude_step: float = 0.5,
        trials_per_level: int = 20,
        isi_s: float = 1.0,
        inter_level_wait_s: float = 5.0,
        trigger_key: int = 0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.stim_electrode = stim_electrode
        self.record_electrode = record_electrode
        self.duration_us = duration_us
        self.polarity_str = polarity
        self.amplitude_min = amplitude_min
        self.amplitude_max = amplitude_max
        self.amplitude_step = amplitude_step
        self.trials_per_level = trials_per_level
        self.isi_s = isi_s
        self.inter_level_wait_s = inter_level_wait_s
        self.trigger_key = trigger_key

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._io_curve_results: List[Dict[str, Any]] = []

        n_steps = round((amplitude_max - amplitude_min) / amplitude_step) + 1
        self._amplitude_levels = [
            round(amplitude_min + i * amplitude_step, 4)
            for i in range(n_steps)
            if round(amplitude_min + i * amplitude_step, 4) <= amplitude_max + 1e-9
        ]

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

            self._phase_io_curve()

            recording_stop = datetime_now()

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _phase_io_curve(self) -> None:
        logger.info(
            "Phase: IO curve sweep on electrode %d -> %d, polarity=%s, duration=%.1f us",
            self.stim_electrode, self.record_electrode, self.polarity_str, self.duration_us,
        )
        logger.info("Amplitude levels: %s", self._amplitude_levels)

        polarity = (
            StimPolarity.PositiveFirst
            if self.polarity_str == "PositiveFirst"
            else StimPolarity.NegativeFirst
        )

        for level_idx, amplitude in enumerate(self._amplitude_levels):
            logger.info(
                "Amplitude level %d/%d: %.2f uA (%d trials)",
                level_idx + 1, len(self._amplitude_levels), amplitude, self.trials_per_level,
            )

            level_spike_counts = []

            for trial_idx in range(self.trials_per_level):
                spike_df = self._stimulate_and_record(
                    electrode_idx=self.stim_electrode,
                    amplitude_ua=amplitude,
                    duration_us=self.duration_us,
                    polarity=polarity,
                    trigger_key=self.trigger_key,
                    level_idx=level_idx,
                    trial_idx=trial_idx,
                )
                n_spikes = len(spike_df) if not spike_df.empty else 0
                level_spike_counts.append(n_spikes)
                logger.info(
                    "  Trial %d/%d: amplitude=%.2f uA, spikes=%d",
                    trial_idx + 1, self.trials_per_level, amplitude, n_spikes,
                )
                if trial_idx < self.trials_per_level - 1:
                    self._wait(self.isi_s)

            mean_spikes = float(np.mean(level_spike_counts)) if level_spike_counts else 0.0
            self._io_curve_results.append({
                "amplitude_ua": amplitude,
                "duration_us": self.duration_us,
                "polarity": self.polarity_str,
                "stim_electrode": self.stim_electrode,
                "record_electrode": self.record_electrode,
                "trials": self.trials_per_level,
                "spike_counts": level_spike_counts,
                "mean_spikes": mean_spikes,
            })
            logger.info(
                "Level %.2f uA complete: mean_spikes=%.2f", amplitude, mean_spikes
            )

            if level_idx < len(self._amplitude_levels) - 1:
                self._wait(self.inter_level_wait_s)

    def _stimulate_and_record(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity,
        trigger_key: int,
        level_idx: int,
        trial_idx: int,
        post_stim_wait_s: float = 0.5,
        recording_window_s: float = 0.5,
    ) -> pd.DataFrame:
        amplitude_ua = min(abs(amplitude_ua), 4.0)
        duration_us = min(abs(duration_us), 400.0)

        a1 = amplitude_ua
        d1 = duration_us
        d2 = d1
        a2 = a1

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

        pattern = np.zeros(16, dtype=np.uint8)
        pattern[trigger_key] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.05)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

        stim_time = datetime_now()

        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            polarity=polarity.name,
            amplitude_level_index=level_idx,
            trial_index=trial_idx,
            timestamp_utc=stim_time.isoformat(),
            trigger_key=trigger_key,
            extra={
                "a1_x_d1": a1 * d1,
                "a2_x_d2": a2 * d2,
                "charge_balanced": abs(a1 * d1 - a2 * d2) < 1e-9,
            },
        ))

        self._wait(post_stim_wait_s)

        query_stop = datetime_now()
        query_start = query_stop - timedelta(seconds=post_stim_wait_s + recording_window_s)
        try:
            spike_df = self.database.get_spike_event_electrode(
                query_start, query_stop, self.record_electrode
            )
        except Exception as exc:
            logger.warning("Spike query failed: %s", exc)
            spike_df = pd.DataFrame()

        return spike_df

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

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "stim_electrode": self.stim_electrode,
            "record_electrode": self.record_electrode,
            "duration_us": self.duration_us,
            "polarity": self.polarity_str,
            "amplitude_levels": self._amplitude_levels,
            "trials_per_level": self.trials_per_level,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "io_curve_results": self._io_curve_results,
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

        io_summary = []
        for entry in self._io_curve_results:
            io_summary.append({
                "amplitude_ua": entry["amplitude_ua"],
                "mean_spikes": entry["mean_spikes"],
                "spike_counts": entry["spike_counts"],
            })

        return {
            "status": "completed",
            "experiment_name": getattr(self.experiment, "exp_name", "unknown"),
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "stim_electrode": self.stim_electrode,
            "record_electrode": self.record_electrode,
            "duration_us": self.duration_us,
            "polarity": self.polarity_str,
            "amplitude_levels": self._amplitude_levels,
            "trials_per_level": self.trials_per_level,
            "total_stimulations": len(self._stimulation_log),
            "io_curve": io_summary,
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
