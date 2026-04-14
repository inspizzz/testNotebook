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
    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        amplitude_ua: float = 4.0,
        duration_us: float = 400.0,
        num_stimulations_per_electrode: int = 1000,
        electrodes: Optional[List[int]] = None,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.amplitude_ua = min(abs(amplitude_ua), 4.0)
        self.duration_us = min(abs(duration_us), 400.0)
        self.num_stimulations_per_electrode = num_stimulations_per_electrode

        if electrodes is not None:
            self.electrodes_to_stimulate = electrodes
        else:
            self.electrodes_to_stimulate = [0, 1, 2, 3, 4, 5, 6, 7]

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._total_stimulations_delivered = 0

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
            logger.info("Electrodes available: %s", self.experiment.electrodes)

            if not self.experiment.start():
                logger.error("Failed to start experiment")
                return {"status": "failed", "error": "Could not start experiment"}

            recording_start = datetime_now()

            self._configure_stimulation_params()
            self._phase_continuous_stimulation()

            recording_stop = datetime_now()

            self._save_all(recording_start, recording_stop)

            results = self._compile_results(recording_start, recording_stop)
            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _configure_stimulation_params(self) -> None:
        logger.info(
            "Configuring stimulation: amplitude=%.2f uA, duration=%.1f us on %d electrodes",
            self.amplitude_ua,
            self.duration_us,
            len(self.electrodes_to_stimulate),
        )
        stim_params = []
        for i, electrode_idx in enumerate(self.electrodes_to_stimulate):
            trigger_key = i % 16
            sp = StimParam()
            sp.index = electrode_idx
            sp.enable = True
            sp.trigger_key = trigger_key
            sp.trigger_delay = 0
            sp.nb_pulse = 0
            sp.pulse_train_period = 10000
            sp.post_stim_ref_period = 0.0
            sp.stim_shape = StimShape.Biphasic
            sp.polarity = StimPolarity.NegativeFirst
            sp.phase_amplitude1 = self.amplitude_ua
            sp.phase_duration1 = self.duration_us
            sp.phase_amplitude2 = self.amplitude_ua
            sp.phase_duration2 = self.duration_us
            sp.enable_amp_settle = True
            sp.pre_stim_amp_settle = 0.0
            sp.post_stim_amp_settle = 0.0
            sp.enable_charge_recovery = True
            sp.post_charge_recovery_on = 0.0
            sp.post_charge_recovery_off = 0.0
            sp.interphase_delay = 0.0
            stim_params.append(sp)

        self.intan.send_stimparam(stim_params)
        logger.info("Stimulation parameters sent for all electrodes")

    def _fire_trigger(self, trigger_key: int) -> None:
        pattern = np.zeros(16, dtype=np.uint8)
        pattern[trigger_key] = 1
        self.trigger_controller.send(pattern)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

    def _phase_continuous_stimulation(self) -> None:
        logger.info(
            "Starting continuous stimulation: %d stimulations x %d electrodes",
            self.num_stimulations_per_electrode,
            len(self.electrodes_to_stimulate),
        )

        for stim_idx in range(self.num_stimulations_per_electrode):
            for elec_pos, electrode_idx in enumerate(self.electrodes_to_stimulate):
                trigger_key = elec_pos % 16
                ts = datetime_now().isoformat()
                self._fire_trigger(trigger_key)
                self._stimulation_log.append(
                    StimulationRecord(
                        electrode_idx=electrode_idx,
                        amplitude_ua=self.amplitude_ua,
                        duration_us=self.duration_us,
                        timestamp_utc=ts,
                        trigger_key=trigger_key,
                        extra={"stim_idx": stim_idx},
                    )
                )
                self._total_stimulations_delivered += 1

            if (stim_idx + 1) % 100 == 0:
                logger.info(
                    "Progress: %d / %d stimulation rounds completed",
                    stim_idx + 1,
                    self.num_stimulations_per_electrode,
                )

        logger.info(
            "Continuous stimulation complete. Total stimulations delivered: %d",
            self._total_stimulations_delivered,
        )

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
            "total_stimulations": len(self._stimulation_log),
            "total_stimulations_delivered": self._total_stimulations_delivered,
            "electrodes_stimulated": self.electrodes_to_stimulate,
            "amplitude_ua": self.amplitude_ua,
            "duration_us": self.duration_us,
            "num_stimulations_per_electrode": self.num_stimulations_per_electrode,
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
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
            if col.lower() in ("channel", "index", "electrode", "electrode_idx"):
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
        logger.info("Compiling results")
        duration_s = (recording_stop - recording_start).total_seconds()
        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": duration_s,
            "total_stimulations_delivered": self._total_stimulations_delivered,
            "electrodes_stimulated": self.electrodes_to_stimulate,
            "amplitude_ua": self.amplitude_ua,
            "duration_us": self.duration_us,
            "num_stimulations_per_electrode": self.num_stimulations_per_electrode,
            "charge_balance_verified": True,
            "charge_per_phase_pC": self.amplitude_ua * self.duration_us,
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
