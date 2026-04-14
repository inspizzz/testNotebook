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
    trial_index: int
    electrode_idx: int
    amplitude_ua: float
    duration_us: float
    polarity: str
    trigger_key: int
    timestamp_utc: str
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
    Maximum-amplitude stimulation experiment.

    Stimulates electrode 14 (most responsive from deep scan: 94% response rate
    to electrode 12) at the platform maximum amplitude (4.0 uA) and maximum
    duration (400 us), charge-balanced, repeated 100 times at 1 Hz.

    Charge balance: A1*D1 = A2*D2
    4.0 uA * 100 us = 4.0 uA * 100 us  (equal phases, both at max amplitude)

    Note: To use amplitude=4.0 with duration=400 on both phases would violate
    charge balance only if phases differ. Here both phases are identical:
    A1=4.0, D1=100, A2=4.0, D2=100 => 400 nC = 400 nC (balanced).

    However, the objective says maximum duration=400 us. To achieve both
    maximum amplitude AND maximum duration while maintaining charge balance:
    A1=4.0, D1=400, A2=4.0, D2=400 => 1600 = 1600 (balanced, both phases
    at max). This is valid since A1*D1 = A2*D2 = 1600.
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        stim_electrode: int = 14,
        amplitude_ua: float = 4.0,
        duration_us: float = 400.0,
        polarity: str = "NegativeFirst",
        num_trials: int = 100,
        iti_seconds: float = 1.0,
        trigger_key: int = 0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.stim_electrode = stim_electrode
        self.amplitude_ua = min(abs(amplitude_ua), 4.0)
        self.duration_us = min(abs(duration_us), 400.0)
        self.polarity_str = polarity
        self.num_trials = num_trials
        self.iti_seconds = iti_seconds
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

            fs_name = self.experiment.exp_name
            electrodes = self.experiment.electrodes

            logger.info("Experiment: %s", fs_name)
            logger.info("Electrodes available: %s", electrodes)
            logger.info(
                "Stimulation plan: electrode=%d, amplitude=%.1f uA, duration=%.1f us, "
                "polarity=%s, trials=%d, ITI=%.2f s",
                self.stim_electrode, self.amplitude_ua, self.duration_us,
                self.polarity_str, self.num_trials, self.iti_seconds,
            )

            if not self.experiment.start():
                logger.error("Failed to start experiment")
                return {"status": "failed", "error": "Could not start experiment"}

            recording_start = datetime_now()

            self._configure_stimulation()
            self._run_stimulation_trials()

            recording_stop = datetime_now()

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _configure_stimulation(self) -> None:
        logger.info("Configuring stimulation parameters on electrode %d", self.stim_electrode)

        polarity_enum = (
            StimPolarity.NegativeFirst
            if self.polarity_str == "NegativeFirst"
            else StimPolarity.PositiveFirst
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

        charge_product = self.amplitude_ua * self.duration_us
        logger.info(
            "Charge balance check: A1*D1=%.1f, A2*D2=%.1f (must be equal)",
            charge_product, charge_product,
        )
        assert math.isclose(
            stim.phase_amplitude1 * stim.phase_duration1,
            stim.phase_amplitude2 * stim.phase_duration2,
            rel_tol=1e-9,
        ), "Charge balance violated!"

        self.intan.send_stimparam([stim])
        logger.info("Stimulation parameters sent to hardware")

    def _run_stimulation_trials(self) -> None:
        logger.info("Starting %d stimulation trials at 1 Hz", self.num_trials)

        pattern_on = np.zeros(16, dtype=np.uint8)
        pattern_on[self.trigger_key] = 1
        pattern_off = np.zeros(16, dtype=np.uint8)

        for trial_idx in range(self.num_trials):
            trial_start = datetime_now()
            logger.info("Trial %d / %d", trial_idx + 1, self.num_trials)

            self.trigger_controller.send(pattern_on)
            self._wait(0.05)
            self.trigger_controller.send(pattern_off)

            timestamp_utc = datetime_now().isoformat()

            self._stimulation_log.append(StimulationRecord(
                trial_index=trial_idx,
                electrode_idx=self.stim_electrode,
                amplitude_ua=self.amplitude_ua,
                duration_us=self.duration_us,
                polarity=self.polarity_str,
                trigger_key=self.trigger_key,
                timestamp_utc=timestamp_utc,
                extra={
                    "charge_balance_product": self.amplitude_ua * self.duration_us,
                },
            ))

            self._trial_results.append({
                "trial_index": trial_idx,
                "timestamp_utc": timestamp_utc,
                "electrode": self.stim_electrode,
                "amplitude_ua": self.amplitude_ua,
                "duration_us": self.duration_us,
            })

            elapsed = (datetime_now() - trial_start).total_seconds()
            remaining_wait = self.iti_seconds - elapsed
            if remaining_wait > 0:
                self._wait(remaining_wait)

        logger.info("All %d trials completed", self.num_trials)

    def _save_all(self, recording_start: datetime, recording_stop: datetime) -> None:
        fs_name = getattr(self.experiment, "exp_name", "unknown")
        saver = DataSaver(self._output_dir, fs_name)

        saver.save_stimulation_log(self._stimulation_log)

        spike_df = self.database.get_spike_event(recording_start, recording_stop, fs_name)
        saver.save_spike_events(spike_df)

        trigger_df = self.database.get_all_triggers(recording_start, recording_stop)
        saver.save_triggers(trigger_df)

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "stim_electrode": self.stim_electrode,
            "amplitude_ua": self.amplitude_ua,
            "duration_us": self.duration_us,
            "polarity": self.polarity_str,
            "num_trials": self.num_trials,
            "iti_seconds": self.iti_seconds,
            "charge_balance_product": self.amplitude_ua * self.duration_us,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df) if not spike_df.empty else 0,
            "total_triggers": len(trigger_df) if not trigger_df.empty else 0,
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
            logger.info("No spike events found; skipping waveform fetch")
            return waveform_records

        electrode_col = None
        for candidate in ("channel", "index", "electrode", "electrode_idx"):
            if candidate in spike_df.columns:
                electrode_col = candidate
                break

        if electrode_col is None:
            logger.warning("Cannot determine electrode column for waveform fetch")
            return waveform_records

        unique_electrodes = spike_df[electrode_col].unique()
        logger.info("Fetching waveforms for %d electrode(s)", len(unique_electrodes))

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
                    logger.info(
                        "Fetched %d waveform rows for electrode %d",
                        len(raw_df), electrode_idx,
                    )
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
            "stim_electrode": self.stim_electrode,
            "amplitude_ua": self.amplitude_ua,
            "duration_us": self.duration_us,
            "polarity": self.polarity_str,
            "num_trials_completed": len(self._stimulation_log),
            "num_trials_requested": self.num_trials,
            "charge_balance_product_nC": self.amplitude_ua * self.duration_us,
            "iti_seconds": self.iti_seconds,
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
