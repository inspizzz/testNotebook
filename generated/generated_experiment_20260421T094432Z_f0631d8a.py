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
    trigger_key: int = 0
    stim_index: int = 0
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
    Rapid burst protocol experiment.

    Delivers charge-balanced biphasic pulses at the maximum safe parameters
    (amplitude=4.0 uA, duration=400 us, charge-balanced) on the most
    responsive electrode (electrode 14, which showed 94% response rate to
    electrode 12 in the deep scan). The requested 5 uA / 500 us exceeds
    hardware safety limits, so parameters are clamped to the platform
    maximum of 4.0 uA / 400 us while preserving charge balance
    (4.0 * 400 = 4.0 * 400). Stimulations are delivered at 100 Hz
    (10 ms inter-stimulus interval) for 10 minutes (60,000 total pulses).
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
        stim_rate_hz: float = 100.0,
        experiment_duration_s: float = 600.0,
        trigger_key: int = 0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        # Clamp parameters to hardware safety limits
        self.stim_electrode = stim_electrode
        self.amplitude_ua = min(abs(amplitude_ua), 4.0)
        self.duration_us = min(abs(duration_us), 400.0)
        self.polarity_str = polarity
        self.stim_rate_hz = stim_rate_hz
        self.experiment_duration_s = experiment_duration_s
        self.trigger_key = trigger_key

        # Derived timing
        self.inter_stim_interval_s = 1.0 / self.stim_rate_hz
        self.total_stimulations = int(self.stim_rate_hz * self.experiment_duration_s)

        # Hardware handles
        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        # Stimulation log
        self._stimulation_log: List[StimulationRecord] = []

        logger.info(
            "Experiment configured: electrode=%d, amplitude=%.2f uA, "
            "duration=%.1f us, rate=%.1f Hz, duration=%.0f s, "
            "total_stims=%d",
            self.stim_electrode,
            self.amplitude_ua,
            self.duration_us,
            self.stim_rate_hz,
            self.experiment_duration_s,
            self.total_stimulations,
        )

    def _wait(self, seconds: float) -> None:
        wait(seconds)

    def run(self) -> Dict[str, Any]:
        """Execute the full rapid burst experiment and return results."""
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
            self._run_burst_protocol(recording_start)

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
        """Configure the stimulation parameters on the Intan hardware."""
        logger.info(
            "Configuring stimulation: electrode=%d, A1=%.2f uA, D1=%.1f us, "
            "A2=%.2f uA, D2=%.1f us, polarity=%s",
            self.stim_electrode,
            self.amplitude_ua,
            self.duration_us,
            self.amplitude_ua,
            self.duration_us,
            self.polarity_str,
        )

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

        # Charge-balanced: A1*D1 == A2*D2
        # 4.0 uA * 400 us == 4.0 uA * 400 us
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

        self.intan.send_stimparam([stim])
        logger.info("Stimulation parameters sent to hardware")

    def _fire_trigger(self) -> None:
        """Fire a single trigger pulse."""
        pattern = np.zeros(16, dtype=np.uint8)
        pattern[self.trigger_key] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.001)
        pattern[self.trigger_key] = 0
        self.trigger_controller.send(pattern)

    def _run_burst_protocol(self, recording_start: datetime) -> None:
        """
        Deliver stimulations at the configured rate for the full duration.

        At 100 Hz the inter-stimulus interval is 10 ms. Each iteration:
          1. Fire trigger.
          2. Log the stimulation.
          3. Wait for the remainder of the 10 ms window.

        Progress is logged every 1000 stimulations (~10 s).
        """
        logger.info(
            "Starting burst protocol: %d stimulations at %.1f Hz over %.0f s",
            self.total_stimulations,
            self.stim_rate_hz,
            self.experiment_duration_s,
        )

        polarity_enum = (
            StimPolarity.NegativeFirst
            if self.polarity_str == "NegativeFirst"
            else StimPolarity.PositiveFirst
        )

        inter_stim_s = self.inter_stim_interval_s
        # Reserve a small overhead for trigger send (~1 ms)
        post_trigger_wait_s = max(0.0, inter_stim_s - 0.002)

        for stim_idx in range(self.total_stimulations):
            stim_time = datetime_now()

            self._fire_trigger()

            self._stimulation_log.append(
                StimulationRecord(
                    electrode_idx=self.stim_electrode,
                    amplitude_ua=self.amplitude_ua,
                    duration_us=self.duration_us,
                    polarity=self.polarity_str,
                    timestamp_utc=stim_time.isoformat(),
                    trigger_key=self.trigger_key,
                    stim_index=stim_idx,
                )
            )

            self._wait(post_trigger_wait_s)

            if (stim_idx + 1) % 1000 == 0:
                elapsed = (datetime_now() - recording_start).total_seconds()
                logger.info(
                    "Progress: %d / %d stimulations delivered (%.1f s elapsed)",
                    stim_idx + 1,
                    self.total_stimulations,
                    elapsed,
                )

        logger.info(
            "Burst protocol complete: %d stimulations delivered",
            len(self._stimulation_log),
        )

    def _save_all(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> None:
        """Persist all raw experiment data for downstream analysis."""
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
            "stim_electrode": self.stim_electrode,
            "amplitude_ua": self.amplitude_ua,
            "duration_us": self.duration_us,
            "polarity": self.polarity_str,
            "stim_rate_hz": self.stim_rate_hz,
            "experiment_duration_s": self.experiment_duration_s,
            "total_stimulations_delivered": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "charge_balance_check": self.amplitude_ua * self.duration_us,
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
                    electrode_idx,
                    exc,
                )

        return waveform_records

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        """Assemble a summary dict to be returned from run()."""
        logger.info("Compiling results")

        duration_s = (recording_stop - recording_start).total_seconds()
        actual_rate = (
            len(self._stimulation_log) / duration_s if duration_s > 0 else 0.0
        )

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
            "target_stim_rate_hz": self.stim_rate_hz,
            "actual_stim_rate_hz": round(actual_rate, 2),
            "total_stimulations_delivered": len(self._stimulation_log),
            "charge_balance_nC": round(self.amplitude_ua * self.duration_us * 1e-3, 4),
        }

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
