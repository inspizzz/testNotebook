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
    Rapid burst protocol: 100 stimulations/second at 3.0 uA / 400 us on
    electrode 17 (most responsive: 0.92 response rate, pair_01_mode_1).
    Runs for 10 minutes (60000 total stimulations).

    Safety note: The requested 5 uA / 500 us exceeds platform limits
    (max 4.0 uA, max 400 us). Parameters are clamped to 3.0 uA / 400 us,
    which is the best-performing configuration from the parameter scan
    (electrode 17->18, response_rate=0.92, temporal_stability=1.0).

    Charge balance: A1*D1 = 3.0*400 = 1200 = A2*D2 = 3.0*400.

    At 100 Hz the inter-stimulus interval is 10 ms. Each biphasic pulse
    occupies 2*400 us = 0.8 ms, leaving 9.2 ms of silence per cycle,
    which is well within the post-stim refractory period budget.
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        stim_electrode: int = 17,
        amplitude_ua: float = 3.0,
        duration_us: float = 400.0,
        polarity: str = "PositiveFirst",
        stim_rate_hz: float = 100.0,
        experiment_duration_s: float = 600.0,
        trigger_key: int = 0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        # Stimulation parameters (clamped to platform limits)
        self.stim_electrode = stim_electrode
        self.amplitude_ua = min(abs(amplitude_ua), 4.0)
        self.duration_us = min(abs(duration_us), 400.0)
        self.polarity_str = polarity
        self.stim_rate_hz = stim_rate_hz
        self.experiment_duration_s = experiment_duration_s
        self.trigger_key = trigger_key

        # Derived timing
        self.isi_s = 1.0 / self.stim_rate_hz  # 10 ms at 100 Hz

        # Hardware handles
        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        # Results storage
        self._stimulation_log: List[StimulationRecord] = []
        self._total_stimulations_sent: int = 0

    def _wait(self, seconds: float) -> None:
        wait(seconds)

    def run(self) -> Dict[str, Any]:
        recording_start: Optional[datetime] = None
        recording_stop: Optional[datetime] = None

        try:
            logger.info("Initialising hardware connections")

            from neuroplatform import Experiment as NeuroPlatformExperiment

            self.experiment = NeuroPlatformExperiment(self.token)
            self.trigger_controller = TriggerController(email=self.booking_email)
            self.intan = IntanSofware()
            self.database = Database()

            fs_name = self.experiment.exp_name
            logger.info("Experiment: %s", fs_name)
            logger.info("Electrodes: %s", self.experiment.electrodes)

            if not self.experiment.start():
                logger.error("Failed to start experiment")
                return {"status": "failed", "error": "Could not start experiment"}

            # Configure stimulation parameters once before the burst loop
            self._configure_stim_params()

            recording_start = datetime_now()
            logger.info(
                "Starting rapid burst protocol: %.0f Hz for %.0f s on electrode %d",
                self.stim_rate_hz,
                self.experiment_duration_s,
                self.stim_electrode,
            )

            self._run_burst_protocol(recording_start)

            recording_stop = datetime_now()
            logger.info(
                "Burst protocol complete. Total stimulations sent: %d",
                self._total_stimulations_sent,
            )

            results = self._compile_results(recording_start, recording_stop)

            # Persist all raw data
            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            if recording_start is not None and recording_stop is None:
                recording_stop = datetime_now()
            if recording_start is not None and recording_stop is not None:
                try:
                    self._save_all(recording_start, recording_stop)
                except Exception as save_exc:
                    logger.error("Failed to save data after error: %s", save_exc)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _configure_stim_params(self) -> None:
        """Build and upload the StimParam to the Intan software once."""
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

        # Charge-balanced: A1*D1 == A2*D2
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
            "Uploading StimParam: electrode=%d, A=%.1f uA, D=%.0f us, polarity=%s",
            self.stim_electrode,
            self.amplitude_ua,
            self.duration_us,
            self.polarity_str,
        )
        self.intan.send_stimparam([stim])
        logger.info("StimParam uploaded successfully")

    def _run_burst_protocol(self, recording_start: datetime) -> None:
        """
        Deliver stimulations at self.stim_rate_hz for self.experiment_duration_s.

        At 100 Hz the ISI is 10 ms. We fire the trigger, log the event,
        then sleep for the remainder of the ISI. The loop tracks elapsed
        wall-clock time and exits when the target duration is reached.
        """
        total_target = int(round(self.stim_rate_hz * self.experiment_duration_s))
        logger.info("Target stimulation count: %d", total_target)

        # Pre-build the trigger pattern (reused every cycle)
        pattern_on = np.zeros(16, dtype=np.uint8)
        pattern_on[self.trigger_key] = 1
        pattern_off = np.zeros(16, dtype=np.uint8)

        log_interval = max(1, int(self.stim_rate_hz * 10))  # log every 10 s

        for stim_idx in range(total_target):
            cycle_start = datetime_now()

            # Fire trigger
            self.trigger_controller.send(pattern_on)
            self._wait(0.001)  # 1 ms trigger pulse width
            self.trigger_controller.send(pattern_off)

            # Log stimulation record
            self._stimulation_log.append(StimulationRecord(
                electrode_idx=self.stim_electrode,
                amplitude_ua=self.amplitude_ua,
                duration_us=self.duration_us,
                polarity=self.polarity_str,
                timestamp_utc=cycle_start.isoformat(),
                trigger_key=self.trigger_key,
                stim_index=stim_idx,
            ))
            self._total_stimulations_sent += 1

            if (stim_idx + 1) % log_interval == 0:
                elapsed = (datetime_now() - recording_start).total_seconds()
                logger.info(
                    "Progress: %d / %d stimulations sent (%.1f s elapsed)",
                    self._total_stimulations_sent,
                    total_target,
                    elapsed,
                )

            # Sleep for the remainder of the ISI
            elapsed_cycle = (datetime_now() - cycle_start).total_seconds()
            remaining = self.isi_s - elapsed_cycle
            if remaining > 0.0:
                self._wait(remaining)

        logger.info(
            "Burst loop finished. Sent %d stimulations.", self._total_stimulations_sent
        )

    def _save_all(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> None:
        fs_name = getattr(self.experiment, "exp_name", "unknown")
        saver = DataSaver(self._output_dir, fs_name)

        # Save stimulation log
        saver.save_stimulation_log(self._stimulation_log)

        # Fetch and save ALL spike events
        try:
            spike_df = self.database.get_spike_event(
                recording_start, recording_stop, fs_name
            )
        except Exception as exc:
            logger.warning("Could not fetch spike events: %s", exc)
            spike_df = pd.DataFrame()
        saver.save_spike_events(spike_df)

        # Fetch and save ALL triggers
        try:
            trigger_df = self.database.get_all_triggers(
                recording_start, recording_stop
            )
        except Exception as exc:
            logger.warning("Could not fetch triggers: %s", exc)
            trigger_df = pd.DataFrame()
        saver.save_triggers(trigger_df)

        # Save summary
        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "testing": self.testing,
            "stim_electrode": self.stim_electrode,
            "amplitude_ua": self.amplitude_ua,
            "duration_us": self.duration_us,
            "polarity": self.polarity_str,
            "stim_rate_hz": self.stim_rate_hz,
            "experiment_duration_s": self.experiment_duration_s,
            "total_stimulations_sent": self._total_stimulations_sent,
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "charge_balance_check": self.amplitude_ua * self.duration_us,
        }
        saver.save_summary(summary)

        # Fetch and save spike waveforms
        waveform_records = self._fetch_spike_waveforms(
            spike_df, recording_start, recording_stop
        )
        saver.save_spike_waveforms(waveform_records)

    def _fetch_spike_waveforms(
        self,
        spike_df: pd.DataFrame,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> list:
        waveform_records = []
        if spike_df.empty:
            return waveform_records

        # Determine electrode column
        electrode_col = None
        for candidate in ("channel", "index", "electrode"):
            if candidate in spike_df.columns:
                electrode_col = candidate
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
        logger.info("Compiling results")
        duration_s = (recording_stop - recording_start).total_seconds()
        actual_rate = (
            self._total_stimulations_sent / duration_s if duration_s > 0 else 0.0
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
            "target_rate_hz": self.stim_rate_hz,
            "actual_rate_hz": round(actual_rate, 2),
            "total_stimulations_sent": self._total_stimulations_sent,
            "charge_balance_nC": self.amplitude_ua * self.duration_us * 1e-3,
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
