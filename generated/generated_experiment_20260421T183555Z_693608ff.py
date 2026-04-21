import numpy as np
import pandas as pd
import json
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
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
    Rapid burst protocol: 100 stimulations/second at 4.0 uA / 400 us on
    electrode 17 (most responsive: 0.92 response rate, pair_01_mode_1).
    Runs for 10 minutes, recording all spikes.

    Safety note: The requested 5 uA / 500 us exceeds platform limits
    (max 4.0 uA, max 400 us). Parameters are clamped to 4.0 uA / 400 us.
    Charge balance: A1*D1 == A2*D2 => 4.0*400 == 4.0*400.

    At 100 Hz the inter-stimulus interval is 10 ms. Each biphasic pulse
    occupies 2 * 400 us = 0.8 ms, leaving 9.2 ms of silence per cycle,
    which is safe for the hardware.
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        stim_electrode: int = 17,
        stim_amplitude_ua: float = 4.0,
        stim_duration_us: float = 400.0,
        stim_rate_hz: float = 100.0,
        experiment_duration_s: float = 600.0,
        trigger_key: int = 0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        # Clamp to hardware limits
        self.stim_electrode = stim_electrode
        self.stim_amplitude_ua = min(abs(stim_amplitude_ua), 4.0)
        self.stim_duration_us = min(abs(stim_duration_us), 400.0)
        self.stim_rate_hz = stim_rate_hz
        self.experiment_duration_s = experiment_duration_s
        self.trigger_key = trigger_key

        # Derived timing
        self.isi_s = 1.0 / self.stim_rate_hz  # 0.01 s at 100 Hz
        self.total_stimulations = int(self.stim_rate_hz * self.experiment_duration_s)

        # Hardware handles
        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        # Stimulation log
        self._stimulation_log: List[StimulationRecord] = []

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

            self._configure_stimulation()
            self._run_burst_protocol()

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
        """Configure the StimParam for the burst electrode and upload to Intan."""
        logger.info(
            "Configuring stimulation: electrode=%d, amplitude=%.2f uA, duration=%.1f us, rate=%.1f Hz",
            self.stim_electrode,
            self.stim_amplitude_ua,
            self.stim_duration_us,
            self.stim_rate_hz,
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
        stim.polarity = StimPolarity.PositiveFirst

        # Charge-balanced: A1*D1 == A2*D2
        stim.phase_amplitude1 = self.stim_amplitude_ua
        stim.phase_duration1 = self.stim_duration_us
        stim.phase_amplitude2 = self.stim_amplitude_ua
        stim.phase_duration2 = self.stim_duration_us

        stim.enable_amp_settle = True
        stim.pre_stim_amp_settle = 0.0
        stim.post_stim_amp_settle = 1000.0
        stim.enable_charge_recovery = True
        stim.post_charge_recovery_on = 0.0
        stim.post_charge_recovery_off = 100.0
        stim.interphase_delay = 0.0

        self.intan.send_stimparam([stim])
        logger.info("StimParam uploaded to Intan")

    def _fire_single_pulse(self) -> None:
        """Fire one trigger pulse and log the stimulation."""
        pattern = np.zeros(16, dtype=np.uint8)
        pattern[self.trigger_key] = 1
        self.trigger_controller.send(pattern)

        ts = datetime_now().isoformat()

        pattern[self.trigger_key] = 0
        self.trigger_controller.send(pattern)

        self._stimulation_log.append(StimulationRecord(
            electrode_idx=self.stim_electrode,
            amplitude_ua=self.stim_amplitude_ua,
            duration_us=self.stim_duration_us,
            polarity="PositiveFirst",
            timestamp_utc=ts,
            trigger_key=self.trigger_key,
        ))

    def _run_burst_protocol(self) -> None:
        """
        Deliver stimulations at self.stim_rate_hz for self.experiment_duration_s.

        At 100 Hz the ISI is 10 ms. We fire a pulse, then wait the remainder
        of the ISI after accounting for the small overhead of the trigger send.
        The inter-pulse wait is kept short (ISI minus a small fixed overhead).
        """
        logger.info(
            "Starting burst protocol: %d stimulations over %.0f s at %.0f Hz",
            self.total_stimulations,
            self.experiment_duration_s,
            self.stim_rate_hz,
        )

        # Log progress every 1000 stimulations
        log_interval = max(1, int(self.stim_rate_hz * 10))  # every 10 s

        for i in range(self.total_stimulations):
            self._fire_single_pulse()

            if (i + 1) % log_interval == 0:
                logger.info(
                    "Burst progress: %d / %d stimulations delivered",
                    i + 1,
                    self.total_stimulations,
                )

            # Wait for the remainder of the ISI
            # The trigger send itself is near-instantaneous; we wait the full ISI
            # minus a small fixed overhead (0.5 ms) to stay on schedule.
            self._wait(self.isi_s - 0.0005)

        logger.info("Burst protocol complete: %d stimulations delivered", self.total_stimulations)

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
            "stim_electrode": self.stim_electrode,
            "stim_amplitude_ua": self.stim_amplitude_ua,
            "stim_duration_us": self.stim_duration_us,
            "stim_rate_hz": self.stim_rate_hz,
            "experiment_duration_s": self.experiment_duration_s,
            "total_stimulations_planned": self.total_stimulations,
            "total_stimulations_delivered": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
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
        return {
            "status": "completed",
            "experiment_name": getattr(self.experiment, "exp_name", "unknown"),
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "stim_electrode": self.stim_electrode,
            "stim_amplitude_ua": self.stim_amplitude_ua,
            "stim_duration_us": self.stim_duration_us,
            "stim_rate_hz": self.stim_rate_hz,
            "total_stimulations_planned": self.total_stimulations,
            "total_stimulations_delivered": len(self._stimulation_log),
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
