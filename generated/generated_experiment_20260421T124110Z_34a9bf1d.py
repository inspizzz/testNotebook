import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
from dataclasses import dataclass, field, asdict
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
    Single-electrode baseline recording experiment.

    Stimulates electrode 7 at 2 uA amplitude, 200 us duration,
    PositiveFirst polarity, once every 2 seconds for 50 repetitions.
    Records all spike events and triggers, then saves all data.
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        stim_electrode: int = 7,
        amplitude_ua: float = 2.0,
        duration_us: float = 200.0,
        polarity: str = "PositiveFirst",
        num_repetitions: int = 50,
        inter_stim_interval_s: float = 2.0,
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
        self.num_repetitions = num_repetitions
        self.inter_stim_interval_s = inter_stim_interval_s
        self.trigger_key = trigger_key

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []

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
            self._run_stimulation_loop()

            recording_stop = datetime_now()

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            if recording_stop is None:
                recording_stop = datetime_now()
            if recording_start is not None:
                try:
                    self._save_all(recording_start, recording_stop)
                except Exception as save_exc:
                    logger.error("Failed to save data on error: %s", save_exc)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _configure_stimulation(self) -> None:
        """Configure the stimulation parameters on the Intan hardware."""
        logger.info(
            "Configuring stimulation: electrode=%d, amplitude=%.2f uA, "
            "duration=%.1f us, polarity=%s",
            self.stim_electrode,
            self.amplitude_ua,
            self.duration_us,
            self.polarity_str,
        )

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

        self.intan.send_stimparam([stim])
        logger.info("Stimulation parameters sent to hardware")

    def _run_stimulation_loop(self) -> None:
        """Run the stimulation loop for the specified number of repetitions."""
        logger.info(
            "Starting stimulation loop: %d repetitions, %.1f s interval",
            self.num_repetitions,
            self.inter_stim_interval_s,
        )

        for rep in range(self.num_repetitions):
            logger.info("Stimulation %d / %d", rep + 1, self.num_repetitions)

            stim_time = datetime_now()

            pattern = np.zeros(16, dtype=np.uint8)
            pattern[self.trigger_key] = 1
            self.trigger_controller.send(pattern)

            self._wait(0.05)

            pattern[self.trigger_key] = 0
            self.trigger_controller.send(pattern)

            self._stimulation_log.append(StimulationRecord(
                electrode_idx=self.stim_electrode,
                amplitude_ua=self.amplitude_ua,
                duration_us=self.duration_us,
                polarity=self.polarity_str,
                timestamp_utc=stim_time.isoformat(),
                trigger_key=self.trigger_key,
                extra={"repetition": rep + 1},
            ))

            if rep < self.num_repetitions - 1:
                self._wait(self.inter_stim_interval_s)

        logger.info("Stimulation loop complete")

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
            "num_repetitions": self.num_repetitions,
            "inter_stim_interval_s": self.inter_stim_interval_s,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
        }
        saver.save_summary(summary)

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

        duration_s = (recording_stop - recording_start).total_seconds()

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": duration_s,
            "stim_electrode": self.stim_electrode,
            "amplitude_ua": self.amplitude_ua,
            "duration_us": self.duration_us,
            "polarity": self.polarity_str,
            "num_repetitions": self.num_repetitions,
            "inter_stim_interval_s": self.inter_stim_interval_s,
            "total_stimulations_delivered": len(self._stimulation_log),
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
