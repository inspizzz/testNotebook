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
    electrode_stim: int
    electrode_resp: int
    amplitude_ua: float
    duration_us: float
    polarity: str
    timestamp_utc: str
    trigger_key: int
    amplitude_level_index: int
    trial_index: int
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AmplitudeLevelResult:
    amplitude_ua: float
    num_trials: int
    responding_trials: int
    total_spikes: int
    response_probability: float
    mean_spike_count: float


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
    Input-output curve experiment on FinalSpark NeuroPlatform.

    Sweeps stimulation amplitude from 0.5 to 4.0 uA in 0.5 uA steps (8 levels)
    using the most responsive electrode pair (17 -> 18, PositiveFirst, 300 us).
    At each amplitude level, 30 stimulations are delivered at 1 Hz.
    Response probability and mean spike count per stimulation are measured.
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        stim_electrode: int = 17,
        resp_electrode: int = 18,
        polarity: str = "PositiveFirst",
        stim_duration_us: float = 300.0,
        amplitudes_ua: Tuple = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0),
        num_trials_per_level: int = 30,
        isi_s: float = 1.0,
        inter_level_wait_s: float = 10.0,
        response_window_ms: float = 50.0,
        trigger_key: int = 0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.stim_electrode = stim_electrode
        self.resp_electrode = resp_electrode
        self.polarity_str = polarity
        self.stim_duration_us = stim_duration_us
        self.amplitudes_ua = list(amplitudes_ua)
        self.num_trials_per_level = num_trials_per_level
        self.isi_s = isi_s
        self.inter_level_wait_s = inter_level_wait_s
        self.response_window_ms = response_window_ms
        self.trigger_key = trigger_key

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._level_results: List[AmplitudeLevelResult] = []
        self._trial_data: List[Dict[str, Any]] = []

    def _wait(self, seconds: float) -> None:
        wait(seconds)

    def run(self) -> Dict[str, Any]:
        """Execute the full input-output curve experiment and return results."""
        try:
            logger.info("Initialising hardware connections")

            from neuroplatform import Experiment as NeuroPlatformExperiment

            self.experiment = NeuroPlatformExperiment(self.token)
            self.trigger_controller = TriggerController(email=self.booking_email)
            self.intan = IntanSofware()
            self.database = Database()

            fs_name = self.experiment.exp_name
            logger.info("Experiment: %s", fs_name)
            logger.info("Electrodes available: %s", self.experiment.electrodes)

            if not self.experiment.start():
                logger.error("Failed to start experiment")
                return {"status": "failed", "error": "Could not start experiment"}

            recording_start = datetime_now()
            logger.info("Recording started at %s", recording_start.isoformat())

            self._configure_stimulation()
            self._run_io_curve()

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

    def _configure_stimulation(self) -> None:
        """Configure the stimulation parameters for the first amplitude level.
        Parameters will be updated per amplitude level during the sweep."""
        logger.info(
            "Configuring stimulation: electrode %d -> %d, polarity=%s, duration=%.1f us",
            self.stim_electrode, self.resp_electrode, self.polarity_str, self.stim_duration_us,
        )

    def _build_stim_param(self, amplitude_ua: float) -> StimParam:
        """Build a charge-balanced StimParam for the given amplitude.

        Charge balance: A1 * D1 = A2 * D2.
        Both phases use the same amplitude and duration (symmetric biphasic).
        """
        amplitude_ua = min(abs(amplitude_ua), 4.0)
        duration_us = min(abs(self.stim_duration_us), 400.0)

        polarity = (
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

        return stim

    def _fire_trigger(self) -> None:
        """Send a single trigger pulse on the configured trigger key."""
        pattern = np.zeros(16, dtype=np.uint8)
        pattern[self.trigger_key] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.02)
        pattern[self.trigger_key] = 0
        self.trigger_controller.send(pattern)

    def _run_io_curve(self) -> None:
        """Sweep amplitude levels and collect response data."""
        logger.info(
            "Starting I/O curve sweep: %d amplitude levels, %d trials each at 1 Hz",
            len(self.amplitudes_ua), self.num_trials_per_level,
        )

        for level_idx, amplitude in enumerate(self.amplitudes_ua):
            logger.info(
                "Amplitude level %d/%d: %.2f uA",
                level_idx + 1, len(self.amplitudes_ua), amplitude,
            )

            stim = self._build_stim_param(amplitude)
            self.intan.send_stimparam([stim])

            level_spike_counts: List[int] = []
            level_responding: int = 0

            for trial_idx in range(self.num_trials_per_level):
                trial_start = datetime_now()
                self._fire_trigger()

                stim_time = datetime_now()

                self._stimulation_log.append(StimulationRecord(
                    electrode_stim=self.stim_electrode,
                    electrode_resp=self.resp_electrode,
                    amplitude_ua=amplitude,
                    duration_us=self.stim_duration_us,
                    polarity=self.polarity_str,
                    timestamp_utc=stim_time.isoformat(),
                    trigger_key=self.trigger_key,
                    amplitude_level_index=level_idx,
                    trial_index=trial_idx,
                ))

                response_window_s = self.response_window_ms / 1000.0
                self._wait(response_window_s)

                query_stop = datetime_now()
                query_start = stim_time

                try:
                    spike_df = self.database.get_spike_event_electrode(
                        query_start, query_stop, self.resp_electrode
                    )
                    if spike_df is not None and not spike_df.empty:
                        spike_count = len(spike_df)
                    else:
                        spike_count = 0
                except Exception as exc:
                    logger.warning("Spike query failed for trial %d: %s", trial_idx, exc)
                    spike_count = 0

                level_spike_counts.append(spike_count)
                if spike_count > 0:
                    level_responding += 1

                self._trial_data.append({
                    "amplitude_level_index": level_idx,
                    "amplitude_ua": amplitude,
                    "trial_index": trial_idx,
                    "spike_count": spike_count,
                    "responded": spike_count > 0,
                    "stim_time_utc": stim_time.isoformat(),
                })

                logger.debug(
                    "  Trial %d/%d: %d spike(s)",
                    trial_idx + 1, self.num_trials_per_level, spike_count,
                )

                elapsed = (datetime_now() - trial_start).total_seconds()
                remaining_isi = self.isi_s - elapsed
                if remaining_isi > 0:
                    self._wait(remaining_isi)

            response_probability = level_responding / self.num_trials_per_level
            mean_spike_count = (
                sum(level_spike_counts) / len(level_spike_counts)
                if level_spike_counts else 0.0
            )

            level_result = AmplitudeLevelResult(
                amplitude_ua=amplitude,
                num_trials=self.num_trials_per_level,
                responding_trials=level_responding,
                total_spikes=sum(level_spike_counts),
                response_probability=response_probability,
                mean_spike_count=mean_spike_count,
            )
            self._level_results.append(level_result)

            logger.info(
                "  Level %.2f uA: response_prob=%.3f, mean_spikes=%.3f",
                amplitude, response_probability, mean_spike_count,
            )

            if level_idx < len(self.amplitudes_ua) - 1:
                logger.info("Waiting %d s between amplitude levels", self.inter_level_wait_s)
                self._wait(self.inter_level_wait_s)

        logger.info("I/O curve sweep complete.")

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        """Assemble a summary dict to be returned from run()."""
        logger.info("Compiling results")

        fs_name = getattr(self.experiment, "exp_name", "unknown")

        io_curve = []
        for r in self._level_results:
            io_curve.append({
                "amplitude_ua": r.amplitude_ua,
                "num_trials": r.num_trials,
                "responding_trials": r.responding_trials,
                "total_spikes": r.total_spikes,
                "response_probability": r.response_probability,
                "mean_spike_count": r.mean_spike_count,
            })

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": fs_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "stim_electrode": self.stim_electrode,
            "resp_electrode": self.resp_electrode,
            "polarity": self.polarity_str,
            "stim_duration_us": self.stim_duration_us,
            "num_amplitude_levels": len(self.amplitudes_ua),
            "num_trials_per_level": self.num_trials_per_level,
            "total_stimulations": len(self._stimulation_log),
            "io_curve": io_curve,
        }

        logger.info("I/O Curve Summary:")
        for entry in io_curve:
            logger.info(
                "  %.2f uA -> P(response)=%.3f, mean_spikes=%.3f",
                entry["amplitude_ua"],
                entry["response_probability"],
                entry["mean_spike_count"],
            )

        return summary

    def _save_all(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> None:
        """Persist all raw experiment data for downstream analysis."""
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

        io_curve = [asdict(r) for r in self._level_results]

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "stim_electrode": self.stim_electrode,
            "resp_electrode": self.resp_electrode,
            "polarity": self.polarity_str,
            "stim_duration_us": self.stim_duration_us,
            "amplitudes_ua": self.amplitudes_ua,
            "num_trials_per_level": self.num_trials_per_level,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df) if not spike_df.empty else 0,
            "total_triggers": len(trigger_df) if not trigger_df.empty else 0,
            "io_curve": io_curve,
            "trial_data": self._trial_data,
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

        electrodes_to_fetch = set()
        electrodes_to_fetch.add(self.resp_electrode)
        electrodes_to_fetch.add(self.stim_electrode)

        if not spike_df.empty:
            channel_col = None
            for col in spike_df.columns:
                if col.lower() in ("channel", "index", "electrode"):
                    channel_col = col
                    break
            if channel_col is not None:
                for e in spike_df[channel_col].unique():
                    electrodes_to_fetch.add(int(e))

        for electrode_idx in electrodes_to_fetch:
            try:
                raw_df = self.database.get_raw_spike(
                    recording_start, recording_stop, int(electrode_idx)
                )
                if raw_df is not None and not raw_df.empty:
                    waveform_records.append({
                        "electrode_idx": int(electrode_idx),
                        "num_waveforms": len(raw_df),
                        "waveform_samples": raw_df.values.tolist(),
                    })
            except Exception as exc:
                logger.warning(
                    "Failed to fetch waveforms for electrode %d: %s",
                    electrode_idx, exc,
                )

        return waveform_records

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
