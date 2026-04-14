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
    polarity: str
    amplitude_level_index: int
    trial_index: int
    timestamp_utc: str
    trigger_key: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AmplitudeLevelResult:
    amplitude_ua: float
    num_trials: int
    num_responding_trials: int
    total_spikes: int
    response_probability: float
    mean_spike_count: float


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
    Input-output curve experiment for FinalSpark NeuroPlatform.

    Sweeps stimulation amplitude from 0.5 to 4.0 uA in 0.5 uA steps (8 levels).
    At each amplitude, delivers 30 stimulations at 1 Hz using the most responsive
    electrode pair (electrode 14 -> electrode 15, based on deep scan results showing
    94% response rate at 1.0 uA / 400 us). Uses 300 us duration and PositiveFirst
    polarity throughout. Measures response probability and mean spike count per
    stimulation for each amplitude level.

    Most responsive pair selection rationale:
    - electrode 14 -> 15: response_rate=0.80 at 2.0uA/300us/PositiveFirst (deep scan pair 4)
    - electrode 14 -> 12: response_rate=0.94 at 1.0uA/400us/NegativeFirst (deep scan pair 3)
    - electrode 14 -> 15 chosen because it shows strong responses across multiple
      amplitude/polarity combinations and the experiment requires PositiveFirst polarity.
      At 1.0uA/300us/PositiveFirst hits_k=4/5 in parameter scan, and at 2.0uA/300us/PositiveFirst
      response_rate=0.80 in deep scan. This pair is expected to show a clear IO curve.
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        stim_electrode: int = 14,
        record_electrode: int = 15,
        stim_duration_us: float = 300.0,
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
        self.record_electrode = record_electrode
        self.stim_duration_us = stim_duration_us
        self.num_trials_per_level = num_trials_per_level
        self.isi_s = isi_s
        self.inter_level_wait_s = inter_level_wait_s
        self.response_window_ms = response_window_ms
        self.trigger_key = trigger_key

        # Amplitude sweep: 0.5 to 4.0 uA in 0.5 uA steps
        self.amplitude_levels: List[float] = [round(0.5 * i, 1) for i in range(1, 9)]

        # Hardware handles
        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        # Results storage
        self._stimulation_log: List[StimulationRecord] = []
        self._level_results: List[AmplitudeLevelResult] = []

        # Per-trial spike counts keyed by amplitude level index
        self._trial_spike_counts: Dict[int, List[int]] = defaultdict(list)

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
            logger.info("Stim electrode: %d -> Record electrode: %d", self.stim_electrode, self.record_electrode)
            logger.info("Amplitude levels: %s uA", self.amplitude_levels)
            logger.info("Duration: %.1f us, Polarity: PositiveFirst", self.stim_duration_us)
            logger.info("Trials per level: %d at %.1f Hz", self.num_trials_per_level, 1.0 / self.isi_s)

            if not self.experiment.start():
                logger.error("Failed to start experiment")
                return {"status": "failed", "error": "Could not start experiment"}

            recording_start = datetime_now()

            self._configure_stimulation()
            self._run_io_curve()

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
        """Pre-configure the stimulation parameters for the stim electrode."""
        logger.info("Configuring stimulation parameters for electrode %d", self.stim_electrode)
        amplitude_ua = self.amplitude_levels[0]
        stim = self._build_stim_param(self.stim_electrode, amplitude_ua, self.stim_duration_us)
        self.intan.send_stimparam([stim])
        logger.info("Initial stim params sent (amplitude=%.1f uA)", amplitude_ua)

    def _build_stim_param(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
    ) -> StimParam:
        """Build a charge-balanced biphasic StimParam with PositiveFirst polarity.

        Charge balance: amplitude1 * duration1 == amplitude2 * duration2.
        Both phases use the same amplitude and duration, so balance is guaranteed.
        """
        amplitude_ua = min(abs(amplitude_ua), 4.0)
        duration_us = min(abs(duration_us), 400.0)

        stim = StimParam()
        stim.index = electrode_idx
        stim.enable = True
        stim.trigger_key = self.trigger_key
        stim.trigger_delay = 0
        stim.nb_pulse = 0
        stim.pulse_train_period = 10000
        stim.post_stim_ref_period = 1000.0
        stim.stim_shape = StimShape.Biphasic
        stim.polarity = StimPolarity.PositiveFirst

        # Charge-balanced: A1*D1 == A2*D2 (equal amplitudes and durations)
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
        """Fire trigger pulse on the configured trigger key."""
        pattern = np.zeros(16, dtype=np.uint8)
        pattern[self.trigger_key] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.02)
        pattern[self.trigger_key] = 0
        self.trigger_controller.send(pattern)

    def _run_io_curve(self) -> None:
        """Run the full input-output curve sweep across all amplitude levels."""
        logger.info("Starting IO curve sweep: %d amplitude levels x %d trials",
                    len(self.amplitude_levels), self.num_trials_per_level)

        for level_idx, amplitude_ua in enumerate(self.amplitude_levels):
            logger.info(
                "Amplitude level %d/%d: %.1f uA",
                level_idx + 1, len(self.amplitude_levels), amplitude_ua
            )

            # Reconfigure stim params for this amplitude level
            stim = self._build_stim_param(self.stim_electrode, amplitude_ua, self.stim_duration_us)
            self.intan.send_stimparam([stim])

            level_spike_counts: List[int] = []

            for trial_idx in range(self.num_trials_per_level):
                trial_start = datetime_now()

                # Fire the trigger
                self._fire_trigger()

                # Log the stimulation
                self._stimulation_log.append(StimulationRecord(
                    electrode_idx=self.stim_electrode,
                    amplitude_ua=amplitude_ua,
                    duration_us=self.stim_duration_us,
                    polarity="PositiveFirst",
                    amplitude_level_index=level_idx,
                    trial_index=trial_idx,
                    timestamp_utc=trial_start.isoformat(),
                    trigger_key=self.trigger_key,
                ))

                # Wait for response window (50 ms) then query spikes
                response_window_s = self.response_window_ms / 1000.0
                self._wait(response_window_s)

                trial_stop = datetime_now()

                # Query spike events on the recording electrode within the response window
                try:
                    spike_df = self.database.get_spike_event_electrode(
                        trial_start,
                        trial_stop,
                        self.record_electrode,
                    )
                    spike_count = len(spike_df) if not spike_df.empty else 0
                except Exception as exc:
                    logger.warning("Failed to query spikes for trial %d: %s", trial_idx, exc)
                    spike_count = 0

                level_spike_counts.append(spike_count)
                self._trial_spike_counts[level_idx].append(spike_count)

                logger.debug(
                    "Level %d, Trial %d/%d: amplitude=%.1f uA, spikes=%d",
                    level_idx + 1, trial_idx + 1, self.num_trials_per_level,
                    amplitude_ua, spike_count
                )

                # Wait for remainder of ISI (1 Hz = 1 s total; subtract response window)
                remaining_isi = self.isi_s - response_window_s
                if remaining_isi > 0:
                    self._wait(remaining_isi)

            # Compute level statistics
            num_responding = sum(1 for c in level_spike_counts if c > 0)
            total_spikes = sum(level_spike_counts)
            response_prob = num_responding / self.num_trials_per_level if self.num_trials_per_level > 0 else 0.0
            mean_spike_count = total_spikes / self.num_trials_per_level if self.num_trials_per_level > 0 else 0.0

            level_result = AmplitudeLevelResult(
                amplitude_ua=amplitude_ua,
                num_trials=self.num_trials_per_level,
                num_responding_trials=num_responding,
                total_spikes=total_spikes,
                response_probability=response_prob,
                mean_spike_count=mean_spike_count,
            )
            self._level_results.append(level_result)

            logger.info(
                "Level %.1f uA: response_prob=%.3f, mean_spikes=%.3f",
                amplitude_ua, response_prob, mean_spike_count
            )

            # Wait between amplitude levels (skip after last level)
            if level_idx < len(self.amplitude_levels) - 1:
                logger.info("Waiting %.1f s before next amplitude level", self.inter_level_wait_s)
                self._wait(self.inter_level_wait_s)

        logger.info("IO curve sweep complete")

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

        level_results_serializable = [asdict(r) for r in self._level_results]

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "stim_electrode": self.stim_electrode,
            "record_electrode": self.record_electrode,
            "stim_duration_us": self.stim_duration_us,
            "polarity": "PositiveFirst",
            "num_amplitude_levels": len(self.amplitude_levels),
            "amplitude_levels_ua": self.amplitude_levels,
            "num_trials_per_level": self.num_trials_per_level,
            "isi_s": self.isi_s,
            "inter_level_wait_s": self.inter_level_wait_s,
            "response_window_ms": self.response_window_ms,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "io_curve_results": level_results_serializable,
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

        io_curve = []
        for result in self._level_results:
            io_curve.append({
                "amplitude_ua": result.amplitude_ua,
                "num_trials": result.num_trials,
                "num_responding_trials": result.num_responding_trials,
                "total_spikes": result.total_spikes,
                "response_probability": result.response_probability,
                "mean_spike_count": result.mean_spike_count,
            })

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "stim_electrode": self.stim_electrode,
            "record_electrode": self.record_electrode,
            "stim_duration_us": self.stim_duration_us,
            "polarity": "PositiveFirst",
            "num_amplitude_levels": len(self.amplitude_levels),
            "num_trials_per_level": self.num_trials_per_level,
            "total_stimulations": len(self._stimulation_log),
            "io_curve": io_curve,
        }

        logger.info("IO Curve Summary:")
        for entry in io_curve:
            logger.info(
                "  %.1f uA -> response_prob=%.3f, mean_spikes=%.3f",
                entry["amplitude_ua"],
                entry["response_probability"],
                entry["mean_spike_count"],
            )

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
