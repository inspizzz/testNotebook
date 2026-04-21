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
    High-power 10 Hz simultaneous stimulation experiment on all 32 electrodes
    for 30 minutes. Uses maximum safe amplitude (4.0 uA) and duration (400 us)
    with charge-balanced biphasic pulses. All 32 electrodes are stimulated in
    rapid succession within each 100 ms inter-stimulus interval window.

    At 10 Hz, one stimulation cycle occurs every 100 ms. Within each cycle,
    all 32 electrodes are stimulated sequentially using trigger keys 0-15
    (two passes of 16 electrodes each), with minimal inter-electrode delay.
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        amplitude_ua: float = 4.0,
        duration_us: float = 400.0,
        stim_frequency_hz: float = 10.0,
        experiment_duration_minutes: float = 30.0,
        num_electrodes: int = 32,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        # Stimulation parameters - enforce safety limits
        self.amplitude_ua = min(abs(amplitude_ua), 4.0)
        self.duration_us = min(abs(duration_us), 400.0)
        self.stim_frequency_hz = stim_frequency_hz
        self.experiment_duration_minutes = experiment_duration_minutes
        self.num_electrodes = num_electrodes

        # Derived timing
        self.cycle_period_s = 1.0 / self.stim_frequency_hz  # 0.1 s at 10 Hz
        self.total_cycles = int(
            self.experiment_duration_minutes * 60.0 * self.stim_frequency_hz
        )  # 30 min * 60 s/min * 10 Hz = 18000 cycles

        # Hardware handles
        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        # Stimulation log
        self._stimulation_log: List[StimulationRecord] = []
        self._total_cycles_completed = 0

    def _wait(self, seconds: float) -> None:
        wait(seconds)

    def run(self) -> Dict[str, Any]:
        """Execute the full 30-minute 10 Hz stimulation experiment."""
        try:
            logger.info("Initialising hardware connections")

            from neuroplatform import Experiment as NeuroPlatformExperiment

            self.experiment = NeuroPlatformExperiment(self.token)
            self.trigger_controller = TriggerController(email=self.booking_email)
            self.intan = IntanSofware()
            self.database = Database()

            logger.info("Experiment: %s", self.experiment.exp_name)
            logger.info("Electrodes: %s", self.experiment.electrodes)
            logger.info(
                "Parameters: amplitude=%.1f uA, duration=%.0f us, freq=%.1f Hz, duration=%.1f min",
                self.amplitude_ua,
                self.duration_us,
                self.stim_frequency_hz,
                self.experiment_duration_minutes,
            )
            logger.info("Total planned cycles: %d", self.total_cycles)

            if not self.experiment.start():
                logger.error("Failed to start experiment")
                return {"status": "failed", "error": "Could not start experiment"}

            recording_start = datetime_now()

            # Configure all electrodes with stimulation parameters
            self._configure_all_electrodes()

            # Run the main stimulation loop
            self._phase_continuous_stimulation()

            recording_stop = datetime_now()

            results = self._compile_results(recording_start, recording_stop)

            # Persist all raw data
            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _configure_all_electrodes(self) -> None:
        """
        Pre-configure all 32 electrodes with stimulation parameters.
        Electrodes are split into two groups of 16, each assigned to
        trigger keys 0-15. This allows rapid sequential triggering.

        Charge balance: amplitude1 * duration1 == amplitude2 * duration2
        With equal amplitudes and equal durations, balance is guaranteed.
        """
        logger.info("Configuring stimulation parameters for all electrodes")

        electrodes = self.experiment.electrodes
        # Use up to num_electrodes electrodes
        active_electrodes = electrodes[:self.num_electrodes]

        # Group 1: first 16 electrodes -> trigger keys 0-15
        # Group 2: next 16 electrodes -> trigger keys 0-15 (second pass)
        self._group1_electrodes = active_electrodes[:16]
        self._group2_electrodes = active_electrodes[16:32]

        # Configure group 1
        params_group1 = []
        for i, elec_idx in enumerate(self._group1_electrodes):
            sp = self._make_stim_param(elec_idx, trigger_key=i)
            params_group1.append(sp)

        if params_group1:
            self.intan.send_stimparam(params_group1)
            logger.info("Configured %d electrodes in group 1", len(params_group1))

        # Configure group 2
        params_group2 = []
        for i, elec_idx in enumerate(self._group2_electrodes):
            sp = self._make_stim_param(elec_idx, trigger_key=i)
            params_group2.append(sp)

        if params_group2:
            self.intan.send_stimparam(params_group2)
            logger.info("Configured %d electrodes in group 2", len(params_group2))

        self._active_electrodes = active_electrodes
        logger.info(
            "Total electrodes configured: %d (group1=%d, group2=%d)",
            len(active_electrodes),
            len(self._group1_electrodes),
            len(self._group2_electrodes),
        )

    def _make_stim_param(self, electrode_idx: int, trigger_key: int) -> StimParam:
        """Create a charge-balanced biphasic StimParam for a given electrode."""
        sp = StimParam()
        sp.index = electrode_idx
        sp.enable = True
        sp.trigger_key = trigger_key
        sp.trigger_delay = 0
        sp.nb_pulse = 0  # single pulse per trigger
        sp.pulse_train_period = 10000
        sp.post_stim_ref_period = 1000.0
        sp.stim_shape = StimShape.Biphasic
        sp.polarity = StimPolarity.PositiveFirst

        # Charge balance: A1*D1 == A2*D2
        # Using equal amplitude and duration on both phases
        sp.phase_amplitude1 = self.amplitude_ua
        sp.phase_duration1 = self.duration_us
        sp.phase_amplitude2 = self.amplitude_ua
        sp.phase_duration2 = self.duration_us

        sp.enable_amp_settle = True
        sp.pre_stim_amp_settle = 0.0
        sp.post_stim_amp_settle = 1000.0
        sp.enable_charge_recovery = True
        sp.post_charge_recovery_on = 0.0
        sp.post_charge_recovery_off = 100.0
        sp.interphase_delay = 0.0
        return sp

    def _fire_trigger_group(self, trigger_keys: List[int], electrodes: List[int], stim_index: int) -> None:
        """
        Fire all triggers in a group simultaneously by setting all bits at once,
        then clearing them.
        """
        pattern = np.zeros(16, dtype=np.uint8)
        for key in trigger_keys:
            if 0 <= key < 16:
                pattern[key] = 1

        self.trigger_controller.send(pattern)
        self._wait(0.005)  # 5 ms pulse width

        # Clear pattern
        clear_pattern = np.zeros(16, dtype=np.uint8)
        self.trigger_controller.send(clear_pattern)

        # Log stimulations for all electrodes in this group
        ts = datetime_now().isoformat()
        for elec_idx in electrodes:
            self._stimulation_log.append(StimulationRecord(
                electrode_idx=elec_idx,
                amplitude_ua=self.amplitude_ua,
                duration_us=self.duration_us,
                polarity="PositiveFirst",
                timestamp_utc=ts,
                trigger_key=trigger_keys[electrodes.index(elec_idx)] if elec_idx in electrodes else 0,
                stim_index=stim_index,
            ))

    def _phase_continuous_stimulation(self) -> None:
        """
        Main stimulation loop: stimulate all electrodes at 10 Hz for 30 minutes.

        Each 100 ms cycle:
          1. Fire group 1 (up to 16 electrodes simultaneously via trigger keys 0-15)
          2. Brief inter-group delay
          3. Fire group 2 (up to 16 electrodes simultaneously via trigger keys 0-15)
          4. Wait for remainder of cycle period

        This ensures all 32 electrodes are stimulated within each 100 ms window.
        """
        logger.info(
            "Starting continuous stimulation: %d cycles at %.1f Hz",
            self.total_cycles,
            self.stim_frequency_hz,
        )

        group1_keys = list(range(len(self._group1_electrodes)))
        group2_keys = list(range(len(self._group2_electrodes)))

        inter_group_delay_s = 0.010  # 10 ms between group 1 and group 2
        log_interval = max(1, self.total_cycles // 20)  # log ~20 times

        for cycle_idx in range(self.total_cycles):
            cycle_start = datetime_now()

            # Fire group 1 - all electrodes simultaneously
            if self._group1_electrodes:
                self._fire_trigger_group(
                    group1_keys,
                    list(self._group1_electrodes),
                    stim_index=cycle_idx,
                )

            # Brief delay between groups
            self._wait(inter_group_delay_s)

            # Fire group 2 - all electrodes simultaneously
            if self._group2_electrodes:
                self._fire_trigger_group(
                    group2_keys,
                    list(self._group2_electrodes),
                    stim_index=cycle_idx,
                )

            self._total_cycles_completed += 1

            # Log progress periodically
            if (cycle_idx + 1) % log_interval == 0:
                elapsed_min = (cycle_idx + 1) / self.stim_frequency_hz / 60.0
                logger.info(
                    "Cycle %d/%d completed (%.1f min elapsed, %d stim records)",
                    cycle_idx + 1,
                    self.total_cycles,
                    elapsed_min,
                    len(self._stimulation_log),
                )

            # Wait for remainder of cycle period
            cycle_elapsed = (datetime_now() - cycle_start).total_seconds()
            remaining = self.cycle_period_s - cycle_elapsed
            if remaining > 0.001:
                self._wait(remaining)

        logger.info(
            "Stimulation complete: %d cycles, %d total stimulation events",
            self._total_cycles_completed,
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

        # Save stimulation log
        saver.save_stimulation_log(self._stimulation_log)

        # Fetch and save ALL spike events for the full experiment duration
        try:
            spike_df = self.database.get_spike_event(
                recording_start, recording_stop, fs_name
            )
        except Exception as exc:
            logger.warning("Failed to fetch spike events: %s", exc)
            spike_df = pd.DataFrame()

        saver.save_spike_events(spike_df)

        # Fetch and save ALL triggers for the full experiment duration
        try:
            trigger_df = self.database.get_all_triggers(
                recording_start, recording_stop
            )
        except Exception as exc:
            logger.warning("Failed to fetch triggers: %s", exc)
            trigger_df = pd.DataFrame()

        saver.save_triggers(trigger_df)

        # Save experiment summary
        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "total_cycles_completed": self._total_cycles_completed,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "amplitude_ua": self.amplitude_ua,
            "duration_us": self.duration_us,
            "stim_frequency_hz": self.stim_frequency_hz,
            "experiment_duration_minutes": self.experiment_duration_minutes,
            "num_electrodes": self.num_electrodes,
            "charge_balance_check": self.amplitude_ua * self.duration_us,
        }
        saver.save_summary(summary)

        # Fetch and save spike waveforms per electrode
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
        """Fetch raw spike waveform data for each electrode that had spikes."""
        waveform_records = []
        if spike_df.empty:
            return waveform_records

        # Determine electrode column
        electrode_col = None
        for col in spike_df.columns:
            if col in ("channel", "index", "electrode"):
                electrode_col = col
                break
            if "electrode" in col.lower() or "channel" in col.lower():
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

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": duration_s,
            "duration_minutes": duration_s / 60.0,
            "total_cycles_completed": self._total_cycles_completed,
            "total_stimulation_events": len(self._stimulation_log),
            "amplitude_ua": self.amplitude_ua,
            "duration_us": self.duration_us,
            "stim_frequency_hz": self.stim_frequency_hz,
            "num_electrodes_stimulated": len(self._active_electrodes) if hasattr(self, "_active_electrodes") else 0,
            "charge_per_phase_nC": self.amplitude_ua * self.duration_us * 1e-3,
            "charge_balance_verified": True,
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
