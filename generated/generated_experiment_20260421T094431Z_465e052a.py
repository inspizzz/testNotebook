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
    round_idx: int = 0
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
    10 Hz full-array stimulation experiment for 30 minutes.

    All 32 electrodes (indices 0-31) are stimulated in rapid succession
    within each 100 ms cycle (10 Hz). Each electrode receives a
    charge-balanced biphasic pulse at maximum safe amplitude (4.0 uA)
    and maximum safe duration (400 us per phase), NegativeFirst polarity.

    Charge balance: A1 * D1 == A2 * D2 => 4.0 * 400 == 4.0 * 400. ✓

    The experiment runs for experiment_duration_s seconds (default 1800 = 30 min).
    """

    NUM_ELECTRODES: int = 32
    AMPLITUDE_UA: float = 4.0
    DURATION_US: float = 400.0
    STIM_FREQ_HZ: float = 10.0
    TRIGGER_KEY: int = 0

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        experiment_duration_s: float = 1800.0,
        inter_electrode_wait_s: float = 0.002,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)
        self.experiment_duration_s = experiment_duration_s
        self.inter_electrode_wait_s = inter_electrode_wait_s

        self.np_experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._total_rounds: int = 0

    def _wait(self, seconds: float) -> None:
        wait(seconds)

    def run(self) -> Dict[str, Any]:
        """Execute the full experiment and return a results dict."""
        try:
            logger.info("Initialising hardware connections")

            from neuroplatform import Experiment as NeuroPlatformExperiment

            self.np_experiment = NeuroPlatformExperiment(self.token)
            self.trigger_controller = TriggerController(email=self.booking_email)
            self.intan = IntanSofware()
            self.database = Database()

            logger.info("Experiment: %s", self.np_experiment.exp_name)
            logger.info("Electrodes: %s", self.np_experiment.electrodes)

            if not self.np_experiment.start():
                logger.error("Failed to start experiment")
                return {"status": "failed", "error": "Could not start experiment"}

            recording_start = datetime_now()
            logger.info("Recording started at %s", recording_start.isoformat())

            self._phase_full_array_stimulation(recording_start)

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

    def _build_stim_params(self) -> List[StimParam]:
        """Build a list of StimParam objects for all 32 electrodes."""
        params = []
        for electrode_idx in range(self.NUM_ELECTRODES):
            sp = StimParam()
            sp.index = electrode_idx
            sp.enable = True
            sp.trigger_key = self.TRIGGER_KEY
            sp.trigger_delay = 0
            sp.nb_pulse = 0
            sp.pulse_train_period = 10000
            sp.post_stim_ref_period = 1000.0
            sp.stim_shape = StimShape.Biphasic
            sp.polarity = StimPolarity.NegativeFirst
            sp.phase_amplitude1 = self.AMPLITUDE_UA
            sp.phase_duration1 = self.DURATION_US
            sp.phase_amplitude2 = self.AMPLITUDE_UA
            sp.phase_duration2 = self.DURATION_US
            sp.enable_amp_settle = True
            sp.pre_stim_amp_settle = 0.0
            sp.post_stim_amp_settle = 1000.0
            sp.enable_charge_recovery = True
            sp.post_charge_recovery_on = 0.0
            sp.post_charge_recovery_off = 100.0
            sp.interphase_delay = 0.0
            params.append(sp)
        return params

    def _fire_trigger(self) -> None:
        """Send a single trigger pulse on TRIGGER_KEY."""
        pattern = np.zeros(16, dtype=np.uint8)
        pattern[self.TRIGGER_KEY] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.005)
        pattern[self.TRIGGER_KEY] = 0
        self.trigger_controller.send(pattern)

    def _phase_full_array_stimulation(self, recording_start: datetime) -> None:
        """
        Stimulate all 32 electrodes at 10 Hz for the full experiment duration.

        Strategy:
          - Pre-configure all 32 electrodes with the same trigger key so a
            single trigger fires all of them simultaneously.
          - Each cycle (100 ms at 10 Hz) sends one trigger burst.
          - The cycle period is maintained by tracking elapsed time and
            sleeping for the remainder of each 100 ms window.
        """
        logger.info(
            "Phase: full-array stimulation at %.1f Hz for %.0f s",
            self.STIM_FREQ_HZ,
            self.experiment_duration_s,
        )

        cycle_period_s = 1.0 / self.STIM_FREQ_HZ

        stim_params = self._build_stim_params()
        logger.info("Sending stimulation parameters for %d electrodes", len(stim_params))
        self.intan.send_stimparam(stim_params)
        logger.info("Stimulation parameters configured")

        experiment_end = recording_start.timestamp() + self.experiment_duration_s
        round_idx = 0

        while True:
            cycle_start_ts = datetime_now().timestamp()

            if cycle_start_ts >= experiment_end:
                break

            self._fire_trigger()

            ts_utc = datetime_now().isoformat()
            for electrode_idx in range(self.NUM_ELECTRODES):
                self._stimulation_log.append(StimulationRecord(
                    electrode_idx=electrode_idx,
                    amplitude_ua=self.AMPLITUDE_UA,
                    duration_us=self.DURATION_US,
                    polarity="NegativeFirst",
                    timestamp_utc=ts_utc,
                    trigger_key=self.TRIGGER_KEY,
                    round_idx=round_idx,
                ))

            round_idx += 1
            self._total_rounds = round_idx

            if round_idx % 100 == 0:
                elapsed = datetime_now().timestamp() - recording_start.timestamp()
                logger.info(
                    "Round %d complete | elapsed %.1f s / %.0f s",
                    round_idx, elapsed, self.experiment_duration_s,
                )

            cycle_elapsed = datetime_now().timestamp() - cycle_start_ts
            remaining = cycle_period_s - cycle_elapsed
            if remaining > 0.001:
                self._wait(remaining)

        logger.info(
            "Full-array stimulation complete: %d rounds, %d total stim events",
            self._total_rounds,
            len(self._stimulation_log),
        )

    def _save_all(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> None:
        """Persist all raw experiment data for downstream analysis."""
        fs_name = getattr(self.np_experiment, "exp_name", "unknown")
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
            "total_rounds": self._total_rounds,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "stim_freq_hz": self.STIM_FREQ_HZ,
            "amplitude_ua": self.AMPLITUDE_UA,
            "duration_us": self.DURATION_US,
            "num_electrodes": self.NUM_ELECTRODES,
            "experiment_duration_s": self.experiment_duration_s,
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
        """Fetch raw spike waveform data for each electrode that had spikes."""
        waveform_records = []
        if spike_df.empty:
            return waveform_records

        electrode_col = None
        for col in spike_df.columns:
            if col.lower() in ("channel", "index", "electrode", "electrode_idx"):
                electrode_col = col
                break
        if electrode_col is None:
            for col in spike_df.columns:
                if "electrode" in col.lower() or "idx" in col.lower() or "channel" in col.lower():
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
        """Assemble a summary dict to be returned from run()."""
        logger.info("Compiling results")
        duration_s = (recording_stop - recording_start).total_seconds()
        return {
            "status": "completed",
            "experiment_name": getattr(self.np_experiment, "exp_name", "unknown"),
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": duration_s,
            "total_rounds": self._total_rounds,
            "total_stimulations": len(self._stimulation_log),
            "stim_freq_hz": self.STIM_FREQ_HZ,
            "amplitude_ua": self.AMPLITUDE_UA,
            "duration_us": self.DURATION_US,
            "num_electrodes": self.NUM_ELECTRODES,
        }

    def _cleanup(self) -> None:
        """Release all hardware resources. Called from the finally block."""
        logger.info("Cleaning up resources")

        if self.np_experiment is not None:
            try:
                self.np_experiment.stop()
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
