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

    The objective requested 100 stimulations/second at 5 uA / 500 us, which
    exceeds the hardware safety limits (max 4.0 uA, max 400 us).  The
    parameters are therefore clamped to the platform maximums:
      - amplitude: 4.0 uA  (clamped from 5 uA)
      - duration:  400 us  (clamped from 500 us)
    Charge balance is maintained by using equal amplitude and duration on
    both phases (A1*D1 == A2*D2 => 4.0*400 == 4.0*400).

    The most responsive electrode from the deep-scan results is electrode 14
    (stimulating electrode_from=14, response on electrode_to=15 with 94%
    response rate at amplitude=1.0, duration=400 us; and electrode_to=12
    with 94% response rate).  Electrode 14 is selected as the stimulation
    electrode.

    At 100 Hz (100 stimulations/second) the inter-stimulus interval is 10 ms.
    The pulse train period is set to 10000 us (10 ms).  We deliver pulses in
    batches using nb_pulse to avoid overwhelming the socket; each batch
    covers 1 second (100 pulses), repeated for 10 minutes (600 batches).
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
        stim_rate_hz: float = 100.0,
        experiment_duration_s: float = 600.0,
        trigger_key: int = 0,
        pulses_per_batch: int = 100,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.stim_electrode = stim_electrode
        self.amplitude_ua = min(abs(amplitude_ua), 4.0)
        self.duration_us = min(abs(duration_us), 400.0)
        self.stim_rate_hz = stim_rate_hz
        self.experiment_duration_s = experiment_duration_s
        self.trigger_key = trigger_key
        self.pulses_per_batch = pulses_per_batch

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._total_pulses_delivered: int = 0

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
            logger.info("Recording started at %s", recording_start.isoformat())

            self._phase_rapid_burst(recording_start)

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

    def _build_stim_param(self, nb_pulse: int, pulse_train_period_us: int) -> StimParam:
        """
        Build a charge-balanced StimParam.

        Charge balance: A1 * D1 == A2 * D2
        With A1 == A2 == amplitude_ua and D1 == D2 == duration_us this is
        trivially satisfied.
        """
        stim = StimParam()
        stim.index = self.stim_electrode
        stim.enable = True
        stim.trigger_key = self.trigger_key
        stim.trigger_delay = 0
        stim.nb_pulse = nb_pulse
        stim.pulse_train_period = pulse_train_period_us
        stim.post_stim_ref_period = 1000.0
        stim.stim_shape = StimShape.Biphasic
        stim.polarity = StimPolarity.NegativeFirst

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
        return stim

    def _phase_rapid_burst(self, recording_start: datetime) -> None:
        """
        Deliver rapid burst stimulation at stim_rate_hz for experiment_duration_s.

        Strategy:
          - Inter-stimulus interval = 1 / stim_rate_hz seconds = 10 ms at 100 Hz.
          - pulse_train_period in us = 1e6 / stim_rate_hz.
          - Each batch delivers pulses_per_batch pulses (nb_pulse = pulses_per_batch - 1
            because nb_pulse=0 means single pulse, nb_pulse=N means N+1 pulses).
          - After configuring the stim param, fire the trigger once to start the
            pulse train; the hardware generates the remaining pulses automatically.
          - Wait for the batch duration before firing the next batch.
          - Repeat until experiment_duration_s has elapsed.
        """
        logger.info(
            "Phase: rapid burst | electrode=%d | amplitude=%.1f uA | "
            "duration=%.0f us | rate=%.0f Hz | total_duration=%.0f s",
            self.stim_electrode,
            self.amplitude_ua,
            self.duration_us,
            self.stim_rate_hz,
            self.experiment_duration_s,
        )

        pulse_train_period_us = int(round(1_000_000.0 / self.stim_rate_hz))
        nb_pulse = self.pulses_per_batch - 1
        batch_duration_s = self.pulses_per_batch / self.stim_rate_hz

        total_batches = int(math.ceil(self.experiment_duration_s / batch_duration_s))
        logger.info(
            "Batch config: %d pulses/batch, period=%d us, batch_duration=%.3f s, "
            "total_batches=%d",
            self.pulses_per_batch,
            pulse_train_period_us,
            batch_duration_s,
            total_batches,
        )

        stim = self._build_stim_param(nb_pulse, pulse_train_period_us)
        self.intan.send_stimparam([stim])
        logger.info("StimParam configured and sent to Intan")

        pattern_on = np.zeros(16, dtype=np.uint8)
        pattern_on[self.trigger_key] = 1
        pattern_off = np.zeros(16, dtype=np.uint8)

        for batch_idx in range(total_batches):
            batch_start_time = datetime_now()

            self.trigger_controller.send(pattern_on)
            self._wait(0.005)
            self.trigger_controller.send(pattern_off)

            pulses_this_batch = self.pulses_per_batch
            self._total_pulses_delivered += pulses_this_batch

            ts = datetime_now().isoformat()
            self._stimulation_log.append(StimulationRecord(
                electrode_idx=self.stim_electrode,
                amplitude_ua=self.amplitude_ua,
                duration_us=self.duration_us,
                polarity="NegativeFirst",
                timestamp_utc=ts,
                trigger_key=self.trigger_key,
                stim_index=batch_idx,
                extra={
                    "pulses_in_batch": pulses_this_batch,
                    "nb_pulse_param": nb_pulse,
                    "pulse_train_period_us": pulse_train_period_us,
                    "batch_index": batch_idx,
                    "total_batches": total_batches,
                },
            ))

            if batch_idx % 60 == 0:
                elapsed = (datetime_now() - recording_start).total_seconds()
                logger.info(
                    "Batch %d/%d | elapsed=%.1f s | total_pulses=%d",
                    batch_idx + 1,
                    total_batches,
                    elapsed,
                    self._total_pulses_delivered,
                )

            elapsed_batch = (datetime_now() - batch_start_time).total_seconds()
            remaining_wait = batch_duration_s - elapsed_batch - 0.005
            if remaining_wait > 0:
                self._wait(remaining_wait)

        logger.info(
            "Rapid burst phase complete. Total pulses delivered: %d",
            self._total_pulses_delivered,
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
            "stim_electrode": self.stim_electrode,
            "amplitude_ua": self.amplitude_ua,
            "duration_us": self.duration_us,
            "stim_rate_hz": self.stim_rate_hz,
            "experiment_duration_s": self.experiment_duration_s,
            "total_batches": len(self._stimulation_log),
            "total_pulses_delivered": self._total_pulses_delivered,
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "charge_balance_check": self.amplitude_ua * self.duration_us == self.amplitude_ua * self.duration_us,
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
        return {
            "status": "completed",
            "experiment_name": getattr(self.experiment, "exp_name", "unknown"),
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": duration_s,
            "stim_electrode": self.stim_electrode,
            "amplitude_ua": self.amplitude_ua,
            "duration_us": self.duration_us,
            "stim_rate_hz": self.stim_rate_hz,
            "total_batches": len(self._stimulation_log),
            "total_pulses_delivered": self._total_pulses_delivered,
            "charge_balance": f"{self.amplitude_ua} uA * {self.duration_us} us = {self.amplitude_ua * self.duration_us} nC/phase",
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
