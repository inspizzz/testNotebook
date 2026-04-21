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
    amplitude_level_index: int
    trial_index: int
    timestamp_utc: str
    trigger_key: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialResult:
    amplitude_ua: float
    trial_index: int
    spike_count: int
    spike_times_ms: List[float]
    timestamp_utc: str


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
    Input-output curve experiment using escalating amplitude stimulation.

    The most responsive electrode from the parameter scan is electrode 17
    (stimulating) with electrode 18 responding, showing response_rate=0.92
    and temporal_stability=1.0 at 3 uA PositiveFirst. We use electrode 17
    as the stimulating electrode with PositiveFirst polarity.

    Amplitude sweep: 1.0 to 4.0 uA in 0.5 uA steps (hardware limit is 4.0 uA).
    Note: The objective requests up to 10 uA but the hardware maximum is 4.0 uA.
    We therefore sweep from 1.0 to 4.0 uA in 0.5 uA steps (7 levels).
    Duration is fixed at 400 us. Charge balance: A1*D1 = A2*D2 (equal phases).
    20 stimulations per amplitude level at 1 Hz.
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        stim_electrode: int = 17,
        resp_electrode: int = 18,
        trigger_key: int = 0,
        stim_duration_us: float = 400.0,
        amplitude_start_ua: float = 1.0,
        amplitude_stop_ua: float = 4.0,
        amplitude_step_ua: float = 0.5,
        trials_per_amplitude: int = 20,
        inter_stim_interval_s: float = 1.0,
        inter_level_pause_s: float = 3.0,
        response_window_ms: float = 100.0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.stim_electrode = stim_electrode
        self.resp_electrode = resp_electrode
        self.trigger_key = trigger_key
        self.stim_duration_us = min(abs(stim_duration_us), 400.0)
        self.amplitude_start_ua = amplitude_start_ua
        self.amplitude_stop_ua = min(amplitude_stop_ua, 4.0)
        self.amplitude_step_ua = amplitude_step_ua
        self.trials_per_amplitude = trials_per_amplitude
        self.inter_stim_interval_s = inter_stim_interval_s
        self.inter_level_pause_s = inter_level_pause_s
        self.response_window_ms = response_window_ms

        self._amplitude_levels: List[float] = []
        n_steps = int(round((self.amplitude_stop_ua - self.amplitude_start_ua) / self.amplitude_step_ua)) + 1
        for i in range(n_steps):
            amp = round(self.amplitude_start_ua + i * self.amplitude_step_ua, 6)
            if amp <= 4.0:
                self._amplitude_levels.append(amp)

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._trial_results: List[TrialResult] = []
        self._io_curve: Dict[float, Dict[str, Any]] = {}

    def _wait(self, seconds: float) -> None:
        wait(seconds)

    def run(self) -> Dict[str, Any]:
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
            logger.info("Amplitude levels: %s", self._amplitude_levels)
            logger.info("Trials per level: %d", self.trials_per_amplitude)
            logger.info("Stim electrode: %d, Resp electrode: %d", self.stim_electrode, self.resp_electrode)

            if not self.experiment.start():
                logger.error("Failed to start experiment")
                return {"status": "failed", "error": "Could not start experiment"}

            recording_start = datetime_now()

            self._phase_io_curve()

            recording_stop = datetime_now()

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            if recording_stop is None:
                recording_stop = datetime_now()
            if recording_start is not None and recording_stop is not None:
                try:
                    self._save_all(recording_start, recording_stop)
                except Exception as save_exc:
                    logger.error("Failed to save data after error: %s", save_exc)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _phase_io_curve(self) -> None:
        logger.info("Phase: input-output curve sweep")
        logger.info("Amplitude levels to test: %s", self._amplitude_levels)

        for level_idx, amplitude_ua in enumerate(self._amplitude_levels):
            logger.info(
                "Amplitude level %d/%d: %.2f uA",
                level_idx + 1,
                len(self._amplitude_levels),
                amplitude_ua,
            )

            level_spike_counts = []
            level_spike_times = []

            for trial_idx in range(self.trials_per_amplitude):
                trial_start = datetime_now()

                spike_df = self._stimulate_and_record(
                    electrode_idx=self.stim_electrode,
                    amplitude_ua=amplitude_ua,
                    duration_us=self.stim_duration_us,
                    polarity=StimPolarity.PositiveFirst,
                    trigger_key=self.trigger_key,
                    amplitude_level_index=level_idx,
                    trial_index=trial_idx,
                )

                trial_end = datetime_now()

                spike_count = 0
                spike_times_ms = []

                if not spike_df.empty:
                    resp_spikes = spike_df
                    if "channel" in spike_df.columns:
                        resp_spikes = spike_df[spike_df["channel"] == self.resp_electrode]
                    spike_count = len(resp_spikes)
                    if spike_count > 0 and "Time" in resp_spikes.columns:
                        stim_time = trial_start
                        for t in resp_spikes["Time"]:
                            try:
                                if hasattr(t, "timestamp"):
                                    latency_ms = (t.timestamp() - stim_time.timestamp()) * 1000.0
                                else:
                                    latency_ms = float("nan")
                                spike_times_ms.append(latency_ms)
                            except Exception:
                                spike_times_ms.append(float("nan"))

                level_spike_counts.append(spike_count)
                level_spike_times.extend(spike_times_ms)

                self._trial_results.append(TrialResult(
                    amplitude_ua=amplitude_ua,
                    trial_index=trial_idx,
                    spike_count=spike_count,
                    spike_times_ms=spike_times_ms,
                    timestamp_utc=trial_start.isoformat(),
                ))

                logger.info(
                    "  Trial %d/%d: amplitude=%.2f uA, spikes=%d",
                    trial_idx + 1,
                    self.trials_per_amplitude,
                    amplitude_ua,
                    spike_count,
                )

                if trial_idx < self.trials_per_amplitude - 1:
                    self._wait(self.inter_stim_interval_s)

            mean_spikes = float(np.mean(level_spike_counts)) if level_spike_counts else 0.0
            response_rate = float(np.mean([1 if c > 0 else 0 for c in level_spike_counts])) if level_spike_counts else 0.0
            mean_latency = float(np.nanmean(level_spike_times)) if level_spike_times else float("nan")

            self._io_curve[amplitude_ua] = {
                "amplitude_ua": amplitude_ua,
                "trials": self.trials_per_amplitude,
                "mean_spike_count": mean_spikes,
                "response_rate": response_rate,
                "mean_latency_ms": mean_latency,
                "spike_counts_per_trial": level_spike_counts,
            }

            logger.info(
                "Level %.2f uA summary: mean_spikes=%.2f, response_rate=%.2f, mean_latency=%.2f ms",
                amplitude_ua,
                mean_spikes,
                response_rate,
                mean_latency,
            )

            if level_idx < len(self._amplitude_levels) - 1:
                self._wait(self.inter_level_pause_s)

        logger.info("Input-output curve sweep complete. %d levels tested.", len(self._amplitude_levels))

    def _stimulate_and_record(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity = StimPolarity.PositiveFirst,
        trigger_key: int = 0,
        amplitude_level_index: int = 0,
        trial_index: int = 0,
        post_stim_wait_s: float = 0.5,
        recording_window_s: float = 0.2,
    ) -> pd.DataFrame:
        amplitude_ua = min(abs(amplitude_ua), 4.0)
        duration_us = min(abs(duration_us), 400.0)

        stim = StimParam()
        stim.index = electrode_idx
        stim.enable = True
        stim.trigger_key = trigger_key
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

        self.intan.send_stimparam([stim])

        stim_timestamp = datetime_now()

        pattern = np.zeros(16, dtype=np.uint8)
        pattern[trigger_key] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.05)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            polarity=polarity.name,
            amplitude_level_index=amplitude_level_index,
            trial_index=trial_index,
            timestamp_utc=stim_timestamp.isoformat(),
            trigger_key=trigger_key,
        ))

        self._wait(post_stim_wait_s)

        query_stop = datetime_now()
        query_start = query_stop - timedelta(seconds=post_stim_wait_s + recording_window_s)

        try:
            spike_df = self.database.get_spike_event(
                query_start, query_stop, self.experiment.exp_name
            )
        except Exception as exc:
            logger.warning("Failed to fetch spike events: %s", exc)
            spike_df = pd.DataFrame()

        return spike_df

    def _save_all(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> None:
        fs_name = getattr(self.experiment, "exp_name", "unknown")
        saver = DataSaver(self._output_dir, fs_name)

        saver.save_stimulation_log(self._stimulation_log)

        try:
            spike_df = self.database.get_spike_event(
                recording_start, recording_stop, fs_name
            )
        except Exception as exc:
            logger.warning("Failed to fetch spike events for save_all: %s", exc)
            spike_df = pd.DataFrame()
        saver.save_spike_events(spike_df)

        try:
            trigger_df = self.database.get_all_triggers(
                recording_start, recording_stop
            )
        except Exception as exc:
            logger.warning("Failed to fetch triggers for save_all: %s", exc)
            trigger_df = pd.DataFrame()
        saver.save_triggers(trigger_df)

        io_curve_serializable = {}
        for amp, data in self._io_curve.items():
            io_curve_serializable[str(amp)] = data

        trial_results_list = [asdict(r) for r in self._trial_results]

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "stim_electrode": self.stim_electrode,
            "resp_electrode": self.resp_electrode,
            "amplitude_levels_ua": self._amplitude_levels,
            "trials_per_amplitude": self.trials_per_amplitude,
            "stim_duration_us": self.stim_duration_us,
            "polarity": "PositiveFirst",
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "io_curve": io_curve_serializable,
            "trial_results": trial_results_list,
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
        waveform_records = []
        if spike_df.empty:
            return waveform_records

        electrode_col = None
        for col in ["channel", "index", "electrode"]:
            if col in spike_df.columns:
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

        io_curve_summary = []
        for amp in self._amplitude_levels:
            if amp in self._io_curve:
                entry = self._io_curve[amp]
                io_curve_summary.append({
                    "amplitude_ua": amp,
                    "mean_spike_count": entry["mean_spike_count"],
                    "response_rate": entry["response_rate"],
                    "mean_latency_ms": entry["mean_latency_ms"],
                })

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "stim_electrode": self.stim_electrode,
            "resp_electrode": self.resp_electrode,
            "amplitude_levels_ua": self._amplitude_levels,
            "trials_per_amplitude": self.trials_per_amplitude,
            "total_stimulations": len(self._stimulation_log),
            "io_curve": io_curve_summary,
        }

        return summary

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
