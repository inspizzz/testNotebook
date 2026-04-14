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
    frequency_hz: float
    trial_index: int
    frequency_block_index: int
    timestamp_utc: str
    trigger_key: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrequencyBlockResult:
    frequency_hz: float
    num_stimulations: int
    num_responses: int
    response_probability: float
    mean_latency_ms: float
    total_spike_count: int
    latencies_ms: List[float] = field(default_factory=list)


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
    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        stim_electrode: int = 0,
        response_electrode: int = 1,
        amplitude_ua: float = 2.0,
        duration_us: float = 200.0,
        polarity: str = "PositiveFirst",
        num_trials_per_frequency: int = 50,
        frequencies_hz: List[float] = None,
        inter_block_wait_s: float = 30.0,
        response_window_ms: float = 50.0,
        trigger_key: int = 0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.stim_electrode = stim_electrode
        self.response_electrode = response_electrode
        self.amplitude_ua = amplitude_ua
        self.duration_us = duration_us
        self.polarity_str = polarity
        self.num_trials_per_frequency = num_trials_per_frequency
        self.frequencies_hz = frequencies_hz if frequencies_hz is not None else [0.5, 1.0, 2.0, 5.0, 10.0]
        self.inter_block_wait_s = inter_block_wait_s
        self.response_window_ms = response_window_ms
        self.trigger_key = trigger_key

        if polarity == "PositiveFirst":
            self.stim_polarity = StimPolarity.PositiveFirst
        else:
            self.stim_polarity = StimPolarity.NegativeFirst

        assert abs(self.amplitude_ua * self.duration_us - self.amplitude_ua * self.duration_us) < 1e-9

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._frequency_block_results: List[FrequencyBlockResult] = []

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

            for block_idx, freq_hz in enumerate(self.frequencies_hz):
                logger.info("Starting frequency block %d: %.2f Hz", block_idx, freq_hz)
                block_result = self._run_frequency_block(freq_hz, block_idx)
                self._frequency_block_results.append(block_result)
                logger.info(
                    "Block %.2f Hz done: response_prob=%.3f, mean_latency=%.2f ms, spike_count=%d",
                    freq_hz,
                    block_result.response_probability,
                    block_result.mean_latency_ms,
                    block_result.total_spike_count,
                )
                if block_idx < len(self.frequencies_hz) - 1:
                    logger.info("Waiting %s s between frequency blocks", self.inter_block_wait_s)
                    self._wait(self.inter_block_wait_s)

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
        stim = self._build_stim_param(self.stim_electrode)
        self.intan.send_stimparam([stim])
        logger.info(
            "Stimulation configured: electrode=%d, amplitude=%.2f uA, duration=%.1f us, polarity=%s",
            self.stim_electrode,
            self.amplitude_ua,
            self.duration_us,
            self.polarity_str,
        )

    def _build_stim_param(self, electrode_idx: int) -> StimParam:
        amplitude_ua = min(abs(self.amplitude_ua), 4.0)
        duration_us = min(abs(self.duration_us), 400.0)

        stim = StimParam()
        stim.index = electrode_idx
        stim.enable = True
        stim.trigger_key = self.trigger_key
        stim.trigger_delay = 0
        stim.nb_pulse = 0
        stim.pulse_train_period = 10000
        stim.post_stim_ref_period = 1000.0
        stim.stim_shape = StimShape.Biphasic
        stim.polarity = self.stim_polarity

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

    def _run_frequency_block(self, freq_hz: float, block_idx: int) -> FrequencyBlockResult:
        inter_stim_interval_s = 1.0 / freq_hz
        response_window_s = self.response_window_ms / 1000.0

        num_responses = 0
        total_spike_count = 0
        latencies_ms: List[float] = []

        for trial_idx in range(self.num_trials_per_frequency):
            stim_time = datetime_now()

            pattern = np.zeros(16, dtype=np.uint8)
            pattern[self.trigger_key] = 1
            self.trigger_controller.send(pattern)
            self._wait(0.005)
            pattern[self.trigger_key] = 0
            self.trigger_controller.send(pattern)

            self._stimulation_log.append(StimulationRecord(
                electrode_idx=self.stim_electrode,
                amplitude_ua=self.amplitude_ua,
                duration_us=self.duration_us,
                polarity=self.polarity_str,
                frequency_hz=freq_hz,
                trial_index=trial_idx,
                frequency_block_index=block_idx,
                timestamp_utc=stim_time.isoformat(),
                trigger_key=self.trigger_key,
            ))

            self._wait(response_window_s)

            query_stop = datetime_now()
            query_start = stim_time

            try:
                spike_df = self.database.get_spike_event_electrode(
                    query_start, query_stop, self.response_electrode
                )

                if not spike_df.empty:
                    time_col = None
                    for col in spike_df.columns:
                        if col.lower() in ("time", "_time", "timestamp"):
                            time_col = col
                            break

                    if time_col is not None:
                        spike_times = pd.to_datetime(spike_df[time_col], utc=True)
                        stim_time_utc = stim_time if stim_time.tzinfo is not None else stim_time.replace(tzinfo=timezone.utc)
                        valid_spikes = spike_times[spike_times >= stim_time_utc]
                        if len(valid_spikes) > 0:
                            num_responses += 1
                            total_spike_count += len(valid_spikes)
                            for st in valid_spikes:
                                latency_ms = (st - stim_time_utc).total_seconds() * 1000.0
                                if 0 < latency_ms <= self.response_window_ms:
                                    latencies_ms.append(latency_ms)
                    else:
                        if len(spike_df) > 0:
                            num_responses += 1
                            total_spike_count += len(spike_df)

            except Exception as exc:
                logger.warning("Error querying spikes for trial %d: %s", trial_idx, exc)

            elapsed_s = (datetime_now() - stim_time).total_seconds()
            remaining_s = inter_stim_interval_s - elapsed_s
            if remaining_s > 0:
                self._wait(remaining_s)

        response_probability = num_responses / self.num_trials_per_frequency if self.num_trials_per_frequency > 0 else 0.0
        mean_latency_ms = float(np.mean(latencies_ms)) if latencies_ms else 0.0

        return FrequencyBlockResult(
            frequency_hz=freq_hz,
            num_stimulations=self.num_trials_per_frequency,
            num_responses=num_responses,
            response_probability=response_probability,
            mean_latency_ms=mean_latency_ms,
            total_spike_count=total_spike_count,
            latencies_ms=latencies_ms,
        )

    def _save_all(self, recording_start: datetime, recording_stop: datetime) -> None:
        fs_name = getattr(self.experiment, "exp_name", "unknown")
        saver = DataSaver(self._output_dir, fs_name)

        saver.save_stimulation_log(self._stimulation_log)

        try:
            spike_df = self.database.get_spike_event(recording_start, recording_stop, fs_name)
        except Exception as exc:
            logger.warning("Failed to fetch spike events: %s", exc)
            spike_df = pd.DataFrame()

        saver.save_spike_events(spike_df)

        try:
            trigger_df = self.database.get_all_triggers(recording_start, recording_stop)
        except Exception as exc:
            logger.warning("Failed to fetch triggers: %s", exc)
            trigger_df = pd.DataFrame()

        saver.save_triggers(trigger_df)

        freq_results_serializable = []
        for r in self._frequency_block_results:
            freq_results_serializable.append({
                "frequency_hz": r.frequency_hz,
                "num_stimulations": r.num_stimulations,
                "num_responses": r.num_responses,
                "response_probability": r.response_probability,
                "mean_latency_ms": r.mean_latency_ms,
                "total_spike_count": r.total_spike_count,
                "latencies_ms": r.latencies_ms,
            })

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "stim_electrode": self.stim_electrode,
            "response_electrode": self.response_electrode,
            "amplitude_ua": self.amplitude_ua,
            "duration_us": self.duration_us,
            "polarity": self.polarity_str,
            "frequencies_hz": self.frequencies_hz,
            "num_trials_per_frequency": self.num_trials_per_frequency,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "frequency_block_results": freq_results_serializable,
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
                    electrode_idx, exc
                )

        return waveform_records

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        logger.info("Compiling results")

        freq_results_serializable = []
        for r in self._frequency_block_results:
            freq_results_serializable.append({
                "frequency_hz": r.frequency_hz,
                "num_stimulations": r.num_stimulations,
                "num_responses": r.num_responses,
                "response_probability": r.response_probability,
                "mean_latency_ms": r.mean_latency_ms,
                "total_spike_count": r.total_spike_count,
            })

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "stim_electrode": self.stim_electrode,
            "response_electrode": self.response_electrode,
            "amplitude_ua": self.amplitude_ua,
            "duration_us": self.duration_us,
            "polarity": self.polarity_str,
            "frequencies_hz": self.frequencies_hz,
            "num_trials_per_frequency": self.num_trials_per_frequency,
            "total_stimulations": len(self._stimulation_log),
            "frequency_block_results": freq_results_serializable,
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
