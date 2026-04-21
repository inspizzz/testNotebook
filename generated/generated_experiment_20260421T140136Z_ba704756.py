import numpy as np
import pandas as pd
import json
import logging
import math
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
    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        stim_electrode: int = 17,
        resp_electrode: int = 18,
        amplitude_ua: float = 2.0,
        duration_us: float = 200.0,
        num_trials: int = 50,
        frequencies_hz: Tuple = (0.5, 1.0, 2.0, 5.0, 10.0),
        inter_block_wait_s: float = 30.0,
        response_window_ms: float = 100.0,
        trigger_key: int = 0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.stim_electrode = stim_electrode
        self.resp_electrode = resp_electrode
        self.amplitude_ua = min(abs(amplitude_ua), 4.0)
        self.duration_us = min(abs(duration_us), 400.0)
        self.num_trials = num_trials
        self.frequencies_hz = list(frequencies_hz)
        self.inter_block_wait_s = inter_block_wait_s
        self.response_window_ms = response_window_ms
        self.trigger_key = trigger_key

        # Charge balance: A1*D1 = A2*D2 with equal amplitudes => D2 = D1
        self.amplitude2_ua = self.amplitude_ua
        self.duration2_us = self.duration_us

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._frequency_results: Dict[float, Dict[str, Any]] = {}

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

            for freq_idx, freq_hz in enumerate(self.frequencies_hz):
                logger.info("Starting frequency block %.2f Hz (%d/%d)",
                            freq_hz, freq_idx + 1, len(self.frequencies_hz))
                self._run_frequency_block(freq_hz)

                if freq_idx < len(self.frequencies_hz) - 1:
                    logger.info("Inter-block wait: %.1f s", self.inter_block_wait_s)
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
        stim = self._build_stim_param()
        self.intan.send_stimparam([stim])
        logger.info(
            "Stimulation configured: electrode=%d, A1=%.2f uA, D1=%.1f us, "
            "A2=%.2f uA, D2=%.1f us, polarity=PositiveFirst",
            self.stim_electrode,
            self.amplitude_ua,
            self.duration_us,
            self.amplitude2_ua,
            self.duration2_us,
        )

    def _build_stim_param(self) -> StimParam:
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
        stim.phase_amplitude1 = self.amplitude_ua
        stim.phase_duration1 = self.duration_us
        stim.phase_amplitude2 = self.amplitude2_ua
        stim.phase_duration2 = self.duration2_us
        stim.enable_amp_settle = True
        stim.pre_stim_amp_settle = 0.0
        stim.post_stim_amp_settle = 1000.0
        stim.enable_charge_recovery = True
        stim.post_charge_recovery_on = 0.0
        stim.post_charge_recovery_off = 100.0
        stim.interphase_delay = 0.0
        return stim

    def _run_frequency_block(self, freq_hz: float) -> None:
        inter_stim_interval_s = 1.0 / freq_hz
        response_window_s = self.response_window_ms / 1000.0

        trial_latencies = []
        trial_spike_counts = []
        responding_trials = 0

        block_start = datetime_now()

        for trial_idx in range(self.num_trials):
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
                polarity="PositiveFirst",
                frequency_hz=freq_hz,
                trial_index=trial_idx,
                timestamp_utc=stim_time.isoformat(),
                trigger_key=self.trigger_key,
            ))

            self._wait(response_window_s)

            query_start = stim_time
            query_stop = datetime_now()

            try:
                spike_df = self.database.get_spike_event(
                    query_start,
                    query_stop,
                    self.experiment.exp_name,
                )
            except Exception as exc:
                logger.warning("Spike query failed for trial %d: %s", trial_idx, exc)
                spike_df = pd.DataFrame()

            trial_spikes = []
            if not spike_df.empty:
                channel_col = None
                for col in spike_df.columns:
                    if col.lower() in ("channel", "index", "electrode"):
                        channel_col = col
                        break

                time_col = None
                for col in spike_df.columns:
                    if col.lower() in ("time", "_time"):
                        time_col = col
                        break

                if channel_col is not None and time_col is not None:
                    resp_spikes = spike_df[spike_df[channel_col] == self.resp_electrode]
                    for _, row in resp_spikes.iterrows():
                        spike_time = row[time_col]
                        if hasattr(spike_time, "timestamp"):
                            latency_ms = (spike_time.timestamp() - stim_time.timestamp()) * 1000.0
                        else:
                            latency_ms = float("nan")
                        if 0 < latency_ms <= self.response_window_ms:
                            trial_spikes.append(latency_ms)

            trial_spike_counts.append(len(trial_spikes))
            if len(trial_spikes) > 0:
                responding_trials += 1
                trial_latencies.extend(trial_spikes)

            elapsed_s = (datetime_now() - stim_time).total_seconds()
            remaining_wait = inter_stim_interval_s - elapsed_s
            if remaining_wait > 0:
                self._wait(remaining_wait)

        block_stop = datetime_now()

        response_probability = responding_trials / self.num_trials if self.num_trials > 0 else 0.0
        mean_latency_ms = float(np.mean(trial_latencies)) if trial_latencies else float("nan")
        total_spike_count = sum(trial_spike_counts)

        self._frequency_results[freq_hz] = {
            "frequency_hz": freq_hz,
            "num_trials": self.num_trials,
            "responding_trials": responding_trials,
            "response_probability": response_probability,
            "mean_latency_ms": mean_latency_ms,
            "total_spike_count": total_spike_count,
            "mean_spikes_per_trial": total_spike_count / self.num_trials if self.num_trials > 0 else 0.0,
            "block_start_utc": block_start.isoformat(),
            "block_stop_utc": block_stop.isoformat(),
        }

        logger.info(
            "Freq %.2f Hz: response_prob=%.3f, mean_latency=%.2f ms, total_spikes=%d",
            freq_hz, response_probability, mean_latency_ms if not math.isnan(mean_latency_ms) else -1,
            total_spike_count,
        )

    def _save_all(self, recording_start: datetime, recording_stop: datetime) -> None:
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

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "stim_electrode": self.stim_electrode,
            "resp_electrode": self.resp_electrode,
            "amplitude_ua": self.amplitude_ua,
            "duration_us": self.duration_us,
            "polarity": "PositiveFirst",
            "num_trials_per_frequency": self.num_trials,
            "frequencies_hz": self.frequencies_hz,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "frequency_results": self._frequency_results,
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

        channel_col = None
        for col in spike_df.columns:
            if col.lower() in ("channel", "index", "electrode"):
                channel_col = col
                break

        if channel_col is None:
            logger.warning("Cannot determine electrode column for waveform fetch")
            return waveform_records

        unique_electrodes = spike_df[channel_col].unique()
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

        freq_summary = []
        for freq_hz in self.frequencies_hz:
            if freq_hz in self._frequency_results:
                freq_summary.append(self._frequency_results[freq_hz])

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "stim_electrode": self.stim_electrode,
            "resp_electrode": self.resp_electrode,
            "amplitude_ua": self.amplitude_ua,
            "duration_us": self.duration_us,
            "polarity": "PositiveFirst",
            "num_trials_per_frequency": self.num_trials,
            "frequencies_hz": self.frequencies_hz,
            "frequency_results": freq_summary,
            "total_stimulations": len(self._stimulation_log),
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
