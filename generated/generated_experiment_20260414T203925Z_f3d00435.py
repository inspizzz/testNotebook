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
    timestamp_utc: str
    trigger_key: int = 0
    phase: str = "probe"
    trial_index: int = 0
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
        baseline_duration_s: float = 600.0,
        probe_phase_duration_s: float = 1200.0,
        probe_interval_s: float = 60.0,
        probe_electrode: int = 0,
        probe_amplitude_ua: float = 2.0,
        probe_duration_us: float = 200.0,
        burst_window_s: float = 0.1,
        burst_threshold_spikes: int = 5,
        ibi_min_s: float = 0.5,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.baseline_duration_s = baseline_duration_s
        self.probe_phase_duration_s = probe_phase_duration_s
        self.probe_interval_s = probe_interval_s
        self.probe_electrode = probe_electrode
        self.probe_amplitude_ua = probe_amplitude_ua
        self.probe_duration_us = probe_duration_us
        self.burst_window_s = burst_window_s
        self.burst_threshold_spikes = burst_threshold_spikes
        self.ibi_min_s = ibi_min_s

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []

        self._baseline_start: Optional[datetime] = None
        self._baseline_stop: Optional[datetime] = None
        self._probe_start: Optional[datetime] = None
        self._probe_stop: Optional[datetime] = None

        self._probe_stim_times: List[str] = []
        self._analysis_results: Dict[str, Any] = {}

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

            self._phase_baseline()
            self._phase_probe_stimulation()

            recording_stop = datetime_now()

            self._analysis_results = self._analyse(recording_start, recording_stop)

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _phase_baseline(self) -> None:
        logger.info("Phase: baseline recording (no stimulation) for %.0f s", self.baseline_duration_s)
        self._baseline_start = datetime_now()
        self._wait(self.baseline_duration_s)
        self._baseline_stop = datetime_now()
        logger.info("Baseline phase complete")

    def _phase_probe_stimulation(self) -> None:
        logger.info(
            "Phase: probe stimulation every %.0f s for %.0f s total",
            self.probe_interval_s,
            self.probe_phase_duration_s,
        )
        self._probe_start = datetime_now()

        num_probes = int(self.probe_phase_duration_s / self.probe_interval_s)
        logger.info("Expected number of probe stimulations: %d", num_probes)

        for trial_idx in range(num_probes):
            logger.info("Probe trial %d / %d", trial_idx + 1, num_probes)
            self._deliver_probe(trial_idx)
            self._wait(self.probe_interval_s)

        self._probe_stop = datetime_now()
        logger.info("Probe stimulation phase complete")

    def _deliver_probe(self, trial_idx: int) -> None:
        amplitude_ua = self.probe_amplitude_ua
        duration_us = self.probe_duration_us

        amplitude_ua = min(abs(amplitude_ua), 4.0)
        duration_us = min(abs(duration_us), 400.0)

        stim = StimParam()
        stim.index = self.probe_electrode
        stim.enable = True
        stim.trigger_key = 0
        stim.trigger_delay = 0
        stim.nb_pulse = 0
        stim.pulse_train_period = 10000
        stim.post_stim_ref_period = 1000.0
        stim.stim_shape = StimShape.Biphasic
        stim.polarity = StimPolarity.PositiveFirst

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

        self.intan.send_stimparam([stim])

        pattern = np.zeros(16, dtype=np.uint8)
        pattern[0] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.05)
        pattern[0] = 0
        self.trigger_controller.send(pattern)

        ts = datetime_now().isoformat()
        self._probe_stim_times.append(ts)
        self._stimulation_log.append(StimulationRecord(
            electrode_idx=self.probe_electrode,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            timestamp_utc=ts,
            trigger_key=0,
            phase="probe",
            trial_index=trial_idx,
        ))

    def _analyse(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        logger.info("Running analysis")
        fs_name = getattr(self.experiment, "exp_name", "unknown")

        try:
            spike_df = self.database.get_spike_event(recording_start, recording_stop, fs_name)
        except Exception as exc:
            logger.warning("Could not fetch spike events for analysis: %s", exc)
            spike_df = pd.DataFrame()

        results: Dict[str, Any] = {}

        if spike_df.empty:
            logger.info("No spike events found for analysis")
            results["note"] = "no spike events"
            return results

        time_col = None
        for candidate in ["Time", "time", "_time", "timestamp"]:
            if candidate in spike_df.columns:
                time_col = candidate
                break

        channel_col = None
        for candidate in ["channel", "index", "electrode", "Channel"]:
            if candidate in spike_df.columns:
                channel_col = candidate
                break

        if time_col is None or channel_col is None:
            logger.warning("Cannot identify time or channel columns in spike_df: %s", list(spike_df.columns))
            results["note"] = "column identification failed"
            return results

        spike_df = spike_df.copy()
        spike_df[time_col] = pd.to_datetime(spike_df[time_col], utc=True)

        baseline_start_dt = self._baseline_start
        baseline_stop_dt = self._baseline_stop
        probe_start_dt = self._probe_start
        probe_stop_dt = self._probe_stop

        if baseline_start_dt is None or baseline_stop_dt is None:
            logger.warning("Baseline timestamps not set")
            return results

        baseline_mask = (spike_df[time_col] >= baseline_start_dt) & (spike_df[time_col] < baseline_stop_dt)
        baseline_spikes = spike_df[baseline_mask]

        baseline_duration_s = (baseline_stop_dt - baseline_start_dt).total_seconds()
        if baseline_duration_s <= 0:
            baseline_duration_s = 1.0

        firing_rates: Dict[int, float] = {}
        unique_channels = spike_df[channel_col].unique()
        for ch in unique_channels:
            ch_baseline = baseline_spikes[baseline_spikes[channel_col] == ch]
            rate = len(ch_baseline) / baseline_duration_s
            firing_rates[int(ch)] = round(rate, 4)

        results["baseline_firing_rates_hz"] = firing_rates
        results["baseline_total_spikes"] = int(len(baseline_spikes))
        results["baseline_duration_s"] = baseline_duration_s

        bursts_baseline = self._detect_bursts(baseline_spikes, time_col, channel_col)
        results["baseline_burst_count"] = len(bursts_baseline)

        if probe_start_dt is not None and probe_stop_dt is not None:
            probe_mask = (spike_df[time_col] >= probe_start_dt) & (spike_df[time_col] < probe_stop_dt)
            probe_spikes = spike_df[probe_mask]
            probe_duration_s = (probe_stop_dt - probe_start_dt).total_seconds()
            if probe_duration_s <= 0:
                probe_duration_s = 1.0

            probe_firing_rates: Dict[int, float] = {}
            for ch in unique_channels:
                ch_probe = probe_spikes[probe_spikes[channel_col] == ch]
                rate = len(ch_probe) / probe_duration_s
                probe_firing_rates[int(ch)] = round(rate, 4)

            results["probe_firing_rates_hz"] = probe_firing_rates
            results["probe_total_spikes"] = int(len(probe_spikes))
            results["probe_duration_s"] = probe_duration_s

            bursts_probe = self._detect_bursts(probe_spikes, time_col, channel_col)
            results["probe_burst_count"] = len(bursts_probe)

            rate_changes: Dict[int, float] = {}
            for ch in unique_channels:
                ch_int = int(ch)
                baseline_r = firing_rates.get(ch_int, 0.0)
                probe_r = probe_firing_rates.get(ch_int, 0.0)
                if baseline_r > 0:
                    rate_changes[ch_int] = round((probe_r - baseline_r) / baseline_r, 4)
                else:
                    rate_changes[ch_int] = float("nan")
            results["firing_rate_change_fraction"] = rate_changes

            evoked_window_s = 0.1
            evoked_counts: List[int] = []
            for stim_ts_str in self._probe_stim_times:
                try:
                    stim_ts = pd.Timestamp(stim_ts_str).tz_convert("UTC")
                    window_end = stim_ts + pd.Timedelta(seconds=evoked_window_s)
                    evoked_mask = (
                        (spike_df[time_col] >= stim_ts) &
                        (spike_df[time_col] < window_end) &
                        (spike_df[channel_col] != self.probe_electrode)
                    )
                    evoked_counts.append(int(evoked_mask.sum()))
                except Exception as exc:
                    logger.warning("Error computing evoked count for stim at %s: %s", stim_ts_str, exc)

            if evoked_counts:
                results["mean_evoked_spikes_per_probe"] = round(float(np.mean(evoked_counts)), 4)
                results["evoked_counts_per_trial"] = evoked_counts

        return results

    def _detect_bursts(
        self,
        spike_df: pd.DataFrame,
        time_col: str,
        channel_col: str,
    ) -> List[Dict[str, Any]]:
        if spike_df.empty:
            return []

        all_times = spike_df[time_col].sort_values().values
        if len(all_times) == 0:
            return []

        bursts: List[Dict[str, Any]] = []
        window_ns = int(self.burst_window_s * 1e9)
        ibi_ns = int(self.ibi_min_s * 1e9)

        i = 0
        n = len(all_times)
        while i < n:
            window_end = all_times[i] + np.timedelta64(window_ns, "ns")
            j = i
            while j < n and all_times[j] <= window_end:
                j += 1
            count_in_window = j - i
            if count_in_window >= self.burst_threshold_spikes:
                burst_start = all_times[i]
                burst_end = all_times[j - 1]
                bursts.append({
                    "start": str(burst_start),
                    "end": str(burst_end),
                    "spike_count": count_in_window,
                })
                next_start = burst_end + np.timedelta64(ibi_ns, "ns")
                while i < n and all_times[i] < next_start:
                    i += 1
            else:
                i += 1

        return bursts

    def _save_all(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> None:
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

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "baseline_duration_s": self.baseline_duration_s,
            "probe_phase_duration_s": self.probe_phase_duration_s,
            "probe_interval_s": self.probe_interval_s,
            "probe_electrode": self.probe_electrode,
            "probe_amplitude_ua": self.probe_amplitude_ua,
            "probe_duration_us": self.probe_duration_us,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "analysis": self._analysis_results,
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

        channel_col = None
        for candidate in ["channel", "index", "electrode", "Channel"]:
            if candidate in spike_df.columns:
                channel_col = candidate
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

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": getattr(self.experiment, "exp_name", "unknown"),
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "baseline_duration_s": self.baseline_duration_s,
            "probe_phase_duration_s": self.probe_phase_duration_s,
            "probe_interval_s": self.probe_interval_s,
            "probe_electrode": self.probe_electrode,
            "probe_amplitude_ua": self.probe_amplitude_ua,
            "probe_duration_us": self.probe_duration_us,
            "total_probe_stimulations": len(self._stimulation_log),
            "analysis": self._analysis_results,
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
