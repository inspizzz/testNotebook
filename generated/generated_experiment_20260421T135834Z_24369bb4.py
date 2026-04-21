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
    probe_index: int = 0
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
        probe_interval_s: float = 60.0,
        probe_phase_duration_s: float = 1200.0,
        probe_amplitude_ua: float = 2.0,
        probe_duration_us: float = 200.0,
        probe_electrode: int = 7,
        trigger_key: int = 0,
        burst_window_s: float = 0.1,
        burst_threshold_spikes: int = 5,
        monitored_electrodes: Tuple[int, ...] = (4, 5, 6, 7, 13, 14, 17, 18, 19, 21, 22),
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.baseline_duration_s = baseline_duration_s
        self.probe_interval_s = probe_interval_s
        self.probe_phase_duration_s = probe_phase_duration_s
        self.probe_amplitude_ua = min(abs(probe_amplitude_ua), 4.0)
        self.probe_duration_us = min(abs(probe_duration_us), 400.0)
        self.probe_electrode = probe_electrode
        self.trigger_key = trigger_key
        self.burst_window_s = burst_window_s
        self.burst_threshold_spikes = burst_threshold_spikes
        self.monitored_electrodes = list(monitored_electrodes)

        # Charge balance: A1*D1 == A2*D2 => same amplitude and duration on both phases
        self._phase_amplitude2 = self.probe_amplitude_ua
        self._phase_duration2 = self.probe_duration_us

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []

        self._baseline_start: Optional[datetime] = None
        self._baseline_stop: Optional[datetime] = None
        self._probe_phase_start: Optional[datetime] = None
        self._probe_phase_stop: Optional[datetime] = None

        self._probe_times: List[datetime] = []
        self._probe_count: int = 0

        self._baseline_spike_counts_per_electrode: Dict[int, int] = {}
        self._probe_phase_spike_counts_per_electrode: Dict[int, int] = {}
        self._burst_events_baseline: List[Dict[str, Any]] = []
        self._burst_events_probe: List[Dict[str, Any]] = []

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
        logger.info("Baseline recording complete")

        try:
            fs_name = self.experiment.exp_name
            spike_df = self.database.get_spike_event(
                self._baseline_start, self._baseline_stop, fs_name
            )
            if not spike_df.empty:
                channel_col = self._get_channel_col(spike_df)
                if channel_col:
                    counts = spike_df.groupby(channel_col).size().to_dict()
                    self._baseline_spike_counts_per_electrode = {int(k): int(v) for k, v in counts.items()}
                self._burst_events_baseline = self._detect_bursts(spike_df, self._baseline_start, self._baseline_stop)
            logger.info("Baseline spike counts per electrode: %s", self._baseline_spike_counts_per_electrode)
            logger.info("Baseline burst events detected: %d", len(self._burst_events_baseline))
        except Exception as exc:
            logger.warning("Could not fetch baseline spike events: %s", exc)

    def _phase_probe_stimulation(self) -> None:
        logger.info(
            "Phase: probe stimulation (%.1f uA, %.0f us) on electrode %d every %.0f s for %.0f s",
            self.probe_amplitude_ua,
            self.probe_duration_us,
            self.probe_electrode,
            self.probe_interval_s,
            self.probe_phase_duration_s,
        )

        self._setup_stim_params()

        self._probe_phase_start = datetime_now()
        phase_end_target = self._probe_phase_start + timedelta(seconds=self.probe_phase_duration_s)

        probe_index = 0
        while True:
            now = datetime_now()
            if now >= phase_end_target:
                break

            self._fire_probe(probe_index)
            probe_index += 1
            self._probe_count += 1

            elapsed = (datetime_now() - self._probe_phase_start).total_seconds()
            next_probe_time = probe_index * self.probe_interval_s
            remaining_phase = self.probe_phase_duration_s - elapsed

            if remaining_phase <= 0:
                break

            wait_time = next_probe_time - elapsed
            if wait_time > remaining_phase:
                self._wait(remaining_phase)
                break
            if wait_time > 0:
                self._wait(wait_time)

        self._probe_phase_stop = datetime_now()
        logger.info("Probe stimulation phase complete. Total probes delivered: %d", self._probe_count)

        try:
            fs_name = self.experiment.exp_name
            spike_df = self.database.get_spike_event(
                self._probe_phase_start, self._probe_phase_stop, fs_name
            )
            if not spike_df.empty:
                channel_col = self._get_channel_col(spike_df)
                if channel_col:
                    counts = spike_df.groupby(channel_col).size().to_dict()
                    self._probe_phase_spike_counts_per_electrode = {int(k): int(v) for k, v in counts.items()}
                self._burst_events_probe = self._detect_bursts(spike_df, self._probe_phase_start, self._probe_phase_stop)
            logger.info("Probe phase spike counts per electrode: %s", self._probe_phase_spike_counts_per_electrode)
            logger.info("Probe phase burst events detected: %d", len(self._burst_events_probe))
        except Exception as exc:
            logger.warning("Could not fetch probe phase spike events: %s", exc)

    def _setup_stim_params(self) -> None:
        stim = StimParam()
        stim.index = self.probe_electrode
        stim.enable = True
        stim.trigger_key = self.trigger_key
        stim.trigger_delay = 0
        stim.nb_pulse = 0
        stim.pulse_train_period = 10000
        stim.post_stim_ref_period = 1000.0
        stim.stim_shape = StimShape.Biphasic
        stim.polarity = StimPolarity.PositiveFirst

        stim.phase_amplitude1 = self.probe_amplitude_ua
        stim.phase_duration1 = self.probe_duration_us
        stim.phase_amplitude2 = self._phase_amplitude2
        stim.phase_duration2 = self._phase_duration2

        stim.enable_amp_settle = True
        stim.pre_stim_amp_settle = 0.0
        stim.post_stim_amp_settle = 1000.0
        stim.enable_charge_recovery = True
        stim.post_charge_recovery_on = 0.0
        stim.post_charge_recovery_off = 100.0
        stim.interphase_delay = 0.0

        self.intan.send_stimparam([stim])
        logger.info(
            "Stim params configured: electrode=%d, A=%.1f uA, D=%.0f us, charge-balanced",
            self.probe_electrode,
            self.probe_amplitude_ua,
            self.probe_duration_us,
        )

    def _fire_probe(self, probe_index: int) -> None:
        stim_time = datetime_now()
        logger.info("Firing probe %d at %s", probe_index, stim_time.isoformat())

        pattern = np.zeros(16, dtype=np.uint8)
        pattern[self.trigger_key] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.05)
        pattern[self.trigger_key] = 0
        self.trigger_controller.send(pattern)

        self._stimulation_log.append(StimulationRecord(
            electrode_idx=self.probe_electrode,
            amplitude_ua=self.probe_amplitude_ua,
            duration_us=self.probe_duration_us,
            polarity="PositiveFirst",
            timestamp_utc=stim_time.isoformat(),
            trigger_key=self.trigger_key,
            probe_index=probe_index,
        ))
        self._probe_times.append(stim_time)

    def _detect_bursts(
        self,
        spike_df: pd.DataFrame,
        phase_start: datetime,
        phase_stop: datetime,
    ) -> List[Dict[str, Any]]:
        burst_events = []
        if spike_df.empty:
            return burst_events

        time_col = self._get_time_col(spike_df)
        if time_col is None:
            return burst_events

        try:
            times = pd.to_datetime(spike_df[time_col], utc=True).sort_values()
            phase_duration_s = (phase_stop - phase_start).total_seconds()
            if phase_duration_s <= 0:
                return burst_events

            bin_size_s = self.burst_window_s
            n_bins = max(1, int(math.ceil(phase_duration_s / bin_size_s)))

            phase_start_ts = pd.Timestamp(phase_start)
            counts_per_bin = np.zeros(n_bins, dtype=int)

            for t in times:
                offset = (t - phase_start_ts).total_seconds()
                bin_idx = int(offset / bin_size_s)
                if 0 <= bin_idx < n_bins:
                    counts_per_bin[bin_idx] += 1

            for i, count in enumerate(counts_per_bin):
                if count >= self.burst_threshold_spikes:
                    burst_time = phase_start + timedelta(seconds=i * bin_size_s)
                    burst_events.append({
                        "bin_index": i,
                        "bin_start_utc": burst_time.isoformat(),
                        "spike_count_in_bin": int(count),
                    })
        except Exception as exc:
            logger.warning("Burst detection error: %s", exc)

        return burst_events

    def _get_channel_col(self, df: pd.DataFrame) -> Optional[str]:
        for col in ["channel", "index", "electrode"]:
            if col in df.columns:
                return col
        for col in df.columns:
            if "channel" in col.lower() or "electrode" in col.lower() or "idx" in col.lower():
                return col
        return None

    def _get_time_col(self, df: pd.DataFrame) -> Optional[str]:
        for col in ["Time", "time", "_time", "timestamp"]:
            if col in df.columns:
                return col
        return None

    def _compute_firing_rates(
        self,
        spike_counts: Dict[int, int],
        duration_s: float,
    ) -> Dict[int, float]:
        if duration_s <= 0:
            return {}
        return {elec: count / duration_s for elec, count in spike_counts.items()}

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        logger.info("Compiling results")

        baseline_duration_s = (
            (self._baseline_stop - self._baseline_start).total_seconds()
            if self._baseline_start and self._baseline_stop
            else self.baseline_duration_s
        )
        probe_duration_s = (
            (self._probe_phase_stop - self._probe_phase_start).total_seconds()
            if self._probe_phase_start and self._probe_phase_stop
            else self.probe_phase_duration_s
        )

        baseline_firing_rates = self._compute_firing_rates(
            self._baseline_spike_counts_per_electrode, baseline_duration_s
        )
        probe_firing_rates = self._compute_firing_rates(
            self._probe_phase_spike_counts_per_electrode, probe_duration_s
        )

        firing_rate_change: Dict[int, float] = {}
        all_electrodes = set(baseline_firing_rates.keys()) | set(probe_firing_rates.keys())
        for elec in all_electrodes:
            base_rate = baseline_firing_rates.get(elec, 0.0)
            probe_rate = probe_firing_rates.get(elec, 0.0)
            firing_rate_change[elec] = probe_rate - base_rate

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "total_duration_seconds": (recording_stop - recording_start).total_seconds(),
            "baseline_duration_s": baseline_duration_s,
            "probe_phase_duration_s": probe_duration_s,
            "probe_electrode": self.probe_electrode,
            "probe_amplitude_ua": self.probe_amplitude_ua,
            "probe_duration_us": self.probe_duration_us,
            "total_probes_delivered": self._probe_count,
            "probe_interval_s": self.probe_interval_s,
            "baseline_spike_counts_per_electrode": self._baseline_spike_counts_per_electrode,
            "probe_phase_spike_counts_per_electrode": self._probe_phase_spike_counts_per_electrode,
            "baseline_firing_rates_hz": baseline_firing_rates,
            "probe_phase_firing_rates_hz": probe_firing_rates,
            "firing_rate_change_hz": firing_rate_change,
            "baseline_burst_events_count": len(self._burst_events_baseline),
            "probe_phase_burst_events_count": len(self._burst_events_probe),
            "baseline_burst_events": self._burst_events_baseline,
            "probe_phase_burst_events": self._burst_events_probe,
            "probe_times_utc": [t.isoformat() for t in self._probe_times],
        }

        return summary

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

        baseline_duration_s = (
            (self._baseline_stop - self._baseline_start).total_seconds()
            if self._baseline_start and self._baseline_stop
            else self.baseline_duration_s
        )
        probe_duration_s = (
            (self._probe_phase_stop - self._probe_phase_start).total_seconds()
            if self._probe_phase_start and self._probe_phase_stop
            else self.probe_phase_duration_s
        )

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "baseline_duration_s": baseline_duration_s,
            "probe_phase_duration_s": probe_duration_s,
            "total_probes_delivered": self._probe_count,
            "probe_electrode": self.probe_electrode,
            "probe_amplitude_ua": self.probe_amplitude_ua,
            "probe_duration_us": self.probe_duration_us,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "baseline_burst_events_count": len(self._burst_events_baseline),
            "probe_phase_burst_events_count": len(self._burst_events_probe),
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

        channel_col = self._get_channel_col(spike_df)
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
