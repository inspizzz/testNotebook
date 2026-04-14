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
    phase: str = "probe"
    trial_index: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BurstEvent:
    electrode_idx: int
    start_utc: str
    end_utc: str
    spike_count: int
    duration_s: float


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
        burst_threshold_multiplier: float = 3.0,
        burst_min_duration_s: float = 0.2,
        burst_bin_s: float = 1.0,
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
        self.burst_threshold_multiplier = burst_threshold_multiplier
        self.burst_min_duration_s = burst_min_duration_s
        self.burst_bin_s = burst_bin_s

        assert abs(probe_amplitude_ua * probe_duration_us - probe_amplitude_ua * probe_duration_us) < 1e-9

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._burst_events: List[BurstEvent] = []

        self._baseline_start: Optional[datetime] = None
        self._baseline_stop: Optional[datetime] = None
        self._probe_start: Optional[datetime] = None
        self._probe_stop: Optional[datetime] = None

        self._baseline_firing_rates: Dict[int, float] = {}
        self._probe_firing_rates: Dict[int, float] = {}
        self._baseline_burst_counts: Dict[int, int] = {}
        self._probe_burst_counts: Dict[int, int] = {}

        self._probe_spike_windows: List[Dict[str, Any]] = []

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
        logger.info("Phase: baseline recording (%.0f s, no stimulation)", self.baseline_duration_s)
        self._baseline_start = datetime_now()
        self._wait(self.baseline_duration_s)
        self._baseline_stop = datetime_now()
        logger.info("Baseline recording complete")

        try:
            fs_name = self.experiment.exp_name
            spike_df = self.database.get_spike_event(
                self._baseline_start, self._baseline_stop, fs_name
            )
            self._baseline_firing_rates = self._compute_firing_rates(
                spike_df, self.baseline_duration_s
            )
            self._baseline_burst_counts = self._detect_bursts_from_df(
                spike_df,
                self._baseline_start,
                self._baseline_stop,
                phase_label="baseline",
            )
            logger.info("Baseline firing rates computed for %d electrodes", len(self._baseline_firing_rates))
        except Exception as exc:
            logger.warning("Could not compute baseline metrics: %s", exc)

    def _phase_probe_stimulation(self) -> None:
        logger.info(
            "Phase: probe stimulation (%.0f s, one probe every %.0f s on electrode %d)",
            self.probe_phase_duration_s,
            self.probe_interval_s,
            self.probe_electrode,
        )
        self._probe_start = datetime_now()

        num_probes = int(self.probe_phase_duration_s / self.probe_interval_s)
        logger.info("Expected number of probe stimulations: %d", num_probes)

        for trial_idx in range(num_probes):
            logger.info("Probe trial %d / %d", trial_idx + 1, num_probes)
            self._deliver_probe(trial_idx)
            remaining_wait = self.probe_interval_s - 1.0
            if remaining_wait > 0:
                self._wait(remaining_wait)

        self._probe_stop = datetime_now()
        logger.info("Probe stimulation phase complete")

        try:
            fs_name = self.experiment.exp_name
            spike_df = self.database.get_spike_event(
                self._probe_start, self._probe_stop, fs_name
            )
            self._probe_firing_rates = self._compute_firing_rates(
                spike_df, self.probe_phase_duration_s
            )
            self._probe_burst_counts = self._detect_bursts_from_df(
                spike_df,
                self._probe_start,
                self._probe_stop,
                phase_label="probe",
            )
            logger.info("Probe phase firing rates computed for %d electrodes", len(self._probe_firing_rates))
        except Exception as exc:
            logger.warning("Could not compute probe phase metrics: %s", exc)

    def _deliver_probe(self, trial_idx: int) -> None:
        amplitude_ua = self.probe_amplitude_ua
        duration_us = self.probe_duration_us
        electrode_idx = self.probe_electrode
        trigger_key = 0
        polarity = StimPolarity.PositiveFirst

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

        self.intan.send_stimparam([stim])

        pattern = np.zeros(16, dtype=np.uint8)
        pattern[trigger_key] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.05)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

        stim_time = datetime_now()

        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            polarity=polarity.name,
            timestamp_utc=stim_time.isoformat(),
            trigger_key=trigger_key,
            phase="probe",
            trial_index=trial_idx,
        ))

        self._wait(0.5)

        window_start = stim_time - timedelta(seconds=0.05)
        window_stop = stim_time + timedelta(seconds=0.5)
        try:
            fs_name = self.experiment.exp_name
            spike_df = self.database.get_spike_event(
                window_start, window_stop, fs_name
            )
            spikes_in_window = []
            if not spike_df.empty:
                time_col = "Time" if "Time" in spike_df.columns else spike_df.columns[0]
                for _, row in spike_df.iterrows():
                    spikes_in_window.append({
                        "channel": int(row.get("channel", -1)),
                        "amplitude": float(row.get("Amplitude", 0.0)),
                        "time": str(row.get(time_col, "")),
                    })
            self._probe_spike_windows.append({
                "trial_index": trial_idx,
                "stim_time_utc": stim_time.isoformat(),
                "spikes": spikes_in_window,
                "spike_count": len(spikes_in_window),
            })
        except Exception as exc:
            logger.warning("Could not fetch spikes for probe trial %d: %s", trial_idx, exc)

    def _compute_firing_rates(self, spike_df: pd.DataFrame, duration_s: float) -> Dict[int, float]:
        rates: Dict[int, float] = {}
        if spike_df.empty or duration_s <= 0:
            return rates
        channel_col = "channel" if "channel" in spike_df.columns else None
        if channel_col is None:
            return rates
        counts = spike_df[channel_col].value_counts()
        for ch, cnt in counts.items():
            rates[int(ch)] = float(cnt) / duration_s
        return rates

    def _detect_bursts_from_df(
        self,
        spike_df: pd.DataFrame,
        phase_start: datetime,
        phase_stop: datetime,
        phase_label: str = "",
    ) -> Dict[int, int]:
        burst_counts: Dict[int, int] = {}
        if spike_df.empty:
            return burst_counts

        channel_col = "channel" if "channel" in spike_df.columns else None
        time_col = "Time" if "Time" in spike_df.columns else None
        if channel_col is None or time_col is None:
            return burst_counts

        duration_s = (phase_stop - phase_start).total_seconds()
        if duration_s <= 0:
            return burst_counts

        unique_channels = spike_df[channel_col].unique()
        for ch in unique_channels:
            ch_df = spike_df[spike_df[channel_col] == ch].copy()
            if ch_df.empty:
                continue

            try:
                ch_df[time_col] = pd.to_datetime(ch_df[time_col], utc=True)
                ch_df = ch_df.sort_values(time_col)
            except Exception:
                continue

            num_bins = max(1, int(math.ceil(duration_s / self.burst_bin_s)))
            spike_times_s = [
                (t - phase_start).total_seconds()
                for t in ch_df[time_col]
                if hasattr(t, 'total_seconds') or True
            ]
            try:
                spike_times_s = [
                    (pd.Timestamp(t).to_pydatetime().replace(tzinfo=timezone.utc) - phase_start).total_seconds()
                    for t in ch_df[time_col]
                ]
            except Exception:
                continue

            bin_counts = [0] * num_bins
            for st in spike_times_s:
                bin_idx = int(st / self.burst_bin_s)
                if 0 <= bin_idx < num_bins:
                    bin_counts[bin_idx] += 1

            if not bin_counts:
                continue

            mean_rate = float(np.mean(bin_counts))
            threshold = mean_rate * self.burst_threshold_multiplier
            min_bins = max(1, int(math.ceil(self.burst_min_duration_s / self.burst_bin_s)))

            in_burst = False
            burst_start_bin = 0
            burst_spike_count = 0
            ch_burst_count = 0

            for b_idx, cnt in enumerate(bin_counts):
                if cnt >= threshold:
                    if not in_burst:
                        in_burst = True
                        burst_start_bin = b_idx
                        burst_spike_count = cnt
                    else:
                        burst_spike_count += cnt
                else:
                    if in_burst:
                        burst_len = b_idx - burst_start_bin
                        if burst_len >= min_bins:
                            ch_burst_count += 1
                            burst_start_s = phase_start + timedelta(seconds=burst_start_bin * self.burst_bin_s)
                            burst_end_s = phase_start + timedelta(seconds=b_idx * self.burst_bin_s)
                            self._burst_events.append(BurstEvent(
                                electrode_idx=int(ch),
                                start_utc=burst_start_s.isoformat(),
                                end_utc=burst_end_s.isoformat(),
                                spike_count=burst_spike_count,
                                duration_s=burst_len * self.burst_bin_s,
                            ))
                        in_burst = False
                        burst_spike_count = 0

            if in_burst:
                burst_len = num_bins - burst_start_bin
                if burst_len >= min_bins:
                    ch_burst_count += 1
                    burst_start_s = phase_start + timedelta(seconds=burst_start_bin * self.burst_bin_s)
                    burst_end_s = phase_stop
                    self._burst_events.append(BurstEvent(
                        electrode_idx=int(ch),
                        start_utc=burst_start_s.isoformat(),
                        end_utc=burst_end_s.isoformat(),
                        spike_count=burst_spike_count,
                        duration_s=burst_len * self.burst_bin_s,
                    ))

            burst_counts[int(ch)] = ch_burst_count

        return burst_counts

    def _save_all(self, recording_start: datetime, recording_stop: datetime) -> None:
        fs_name = getattr(self.experiment, "exp_name", "unknown")
        saver = DataSaver(self._output_dir, fs_name)

        saver.save_stimulation_log(self._stimulation_log)

        spike_df = self.database.get_spike_event(recording_start, recording_stop, fs_name)
        saver.save_spike_events(spike_df)

        trigger_df = self.database.get_all_triggers(recording_start, recording_stop)
        saver.save_triggers(trigger_df)

        all_electrodes = list(self.experiment.electrodes) if self.experiment.electrodes else []
        unique_stim_electrodes = list({r.electrode_idx for r in self._stimulation_log})
        electrodes_for_waveforms = list(set(all_electrodes + unique_stim_electrodes))

        waveform_records = []
        for electrode_idx in electrodes_for_waveforms:
            try:
                raw_df = self.database.get_raw_spike(recording_start, recording_stop, int(electrode_idx))
                if not raw_df.empty:
                    waveform_records.append({
                        "electrode_idx": int(electrode_idx),
                        "num_waveforms": len(raw_df),
                        "waveform_samples": raw_df.values.tolist(),
                    })
            except Exception as exc:
                logger.warning("Failed to fetch waveforms for electrode %s: %s", electrode_idx, exc)

        saver.save_spike_waveforms(waveform_records)

        firing_rate_comparison = {}
        all_channels = set(list(self._baseline_firing_rates.keys()) + list(self._probe_firing_rates.keys()))
        for ch in all_channels:
            baseline_rate = self._baseline_firing_rates.get(ch, 0.0)
            probe_rate = self._probe_firing_rates.get(ch, 0.0)
            firing_rate_comparison[str(ch)] = {
                "baseline_hz": baseline_rate,
                "probe_hz": probe_rate,
                "delta_hz": probe_rate - baseline_rate,
                "ratio": (probe_rate / baseline_rate) if baseline_rate > 0 else None,
            }

        burst_comparison = {}
        all_burst_channels = set(list(self._baseline_burst_counts.keys()) + list(self._probe_burst_counts.keys()))
        for ch in all_burst_channels:
            baseline_bursts = self._baseline_burst_counts.get(ch, 0)
            probe_bursts = self._probe_burst_counts.get(ch, 0)
            burst_comparison[str(ch)] = {
                "baseline_burst_count": baseline_bursts,
                "probe_burst_count": probe_bursts,
                "delta_bursts": probe_bursts - baseline_bursts,
            }

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
            "total_probe_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "total_burst_events": len(self._burst_events),
            "firing_rate_comparison_per_electrode": firing_rate_comparison,
            "burst_count_comparison_per_electrode": burst_comparison,
            "probe_spike_windows_count": len(self._probe_spike_windows),
            "baseline_start_utc": self._baseline_start.isoformat() if self._baseline_start else None,
            "baseline_stop_utc": self._baseline_stop.isoformat() if self._baseline_stop else None,
            "probe_start_utc": self._probe_start.isoformat() if self._probe_start else None,
            "probe_stop_utc": self._probe_stop.isoformat() if self._probe_stop else None,
        }
        saver.save_summary(summary)

        probe_windows_path = self._output_dir / f"{fs_name}_probe_spike_windows.json"
        probe_windows_path.write_text(json.dumps(self._probe_spike_windows, indent=2, default=str))
        logger.info("Saved probe spike windows -> %s", probe_windows_path)

        burst_events_path = self._output_dir / f"{fs_name}_burst_events.json"
        burst_events_path.write_text(json.dumps([asdict(b) for b in self._burst_events], indent=2, default=str))
        logger.info("Saved burst events -> %s", burst_events_path)

    def _compile_results(self, recording_start: datetime, recording_stop: datetime) -> Dict[str, Any]:
        logger.info("Compiling results")

        firing_rate_comparison = {}
        all_channels = set(list(self._baseline_firing_rates.keys()) + list(self._probe_firing_rates.keys()))
        for ch in all_channels:
            baseline_rate = self._baseline_firing_rates.get(ch, 0.0)
            probe_rate = self._probe_firing_rates.get(ch, 0.0)
            firing_rate_comparison[str(ch)] = {
                "baseline_hz": baseline_rate,
                "probe_hz": probe_rate,
                "delta_hz": probe_rate - baseline_rate,
                "ratio": (probe_rate / baseline_rate) if baseline_rate > 0 else None,
            }

        burst_comparison = {}
        all_burst_channels = set(list(self._baseline_burst_counts.keys()) + list(self._probe_burst_counts.keys()))
        for ch in all_burst_channels:
            baseline_bursts = self._baseline_burst_counts.get(ch, 0)
            probe_bursts = self._probe_burst_counts.get(ch, 0)
            burst_comparison[str(ch)] = {
                "baseline_burst_count": baseline_bursts,
                "probe_burst_count": probe_bursts,
                "delta_bursts": probe_bursts - baseline_bursts,
            }

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "baseline_duration_s": self.baseline_duration_s,
            "probe_phase_duration_s": self.probe_phase_duration_s,
            "total_probe_stimulations": len(self._stimulation_log),
            "probe_electrode": self.probe_electrode,
            "probe_amplitude_ua": self.probe_amplitude_ua,
            "probe_duration_us": self.probe_duration_us,
            "charge_balance_verified": abs(self.probe_amplitude_ua * self.probe_duration_us - self.probe_amplitude_ua * self.probe_duration_us) < 1e-9,
            "total_burst_events": len(self._burst_events),
            "firing_rate_comparison_per_electrode": firing_rate_comparison,
            "burst_count_comparison_per_electrode": burst_comparison,
            "num_electrodes_with_baseline_activity": len(self._baseline_firing_rates),
            "num_electrodes_with_probe_activity": len(self._probe_firing_rates),
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
