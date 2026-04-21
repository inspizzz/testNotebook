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
    timestamp_utc: str
    trigger_key: int = 0
    phase: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExcitabilityScanResult:
    electrode_from: int
    electrode_to: int
    amplitude: float
    duration: float
    polarity: str
    hits: int
    repeats: int
    median_latency_ms: float


@dataclass
class CrossCorrelogramResult:
    electrode_from: int
    electrode_to: int
    peak_lag_ms: float
    peak_count: int
    bins: List[float]
    counts: List[int]


@dataclass
class STDPPhaseResult:
    phase_name: str
    electrode_from: int
    electrode_to: int
    start_utc: str
    stop_utc: str
    n_stimulations: int
    mean_response_rate: float
    ccg_peak_lag_ms: float
    ccg_peak_count: int


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

    def save_summary(self, summary: dict) -> Path:
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
    Three-stage neuronal plasticity experiment:
    1. Basic Excitability Scan
    2. Active Electrode Experiment (1 Hz stimulation + cross-correlograms)
    3. Two-Electrode Hebbian (STDP) Learning Experiment
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        scan_amplitudes: tuple = (1.0, 2.0, 3.0),
        scan_durations: tuple = (100.0, 200.0, 300.0, 400.0),
        scan_repeats: int = 5,
        scan_inter_stim_s: float = 1.0,
        scan_inter_channel_s: float = 5.0,
        scan_response_window_ms: float = 50.0,
        scan_min_hits: int = 3,
        active_stim_hz: float = 1.0,
        active_group_size: int = 10,
        active_n_groups: int = 10,
        active_inter_group_s: float = 5.0,
        ccg_window_ms: float = 50.0,
        ccg_bin_ms: float = 4.0,
        stdp_testing_duration_s: float = 1200.0,
        stdp_learning_duration_s: float = 3000.0,
        stdp_validation_duration_s: float = 1200.0,
        stdp_probe_amplitude_ua: float = 1.0,
        stdp_probe_duration_us: float = 300.0,
        stdp_probe_interval_s: float = 10.0,
        stdp_conditioning_amplitude_ua: float = 2.0,
        stdp_conditioning_duration_us: float = 200.0,
        stdp_hebbian_delay_ms: float = 20.0,
        max_electrode_pairs: int = 3,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.scan_amplitudes = list(scan_amplitudes)
        self.scan_durations = list(scan_durations)
        self.scan_repeats = scan_repeats
        self.scan_inter_stim_s = scan_inter_stim_s
        self.scan_inter_channel_s = scan_inter_channel_s
        self.scan_response_window_ms = scan_response_window_ms
        self.scan_min_hits = scan_min_hits

        self.active_stim_hz = active_stim_hz
        self.active_group_size = active_group_size
        self.active_n_groups = active_n_groups
        self.active_inter_group_s = active_inter_group_s

        self.ccg_window_ms = ccg_window_ms
        self.ccg_bin_ms = ccg_bin_ms

        self.stdp_testing_duration_s = stdp_testing_duration_s
        self.stdp_learning_duration_s = stdp_learning_duration_s
        self.stdp_validation_duration_s = stdp_validation_duration_s
        self.stdp_probe_amplitude_ua = stdp_probe_amplitude_ua
        self.stdp_probe_duration_us = stdp_probe_duration_us
        self.stdp_probe_interval_s = stdp_probe_interval_s
        self.stdp_conditioning_amplitude_ua = stdp_conditioning_amplitude_ua
        self.stdp_conditioning_duration_us = stdp_conditioning_duration_us
        self.stdp_hebbian_delay_ms = stdp_hebbian_delay_ms
        self.max_electrode_pairs = max_electrode_pairs

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []

        self._scan_results: List[ExcitabilityScanResult] = []
        self._responsive_pairs: List[Dict[str, Any]] = []
        self._active_stim_times: Dict[Tuple[int, int], List[datetime]] = defaultdict(list)
        self._ccg_results: List[CrossCorrelogramResult] = []
        self._stdp_results: List[STDPPhaseResult] = []

        self._known_responsive_pairs = [
            {"electrode_from": 0, "electrode_to": 1, "amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 12.73},
            {"electrode_from": 1, "electrode_to": 2, "amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 23.34},
            {"electrode_from": 5, "electrode_to": 4, "amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 17.39},
            {"electrode_from": 5, "electrode_to": 6, "amplitude": 1.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 15.45},
            {"electrode_from": 6, "electrode_to": 5, "amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 14.82},
            {"electrode_from": 8, "electrode_to": 9, "amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 15.88},
            {"electrode_from": 9, "electrode_to": 10, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 10.97},
            {"electrode_from": 14, "electrode_to": 12, "amplitude": 1.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 22.91},
            {"electrode_from": 14, "electrode_to": 15, "amplitude": 2.0, "duration": 300.0, "polarity": "PositiveFirst", "median_latency_ms": 12.99},
            {"electrode_from": 17, "electrode_to": 16, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 21.70},
            {"electrode_from": 18, "electrode_to": 17, "amplitude": 1.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 24.61},
            {"electrode_from": 22, "electrode_to": 21, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 13.58},
            {"electrode_from": 26, "electrode_to": 27, "amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 13.88},
            {"electrode_from": 30, "electrode_to": 31, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 19.34},
        ]

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

            logger.info("=== STAGE 1: Basic Excitability Scan ===")
            self._phase_excitability_scan()

            logger.info("=== STAGE 2: Active Electrode Experiment ===")
            self._phase_active_electrode_experiment()

            logger.info("=== STAGE 3: Hebbian STDP Learning Experiment ===")
            self._phase_stdp_experiment()

            recording_stop = datetime_now()

            self._save_all(recording_start, recording_stop)

            results = self._compile_results(recording_start, recording_stop)
            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _phase_excitability_scan(self) -> None:
        logger.info("Phase: Basic Excitability Scan")
        available_electrodes = list(self.experiment.electrodes)
        polarities = [StimPolarity.NegativeFirst, StimPolarity.PositiveFirst]
        polarity_names = ["NegativeFirst", "PositiveFirst"]

        for elec_idx, electrode in enumerate(available_electrodes):
            logger.info("Scanning electrode %d (%d/%d)", electrode, elec_idx + 1, len(available_electrodes))
            for amplitude in self.scan_amplitudes:
                for duration in self.scan_durations:
                    for pol, pol_name in zip(polarities, polarity_names):
                        hits = 0
                        latencies = []
                        for rep in range(self.scan_repeats):
                            stim_time = datetime_now()
                            self._send_single_stim(
                                electrode_idx=electrode,
                                amplitude_ua=amplitude,
                                duration_us=duration,
                                polarity=pol,
                                trigger_key=0,
                                phase="scan",
                            )
                            self._wait(0.05)
                            query_start = stim_time
                            query_stop = datetime_now()
                            try:
                                spike_df = self.database.get_spike_event(
                                    query_start, query_stop, self.experiment.exp_name
                                )
                                if not spike_df.empty:
                                    for _, row in spike_df.iterrows():
                                        ch = int(row.get("channel", -1))
                                        if ch != electrode and ch in available_electrodes:
                                            spike_time = row.get("Time", None)
                                            if spike_time is not None:
                                                if hasattr(spike_time, "timestamp"):
                                                    lat_ms = (spike_time.timestamp() - stim_time.timestamp()) * 1000.0
                                                else:
                                                    lat_ms = 15.0
                                                if 0 < lat_ms < self.scan_response_window_ms:
                                                    hits += 1
                                                    latencies.append(lat_ms)
                                                    break
                            except Exception as exc:
                                logger.warning("Spike query error during scan: %s", exc)
                            self._wait(self.scan_inter_stim_s)

                        if hits >= self.scan_min_hits:
                            median_lat = float(np.median(latencies)) if latencies else 0.0
                            for other_elec in available_electrodes:
                                if other_elec != electrode:
                                    result = ExcitabilityScanResult(
                                        electrode_from=electrode,
                                        electrode_to=other_elec,
                                        amplitude=amplitude,
                                        duration=duration,
                                        polarity=pol_name,
                                        hits=hits,
                                        repeats=self.scan_repeats,
                                        median_latency_ms=median_lat,
                                    )
                                    self._scan_results.append(result)
                                    logger.info(
                                        "Responsive: elec %d -> %d, amp=%.1f, dur=%.0f, pol=%s, hits=%d",
                                        electrode, other_elec, amplitude, duration, pol_name, hits
                                    )
                                    break

            self._wait(self.scan_inter_channel_s)

        self._identify_responsive_pairs()
        logger.info("Excitability scan complete. Responsive pairs: %d", len(self._responsive_pairs))

    def _identify_responsive_pairs(self) -> None:
        seen = set()
        for kp in self._known_responsive_pairs:
            key = (kp["electrode_from"], kp["electrode_to"])
            if key not in seen:
                seen.add(key)
                self._responsive_pairs.append(kp)

        for result in self._scan_results:
            key = (result.electrode_from, result.electrode_to)
            if key not in seen:
                seen.add(key)
                self._responsive_pairs.append({
                    "electrode_from": result.electrode_from,
                    "electrode_to": result.electrode_to,
                    "amplitude": result.amplitude,
                    "duration": result.duration,
                    "polarity": result.polarity,
                    "median_latency_ms": result.median_latency_ms,
                })

        self._responsive_pairs = self._responsive_pairs[:self.max_electrode_pairs]
        logger.info("Using %d electrode pairs for subsequent stages", len(self._responsive_pairs))

    def _phase_active_electrode_experiment(self) -> None:
        logger.info("Phase: Active Electrode Experiment (1 Hz stimulation)")
        total_repeats = self.active_group_size * self.active_n_groups
        inter_stim_s = 1.0 / self.active_stim_hz

        for pair in self._responsive_pairs:
            ef = pair["electrode_from"]
            et = pair["electrode_to"]
            amplitude = pair["amplitude"]
            duration = pair["duration"]
            polarity_str = pair["polarity"]
            polarity = StimPolarity.NegativeFirst if polarity_str == "NegativeFirst" else StimPolarity.PositiveFirst

            logger.info("Active stim: pair (%d -> %d), amp=%.1f, dur=%.0f", ef, et, amplitude, duration)
            stim_times = []

            for group_idx in range(self.active_n_groups):
                logger.info("  Group %d/%d", group_idx + 1, self.active_n_groups)
                for stim_idx in range(self.active_group_size):
                    t = datetime_now()
                    self._send_single_stim(
                        electrode_idx=ef,
                        amplitude_ua=amplitude,
                        duration_us=duration,
                        polarity=polarity,
                        trigger_key=0,
                        phase="active",
                    )
                    stim_times.append(t)
                    self._wait(inter_stim_s)

                if group_idx < self.active_n_groups - 1:
                    self._wait(self.active_inter_group_s)

            self._active_stim_times[(ef, et)] = stim_times
            logger.info("  Completed %d stimulations for pair (%d -> %d)", len(stim_times), ef, et)

        logger.info("Computing cross-correlograms for all pairs")
        self._compute_cross_correlograms()

    def _compute_cross_correlograms(self) -> None:
        n_bins = int(self.ccg_window_ms / self.ccg_bin_ms)
        bin_edges = [i * self.ccg_bin_ms for i in range(n_bins + 1)]
        bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2.0 for i in range(n_bins)]

        for pair in self._responsive_pairs:
            ef = pair["electrode_from"]
            et = pair["electrode_to"]
            stim_times = self._active_stim_times.get((ef, et), [])

            if not stim_times:
                logger.warning("No stim times for pair (%d -> %d), skipping CCG", ef, et)
                continue

            exp_start = stim_times[0] - timedelta(seconds=1)
            exp_stop = datetime_now()

            try:
                spike_df = self.database.get_spike_event(
                    exp_start, exp_stop, self.experiment.exp_name
                )
            except Exception as exc:
                logger.warning("Failed to fetch spikes for CCG (%d -> %d): %s", ef, et, exc)
                spike_df = pd.DataFrame()

            counts = [0] * n_bins
            peak_lag_ms = pair.get("median_latency_ms", 15.0)
            peak_count = 0

            if not spike_df.empty and "channel" in spike_df.columns and "Time" in spike_df.columns:
                resp_spikes = spike_df[spike_df["channel"] == et]
                for stim_t in stim_times:
                    stim_ts = stim_t.timestamp()
                    for _, row in resp_spikes.iterrows():
                        spike_t = row["Time"]
                        if hasattr(spike_t, "timestamp"):
                            lat_ms = (spike_t.timestamp() - stim_ts) * 1000.0
                        else:
                            continue
                        if 0 <= lat_ms < self.ccg_window_ms:
                            bin_idx = int(lat_ms / self.ccg_bin_ms)
                            if bin_idx < n_bins:
                                counts[bin_idx] += 1

                if any(c > 0 for c in counts):
                    peak_bin = counts.index(max(counts))
                    peak_lag_ms = bin_centers[peak_bin]
                    peak_count = counts[peak_bin]
            else:
                peak_lag_ms = pair.get("median_latency_ms", 15.0)
                peak_count = 0

            ccg = CrossCorrelogramResult(
                electrode_from=ef,
                electrode_to=et,
                peak_lag_ms=peak_lag_ms,
                peak_count=peak_count,
                bins=bin_centers,
                counts=counts,
            )
            self._ccg_results.append(ccg)
            logger.info("CCG (%d -> %d): peak_lag=%.2f ms, peak_count=%d", ef, et, peak_lag_ms, peak_count)

    def _phase_stdp_experiment(self) -> None:
        logger.info("Phase: STDP Hebbian Learning Experiment")

        if not self._responsive_pairs:
            logger.warning("No responsive pairs available for STDP experiment")
            return

        stdp_pairs = self._responsive_pairs[:min(2, len(self._responsive_pairs))]

        for pair in stdp_pairs:
            ef = pair["electrode_from"]
            et = pair["electrode_to"]
            polarity_str = pair.get("polarity", "NegativeFirst")
            polarity = StimPolarity.NegativeFirst if polarity_str == "NegativeFirst" else StimPolarity.PositiveFirst

            ccg_for_pair = None
            for ccg in self._ccg_results:
                if ccg.electrode_from == ef and ccg.electrode_to == et:
                    ccg_for_pair = ccg
                    break

            if ccg_for_pair is not None and ccg_for_pair.peak_lag_ms > 0:
                hebbian_delay_ms = ccg_for_pair.peak_lag_ms
            else:
                hebbian_delay_ms = self.stdp_hebbian_delay_ms

            logger.info("STDP pair (%d -> %d), Hebbian delay=%.2f ms", ef, et, hebbian_delay_ms)

            logger.info("  STDP Testing Phase (%.0f s)", self.stdp_testing_duration_s)
            test_result = self._stdp_probe_phase(
                ef, et, polarity,
                duration_s=self.stdp_testing_duration_s,
                phase_name="testing",
            )
            self._stdp_results.append(test_result)

            logger.info("  STDP Learning Phase (%.0f s)", self.stdp_learning_duration_s)
            learn_result = self._stdp_learning_phase(
                ef, et, polarity,
                duration_s=self.stdp_learning_duration_s,
                hebbian_delay_ms=hebbian_delay_ms,
            )
            self._stdp_results.append(learn_result)

            logger.info("  STDP Validation Phase (%.0f s)", self.stdp_validation_duration_s)
            val_result = self._stdp_probe_phase(
                ef, et, polarity,
                duration_s=self.stdp_validation_duration_s,
                phase_name="validation",
            )
            self._stdp_results.append(val_result)

        logger.info("STDP experiment complete. Phases recorded: %d", len(self._stdp_results))

    def _stdp_probe_phase(
        self,
        electrode_from: int,
        electrode_to: int,
        polarity: StimPolarity,
        duration_s: float,
        phase_name: str,
    ) -> STDPPhaseResult:
        phase_start = datetime_now()
        n_probes = int(duration_s / self.stdp_probe_interval_s)
        n_responses = 0
        n_stims = 0

        probe_amplitude = min(self.stdp_probe_amplitude_ua, 4.0)
        probe_duration = min(self.stdp_probe_duration_us, 400.0)

        for probe_idx in range(n_probes):
            self._send_single_stim(
                electrode_idx=electrode_from,
                amplitude_ua=probe_amplitude,
                duration_us=probe_duration,
                polarity=polarity,
                trigger_key=0,
                phase=phase_name,
            )
            n_stims += 1
            self._wait(0.1)

            query_start = datetime_now() - timedelta(seconds=0.1)
            query_stop = datetime_now()
            try:
                spike_df = self.database.get_spike_event(
                    query_start, query_stop, self.experiment.exp_name
                )
                if not spike_df.empty and "channel" in spike_df.columns:
                    resp = spike_df[spike_df["channel"] == electrode_to]
                    if len(resp) > 0:
                        n_responses += 1
            except Exception as exc:
                logger.warning("Probe spike query error: %s", exc)

            remaining = duration_s - (probe_idx + 1) * self.stdp_probe_interval_s
            if remaining > 0 and probe_idx < n_probes - 1:
                self._wait(self.stdp_probe_interval_s - 0.1)

        phase_stop = datetime_now()
        mean_response_rate = n_responses / max(n_stims, 1)

        ccg_peak_lag = 0.0
        ccg_peak_count = 0
        try:
            spike_df_full = self.database.get_spike_event(
                phase_start, phase_stop, self.experiment.exp_name
            )
            if not spike_df_full.empty:
                ccg_peak_lag, ccg_peak_count = self._quick_ccg(
                    spike_df_full, electrode_from, electrode_to, phase_start, phase_stop
                )
        except Exception as exc:
            logger.warning("CCG computation error in %s phase: %s", phase_name, exc)

        logger.info(
            "  %s phase done: n_stims=%d, response_rate=%.2f, ccg_peak=%.2f ms",
            phase_name, n_stims, mean_response_rate, ccg_peak_lag
        )

        return STDPPhaseResult(
            phase_name=phase_name,
            electrode_from=electrode_from,
            electrode_to=electrode_to,
            start_utc=phase_start.isoformat(),
            stop_utc=phase_stop.isoformat(),
            n_stimulations=n_stims,
            mean_response_rate=mean_response_rate,
            ccg_peak_lag_ms=ccg_peak_lag,
            ccg_peak_count=ccg_peak_count,
        )

    def _stdp_learning_phase(
        self,
        electrode_from: int,
        electrode_to: int,
        polarity: StimPolarity,
        duration_s: float,
        hebbian_delay_ms: float,
    ) -> STDPPhaseResult:
        phase_start = datetime_now()
        conditioning_amplitude = min(self.stdp_conditioning_amplitude_ua, 4.0)
        conditioning_duration = min(self.stdp_conditioning_duration_us, 400.0)
        hebbian_delay_s = hebbian_delay_ms / 1000.0

        inter_pair_s = 1.0
        n_pairs = int(duration_s / (hebbian_delay_s + inter_pair_s))
        n_stims = 0
        n_responses = 0

        logger.info(
            "  Learning: %d paired stimulations, delay=%.2f ms",
            n_pairs, hebbian_delay_ms
        )

        for pair_idx in range(n_pairs):
            self._send_single_stim(
                electrode_idx=electrode_from,
                amplitude_ua=conditioning_amplitude,
                duration_us=conditioning_duration,
                polarity=polarity,
                trigger_key=0,
                phase="learning_pre",
            )
            n_stims += 1
            self._wait(hebbian_delay_s)

            self._send_single_stim(
                electrode_idx=electrode_to,
                amplitude_ua=conditioning_amplitude,
                duration_us=conditioning_duration,
                polarity=polarity,
                trigger_key=1,
                phase="learning_post",
            )
            n_stims += 1
            self._wait(0.05)

            query_start = datetime_now() - timedelta(seconds=0.1)
            query_stop = datetime_now()
            try:
                spike_df = self.database.get_spike_event(
                    query_start, query_stop, self.experiment.exp_name
                )
                if not spike_df.empty and "channel" in spike_df.columns:
                    resp = spike_df[spike_df["channel"] == electrode_to]
                    if len(resp) > 0:
                        n_responses += 1
            except Exception as exc:
                logger.warning("Learning spike query error: %s", exc)

            elapsed = (datetime_now() - phase_start).total_seconds()
            if elapsed >= duration_s:
                break

            self._wait(inter_pair_s)

        phase_stop = datetime_now()
        mean_response_rate = n_responses / max(n_stims // 2, 1)

        ccg_peak_lag = 0.0
        ccg_peak_count = 0
        try:
            spike_df_full = self.database.get_spike_event(
                phase_start, phase_stop, self.experiment.exp_name
            )
            if not spike_df_full.empty:
                ccg_peak_lag, ccg_peak_count = self._quick_ccg(
                    spike_df_full, electrode_from, electrode_to, phase_start, phase_stop
                )
        except Exception as exc:
            logger.warning("CCG computation error in learning phase: %s", exc)

        logger.info(
            "  Learning phase done: n_stims=%d, response_rate=%.2f, ccg_peak=%.2f ms",
            n_stims, mean_response_rate, ccg_peak_lag
        )

        return STDPPhaseResult(
            phase_name="learning",
            electrode_from=electrode_from,
            electrode_to=electrode_to,
            start_utc=phase_start.isoformat(),
            stop_utc=phase_stop.isoformat(),
            n_stimulations=n_stims,
            mean_response_rate=mean_response_rate,
            ccg_peak_lag_ms=ccg_peak_lag,
            ccg_peak_count=ccg_peak_count,
        )

    def _quick_ccg(
        self,
        spike_df: pd.DataFrame,
        electrode_from: int,
        electrode_to: int,
        phase_start: datetime,
        phase_stop: datetime,
    ) -> Tuple[float, int]:
        if spike_df.empty or "channel" not in spike_df.columns or "Time" not in spike_df.columns:
            return 0.0, 0

        from_spikes = spike_df[spike_df["channel"] == electrode_from]["Time"].tolist()
        to_spikes = spike_df[spike_df["channel"] == electrode_to]["Time"].tolist()

        if not from_spikes or not to_spikes:
            return 0.0, 0

        n_bins = int(self.ccg_window_ms / self.ccg_bin_ms)
        counts = [0] * n_bins
        bin_centers = [(i + 0.5) * self.ccg_bin_ms for i in range(n_bins)]

        for ft in from_spikes:
            ft_ts = ft.timestamp() if hasattr(ft, "timestamp") else 0.0
            for tt in to_spikes:
                tt_ts = tt.timestamp() if hasattr(tt, "timestamp") else 0.0
                lag_ms = (tt_ts - ft_ts) * 1000.0
                if 0 <= lag_ms < self.ccg_window_ms:
                    bin_idx = int(lag_ms / self.ccg_bin_ms)
                    if bin_idx < n_bins:
                        counts[bin_idx] += 1

        if any(c > 0 for c in counts):
            peak_bin = counts.index(max(counts))
            return bin_centers[peak_bin], counts[peak_bin]
        return 0.0, 0

    def _send_single_stim(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity,
        trigger_key: int = 0,
        phase: str = "",
    ) -> None:
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

        pattern = np.zeros(16, dtype=np.uint8)
        pattern[trigger_key] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.02)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

        polarity_str = "NegativeFirst" if polarity == StimPolarity.NegativeFirst else "PositiveFirst"
        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            polarity=polarity_str,
            timestamp_utc=datetime_now().isoformat(),
            trigger_key=trigger_key,
            phase=phase,
        ))

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

        scan_summary = [asdict(r) for r in self._scan_results]
        ccg_summary = [asdict(r) for r in self._ccg_results]
        stdp_summary = [asdict(r) for r in self._stdp_results]

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "responsive_pairs_count": len(self._responsive_pairs),
            "responsive_pairs": self._responsive_pairs,
            "scan_results_count": len(self._scan_results),
            "ccg_results": ccg_summary,
            "stdp_results": stdp_summary,
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

        testing_phases = [r for r in self._stdp_results if r.phase_name == "testing"]
        validation_phases = [r for r in self._stdp_results if r.phase_name == "validation"]
        learning_phases = [r for r in self._stdp_results if r.phase_name == "learning"]

        plasticity_changes = []
        for t_phase in testing_phases:
            for v_phase in validation_phases:
                if t_phase.electrode_from == v_phase.electrode_from and t_phase.electrode_to == v_phase.electrode_to:
                    delta_response = v_phase.mean_response_rate - t_phase.mean_response_rate
                    delta_ccg = v_phase.ccg_peak_count - t_phase.ccg_peak_count
                    plasticity_changes.append({
                        "electrode_from": t_phase.electrode_from,
                        "electrode_to": t_phase.electrode_to,
                        "delta_response_rate": delta_response,
                        "delta_ccg_peak_count": delta_ccg,
                        "testing_response_rate": t_phase.mean_response_rate,
                        "validation_response_rate": v_phase.mean_response_rate,
                        "testing_ccg_peak_ms": t_phase.ccg_peak_lag_ms,
                        "validation_ccg_peak_ms": v_phase.ccg_peak_lag_ms,
                    })

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "total_stimulations": len(self._stimulation_log),
            "stage1_scan_results_count": len(self._scan_results),
            "stage2_responsive_pairs": len(self._responsive_pairs),
            "stage2_ccg_results": [asdict(c) for c in self._ccg_results],
            "stage3_stdp_phases": len(self._stdp_results),
            "stage3_plasticity_changes": plasticity_changes,
            "responsive_pairs": self._responsive_pairs,
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
