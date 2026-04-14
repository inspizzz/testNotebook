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
    polarity: str = "NegativeFirst"
    phase: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PairConfig:
    electrode_from: int
    electrode_to: int
    amplitude: float
    duration: float
    polarity: str
    median_latency_ms: float
    hits_k: int
    repeats_n: int
    response_rate: float = 1.0
    mean_latency_ms: float = 0.0


class DataSaver:
    def __init__(self, output_dir: Path, fs_name: str) -> None:
        self._dir = output_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._prefix = self._dir / f"{fs_name}_{timestamp}"

    def save_stimulation_log(self, stimulations: list) -> Path:
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

    def save_analysis(self, analysis: dict, suffix: str = "analysis") -> Path:
        path = Path(f"{self._prefix}_{suffix}.json")
        path.write_text(json.dumps(analysis, indent=2, default=str))
        logger.info("Saved analysis -> %s", path)
        return path


def _select_best_pairs_from_scan(scan_data: list) -> List[PairConfig]:
    pair_map: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
    for entry in scan_data:
        key = (entry["electrode_from"], entry["electrode_to"])
        pair_map[key].append(entry)

    best_pairs = []
    for (ef, et), entries in pair_map.items():
        best = max(entries, key=lambda e: (e["hits_k"], -e.get("response_entropy", 1.0), e["stimulation"]["duration"]))
        best_pairs.append(PairConfig(
            electrode_from=ef,
            electrode_to=et,
            amplitude=best["stimulation"]["amplitude"],
            duration=best["stimulation"]["duration"],
            polarity=best["stimulation"]["polarity"],
            median_latency_ms=best["median_latency_ms"],
            hits_k=best["hits_k"],
            repeats_n=best["repeats_n"],
        ))

    best_pairs.sort(key=lambda p: (-p.hits_k, p.median_latency_ms))
    return best_pairs


def _select_deep_scan_pairs(deep_scan: list) -> List[PairConfig]:
    pairs = []
    for entry in deep_scan:
        if entry.get("response_rate", 0) >= 0.9:
            pairs.append(PairConfig(
                electrode_from=entry["stim_electrode"],
                electrode_to=entry["resp_electrode"],
                amplitude=entry["amplitude"],
                duration=entry["duration"],
                polarity=entry["polarity"],
                median_latency_ms=entry["median_latency_ms"],
                hits_k=5,
                repeats_n=5,
                response_rate=entry["response_rate"],
                mean_latency_ms=entry.get("mean_latency_ms", 0.0),
            ))
    pairs.sort(key=lambda p: (-p.response_rate, p.median_latency_ms))
    return pairs


def _compute_wasserstein_1d(dist_a: List[float], dist_b: List[float]) -> float:
    if not dist_a or not dist_b:
        return 0.0
    a_sorted = sorted(dist_a)
    b_sorted = sorted(dist_b)
    all_vals = sorted(set(a_sorted + b_sorted))
    if len(all_vals) < 2:
        return 0.0

    def ecdf(data, x):
        count = 0
        for v in data:
            if v <= x:
                count += 1
        return count / len(data)

    emd = 0.0
    for i in range(len(all_vals) - 1):
        fa = ecdf(a_sorted, all_vals[i])
        fb = ecdf(b_sorted, all_vals[i])
        emd += abs(fa - fb) * (all_vals[i + 1] - all_vals[i])
    return emd


def _fit_gaussian_mixture(latencies: List[float], n_components: int) -> Dict[str, Any]:
    if not latencies or len(latencies) < n_components:
        return {"n_components": n_components, "bic": float("inf"), "means": [], "stds": [], "weights": []}

    data = np.array(latencies, dtype=float)
    n = len(data)

    indices = np.linspace(0, n - 1, n_components, dtype=int)
    means = [data[i] for i in indices]
    stds = [max(np.std(data) / n_components, 0.5) for _ in range(n_components)]
    weights = [1.0 / n_components] * n_components

    for iteration in range(50):
        resp = np.zeros((n, n_components))
        for k in range(n_components):
            if stds[k] < 0.01:
                stds[k] = 0.5
            for i in range(n):
                diff = data[i] - means[k]
                resp[i, k] = weights[k] * math.exp(-0.5 * (diff / stds[k]) ** 2) / (stds[k] * math.sqrt(2 * math.pi) + 1e-30)

        row_sums = resp.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-30)
        resp = resp / row_sums

        for k in range(n_components):
            nk = resp[:, k].sum()
            if nk < 1e-10:
                continue
            weights[k] = nk / n
            means[k] = (resp[:, k] * data).sum() / nk
            stds[k] = max(math.sqrt(((resp[:, k] * (data - means[k]) ** 2).sum()) / nk), 0.1)

    log_likelihood = 0.0
    for i in range(n):
        p = 0.0
        for k in range(n_components):
            diff = data[i] - means[k]
            p += weights[k] * math.exp(-0.5 * (diff / stds[k]) ** 2) / (stds[k] * math.sqrt(2 * math.pi) + 1e-30)
        log_likelihood += math.log(max(p, 1e-300))

    num_params = 3 * n_components - 1
    bic = -2 * log_likelihood + num_params * math.log(n)

    return {
        "n_components": n_components,
        "bic": bic,
        "means": means,
        "stds": stds,
        "weights": weights,
        "log_likelihood": log_likelihood,
    }


def _fit_gamma(latencies: List[float]) -> Dict[str, Any]:
    if not latencies or len(latencies) < 2:
        return {"distribution": "gamma", "bic": float("inf"), "shape": 0, "scale": 0}

    data = np.array(latencies, dtype=float)
    data = data[data > 0]
    if len(data) < 2:
        return {"distribution": "gamma", "bic": float("inf"), "shape": 0, "scale": 0}

    n = len(data)
    mean_val = np.mean(data)
    var_val = np.var(data)
    if var_val < 1e-10:
        var_val = 1.0

    shape = (mean_val ** 2) / var_val
    scale = var_val / mean_val

    if shape <= 0 or scale <= 0:
        return {"distribution": "gamma", "bic": float("inf"), "shape": 0, "scale": 0}

    log_likelihood = 0.0
    for x in data:
        if x <= 0:
            continue
        lp = (shape - 1) * math.log(x) - x / scale - shape * math.log(scale) - math.lgamma(shape)
        log_likelihood += lp

    bic = -2 * log_likelihood + 2 * math.log(n)

    return {
        "distribution": "gamma",
        "bic": bic,
        "shape": shape,
        "scale": scale,
        "log_likelihood": log_likelihood,
    }


def _best_fit_model(latencies: List[float]) -> Dict[str, Any]:
    models = []
    for nc in [1, 2, 3]:
        models.append(_fit_gaussian_mixture(latencies, nc))
    gamma_fit = _fit_gamma(latencies)
    models.append(gamma_fit)

    best = min(models, key=lambda m: m.get("bic", float("inf")))
    return best


class Experiment:
    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        scan_repeats: int = 5,
        scan_inter_stim_s: float = 1.0,
        scan_inter_channel_s: float = 5.0,
        scan_amplitudes: Optional[List[float]] = None,
        scan_durations: Optional[List[float]] = None,
        scan_min_hits: int = 3,
        active_total_repeats: int = 100,
        active_group_size: int = 10,
        active_inter_stim_s: float = 1.0,
        active_inter_group_s: float = 5.0,
        hebbian_test_duration_min: float = 20.0,
        hebbian_learn_duration_min: float = 50.0,
        hebbian_valid_duration_min: float = 20.0,
        hebbian_stim_interval_s: float = 2.0,
        hebbian_probe_interval_stims: int = 30,
        ccg_window_ms: float = 50.0,
        ccg_bin_ms: float = 0.5,
        max_pairs_for_hebbian: int = 3,
        max_scan_electrodes: int = 8,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.scan_repeats = scan_repeats
        self.scan_inter_stim_s = scan_inter_stim_s
        self.scan_inter_channel_s = scan_inter_channel_s
        self.scan_amplitudes = scan_amplitudes or [1.0, 2.0, 3.0]
        self.scan_durations = scan_durations or [100.0, 200.0, 300.0, 400.0]
        self.scan_min_hits = scan_min_hits

        self.active_total_repeats = active_total_repeats
        self.active_group_size = active_group_size
        self.active_inter_stim_s = active_inter_stim_s
        self.active_inter_group_s = active_inter_group_s

        self.hebbian_test_duration_min = hebbian_test_duration_min
        self.hebbian_learn_duration_min = hebbian_learn_duration_min
        self.hebbian_valid_duration_min = hebbian_valid_duration_min
        self.hebbian_stim_interval_s = hebbian_stim_interval_s
        self.hebbian_probe_interval_stims = hebbian_probe_interval_stims

        self.ccg_window_ms = ccg_window_ms
        self.ccg_bin_ms = ccg_bin_ms
        self.max_pairs_for_hebbian = max_pairs_for_hebbian
        self.max_scan_electrodes = max_scan_electrodes

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []

        self._scan_results: Dict[str, Any] = {}
        self._active_results: Dict[str, Any] = {}
        self._hebbian_results: Dict[str, Any] = {}

        self._prior_scan_pairs = _select_best_pairs_from_scan(self._get_reliable_connections())
        self._deep_scan_pairs = _select_deep_scan_pairs(self._get_deep_scan_summaries())

    def _get_reliable_connections(self) -> list:
        return [
            {"electrode_from": 1, "electrode_to": 2, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 6.5, "response_entropy": 0.0, "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst"}},
            {"electrode_from": 5, "electrode_to": 4, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 5.5, "response_entropy": 0.0, "stimulation": {"amplitude": 2.0, "duration": 300.0, "polarity": "PositiveFirst"}},
            {"electrode_from": 5, "electrode_to": 6, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 5.0, "response_entropy": 0.0, "stimulation": {"amplitude": 1.0, "duration": 400.0, "polarity": "NegativeFirst"}},
            {"electrode_from": 6, "electrode_to": 5, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 5.0, "response_entropy": 0.0, "stimulation": {"amplitude": 1.0, "duration": 400.0, "polarity": "NegativeFirst"}},
            {"electrode_from": 8, "electrode_to": 9, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 4.0, "response_entropy": 0.0, "stimulation": {"amplitude": 2.0, "duration": 400.0, "polarity": "NegativeFirst"}},
            {"electrode_from": 14, "electrode_to": 12, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 4.0, "response_entropy": 0.0, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
            {"electrode_from": 14, "electrode_to": 15, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 6.5, "response_entropy": 0.0, "stimulation": {"amplitude": 2.0, "duration": 400.0, "polarity": "PositiveFirst"}},
            {"electrode_from": 18, "electrode_to": 17, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 4.0, "response_entropy": 0.0, "stimulation": {"amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst"}},
            {"electrode_from": 17, "electrode_to": 16, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 4.0, "response_entropy": 0.0, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
            {"electrode_from": 31, "electrode_to": 30, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 4.0, "response_entropy": 0.0, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
        ]

    def _get_deep_scan_summaries(self) -> list:
        return [
            {"pair_index": 1, "stim_electrode": 1, "resp_electrode": 2, "amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst", "stimulations_n": 100, "responding_trials_n": 100, "spikes_in_window_n": 243, "response_rate": 1.0, "mean_latency_ms": 7.601, "median_latency_ms": 7.0},
            {"pair_index": 3, "stim_electrode": 5, "resp_electrode": 6, "amplitude": 1.0, "duration": 400.0, "polarity": "NegativeFirst", "stimulations_n": 100, "responding_trials_n": 99, "spikes_in_window_n": 99, "response_rate": 0.99, "mean_latency_ms": 5.152, "median_latency_ms": 5.0},
            {"pair_index": 4, "stim_electrode": 6, "resp_electrode": 5, "amplitude": 1.0, "duration": 400.0, "polarity": "NegativeFirst", "stimulations_n": 100, "responding_trials_n": 100, "spikes_in_window_n": 140, "response_rate": 1.0, "mean_latency_ms": 6.907, "median_latency_ms": 5.0},
            {"pair_index": 5, "stim_electrode": 8, "resp_electrode": 9, "amplitude": 2.0, "duration": 200.0, "polarity": "NegativeFirst", "stimulations_n": 100, "responding_trials_n": 100, "spikes_in_window_n": 246, "response_rate": 1.0, "mean_latency_ms": 7.565, "median_latency_ms": 6.5},
            {"pair_index": 19, "stim_electrode": 18, "resp_electrode": 17, "amplitude": 1.0, "duration": 300.0, "polarity": "PositiveFirst", "stimulations_n": 100, "responding_trials_n": 100, "spikes_in_window_n": 100, "response_rate": 1.0, "mean_latency_ms": 6.655, "median_latency_ms": 6.5},
            {"pair_index": 15, "stim_electrode": 0, "resp_electrode": 1, "amplitude": 2.0, "duration": 400.0, "polarity": "PositiveFirst", "stimulations_n": 100, "responding_trials_n": 99, "spikes_in_window_n": 99, "response_rate": 0.99, "mean_latency_ms": 6.146, "median_latency_ms": 6.0},
        ]

    def _wait(self, seconds: float) -> None:
        if not self.testing:
            time.sleep(seconds)

    def _get_polarity(self, polarity_str: str) -> StimPolarity:
        if polarity_str == "PositiveFirst":
            return StimPolarity.PositiveFirst
        return StimPolarity.NegativeFirst

    def _make_stim_param(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity = StimPolarity.NegativeFirst,
        trigger_key: int = 0,
    ) -> StimParam:
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
        return stim

    def _fire_trigger(self, trigger_key: int = 0) -> None:
        pattern = np.zeros(16, dtype=np.uint8)
        pattern[trigger_key] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.05)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

    def _stimulate_single(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity = StimPolarity.NegativeFirst,
        trigger_key: int = 0,
        phase: str = "",
    ) -> None:
        stim = self._make_stim_param(electrode_idx, amplitude_ua, duration_us, polarity, trigger_key)
        self.intan.send_stimparam([stim])
        self._fire_trigger(trigger_key)

        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=min(abs(amplitude_ua), 4.0),
            duration_us=min(abs(duration_us), 400.0),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            trigger_key=trigger_key,
            polarity=polarity.name,
            phase=phase,
        ))

    def _stimulate_pair(
        self,
        electrode_pre: int,
        electrode_post: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity,
        delay_ms: float,
        trigger_key_pre: int = 0,
        trigger_key_post: int = 1,
        phase: str = "",
    ) -> None:
        stim_pre = self._make_stim_param(electrode_pre, amplitude_ua, duration_us, polarity, trigger_key_pre)
        delay_us = int(delay_ms * 1000.0)
        stim_post = self._make_stim_param(electrode_post, amplitude_ua, duration_us, polarity, trigger_key_post)
        stim_post.trigger_key = trigger_key_pre
        stim_post.trigger_delay = delay_us

        self.intan.send_stimparam([stim_pre, stim_post])
        self._fire_trigger(trigger_key_pre)

        now_str = datetime.now(timezone.utc).isoformat()
        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_pre,
            amplitude_ua=min(abs(amplitude_ua), 4.0),
            duration_us=min(abs(duration_us), 400.0),
            timestamp_utc=now_str,
            trigger_key=trigger_key_pre,
            polarity=polarity.name,
            phase=phase,
            extra={"paired_with": electrode_post, "delay_ms": delay_ms},
        ))
        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_post,
            amplitude_ua=min(abs(amplitude_ua), 4.0),
            duration_us=min(abs(duration_us), 400.0),
            timestamp_utc=now_str,
            trigger_key=trigger_key_pre,
            polarity=polarity.name,
            phase=phase,
            extra={"paired_with": electrode_pre, "delay_ms": -delay_ms},
        ))

    def _get_spikes_in_window(
        self,
        stim_time: datetime,
        resp_electrode: int,
        window_ms: float = 50.0,
    ) -> List[float]:
        start = stim_time - timedelta(milliseconds=5)
        stop = stim_time + timedelta(milliseconds=window_ms + 10)
        df = self.database.get_spike_event_electrode(start, stop, resp_electrode)
        latencies = []
        if not df.empty and "Time" in df.columns:
            for _, row in df.iterrows():
                spike_time = pd.to_datetime(row["Time"], utc=True)
                lat_ms = (spike_time - stim_time).total_seconds() * 1000.0
                if 1.0 <= lat_ms <= window_ms:
                    latencies.append(lat_ms)
        return latencies

    def _compute_ccg(
        self,
        stim_times: List[datetime],
        resp_electrode: int,
        window_ms: float = 50.0,
        bin_ms: float = 0.5,
    ) -> Dict[str, Any]:
        n_bins = int(2 * window_ms / bin_ms)
        counts = [0] * n_bins
        all_latencies = []

        for st in stim_times:
            start = st - timedelta(milliseconds=window_ms + 5)
            stop = st + timedelta(milliseconds=window_ms + 5)
            df = self.database.get_spike_event_electrode(start, stop, resp_electrode)
            if not df.empty and "Time" in df.columns:
                for _, row in df.iterrows():
                    spike_time = pd.to_datetime(row["Time"], utc=True)
                    lat_ms = (spike_time - st).total_seconds() * 1000.0
                    if -window_ms <= lat_ms <= window_ms:
                        bin_idx = int((lat_ms + window_ms) / bin_ms)
                        bin_idx = min(bin_idx, n_bins - 1)
                        counts[bin_idx] += 1
                        if lat_ms > 0:
                            all_latencies.append(lat_ms)

        bin_edges = [-window_ms + i * bin_ms for i in range(n_bins + 1)]
        bin_centers = [-window_ms + (i + 0.5) * bin_ms for i in range(n_bins)]

        return {
            "counts": counts,
            "bin_centers": bin_centers,
            "bin_edges": bin_edges,
            "all_latencies": all_latencies,
            "n_stim": len(stim_times),
        }

    def _phase1_excitability_scan(self) -> Dict[str, Any]:
        logger.info("=== Phase 1: Basic Excitability Scan ===")
        electrodes = self.experiment.electrodes[:self.max_scan_electrodes]
        polarities = [StimPolarity.NegativeFirst, StimPolarity.PositiveFirst]
        results = []
        responsive_pairs = []

        for ch_idx, electrode in enumerate(electrodes):
            logger.info("Scanning electrode %d (%d/%d)", electrode, ch_idx + 1, len(electrodes))
            for amplitude in self.scan_amplitudes:
                for duration in self.scan_durations:
                    for polarity in polarities:
                        hit_count = 0
                        latencies = []
                        for rep in range(self.scan_repeats):
                            stim_time = datetime.now(timezone.utc)
                            self._stimulate_single(
                                electrode, amplitude, duration, polarity,
                                trigger_key=0, phase="scan"
                            )
                            self._wait(self.scan_inter_stim_s)

                            lats = self._get_spikes_in_window(stim_time, electrode, window_ms=self.ccg_window_ms)
                            if len(lats) > 0:
                                hit_count += 1
                                latencies.extend(lats)

                        record = {
                            "electrode": electrode,
                            "amplitude": amplitude,
                            "duration": duration,
                            "polarity": polarity.name,
                            "hits": hit_count,
                            "repeats": self.scan_repeats,
                            "latencies": latencies,
                            "median_latency": float(np.median(latencies)) if latencies else None,
                        }
                        results.append(record)

                        if self.scan_min_hits <= hit_count <= self.scan_repeats:
                            responsive_pairs.append(record)

            if ch_idx < len(electrodes) - 1:
                self._wait(self.scan_inter_channel_s)

        self._scan_results = {
            "all_results": results,
            "responsive": responsive_pairs,
            "electrodes_scanned": electrodes,
        }
        logger.info("Phase 1 complete: %d responsive parameter sets found", len(responsive_pairs))
        return self._scan_results

    def _phase2_active_electrode(self) -> Dict[str, Any]:
        logger.info("=== Phase 2: Active Electrode Experiment ===")

        pairs_to_use = self._deep_scan_pairs[:6] if self._deep_scan_pairs else self._prior_scan_pairs[:6]
        if not pairs_to_use:
            logger.warning("No responsive pairs available for Phase 2")
            return {"pairs": [], "error": "no_pairs"}

        pair_results = []

        for pair_idx, pair in enumerate(pairs_to_use):
            logger.info("Active electrode pair %d: %d -> %d (amp=%.1f, dur=%.0f, pol=%s)",
                        pair_idx, pair.electrode_from, pair.electrode_to,
                        pair.amplitude, pair.duration, pair.polarity)

            polarity = self._get_polarity(pair.polarity)
            stim_times = []
            n_groups = self.active_total_repeats // self.active_group_size

            for group in range(n_groups):
                for stim_i in range(self.active_group_size):
                    stim_time = datetime.now(timezone.utc)
                    stim_times.append(stim_time)
                    self._stimulate_single(
                        pair.electrode_from, pair.amplitude, pair.duration,
                        polarity, trigger_key=0, phase="active"
                    )
                    self._wait(self.active_inter_stim_s)

                if group < n_groups - 1:
                    self._wait(self.active_inter_group_s)

            ccg = self._compute_ccg(stim_times, pair.electrode_to, self.ccg_window_ms, self.ccg_bin_ms)

            model_fit = _best_fit_model(ccg["all_latencies"])

            synaptic_delay = pair.median_latency_ms
            if model_fit.get("means") and isinstance(model_fit["means"], list) and len(model_fit["means"]) > 0:
                synaptic_delay = min(model_fit["means"])

            pair_result = {
                "pair_index": pair_idx,
                "electrode_from": pair.electrode_from,
                "electrode_to": pair.electrode_to,
                "amplitude": pair.amplitude,
                "duration": pair.duration,
                "polarity": pair.polarity,
                "n_stim": len(stim_times),
                "ccg_counts": ccg["counts"],
                "ccg_bin_centers": ccg["bin_centers"],
                "all_latencies": ccg["all_latencies"],
                "model_fit": model_fit,
                "estimated_synaptic_delay_ms": synaptic_delay,
                "median_latency_ms": pair.median_latency_ms,
            }
            pair_results.append(pair_result)

        self._active_results = {"pairs": pair_results}
        logger.info("Phase 2 complete: %d pairs characterized", len(pair_results))
        return self._active_results

    def _phase3_hebbian_learning(self) -> Dict[str, Any]:
        logger.info("=== Phase 3: Two-Electrode Hebbian Learning ===")

        if not self._active_results.get("pairs"):
            logger.warning("No active pairs for Hebbian learning")
            return {"pairs": [], "error": "no_active_pairs"}

        active_pairs = self._active_results["pairs"]
        pairs_for_hebbian = active_pairs[:self.max_pairs_for_hebbian]

        hebbian_results = []

        for hp_idx, ap in enumerate(pairs_for_hebbian):
            e_pre = ap["electrode_from"]
            e_post = ap["electrode_to"]
            amplitude = ap["amplitude"]
            duration = ap["duration"]
            polarity = self._get_polarity(ap["polarity"])
            delay_ms = ap["estimated_synaptic_delay_ms"]

            logger.info("Hebbian pair %d: %d -> %d, delay=%.1f ms", hp_idx, e_pre, e_post, delay_ms)

            testing_stim_times = []
            testing_start = datetime.now(timezone.utc)
            test_end_time = testing_start + timedelta(minutes=self.hebbian_test_duration_min)
            stim_count = 0

            logger.info("  Testing phase (%.0f min)", self.hebbian_test_duration_min)
            while datetime.now(timezone.utc) < test_end_time:
                stim_time = datetime.now(timezone.utc)
                testing_stim_times.append(stim_time)
                self._stimulate_single(
                    e_pre, amplitude, duration, polarity,
                    trigger_key=0, phase="hebbian_test"
                )
                stim_count += 1
                self._wait(self.hebbian_stim_interval_s)

                if stim_count > 600:
                    break

            testing_stop = datetime.now(timezone.utc)

            testing_ccg = self._compute_ccg(testing_stim_times, e_post, self.ccg_window_ms, self.ccg_bin_ms)
            testing_latencies = testing_ccg["all_latencies"]

            logger.info("  Learning phase (%.0f min)", self.hebbian_learn_duration_min)
            learning_start = datetime.now(timezone.utc)
            learn_end_time = learning_start + timedelta(minutes=self.hebbian_learn_duration_min)
            learn_stim_count = 0
            learning_probe_times = []

            while datetime.now(timezone.utc) < learn_end_time:
                learn_stim_count += 1

                if learn_stim_count % self.hebbian_probe_interval_stims == 0:
                    probe_time = datetime.now(timezone.utc)
                    learning_probe_times.append(probe_time)
                    self._stimulate_single(
                        e_pre, amplitude, duration, polarity,
                        trigger_key=0, phase="hebbian_learn_probe"
                    )
                else:
                    self._stimulate_pair(
                        e_pre, e_post, amplitude, duration, polarity,
                        delay_ms=delay_ms,
                        trigger_key_pre=0, trigger_key_post=1,
                        phase="hebbian_learn"
                    )

                self._wait(self.hebbian_stim_interval_s)

                if learn_stim_count > 1500:
                    break

            learning_stop = datetime.now(timezone.utc)

            learning_probe_ccg = None
            if learning_probe_times:
                learning_probe_ccg = self._compute_ccg(learning_probe_times, e_post, self.ccg_window_ms, self.ccg_bin_ms)

            logger.info("  Validation phase (%.0f min)", self.hebbian_valid_duration_min)
            validation_stim_times = []
            validation_start = datetime.now(timezone.utc)
            valid_end_time = validation_start + timedelta(minutes=self.hebbian_valid_duration_min)
            valid_stim_count = 0

            while datetime.now(timezone.utc) < valid_end_time:
                stim_time = datetime.now(timezone.utc)
                validation_stim_times.append(stim_time)
                self._stimulate_single(
                    e_pre, amplitude, duration, polarity,
                    trigger_key=0, phase="hebbian_valid"
                )
                valid_stim_count += 1
                self._wait(self.hebbian_stim_interval_s)

                if valid_stim_count > 600:
                    break

            validation_stop = datetime.now(timezone.utc)

            validation_ccg = self._compute_ccg(validation_stim_times, e_post, self.ccg_window_ms, self.ccg_bin_ms)
            validation_latencies = validation_ccg["all_latencies"]

            emd = _compute_wasserstein_1d(testing_latencies, validation_latencies)

            testing_model = _best_fit_model(testing_latencies)
            validation_model = _best_fit_model(validation_latencies)

            mean_shift = 0.0
            if testing_latencies and validation_latencies:
                mean_shift = float(np.mean(validation_latencies)) - float(np.mean(testing_latencies))

            pair_hebbian = {
                "pair_index": hp_idx,
                "electrode_pre": e_pre,
                "electrode_post": e_post,
                "amplitude": amplitude,
                "duration": duration,
                "polarity": ap["polarity"],
                "hebbian_delay_ms": delay_ms,
                "testing_phase": {
                    "start": testing_start.isoformat(),
                    "stop": testing_stop.isoformat(),
                    "n_stim": len(testing_stim_times),
                    "ccg_counts": testing_ccg["counts"],
                    "ccg_bin_centers": testing_ccg["bin_centers"],
                    "latencies": testing_latencies,
                    "model_fit": testing_model,
                },
                "learning_phase": {
                    "start": learning_start.isoformat(),
                    "stop": learning_stop.isoformat(),
                    "n_paired_stim": learn_stim_count,
                    "n_probes": len(learning_probe_times),
                    "probe_ccg": {
                        "counts": learning_probe_ccg["counts"] if learning_probe_ccg else [],
                        "latencies": learning_probe_ccg["all_latencies"] if learning_probe_ccg else [],
                    },
                },
                "validation_phase": {
                    "start": validation_start.isoformat(),
                    "stop": validation_stop.isoformat(),
                    "n_stim": len(validation_stim_times),
                    "ccg_counts": validation_ccg["counts"],
                    "ccg_bin_centers": validation_ccg["bin_centers"],
                    "latencies": validation_latencies,
                    "model_fit": validation_model,
                },
                "earth_movers_distance": emd,
                "mean_latency_shift_ms": mean_shift,
                "plasticity_detected": emd > 0.5,
            }
            hebbian_results.append(pair_hebbian)
            logger.info("  Pair %d: EMD=%.3f, mean_shift=%.2f ms", hp_idx, emd, mean_shift)

        self._hebbian_results = {"pairs": hebbian_results}
        logger.info("Phase 3 complete: %d pairs tested for Hebbian plasticity", len(hebbian_results))
        return self._hebbian_results

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

            recording_start = datetime.now(timezone.utc)

            scan_results = self._phase1_excitability_scan()

            active_results = self._phase2_active_electrode()

            hebbian_results = self._phase3_hebbian_learning()

            recording_stop = datetime.now(timezone.utc)

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        logger.info("Compiling results")

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "total_stimulations": len(self._stimulation_log),
            "phase1_scan": {
                "electrodes_scanned": self._scan_results.get("electrodes_scanned", []),
                "responsive_count": len(self._scan_results.get("responsive", [])),
            },
            "phase2_active": {
                "pairs_tested": len(self._active_results.get("pairs", [])),
                "pair_summaries": [],
            },
            "phase3_hebbian": {
                "pairs_tested": len(self._hebbian_results.get("pairs", [])),
                "pair_summaries": [],
            },
        }

        for ap in self._active_results.get("pairs", []):
            summary["phase2_active"]["pair_summaries"].append({
                "from": ap["electrode_from"],
                "to": ap["electrode_to"],
                "n_latencies": len(ap.get("all_latencies", [])),
                "estimated_delay_ms": ap.get("estimated_synaptic_delay_ms"),
                "model_type": ap.get("model_fit", {}).get("n_components", "gamma"),
            })

        for hp in self._hebbian_results.get("pairs", []):
            summary["phase3_hebbian"]["pair_summaries"].append({
                "pre": hp["electrode_pre"],
                "post": hp["electrode_post"],
                "emd": hp["earth_movers_distance"],
                "mean_shift_ms": hp["mean_latency_shift_ms"],
                "plasticity_detected": hp["plasticity_detected"],
                "testing_n_latencies": len(hp["testing_phase"].get("latencies", [])),
                "validation_n_latencies": len(hp["validation_phase"].get("latencies", [])),
            })

        return summary

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
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "phase1_responsive": len(self._scan_results.get("responsive", [])),
            "phase2_pairs": len(self._active_results.get("pairs", [])),
            "phase3_pairs": len(self._hebbian_results.get("pairs", [])),
        }
        saver.save_summary(summary)

        waveform_records = self._fetch_spike_waveforms(
            fs_name, spike_df, trigger_df, recording_start, recording_stop
        )
        saver.save_spike_waveforms(waveform_records)

        analysis = {
            "scan_results": self._scan_results,
            "active_results": self._active_results,
            "hebbian_results": self._hebbian_results,
        }
        saver.save_analysis(analysis, suffix="full_analysis")

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
            if col in ("channel", "index"):
                electrode_col = col
                break
            if "electrode" in col.lower() or "idx" in col.lower():
                electrode_col = col
                break

        if electrode_col is None:
            logger.warning("Cannot determine electrode column for waveform fetch")
            return waveform_records

        unique_electrodes = spike_df[electrode_col].unique()
        max_waveform_electrodes = 10
        for electrode_idx in unique_electrodes[:max_waveform_electrodes]:
            try:
                raw_df = self.database.get_raw_spike(
                    recording_start, recording_stop, int(electrode_idx)
                )
                if not raw_df.empty:
                    waveform_records.append({
                        "electrode_idx": int(electrode_idx),
                        "num_waveforms": len(raw_df),
                        "waveform_samples": raw_df.values.tolist()[:100],
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
