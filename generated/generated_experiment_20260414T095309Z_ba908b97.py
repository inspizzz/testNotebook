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
    phase: str = ""
    pair_id: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpikeResponse:
    stim_electrode: int
    resp_electrode: int
    latency_ms: float
    trial_idx: int
    phase: str
    timestamp_utc: str


class DataSaver:
    def __init__(self, output_dir: Path, fs_name: str) -> None:
        self._dir = output_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime_now().strftime("%Y%m%dT%H%M%SZ")
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

    def save_analysis(self, analysis: dict) -> Path:
        path = Path(f"{self._prefix}_analysis.json")
        path.write_text(json.dumps(analysis, indent=2, default=str))
        logger.info("Saved analysis -> %s", path)
        return path


RELIABLE_CONNECTIONS = [
    {"electrode_from": 0, "electrode_to": 1, "hits_k": 5, "median_latency_ms": 12.73, "stimulation": {"amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 1, "electrode_to": 2, "hits_k": 5, "median_latency_ms": 23.34, "stimulation": {"amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 4, "electrode_to": 3, "hits_k": 5, "median_latency_ms": 22.44, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 5, "electrode_to": 4, "hits_k": 5, "median_latency_ms": 17.39, "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 5, "electrode_to": 6, "hits_k": 5, "median_latency_ms": 15.45, "stimulation": {"amplitude": 1.0, "duration": 400.0, "polarity": "PositiveFirst"}},
    {"electrode_from": 6, "electrode_to": 5, "hits_k": 5, "median_latency_ms": 14.82, "stimulation": {"amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 8, "electrode_to": 9, "hits_k": 5, "median_latency_ms": 15.88, "stimulation": {"amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 9, "electrode_to": 10, "hits_k": 5, "median_latency_ms": 10.97, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 9, "electrode_to": 11, "hits_k": 5, "median_latency_ms": 16.17, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 10, "electrode_to": 11, "hits_k": 5, "median_latency_ms": 14.75, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 13, "electrode_to": 11, "hits_k": 5, "median_latency_ms": 15.95, "stimulation": {"amplitude": 2.0, "duration": 300.0, "polarity": "PositiveFirst"}},
    {"electrode_from": 13, "electrode_to": 12, "hits_k": 5, "median_latency_ms": 24.03, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 13, "electrode_to": 14, "hits_k": 5, "median_latency_ms": 20.16, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 14, "electrode_to": 12, "hits_k": 5, "median_latency_ms": 22.37, "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 14, "electrode_to": 15, "hits_k": 5, "median_latency_ms": 13.20, "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 17, "electrode_to": 16, "hits_k": 5, "median_latency_ms": 21.56, "stimulation": {"amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 17, "electrode_to": 18, "hits_k": 5, "median_latency_ms": 11.19, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 18, "electrode_to": 17, "hits_k": 5, "median_latency_ms": 24.71, "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 20, "electrode_to": 22, "hits_k": 5, "median_latency_ms": 22.42, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 21, "electrode_to": 22, "hits_k": 5, "median_latency_ms": 18.66, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 22, "electrode_to": 21, "hits_k": 5, "median_latency_ms": 13.58, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 24, "electrode_to": 25, "hits_k": 5, "median_latency_ms": 13.18, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 26, "electrode_to": 27, "hits_k": 5, "median_latency_ms": 13.88, "stimulation": {"amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 27, "electrode_to": 28, "hits_k": 5, "median_latency_ms": 14.51, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 28, "electrode_to": 29, "hits_k": 5, "median_latency_ms": 17.74, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 30, "electrode_to": 31, "hits_k": 5, "median_latency_ms": 19.34, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 31, "electrode_to": 30, "hits_k": 5, "median_latency_ms": 18.87, "stimulation": {"amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst"}},
]

DEEP_SCAN_PAIRS = [
    {"pair_index": 1, "stim_electrode": 1, "resp_electrode": 2, "amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 23.83, "response_rate": 0.79},
    {"pair_index": 2, "stim_electrode": 6, "resp_electrode": 5, "amplitude": 2.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 15.245, "response_rate": 0.80},
    {"pair_index": 3, "stim_electrode": 14, "resp_electrode": 12, "amplitude": 1.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 22.72, "response_rate": 0.94},
    {"pair_index": 4, "stim_electrode": 14, "resp_electrode": 15, "amplitude": 2.0, "duration": 300.0, "polarity": "PositiveFirst", "median_latency_ms": 12.84, "response_rate": 0.80},
    {"pair_index": 5, "stim_electrode": 17, "resp_electrode": 16, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 21.58, "response_rate": 0.90},
    {"pair_index": 7, "stim_electrode": 18, "resp_electrode": 17, "amplitude": 1.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 25.075, "response_rate": 0.89},
    {"pair_index": 8, "stim_electrode": 22, "resp_electrode": 21, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 14.03, "response_rate": 0.93},
    {"pair_index": 9, "stim_electrode": 24, "resp_electrode": 25, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 13.17, "response_rate": 0.81},
    {"pair_index": 10, "stim_electrode": 30, "resp_electrode": 31, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 19.18, "response_rate": 0.85},
    {"pair_index": 12, "stim_electrode": 0, "resp_electrode": 1, "amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 12.35, "response_rate": 0.83},
]


def _polarity_from_str(s: str) -> StimPolarity:
    if s == "PositiveFirst":
        return StimPolarity.PositiveFirst
    return StimPolarity.NegativeFirst


def _compute_charge_balanced_params(amplitude: float, duration: float) -> Tuple[float, float, float, float]:
    a1 = min(abs(amplitude), 4.0)
    d1 = min(abs(duration), 400.0)
    a2 = a1
    d2 = d1
    return a1, d1, a2, d2


def _wasserstein_distance(u: List[float], v: List[float]) -> float:
    if not u or not v:
        return 0.0
    u_sorted = sorted(u)
    v_sorted = sorted(v)
    all_vals = sorted(set(u_sorted + v_sorted))
    n_u = len(u_sorted)
    n_v = len(v_sorted)
    u_cdf = []
    v_cdf = []
    ui = 0
    vi = 0
    for val in all_vals:
        while ui < n_u and u_sorted[ui] <= val:
            ui += 1
        while vi < n_v and v_sorted[vi] <= val:
            vi += 1
        u_cdf.append(ui / n_u)
        v_cdf.append(vi / n_v)
    dist = 0.0
    for i in range(len(all_vals) - 1):
        dist += abs(u_cdf[i] - v_cdf[i]) * (all_vals[i + 1] - all_vals[i])
    return dist


def _fit_gaussian(data: List[float]) -> Dict[str, float]:
    if len(data) < 2:
        return {"mean": data[0] if data else 0.0, "std": 1.0, "weight": 1.0}
    n = len(data)
    mean = sum(data) / n
    var = sum((x - mean) ** 2 for x in data) / max(n - 1, 1)
    std = math.sqrt(var) if var > 0 else 1.0
    return {"mean": mean, "std": std, "weight": 1.0}


def _gaussian_log_likelihood(data: List[float], mean: float, std: float) -> float:
    if std <= 0:
        return -1e18
    ll = 0.0
    for x in data:
        z = (x - mean) / std
        ll += -0.5 * z * z - math.log(std) - 0.5 * math.log(2 * math.pi)
    return ll


def _fit_gmm_k(data: List[float], k: int, n_iter: int = 50) -> Dict[str, Any]:
    if len(data) < k:
        return {"k": k, "bic": 1e18, "components": [], "log_likelihood": -1e18}
    n = len(data)
    mn = min(data)
    mx = max(data)
    span = mx - mn if mx > mn else 1.0
    means = [mn + span * (i + 1) / (k + 1) for i in range(k)]
    stds = [span / (2 * k)] * k
    weights = [1.0 / k] * k
    for _ in range(n_iter):
        responsibilities = []
        for x in data:
            row = []
            for j in range(k):
                if stds[j] <= 0:
                    row.append(0.0)
                    continue
                z = (x - means[j]) / stds[j]
                p = weights[j] * math.exp(-0.5 * z * z) / (stds[j] * math.sqrt(2 * math.pi))
                row.append(p)
            total = sum(row)
            if total <= 0:
                row = [1.0 / k] * k
            else:
                row = [r / total for r in row]
            responsibilities.append(row)
        new_weights = [sum(responsibilities[i][j] for i in range(n)) / n for j in range(k)]
        new_means = []
        for j in range(k):
            denom = sum(responsibilities[i][j] for i in range(n))
            if denom <= 0:
                new_means.append(means[j])
            else:
                new_means.append(sum(responsibilities[i][j] * data[i] for i in range(n)) / denom)
        new_stds = []
        for j in range(k):
            denom = sum(responsibilities[i][j] for i in range(n))
            if denom <= 0:
                new_stds.append(stds[j])
            else:
                var = sum(responsibilities[i][j] * (data[i] - new_means[j]) ** 2 for i in range(n)) / denom
                new_stds.append(math.sqrt(var) if var > 0 else 1e-3)
        means = new_means
        stds = new_stds
        weights = new_weights
    ll = 0.0
    for x in data:
        p = 0.0
        for j in range(k):
            if stds[j] <= 0:
                continue
            z = (x - means[j]) / stds[j]
            p += weights[j] * math.exp(-0.5 * z * z) / (stds[j] * math.sqrt(2 * math.pi))
        ll += math.log(p) if p > 0 else -1e6
    n_params = 3 * k - 1
    bic = n_params * math.log(n) - 2 * ll
    components = [{"mean": means[j], "std": stds[j], "weight": weights[j]} for j in range(k)]
    return {"k": k, "bic": bic, "components": components, "log_likelihood": ll}


def _fit_gamma(data: List[float]) -> Dict[str, Any]:
    if len(data) < 2:
        return {"shape": 1.0, "scale": 1.0, "log_likelihood": -1e18, "bic": 1e18}
    n = len(data)
    data_pos = [max(x, 1e-6) for x in data]
    mean = sum(data_pos) / n
    log_mean = math.log(mean)
    mean_log = sum(math.log(x) for x in data_pos) / n
    s = log_mean - mean_log
    if s <= 0:
        s = 1e-6
    shape = (3 - s + math.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
    scale = mean / shape
    ll = 0.0
    for x in data_pos:
        try:
            log_gamma_shape = math.lgamma(shape)
            ll += (shape - 1) * math.log(x) - x / scale - shape * math.log(scale) - log_gamma_shape
        except Exception:
            ll += -1e6
    n_params = 2
    bic = n_params * math.log(n) - 2 * ll
    return {"shape": shape, "scale": scale, "log_likelihood": ll, "bic": bic}


def _select_best_fit(data: List[float]) -> Dict[str, Any]:
    if len(data) < 3:
        return {"model": "unimodal_gaussian", "synaptic_delay_ms": sum(data) / len(data) if data else 0.0}
    results = {}
    for k in [1, 2, 3]:
        results[f"gmm_{k}"] = _fit_gmm_k(data, k)
    results["gamma"] = _fit_gamma(data)
    best_model = min(results, key=lambda m: results[m]["bic"])
    best = results[best_model]
    if best_model.startswith("gmm"):
        comps = best.get("components", [])
        if comps:
            dominant = max(comps, key=lambda c: c["weight"])
            delay = dominant["mean"]
        else:
            delay = sum(data) / len(data)
    else:
        delay = best.get("shape", 1.0) * best.get("scale", 1.0)
    return {
        "model": best_model,
        "bic": best.get("bic", 1e18),
        "synaptic_delay_ms": delay,
        "fit_details": best,
        "all_fits": {k: {"bic": v["bic"]} for k, v in results.items()},
    }


class Experiment:
    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        scan_amplitudes: List[float] = None,
        scan_durations: List[float] = None,
        scan_repeats: int = 5,
        active_electrode_repeats: int = 100,
        active_electrode_group_size: int = 10,
        active_electrode_group_pause_s: float = 5.0,
        testing_phase_minutes: float = 20.0,
        learning_phase_minutes: float = 50.0,
        validation_phase_minutes: float = 20.0,
        max_pairs_for_stdp: int = 3,
        inter_stim_s: float = 1.0,
        inter_channel_s: float = 5.0,
        response_window_ms: float = 50.0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.scan_amplitudes = scan_amplitudes if scan_amplitudes is not None else [1.0, 2.0, 3.0]
        self.scan_durations = scan_durations if scan_durations is not None else [100.0, 200.0, 300.0, 400.0]
        self.scan_repeats = scan_repeats
        self.active_electrode_repeats = active_electrode_repeats
        self.active_electrode_group_size = active_electrode_group_size
        self.active_electrode_group_pause_s = active_electrode_group_pause_s
        self.testing_phase_minutes = testing_phase_minutes
        self.learning_phase_minutes = learning_phase_minutes
        self.validation_phase_minutes = validation_phase_minutes
        self.max_pairs_for_stdp = max_pairs_for_stdp
        self.inter_stim_s = inter_stim_s
        self.inter_channel_s = inter_channel_s
        self.response_window_ms = response_window_ms

        self.np_experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._spike_responses: List[SpikeResponse] = []

        self._scan_results: Dict[str, Any] = {}
        self._responsive_pairs: List[Dict[str, Any]] = []
        self._active_electrode_results: Dict[str, Any] = {}
        self._synaptic_delays: Dict[str, float] = {}
        self._stdp_results: Dict[str, Any] = {}
        self._analysis_results: Dict[str, Any] = {}

        self._recording_start: Optional[datetime] = None
        self._recording_stop: Optional[datetime] = None

    def _wait(self, seconds: float) -> None:
        wait(seconds)

    def run(self) -> Dict[str, Any]:
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

            self._recording_start = datetime_now()

            logger.info("=== STAGE 1: Basic Excitability Scan ===")
            self._phase_basic_excitability_scan()

            logger.info("=== STAGE 2: Active Electrode Experiment ===")
            self._phase_active_electrode_experiment()

            logger.info("=== STAGE 3: Two-Electrode Hebbian Learning (STDP) ===")
            self._phase_stdp_experiment()

            self._recording_stop = datetime_now()

            results = self._compile_results(self._recording_start, self._recording_stop)

            self._save_all(self._recording_start, self._recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            if self._recording_stop is None:
                self._recording_stop = datetime_now()
            try:
                if self._recording_start is not None and self._recording_stop is not None:
                    self._save_all(self._recording_start, self._recording_stop)
            except Exception as save_exc:
                logger.error("Failed to save data on error: %s", save_exc)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _phase_basic_excitability_scan(self) -> None:
        logger.info("Phase 1: Basic Excitability Scan")
        available_electrodes = list(self.np_experiment.electrodes) if self.np_experiment.electrodes else list(range(32))
        polarities = [StimPolarity.NegativeFirst, StimPolarity.PositiveFirst]
        polarity_names = ["NegativeFirst", "PositiveFirst"]

        scan_data: Dict[int, List[Dict]] = defaultdict(list)

        for electrode_idx in available_electrodes:
            logger.info("Scanning electrode %d", electrode_idx)
            for amplitude in self.scan_amplitudes:
                for duration in self.scan_durations:
                    for pol, pol_name in zip(polarities, polarity_names):
                        hits = 0
                        latencies = []
                        for rep in range(self.scan_repeats):
                            t_stim = datetime_now()
                            self._send_single_stim(
                                electrode_idx=electrode_idx,
                                amplitude=amplitude,
                                duration=duration,
                                polarity=pol,
                                trigger_key=0,
                                phase="scan",
                                pair_id=f"scan_e{electrode_idx}",
                            )
                            self._wait(0.05)
                            t_query_start = datetime_now()
                            t_query_stop = datetime_now()
                            try:
                                spike_df = self.database.get_spike_event(
                                    t_stim,
                                    t_query_stop,
                                    self.np_experiment.exp_name,
                                )
                                if not spike_df.empty:
                                    time_col = self._get_time_col(spike_df)
                                    if time_col is not None:
                                        window_end = t_stim + timedelta(milliseconds=self.response_window_ms)
                                        mask = spike_df[time_col] >= t_stim
                                        mask &= spike_df[time_col] <= window_end
                                        ch_col = self._get_channel_col(spike_df)
                                        if ch_col is not None:
                                            mask &= spike_df[ch_col] != electrode_idx
                                        resp = spike_df[mask]
                                        if len(resp) > 0:
                                            hits += 1
                                            lat = (resp[time_col].iloc[0] - t_stim).total_seconds() * 1000.0
                                            latencies.append(lat)
                            except Exception as e:
                                logger.warning("Spike query error: %s", e)
                            if rep < self.scan_repeats - 1:
                                self._wait(self.inter_stim_s)

                        scan_data[electrode_idx].append({
                            "amplitude": amplitude,
                            "duration": duration,
                            "polarity": pol_name,
                            "hits": hits,
                            "repeats": self.scan_repeats,
                            "latencies_ms": latencies,
                            "median_latency_ms": float(np.median(latencies)) if latencies else 0.0,
                            "consistent": hits >= 3,
                        })
            self._wait(self.inter_channel_s)

        self._scan_results = dict(scan_data)

        responsive_electrodes = set()
        for elec, records in scan_data.items():
            for rec in records:
                if rec["consistent"]:
                    responsive_electrodes.add(elec)

        logger.info("Responsive electrodes from scan: %s", responsive_electrodes)

        self._responsive_pairs = []
        for conn in RELIABLE_CONNECTIONS:
            ef = conn["electrode_from"]
            et = conn["electrode_to"]
            if ef in available_electrodes and et in available_electrodes:
                self._responsive_pairs.append(conn)

        logger.info("Responsive pairs (from prior scan + current): %d", len(self._responsive_pairs))

    def _phase_active_electrode_experiment(self) -> None:
        logger.info("Phase 2: Active Electrode Experiment")
        pairs_to_use = self._responsive_pairs[:self.max_pairs_for_stdp * 2]
        if not pairs_to_use:
            pairs_to_use = DEEP_SCAN_PAIRS[:5]

        active_results: Dict[str, Any] = {}
        latency_distributions: Dict[str, List[float]] = {}

        for pair in pairs_to_use:
            ef = pair.get("electrode_from", pair.get("stim_electrode", 0))
            et = pair.get("electrode_to", pair.get("resp_electrode", 1))
            stim_info = pair.get("stimulation", {})
            amplitude = stim_info.get("amplitude", pair.get("amplitude", 2.0))
            duration = stim_info.get("duration", pair.get("duration", 300.0))
            polarity_str = stim_info.get("polarity", pair.get("polarity", "NegativeFirst"))
            polarity = _polarity_from_str(polarity_str)
            pair_id = f"e{ef}_e{et}"

            logger.info("Active electrode experiment: pair %s", pair_id)
            latencies: List[float] = []
            stim_times: List[datetime] = []

            n_groups = self.active_electrode_repeats // self.active_electrode_group_size
            total_done = 0

            for group_idx in range(n_groups):
                for stim_in_group in range(self.active_electrode_group_size):
                    if total_done >= self.active_electrode_repeats:
                        break
                    t_stim = datetime_now()
                    stim_times.append(t_stim)
                    self._send_single_stim(
                        electrode_idx=ef,
                        amplitude=amplitude,
                        duration=duration,
                        polarity=polarity,
                        trigger_key=1,
                        phase="active_electrode",
                        pair_id=pair_id,
                    )
                    self._wait(0.05)
                    try:
                        t_query_stop = datetime_now()
                        spike_df = self.database.get_spike_event(
                            t_stim,
                            t_query_stop,
                            self.np_experiment.exp_name,
                        )
                        if not spike_df.empty:
                            time_col = self._get_time_col(spike_df)
                            ch_col = self._get_channel_col(spike_df)
                            if time_col is not None and ch_col is not None:
                                window_end = t_stim + timedelta(milliseconds=self.response_window_ms)
                                mask = (spike_df[time_col] >= t_stim) & (spike_df[time_col] <= window_end)
                                mask &= spike_df[ch_col] == et
                                resp = spike_df[mask]
                                if len(resp) > 0:
                                    lat = (resp[time_col].iloc[0] - t_stim).total_seconds() * 1000.0
                                    latencies.append(lat)
                                    self._spike_responses.append(SpikeResponse(
                                        stim_electrode=ef,
                                        resp_electrode=et,
                                        latency_ms=lat,
                                        trial_idx=total_done,
                                        phase="active_electrode",
                                        timestamp_utc=t_stim.isoformat(),
                                    ))
                    except Exception as e:
                        logger.warning("Spike query error in active electrode: %s", e)
                    total_done += 1
                    self._wait(1.0 - 0.05)

                if group_idx < n_groups - 1:
                    self._wait(self.active_electrode_group_pause_s)

            latency_distributions[pair_id] = latencies
            fit_result = _select_best_fit(latencies) if len(latencies) >= 3 else {
                "model": "insufficient_data",
                "synaptic_delay_ms": pair.get("median_latency_ms", 15.0),
            }
            self._synaptic_delays[pair_id] = fit_result["synaptic_delay_ms"]

            active_results[pair_id] = {
                "electrode_from": ef,
                "electrode_to": et,
                "total_stimulations": total_done,
                "responding_trials": len(latencies),
                "response_rate": len(latencies) / max(total_done, 1),
                "latencies_ms": latencies,
                "fit_result": fit_result,
                "synaptic_delay_ms": self._synaptic_delays[pair_id],
            }
            logger.info("Pair %s: synaptic delay = %.2f ms, response_rate = %.2f",
                        pair_id, self._synaptic_delays[pair_id],
                        active_results[pair_id]["response_rate"])

        self._active_electrode_results = active_results

    def _phase_stdp_experiment(self) -> None:
        logger.info("Phase 3: STDP Hebbian Learning Experiment")
        stdp_pairs = []
        for pair in self._responsive_pairs[:self.max_pairs_for_stdp]:
            ef = pair.get("electrode_from", pair.get("stim_electrode", 0))
            et = pair.get("electrode_to", pair.get("resp_electrode", 1))
            stim_info = pair.get("stimulation", {})
            amplitude = stim_info.get("amplitude", pair.get("amplitude", 2.0))
            duration = stim_info.get("duration", pair.get("duration", 300.0))
            polarity_str = stim_info.get("polarity", pair.get("polarity", "NegativeFirst"))
            pair_id = f"e{ef}_e{et}"
            delay_ms = self._synaptic_delays.get(pair_id, pair.get("median_latency_ms", 15.0))
            stdp_pairs.append({
                "electrode_from": ef,
                "electrode_to": et,
                "amplitude": amplitude,
                "duration": duration,
                "polarity": polarity_str,
                "pair_id": pair_id,
                "hebbian_delay_ms": delay_ms,
            })

        if not stdp_pairs:
            for p in DEEP_SCAN_PAIRS[:self.max_pairs_for_stdp]:
                pair_id = f"e{p['stim_electrode']}_e{p['resp_electrode']}"
                stdp_pairs.append({
                    "electrode_from": p["stim_electrode"],
                    "electrode_to": p["resp_electrode"],
                    "amplitude": p["amplitude"],
                    "duration": p["duration"],
                    "polarity": p["polarity"],
                    "pair_id": pair_id,
                    "hebbian_delay_ms": p["median_latency_ms"],
                })

        stdp_results: Dict[str, Any] = {}

        for pair_info in stdp_pairs:
            ef = pair_info["electrode_from"]
            et = pair_info["electrode_to"]
            pair_id = pair_info["pair_id"]
            amplitude = pair_info["amplitude"]
            duration = pair_info["duration"]
            polarity = _polarity_from_str(pair_info["polarity"])
            hebbian_delay_s = pair_info["hebbian_delay_ms"] / 1000.0

            logger.info("STDP pair %s, Hebbian delay=%.2f ms", pair_id, pair_info["hebbian_delay_ms"])

            testing_latencies: List[float] = []
            validation_latencies: List[float] = []

            logger.info("STDP Testing phase: %s (%.1f min)", pair_id, self.testing_phase_minutes)
            testing_phase_start = datetime_now()
            testing_phase_duration_s = self.testing_phase_minutes * 60.0
            probe_interval_s = 5.0
            probes_done = 0
            while True:
                elapsed = (datetime_now() - testing_phase_start).total_seconds()
                if elapsed >= testing_phase_duration_s:
                    break
                t_stim = datetime_now()
                self._send_single_stim(
                    electrode_idx=ef,
                    amplitude=amplitude,
                    duration=duration,
                    polarity=polarity,
                    trigger_key=2,
                    phase="stdp_testing",
                    pair_id=pair_id,
                )
                self._wait(0.05)
                try:
                    t_query_stop = datetime_now()
                    spike_df = self.database.get_spike_event(
                        t_stim, t_query_stop, self.np_experiment.exp_name
                    )
                    if not spike_df.empty:
                        time_col = self._get_time_col(spike_df)
                        ch_col = self._get_channel_col(spike_df)
                        if time_col is not None and ch_col is not None:
                            window_end = t_stim + timedelta(milliseconds=self.response_window_ms)
                            mask = (spike_df[time_col] >= t_stim) & (spike_df[time_col] <= window_end)
                            mask &= spike_df[ch_col] == et
                            resp = spike_df[mask]
                            if len(resp) > 0:
                                lat = (resp[time_col].iloc[0] - t_stim).total_seconds() * 1000.0
                                testing_latencies.append(lat)
                except Exception as e:
                    logger.warning("Spike query error in STDP testing: %s", e)
                probes_done += 1
                self._wait(probe_interval_s - 0.05)

            logger.info("STDP Learning phase: %s (%.1f min)", pair_id, self.learning_phase_minutes)
            learning_phase_start = datetime_now()
            learning_phase_duration_s = self.learning_phase_minutes * 60.0
            learning_interval_s = 1.0
            learning_probe_every = 20
            learning_count = 0
            while True:
                elapsed = (datetime_now() - learning_phase_start).total_seconds()
                if elapsed >= learning_phase_duration_s:
                    break
                t_pre = datetime_now()
                self._send_single_stim(
                    electrode_idx=ef,
                    amplitude=amplitude,
                    duration=duration,
                    polarity=polarity,
                    trigger_key=3,
                    phase="stdp_learning_pre",
                    pair_id=pair_id,
                )
                self._wait(hebbian_delay_s)
                self._send_single_stim(
                    electrode_idx=et,
                    amplitude=min(amplitude, 2.0),
                    duration=duration,
                    polarity=polarity,
                    trigger_key=4,
                    phase="stdp_learning_post",
                    pair_id=pair_id,
                )
                learning_count += 1
                if learning_count % learning_probe_every == 0:
                    self._wait(0.05)
                    try:
                        t_query_stop = datetime_now()
                        spike_df = self.database.get_spike_event(
                            t_pre, t_query_stop, self.np_experiment.exp_name
                        )
                        if not spike_df.empty:
                            time_col = self._get_time_col(spike_df)
                            ch_col = self._get_channel_col(spike_df)
                            if time_col is not None and ch_col is not None:
                                window_end = t_pre + timedelta(milliseconds=self.response_window_ms)
                                mask = (spike_df[time_col] >= t_pre) & (spike_df[time_col] <= window_end)
                                mask &= spike_df[ch_col] == et
                                resp = spike_df[mask]
                                if len(resp) > 0:
                                    lat = (resp[time_col].iloc[0] - t_pre).total_seconds() * 1000.0
                                    self._spike_responses.append(SpikeResponse(
                                        stim_electrode=ef,
                                        resp_electrode=et,
                                        latency_ms=lat,
                                        trial_idx=learning_count,
                                        phase="stdp_learning_probe",
                                        timestamp_utc=t_pre.isoformat(),
                                    ))
                    except Exception as e:
                        logger.warning("Spike query error in STDP learning probe: %s", e)
                remaining = learning_interval_s - hebbian_delay_s - 0.05
                if remaining > 0:
                    self._wait(remaining)

            logger.info("STDP Validation phase: %s (%.1f min)", pair_id, self.validation_phase_minutes)
            validation_phase_start = datetime_now()
            validation_phase_duration_s = self.validation_phase_minutes * 60.0
            val_probes_done = 0
            while True:
                elapsed = (datetime_now() - validation_phase_start).total_seconds()
                if elapsed >= validation_phase_duration_s:
                    break
                t_stim = datetime_now()
                self._send_single_stim(
                    electrode_idx=ef,
                    amplitude=amplitude,
                    duration=duration,
                    polarity=polarity,
                    trigger_key=5,
                    phase="stdp_validation",
                    pair_id=pair_id,
                )
                self._wait(0.05)
                try:
                    t_query_stop = datetime_now()
                    spike_df = self.database.get_spike_event(
                        t_stim, t_query_stop, self.np_experiment.exp_name
                    )
                    if not spike_df.empty:
                        time_col = self._get_time_col(spike_df)
                        ch_col = self._get_channel_col(spike_df)
                        if time_col is not None and ch_col is not None:
                            window_end = t_stim + timedelta(milliseconds=self.response_window_ms)
                            mask = (spike_df[time_col] >= t_stim) & (spike_df[time_col] <= window_end)
                            mask &= spike_df[ch_col] == et
                            resp = spike_df[mask]
                            if len(resp) > 0:
                                lat = (resp[time_col].iloc[0] - t_stim).total_seconds() * 1000.0
                                validation_latencies.append(lat)
                except Exception as e:
                    logger.warning("Spike query error in STDP validation: %s", e)
                val_probes_done += 1
                self._wait(probe_interval_s - 0.05)

            emd = _wasserstein_distance(testing_latencies, validation_latencies)
            testing_fit = _select_best_fit(testing_latencies) if len(testing_latencies) >= 3 else {"model": "insufficient_data", "synaptic_delay_ms": 0.0}
            validation_fit = _select_best_fit(validation_latencies) if len(validation_latencies) >= 3 else {"model": "insufficient_data", "synaptic_delay_ms": 0.0}

            plasticity_detected = emd > 2.0

            stdp_results[pair_id] = {
                "electrode_from": ef,
                "electrode_to": et,
                "hebbian_delay_ms": pair_info["hebbian_delay_ms"],
                "testing_latencies_ms": testing_latencies,
                "validation_latencies_ms": validation_latencies,
                "testing_n": len(testing_latencies),
                "validation_n": len(validation_latencies),
                "wasserstein_distance": emd,
                "plasticity_detected": plasticity_detected,
                "testing_fit": testing_fit,
                "validation_fit": validation_fit,
                "learning_stimulations": learning_count,
            }
            logger.info("Pair %s: EMD=%.3f, plasticity_detected=%s", pair_id, emd, plasticity_detected)

        self._stdp_results = stdp_results

        self._analysis_results = {
            "scan_responsive_pairs": len(self._responsive_pairs),
            "active_electrode_pairs": len(self._active_electrode_results),
            "stdp_pairs": len(stdp_results),
            "synaptic_delays": self._synaptic_delays,
            "plasticity_summary": {
                pid: {
                    "wasserstein_distance": v["wasserstein_distance"],
                    "plasticity_detected": v["plasticity_detected"],
                }
                for pid, v in stdp_results.items()
            },
        }

    def _send_single_stim(
        self,
        electrode_idx: int,
        amplitude: float,
        duration: float,
        polarity: StimPolarity,
        trigger_key: int = 0,
        phase: str = "",
        pair_id: str = "",
    ) -> None:
        a1, d1, a2, d2 = _compute_charge_balanced_params(amplitude, duration)

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
        stim.phase_amplitude1 = a1
        stim.phase_duration1 = d1
        stim.phase_amplitude2 = a2
        stim.phase_duration2 = d2
        stim.enable_amp_settle = True
        stim.pre_stim_amp_settle = 0.0
        stim.post_stim_amp_settle = 1000.0
        stim.enable_charge_recovery = True
        stim.post_charge_recovery_on = 0.0
        stim.post_charge_recovery_off = 100.0
        stim.interphase_delay = 0.0

        self.intan.send_stimparam([stim])

        pattern = np.zeros(16, dtype=np.uint8)
        pattern[trigger_key % 16] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.01)
        pattern[trigger_key % 16] = 0
        self.trigger_controller.send(pattern)

        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=a1,
            duration_us=d1,
            polarity=polarity.name,
            timestamp_utc=datetime_now().isoformat(),
            trigger_key=trigger_key,
            phase=phase,
            pair_id=pair_id,
        ))

    def _get_time_col(self, df: pd.DataFrame) -> Optional[str]:
        for col in ["Time", "time", "_time", "timestamp"]:
            if col in df.columns:
                return col
        return None

    def _get_channel_col(self, df: pd.DataFrame) -> Optional[str]:
        for col in ["channel", "Channel", "index", "electrode", "ch"]:
            if col in df.columns:
                return col
        return None

    def _save_all(self, recording_start: datetime, recording_stop: datetime) -> None:
        fs_name = getattr(self.np_experiment, "exp_name", "unknown") if self.np_experiment else "unknown"
        saver = DataSaver(self._output_dir, fs_name)

        saver.save_stimulation_log(self._stimulation_log)

        try:
            spike_df = self.database.get_spike_event(recording_start, recording_stop, fs_name)
        except Exception as e:
            logger.warning("Failed to fetch spike events: %s", e)
            spike_df = pd.DataFrame()
        saver.save_spike_events(spike_df)

        try:
            trigger_df = self.database.get_all_triggers(recording_start, recording_stop)
        except Exception as e:
            logger.warning("Failed to fetch triggers: %s", e)
            trigger_df = pd.DataFrame()
        saver.save_triggers(trigger_df)

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "responsive_pairs_found": len(self._responsive_pairs),
            "active_electrode_pairs": len(self._active_electrode_results),
            "stdp_pairs": len(self._stdp_results),
            "synaptic_delays": self._synaptic_delays,
            "plasticity_summary": {
                pid: {
                    "wasserstein_distance": v.get("wasserstein_distance", 0.0),
                    "plasticity_detected": v.get("plasticity_detected", False),
                }
                for pid, v in self._stdp_results.items()
            },
        }
        saver.save_summary(summary)

        waveform_records = self._fetch_spike_waveforms(
            fs_name, spike_df, trigger_df, recording_start, recording_stop
        )
        saver.save_spike_waveforms(waveform_records)

        analysis = {
            "scan_results_electrode_count": len(self._scan_results),
            "active_electrode_results": {
                pid: {
                    "electrode_from": v["electrode_from"],
                    "electrode_to": v["electrode_to"],
                    "response_rate": v["response_rate"],
                    "synaptic_delay_ms": v["synaptic_delay_ms"],
                    "fit_model": v["fit_result"].get("model", ""),
                }
                for pid, v in self._active_electrode_results.items()
            },
            "stdp_results": {
                pid: {
                    "electrode_from": v["electrode_from"],
                    "electrode_to": v["electrode_to"],
                    "hebbian_delay_ms": v["hebbian_delay_ms"],
                    "testing_n": v["testing_n"],
                    "validation_n": v["validation_n"],
                    "wasserstein_distance": v["wasserstein_distance"],
                    "plasticity_detected": v["plasticity_detected"],
                    "testing_fit_model": v["testing_fit"].get("model", ""),
                    "validation_fit_model": v["validation_fit"].get("model", ""),
                }
                for pid, v in self._stdp_results.items()
            },
        }
        saver.save_analysis(analysis)

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

        ch_col = self._get_channel_col(spike_df)
        if ch_col is None:
            logger.warning("Cannot determine electrode column for waveform fetch")
            return waveform_records

        unique_electrodes = spike_df[ch_col].unique()
        for electrode_idx in unique_electrodes:
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

        return waveform_records

    def _compile_results(self, recording_start: datetime, recording_stop: datetime) -> Dict[str, Any]:
        logger.info("Compiling results")
        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": getattr(self.np_experiment, "exp_name", "unknown"),
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "total_stimulations": len(self._stimulation_log),
            "responsive_pairs_found": len(self._responsive_pairs),
            "active_electrode_pairs_analysed": len(self._active_electrode_results),
            "stdp_pairs_analysed": len(self._stdp_results),
            "synaptic_delays_ms": self._synaptic_delays,
            "plasticity_results": {
                pid: {
                    "wasserstein_distance": v.get("wasserstein_distance", 0.0),
                    "plasticity_detected": v.get("plasticity_detected", False),
                    "testing_n": v.get("testing_n", 0),
                    "validation_n": v.get("validation_n", 0),
                }
                for pid, v in self._stdp_results.items()
            },
        }
        return summary

    def _cleanup(self) -> None:
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
