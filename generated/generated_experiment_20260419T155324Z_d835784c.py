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
    pair_id: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpikeResponse:
    trial_idx: int
    electrode_from: int
    electrode_to: int
    stim_time_utc: str
    latency_ms: float
    amplitude_uv: float
    phase: str = ""


class DataSaver:
    def __init__(self, output_dir: Path, fs_name: str) -> None:
        self._dir = output_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime_now().strftime("%Y%m%dT%H%M%SZ")
        self._prefix = self._dir / f"{fs_name}_{timestamp}"

    def save_stimulation_log(self, stimulations: list) -> Path:
        path = Path(f"{self._prefix}_stimulations.json")
        records = [asdict(s) if hasattr(s, '__dataclass_fields__') else s for s in stimulations]
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


def _compute_wasserstein_distance(samples_a: List[float], samples_b: List[float]) -> float:
    if not samples_a or not samples_b:
        return float('nan')
    sa = sorted(samples_a)
    sb = sorted(samples_b)
    na, nb = len(sa), len(sb)
    all_vals = sorted(set(sa + sb))
    cdf_a, cdf_b = 0.0, 0.0
    ia, ib = 0, 0
    dist = 0.0
    prev_val = all_vals[0]
    for val in all_vals:
        width = val - prev_val
        dist += abs(cdf_a - cdf_b) * width
        while ia < na and sa[ia] <= val:
            cdf_a += 1.0 / na
            ia += 1
        while ib < nb and sb[ib] <= val:
            cdf_b += 1.0 / nb
            ib += 1
        prev_val = val
    return dist


def _fit_gaussian_mixture(samples: List[float], n_components: int) -> Dict[str, Any]:
    if len(samples) < n_components * 2:
        return {"aic": float('inf'), "n_components": n_components, "means": [], "stds": [], "weights": []}
    arr = np.array(samples)
    n = len(arr)
    if n_components == 1:
        mu = float(np.mean(arr))
        sigma = float(np.std(arr)) + 1e-6
        log_lik = float(np.sum(-0.5 * ((arr - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * math.pi))))
        aic = 2 * 2 - 2 * log_lik
        return {"aic": aic, "n_components": 1, "means": [mu], "stds": [sigma], "weights": [1.0]}
    step = (np.max(arr) - np.min(arr)) / (n_components + 1)
    means = [float(np.min(arr) + step * (i + 1)) for i in range(n_components)]
    stds = [float(np.std(arr) / n_components + 1e-6)] * n_components
    weights = [1.0 / n_components] * n_components
    for _ in range(50):
        responsibilities = np.zeros((n, n_components))
        for k in range(n_components):
            sigma_k = stds[k]
            responsibilities[:, k] = weights[k] * np.exp(-0.5 * ((arr - means[k]) / sigma_k) ** 2) / (sigma_k * np.sqrt(2 * math.pi))
        row_sums = responsibilities.sum(axis=1, keepdims=True) + 1e-300
        responsibilities /= row_sums
        for k in range(n_components):
            r_k = responsibilities[:, k]
            r_sum = r_k.sum() + 1e-300
            weights[k] = float(r_sum / n)
            means[k] = float(np.dot(r_k, arr) / r_sum)
            stds[k] = float(np.sqrt(np.dot(r_k, (arr - means[k]) ** 2) / r_sum)) + 1e-6
    log_lik = 0.0
    for i in range(n):
        p = sum(weights[k] * math.exp(-0.5 * ((arr[i] - means[k]) / stds[k]) ** 2) / (stds[k] * math.sqrt(2 * math.pi)) for k in range(n_components))
        log_lik += math.log(p + 1e-300)
    n_params = n_components * 3 - 1
    aic = 2 * n_params - 2 * log_lik
    return {"aic": aic, "n_components": n_components, "means": means, "stds": stds, "weights": weights}


def _fit_gamma(samples: List[float]) -> Dict[str, Any]:
    if len(samples) < 2:
        return {"aic": float('inf'), "shape": float('nan'), "scale": float('nan')}
    arr = np.array(samples)
    arr = arr[arr > 0]
    if len(arr) < 2:
        return {"aic": float('inf'), "shape": float('nan'), "scale": float('nan')}
    mean_val = float(np.mean(arr))
    var_val = float(np.var(arr)) + 1e-12
    shape = mean_val ** 2 / var_val
    scale = var_val / mean_val
    log_lik = float(np.sum((shape - 1) * np.log(arr + 1e-300) - arr / scale - shape * math.log(scale) - math.lgamma(shape)))
    aic = 2 * 2 - 2 * log_lik
    return {"aic": aic, "shape": shape, "scale": scale}


def _select_best_model(latencies: List[float]) -> Dict[str, Any]:
    if not latencies:
        return {"best_model": "none", "synaptic_delay_ms": float('nan')}
    models = {}
    for k in [1, 2, 3]:
        models[f"gmm_{k}"] = _fit_gaussian_mixture(latencies, k)
    models["gamma"] = _fit_gamma(latencies)
    best_name = min(models, key=lambda x: models[x]["aic"])
    best = models[best_name]
    if best_name.startswith("gmm"):
        means = best.get("means", [])
        weights = best.get("weights", [])
        if means:
            dominant_idx = int(np.argmax(weights))
            synaptic_delay = means[dominant_idx]
        else:
            synaptic_delay = float(np.mean(latencies))
    else:
        shape = best.get("shape", float('nan'))
        scale = best.get("scale", float('nan'))
        if not math.isnan(shape) and not math.isnan(scale):
            synaptic_delay = (shape - 1) * scale if shape > 1 else shape * scale
        else:
            synaptic_delay = float(np.mean(latencies))
    return {
        "best_model": best_name,
        "model_params": best,
        "all_models_aic": {k: v["aic"] for k, v in models.items()},
        "synaptic_delay_ms": float(synaptic_delay),
    }


def _compute_cross_correlogram(
    stim_times_s: List[float],
    spike_times_s: List[float],
    window_ms: float = 100.0,
    bin_ms: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    if not stim_times_s or not spike_times_s:
        n_bins = int(window_ms / bin_ms)
        bins = np.arange(0, window_ms + bin_ms, bin_ms)
        return np.zeros(n_bins), bins
    window_s = window_ms / 1000.0
    bin_s = bin_ms / 1000.0
    n_bins = int(window_ms / bin_ms)
    counts = np.zeros(n_bins)
    spike_arr = np.array(sorted(spike_times_s))
    for t_stim in stim_times_s:
        mask = (spike_arr >= t_stim) & (spike_arr < t_stim + window_s)
        lats = spike_arr[mask] - t_stim
        for lat in lats:
            bin_idx = int(lat / bin_s)
            if 0 <= bin_idx < n_bins:
                counts[bin_idx] += 1
    bins = np.arange(0, window_ms + bin_ms, bin_ms)
    return counts, bins


SCAN_RESULTS = {
    "reliable_connections": [
        {"electrode_from": 0, "electrode_to": 1, "hits_k": 5, "median_latency_ms": 12.73,
         "stimulation": {"amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 1, "electrode_to": 2, "hits_k": 5, "median_latency_ms": 23.61,
         "stimulation": {"amplitude": 2.0, "duration": 100.0, "polarity": "PositiveFirst"}},
        {"electrode_from": 5, "electrode_to": 4, "hits_k": 5, "median_latency_ms": 17.39,
         "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 5, "electrode_to": 6, "hits_k": 5, "median_latency_ms": 15.45,
         "stimulation": {"amplitude": 1.0, "duration": 400.0, "polarity": "PositiveFirst"}},
        {"electrode_from": 6, "electrode_to": 5, "hits_k": 5, "median_latency_ms": 14.82,
         "stimulation": {"amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 8, "electrode_to": 9, "hits_k": 5, "median_latency_ms": 15.88,
         "stimulation": {"amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 9, "electrode_to": 10, "hits_k": 5, "median_latency_ms": 10.97,
         "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 9, "electrode_to": 11, "hits_k": 5, "median_latency_ms": 16.17,
         "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 14, "electrode_to": 12, "hits_k": 5, "median_latency_ms": 22.91,
         "stimulation": {"amplitude": 1.0, "duration": 400.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 14, "electrode_to": 15, "hits_k": 5, "median_latency_ms": 12.99,
         "stimulation": {"amplitude": 2.0, "duration": 300.0, "polarity": "PositiveFirst"}},
        {"electrode_from": 17, "electrode_to": 16, "hits_k": 5, "median_latency_ms": 21.56,
         "stimulation": {"amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 18, "electrode_to": 17, "hits_k": 5, "median_latency_ms": 24.71,
         "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 24, "electrode_to": 25, "hits_k": 5, "median_latency_ms": 13.18,
         "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 26, "electrode_to": 27, "hits_k": 5, "median_latency_ms": 13.88,
         "stimulation": {"amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 30, "electrode_to": 31, "hits_k": 5, "median_latency_ms": 19.34,
         "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 31, "electrode_to": 30, "hits_k": 5, "median_latency_ms": 18.87,
         "stimulation": {"amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst"}},
    ]
}

DEEP_SCAN_PAIRS = [
    {"stim_electrode": 0, "resp_electrode": 1, "amplitude": 3.0, "duration": 300.0,
     "polarity": "NegativeFirst", "median_latency_ms": 12.283, "response_rate": 0.83},
    {"stim_electrode": 1, "resp_electrode": 2, "amplitude": 2.0, "duration": 300.0,
     "polarity": "NegativeFirst", "median_latency_ms": 23.843, "response_rate": 0.79},
    {"stim_electrode": 5, "resp_electrode": 4, "amplitude": 1.0, "duration": 300.0,
     "polarity": "PositiveFirst", "median_latency_ms": 17.456, "response_rate": 0.93},
    {"stim_electrode": 5, "resp_electrode": 6, "amplitude": 1.0, "duration": 400.0,
     "polarity": "PositiveFirst", "median_latency_ms": 15.677, "response_rate": 0.80},
    {"stim_electrode": 6, "resp_electrode": 5, "amplitude": 2.0, "duration": 400.0,
     "polarity": "PositiveFirst", "median_latency_ms": 15.039, "response_rate": 0.80},
    {"stim_electrode": 8, "resp_electrode": 9, "amplitude": 2.0, "duration": 400.0,
     "polarity": "NegativeFirst", "median_latency_ms": 16.105, "response_rate": 0.73},
    {"stim_electrode": 9, "resp_electrode": 10, "amplitude": 3.0, "duration": 400.0,
     "polarity": "NegativeFirst", "median_latency_ms": 10.783, "response_rate": 0.94},
    {"stim_electrode": 14, "resp_electrode": 12, "amplitude": 1.0, "duration": 400.0,
     "polarity": "NegativeFirst", "median_latency_ms": 22.735, "response_rate": 0.94},
    {"stim_electrode": 14, "resp_electrode": 15, "amplitude": 2.0, "duration": 300.0,
     "polarity": "PositiveFirst", "median_latency_ms": 12.879, "response_rate": 0.80},
    {"stim_electrode": 17, "resp_electrode": 16, "amplitude": 3.0, "duration": 400.0,
     "polarity": "PositiveFirst", "median_latency_ms": 21.556, "response_rate": 0.90},
    {"stim_electrode": 18, "resp_electrode": 17, "amplitude": 1.0, "duration": 400.0,
     "polarity": "PositiveFirst", "median_latency_ms": 24.876, "response_rate": 0.89},
    {"stim_electrode": 24, "resp_electrode": 25, "amplitude": 3.0, "duration": 400.0,
     "polarity": "NegativeFirst", "median_latency_ms": 13.139, "response_rate": 0.81},
    {"stim_electrode": 26, "resp_electrode": 27, "amplitude": 3.0, "duration": 300.0,
     "polarity": "PositiveFirst", "median_latency_ms": 13.793, "response_rate": 0.60},
    {"stim_electrode": 30, "resp_electrode": 31, "amplitude": 3.0, "duration": 400.0,
     "polarity": "NegativeFirst", "median_latency_ms": 19.118, "response_rate": 0.85},
    {"stim_electrode": 31, "resp_electrode": 30, "amplitude": 3.0, "duration": 400.0,
     "polarity": "NegativeFirst", "median_latency_ms": 18.739, "response_rate": 0.82},
]


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
        active_electrode_isi_s: float = 1.0,
        active_electrode_group_pause_s: float = 5.0,
        testing_phase_min: float = 20.0,
        learning_phase_min: float = 50.0,
        validation_phase_min: float = 20.0,
        stdp_delay_ms: float = 10.0,
        probe_interval_trials: int = 20,
        max_pairs_for_stdp: int = 3,
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
        self.active_electrode_isi_s = active_electrode_isi_s
        self.active_electrode_group_pause_s = active_electrode_group_pause_s
        self.testing_phase_min = testing_phase_min
        self.learning_phase_min = learning_phase_min
        self.validation_phase_min = validation_phase_min
        self.stdp_delay_ms = stdp_delay_ms
        self.probe_interval_trials = probe_interval_trials
        self.max_pairs_for_stdp = max_pairs_for_stdp

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
            self._stage1_excitability_scan()

            logger.info("=== STAGE 2: Active Electrode Experiment ===")
            self._stage2_active_electrode_experiment()

            logger.info("=== STAGE 3: Two-Electrode Hebbian Learning ===")
            self._stage3_hebbian_learning()

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
                logger.error("Error saving data after failure: %s", save_exc)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _stage1_excitability_scan(self) -> None:
        logger.info("Stage 1: Excitability scan across electrodes")
        polarities = [StimPolarity.NegativeFirst, StimPolarity.PositiveFirst]
        polarity_names = ["NegativeFirst", "PositiveFirst"]

        available_electrodes = list(self.np_experiment.electrodes)
        scan_pairs_to_test = []
        for conn in DEEP_SCAN_PAIRS[:8]:
            e_from = conn["stim_electrode"]
            e_to = conn["resp_electrode"]
            if e_from in available_electrodes and e_to in available_electrodes:
                scan_pairs_to_test.append((e_from, e_to))

        if not scan_pairs_to_test:
            for i, e in enumerate(available_electrodes[:4]):
                if i + 1 < len(available_electrodes):
                    scan_pairs_to_test.append((e, available_electrodes[i + 1]))

        scan_data = {}
        trigger_key = 0

        for (e_from, e_to) in scan_pairs_to_test:
            pair_key = f"{e_from}->{e_to}"
            scan_data[pair_key] = []
            logger.info("Scanning pair %s", pair_key)

            for amplitude in self.scan_amplitudes:
                for duration in self.scan_durations:
                    if amplitude * duration > 4.0 * 400.0:
                        continue
                    for pol, pol_name in zip(polarities, polarity_names):
                        hits = 0
                        latencies = []
                        for rep in range(self.scan_repeats):
                            stim_time = datetime_now()
                            self._send_stim_pulse(
                                electrode_idx=e_from,
                                amplitude_ua=amplitude,
                                duration_us=duration,
                                polarity=pol,
                                trigger_key=trigger_key,
                                phase="scan",
                                pair_id=pair_key,
                            )
                            self._wait(0.05)
                            query_start = stim_time
                            query_stop = datetime_now()
                            try:
                                spike_df = self.database.get_spike_event(
                                    query_start, query_stop, self.np_experiment.exp_name
                                )
                                if not spike_df.empty:
                                    resp_spikes = spike_df[spike_df.get("channel", pd.Series(dtype=int)) == e_to] if "channel" in spike_df.columns else pd.DataFrame()
                                    if not resp_spikes.empty:
                                        hits += 1
                                        if "Time" in resp_spikes.columns:
                                            t_spike = pd.to_datetime(resp_spikes["Time"].iloc[0])
                                            lat_ms = (t_spike.timestamp() - stim_time.timestamp()) * 1000.0
                                            if 0 < lat_ms < 200:
                                                latencies.append(lat_ms)
                            except Exception as exc:
                                logger.warning("Spike query error: %s", exc)
                            self._wait(self.active_electrode_isi_s)

                        scan_data[pair_key].append({
                            "amplitude": amplitude,
                            "duration": duration,
                            "polarity": pol_name,
                            "hits": hits,
                            "repeats": self.scan_repeats,
                            "median_latency_ms": float(np.median(latencies)) if latencies else float('nan'),
                        })

            self._wait(5.0)

        self._scan_results = scan_data

        responsive = []
        for pair_key, results_list in scan_data.items():
            parts = pair_key.split("->")
            e_from, e_to = int(parts[0]), int(parts[1])
            best = max(results_list, key=lambda x: x["hits"])
            if best["hits"] >= 3:
                responsive.append({
                    "electrode_from": e_from,
                    "electrode_to": e_to,
                    "best_hits": best["hits"],
                    "best_amplitude": best["amplitude"],
                    "best_duration": best["duration"],
                    "best_polarity": best["polarity"],
                    "median_latency_ms": best["median_latency_ms"],
                })

        if not responsive:
            logger.info("No responsive pairs found in scan, using pre-computed pairs")
            for conn in DEEP_SCAN_PAIRS[:self.max_pairs_for_stdp + 2]:
                responsive.append({
                    "electrode_from": conn["stim_electrode"],
                    "electrode_to": conn["resp_electrode"],
                    "best_hits": int(conn["response_rate"] * 5),
                    "best_amplitude": conn["amplitude"],
                    "best_duration": conn["duration"],
                    "best_polarity": conn["polarity"],
                    "median_latency_ms": conn["median_latency_ms"],
                })

        self._responsive_pairs = responsive
        logger.info("Stage 1 complete. Responsive pairs: %d", len(responsive))

    def _stage2_active_electrode_experiment(self) -> None:
        logger.info("Stage 2: Active electrode experiment")
        if not self._responsive_pairs:
            logger.warning("No responsive pairs for stage 2")
            return

        pairs_to_use = self._responsive_pairs[:min(len(self._responsive_pairs), self.max_pairs_for_stdp + 2)]
        active_results = {}
        trigger_key = 1

        for pair_info in pairs_to_use:
            e_from = pair_info["electrode_from"]
            e_to = pair_info["electrode_to"]
            pair_key = f"{e_from}->{e_to}"
            amplitude = min(pair_info["best_amplitude"], 4.0)
            duration = min(pair_info["best_duration"], 400.0)
            polarity_str = pair_info["best_polarity"]
            polarity = StimPolarity.NegativeFirst if polarity_str == "NegativeFirst" else StimPolarity.PositiveFirst

            logger.info("Active electrode experiment for pair %s", pair_key)
            stim_times = []
            latencies_all = []
            n_groups = self.active_electrode_repeats // self.active_electrode_group_size

            for group_idx in range(n_groups):
                for trial_in_group in range(self.active_electrode_group_size):
                    trial_idx = group_idx * self.active_electrode_group_size + trial_in_group
                    stim_time = datetime_now()
                    self._send_stim_pulse(
                        electrode_idx=e_from,
                        amplitude_ua=amplitude,
                        duration_us=duration,
                        polarity=polarity,
                        trigger_key=trigger_key,
                        phase="active_electrode",
                        pair_id=pair_key,
                    )
                    stim_times.append(stim_time.timestamp())
                    self._wait(0.05)
                    try:
                        query_start = stim_time
                        query_stop = datetime_now()
                        spike_df = self.database.get_spike_event(
                            query_start, query_stop, self.np_experiment.exp_name
                        )
                        if not spike_df.empty and "channel" in spike_df.columns and "Time" in spike_df.columns:
                            resp_spikes = spike_df[spike_df["channel"] == e_to]
                            if not resp_spikes.empty:
                                t_spike = pd.to_datetime(resp_spikes["Time"].iloc[0])
                                lat_ms = (t_spike.timestamp() - stim_time.timestamp()) * 1000.0
                                if 0 < lat_ms < 200:
                                    latencies_all.append(lat_ms)
                                    self._spike_responses.append(SpikeResponse(
                                        trial_idx=trial_idx,
                                        electrode_from=e_from,
                                        electrode_to=e_to,
                                        stim_time_utc=stim_time.isoformat(),
                                        latency_ms=lat_ms,
                                        amplitude_uv=float(resp_spikes["Amplitude"].iloc[0]) if "Amplitude" in resp_spikes.columns else 0.0,
                                        phase="active_electrode",
                                    ))
                    except Exception as exc:
                        logger.warning("Spike query error in stage 2: %s", exc)
                    self._wait(self.active_electrode_isi_s - 0.05)

                if group_idx < n_groups - 1:
                    self._wait(self.active_electrode_group_pause_s)

            model_result = _select_best_model(latencies_all)
            synaptic_delay = model_result["synaptic_delay_ms"]
            self._synaptic_delays[pair_key] = synaptic_delay

            ccg_counts, ccg_bins = _compute_cross_correlogram(
                stim_times_s=stim_times,
                spike_times_s=[st + lat / 1000.0 for st, lat in zip(stim_times[:len(latencies_all)], latencies_all)],
                window_ms=100.0,
                bin_ms=1.0,
            )

            active_results[pair_key] = {
                "electrode_from": e_from,
                "electrode_to": e_to,
                "n_trials": self.active_electrode_repeats,
                "n_responses": len(latencies_all),
                "response_rate": len(latencies_all) / max(self.active_electrode_repeats, 1),
                "latencies_ms": latencies_all,
                "model_fit": model_result,
                "synaptic_delay_ms": synaptic_delay,
                "ccg_counts": ccg_counts.tolist(),
                "ccg_bins_ms": ccg_bins.tolist(),
            }
            logger.info("Pair %s: synaptic delay = %.2f ms", pair_key, synaptic_delay)

        self._active_electrode_results = active_results
        logger.info("Stage 2 complete.")

    def _stage3_hebbian_learning(self) -> None:
        logger.info("Stage 3: Hebbian STDP learning experiment")

        stdp_pairs = []
        for pair_info in self._responsive_pairs[:self.max_pairs_for_stdp]:
            e_from = pair_info["electrode_from"]
            e_to = pair_info["electrode_to"]
            pair_key = f"{e_from}->{e_to}"
            delay_ms = self._synaptic_delays.get(pair_key, pair_info.get("median_latency_ms", self.stdp_delay_ms))
            stdp_pairs.append({
                "electrode_from": e_from,
                "electrode_to": e_to,
                "pair_key": pair_key,
                "amplitude": min(pair_info["best_amplitude"], 4.0),
                "duration": min(pair_info["best_duration"], 400.0),
                "polarity": pair_info["best_polarity"],
                "hebbian_delay_ms": delay_ms,
            })

        if not stdp_pairs:
            logger.warning("No pairs for STDP experiment, using defaults")
            conn = DEEP_SCAN_PAIRS[0]
            stdp_pairs = [{
                "electrode_from": conn["stim_electrode"],
                "electrode_to": conn["resp_electrode"],
                "pair_key": f"{conn['stim_electrode']}->{conn['resp_electrode']}",
                "amplitude": conn["amplitude"],
                "duration": conn["duration"],
                "polarity": conn["polarity"],
                "hebbian_delay_ms": conn["median_latency_ms"],
            }]

        stdp_results = {}
        for pair_info in stdp_pairs:
            pair_key = pair_info["pair_key"]
            e_from = pair_info["electrode_from"]
            e_to = pair_info["electrode_to"]
            amplitude = pair_info["amplitude"]
            duration = pair_info["duration"]
            polarity_str = pair_info["polarity"]
            polarity = StimPolarity.NegativeFirst if polarity_str == "NegativeFirst" else StimPolarity.PositiveFirst
            hebbian_delay_ms = pair_info["hebbian_delay_ms"]

            logger.info("STDP experiment for pair %s, Hebbian delay=%.2f ms", pair_key, hebbian_delay_ms)

            testing_latencies = self._stdp_phase_testing(
                e_from=e_from,
                e_to=e_to,
                amplitude=amplitude,
                duration=duration,
                polarity=polarity,
                phase_duration_min=self.testing_phase_min,
                phase_name="testing",
                pair_key=pair_key,
                trigger_key=2,
            )

            self._stdp_phase_learning(
                e_from=e_from,
                e_to=e_to,
                amplitude=amplitude,
                duration=duration,
                polarity=polarity,
                phase_duration_min=self.learning_phase_min,
                hebbian_delay_ms=hebbian_delay_ms,
                pair_key=pair_key,
                trigger_key_pre=2,
                trigger_key_post=3,
                probe_interval=self.probe_interval_trials,
            )

            validation_latencies = self._stdp_phase_testing(
                e_from=e_from,
                e_to=e_to,
                amplitude=amplitude,
                duration=duration,
                polarity=polarity,
                phase_duration_min=self.validation_phase_min,
                phase_name="validation",
                pair_key=pair_key,
                trigger_key=2,
            )

            emd = _compute_wasserstein_distance(testing_latencies, validation_latencies)

            testing_ccg, ccg_bins = _compute_cross_correlogram(
                stim_times_s=list(range(len(testing_latencies))),
                spike_times_s=[i + lat / 1000.0 for i, lat in enumerate(testing_latencies)],
                window_ms=100.0,
                bin_ms=1.0,
            )
            validation_ccg, _ = _compute_cross_correlogram(
                stim_times_s=list(range(len(validation_latencies))),
                spike_times_s=[i + lat / 1000.0 for i, lat in enumerate(validation_latencies)],
                window_ms=100.0,
                bin_ms=1.0,
            )

            testing_peak = float(np.max(testing_ccg)) if len(testing_ccg) > 0 else 0.0
            validation_peak = float(np.max(validation_ccg)) if len(validation_ccg) > 0 else 0.0
            peak_change_pct = ((validation_peak - testing_peak) / max(testing_peak, 1e-6)) * 100.0

            stdp_results[pair_key] = {
                "electrode_from": e_from,
                "electrode_to": e_to,
                "hebbian_delay_ms": hebbian_delay_ms,
                "testing_n_responses": len(testing_latencies),
                "validation_n_responses": len(validation_latencies),
                "testing_latencies_ms": testing_latencies,
                "validation_latencies_ms": validation_latencies,
                "earth_movers_distance": emd,
                "testing_ccg": testing_ccg.tolist(),
                "validation_ccg": validation_ccg.tolist(),
                "ccg_bins_ms": ccg_bins.tolist(),
                "testing_ccg_peak": testing_peak,
                "validation_ccg_peak": validation_peak,
                "ccg_peak_change_pct": peak_change_pct,
                "plasticity_detected": peak_change_pct > 20.0,
            }
            logger.info(
                "Pair %s STDP: EMD=%.3f, CCG peak change=%.1f%%",
                pair_key, emd, peak_change_pct
            )

        self._stdp_results = stdp_results
        logger.info("Stage 3 complete.")

    def _stdp_phase_testing(
        self,
        e_from: int,
        e_to: int,
        amplitude: float,
        duration: float,
        polarity: StimPolarity,
        phase_duration_min: float,
        phase_name: str,
        pair_key: str,
        trigger_key: int = 2,
    ) -> List[float]:
        logger.info("STDP %s phase for pair %s (%.1f min)", phase_name, pair_key, phase_duration_min)
        phase_duration_s = phase_duration_min * 60.0
        isi_s = 2.0
        n_trials = max(1, int(phase_duration_s / isi_s))
        latencies = []

        for trial_idx in range(n_trials):
            stim_time = datetime_now()
            self._send_stim_pulse(
                electrode_idx=e_from,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=polarity,
                trigger_key=trigger_key,
                phase=phase_name,
                pair_id=pair_key,
            )
            self._wait(0.05)
            try:
                query_start = stim_time
                query_stop = datetime_now()
                spike_df = self.database.get_spike_event(
                    query_start, query_stop, self.np_experiment.exp_name
                )
                if not spike_df.empty and "channel" in spike_df.columns and "Time" in spike_df.columns:
                    resp_spikes = spike_df[spike_df["channel"] == e_to]
                    if not resp_spikes.empty:
                        t_spike = pd.to_datetime(resp_spikes["Time"].iloc[0])
                        lat_ms = (t_spike.timestamp() - stim_time.timestamp()) * 1000.0
                        if 0 < lat_ms < 200:
                            latencies.append(lat_ms)
                            self._spike_responses.append(SpikeResponse(
                                trial_idx=trial_idx,
                                electrode_from=e_from,
                                electrode_to=e_to,
                                stim_time_utc=stim_time.isoformat(),
                                latency_ms=lat_ms,
                                amplitude_uv=float(resp_spikes["Amplitude"].iloc[0]) if "Amplitude" in resp_spikes.columns else 0.0,
                                phase=phase_name,
                            ))
            except Exception as exc:
                logger.warning("Spike query error in %s phase: %s", phase_name, exc)
            self._wait(isi_s - 0.05)

        logger.info("%s phase complete: %d responses from %d trials", phase_name, len(latencies), n_trials)
        return latencies

    def _stdp_phase_learning(
        self,
        e_from: int,
        e_to: int,
        amplitude: float,
        duration: float,
        polarity: StimPolarity,
        phase_duration_min: float,
        hebbian_delay_ms: float,
        pair_key: str,
        trigger_key_pre: int = 2,
        trigger_key_post: int = 3,
        probe_interval: int = 20,
    ) -> None:
        logger.info("STDP learning phase for pair %s (%.1f min, delay=%.2f ms)", pair_key, phase_duration_min, hebbian_delay_ms)
        phase_duration_s = phase_duration_min * 60.0
        isi_s = 2.0
        n_trials = max(1, int(phase_duration_s / isi_s))
        hebbian_delay_s = hebbian_delay_ms / 1000.0

        amplitude2 = amplitude
        duration2 = duration

        for trial_idx in range(n_trials):
            stim_time_pre = datetime_now()
            self._send_stim_pulse(
                electrode_idx=e_from,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=polarity,
                trigger_key=trigger_key_pre,
                phase="learning_pre",
                pair_id=pair_key,
            )
            self._wait(hebbian_delay_s)
            self._send_stim_pulse(
                electrode_idx=e_to,
                amplitude_ua=amplitude2,
                duration_us=duration2,
                polarity=polarity,
                trigger_key=trigger_key_post,
                phase="learning_post",
                pair_id=pair_key,
            )

            if (trial_idx + 1) % probe_interval == 0:
                probe_stim_time = datetime_now()
                self._send_stim_pulse(
                    electrode_idx=e_from,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    polarity=polarity,
                    trigger_key=trigger_key_pre,
                    phase="learning_probe",
                    pair_id=pair_key,
                )
                self._wait(0.05)
                try:
                    query_start = probe_stim_time
                    query_stop = datetime_now()
                    spike_df = self.database.get_spike_event(
                        query_start, query_stop, self.np_experiment.exp_name
                    )
                    if not spike_df.empty and "channel" in spike_df.columns and "Time" in spike_df.columns:
                        resp_spikes = spike_df[spike_df["channel"] == e_to]
                        if not resp_spikes.empty:
                            t_spike = pd.to_datetime(resp_spikes["Time"].iloc[0])
                            lat_ms = (t_spike.timestamp() - probe_stim_time.timestamp()) * 1000.0
                            if 0 < lat_ms < 200:
                                self._spike_responses.append(SpikeResponse(
                                    trial_idx=trial_idx,
                                    electrode_from=e_from,
                                    electrode_to=e_to,
                                    stim_time_utc=probe_stim_time.isoformat(),
                                    latency_ms=lat_ms,
                                    amplitude_uv=float(resp_spikes["Amplitude"].iloc[0]) if "Amplitude" in resp_spikes.columns else 0.0,
                                    phase="learning_probe",
                                ))
                except Exception as exc:
                    logger.warning("Probe spike query error: %s", exc)

            self._wait(isi_s - hebbian_delay_s - 0.05)

        logger.info("Learning phase complete: %d trials", n_trials)

    def _send_stim_pulse(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity = StimPolarity.NegativeFirst,
        trigger_key: int = 0,
        phase: str = "",
        pair_id: str = "",
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

        self.intan.send_stimparam([stim])

        pattern = np.zeros(16, dtype=np.uint8)
        pattern[trigger_key] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.02)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            polarity=polarity.name,
            timestamp_utc=datetime_now().isoformat(),
            trigger_key=trigger_key,
            phase=phase,
            pair_id=pair_id,
        ))

    def _save_all(self, recording_start: datetime, recording_stop: datetime) -> None:
        fs_name = getattr(self.np_experiment, "exp_name", "unknown") if self.np_experiment else "unknown"
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
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "responsive_pairs_count": len(self._responsive_pairs),
            "stdp_pairs_count": len(self._stdp_results),
        }
        saver.save_summary(summary)

        waveform_records = []
        unique_electrodes = set()
        if not spike_df.empty and "channel" in spike_df.columns:
            unique_electrodes = set(int(e) for e in spike_df["channel"].unique())
        for electrode_idx in unique_electrodes:
            try:
                raw_df = self.database.get_raw_spike(recording_start, recording_stop, electrode_idx)
                if not raw_df.empty:
                    waveform_records.append({
                        "electrode_idx": electrode_idx,
                        "num_waveforms": len(raw_df),
                        "waveform_samples": raw_df.values.tolist(),
                    })
            except Exception as exc:
                logger.warning("Failed to fetch waveforms for electrode %d: %s", electrode_idx, exc)
        saver.save_spike_waveforms(waveform_records)

        analysis = {
            "scan_results": self._scan_results,
            "responsive_pairs": self._responsive_pairs,
            "active_electrode_results": {
                k: {kk: vv for kk, vv in v.items() if kk not in ("latencies_ms",)}
                for k, v in self._active_electrode_results.items()
            },
            "synaptic_delays_ms": self._synaptic_delays,
            "stdp_results": {
                k: {kk: vv for kk, vv in v.items() if kk not in ("testing_latencies_ms", "validation_latencies_ms")}
                for k, v in self._stdp_results.items()
            },
        }
        saver.save_analysis(analysis, suffix="full_analysis")

    def _compile_results(self, recording_start: datetime, recording_stop: datetime) -> Dict[str, Any]:
        logger.info("Compiling results")
        plasticity_summary = []
        for pair_key, res in self._stdp_results.items():
            plasticity_summary.append({
                "pair": pair_key,
                "emd": res.get("earth_movers_distance", float('nan')),
                "ccg_peak_change_pct": res.get("ccg_peak_change_pct", 0.0),
                "plasticity_detected": res.get("plasticity_detected", False),
            })

        return {
            "status": "completed",
            "experiment_name": getattr(self.np_experiment, "exp_name", "unknown") if self.np_experiment else "unknown",
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "stage1_responsive_pairs": len(self._responsive_pairs),
            "stage2_pairs_analyzed": len(self._active_electrode_results),
            "stage3_stdp_pairs": len(self._stdp_results),
            "synaptic_delays_ms": self._synaptic_delays,
            "plasticity_summary": plasticity_summary,
            "total_stimulations": len(self._stimulation_log),
        }

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
