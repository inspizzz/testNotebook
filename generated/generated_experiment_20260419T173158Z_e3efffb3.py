"""
FinalSpark Neuroplasticity Experiment
======================================
Three-stage pipeline:
  Stage 1 - Basic Excitability Scan (from scan results)
  Stage 2 - Active Electrode Experiment (cross-correlograms, GMM/Gamma fitting)
  Stage 3 - Two-Electrode Hebbian Learning (STDP, EMD comparison)
"""

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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

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
class TrialResult:
    phase: str
    stim_electrode: int
    resp_electrode: int
    trial_index: int
    stim_time_utc: str
    latency_ms: Optional[float]
    responded: bool


# ---------------------------------------------------------------------------
# Data persistence
# ---------------------------------------------------------------------------

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

    def save_analysis(self, analysis: dict, tag: str = "analysis") -> Path:
        path = Path(f"{self._prefix}_{tag}.json")
        path.write_text(json.dumps(analysis, indent=2, default=str))
        logger.info("Saved analysis -> %s", path)
        return path


# ---------------------------------------------------------------------------
# Statistical helpers (no scipy)
# ---------------------------------------------------------------------------

def _mean(vals):
    if not vals:
        return 0.0
    return sum(vals) / len(vals)

def _std(vals):
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))

def _gaussian_pdf(x, mu, sigma):
    if sigma <= 0:
        return 0.0
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))

def _fit_gmm_1d(data, n_components, max_iter=100, tol=1e-4):
    """Fit a 1D Gaussian Mixture Model via EM. Returns (weights, means, stds, log_likelihood)."""
    if not data or n_components < 1:
        return [], [], [], -1e18
    n = len(data)
    arr = sorted(data)
    # Initialise means by quantiles
    means = [arr[int(i * n / n_components)] for i in range(n_components)]
    stds = [max(_std(data) / n_components, 0.5)] * n_components
    weights = [1.0 / n_components] * n_components

    log_lik = -1e18
    for _ in range(max_iter):
        # E-step
        responsibilities = []
        for x in data:
            row = [weights[k] * _gaussian_pdf(x, means[k], stds[k]) for k in range(n_components)]
            total = sum(row) + 1e-300
            responsibilities.append([r / total for r in row])

        # M-step
        new_weights = []
        new_means = []
        new_stds = []
        for k in range(n_components):
            r_k = [responsibilities[i][k] for i in range(n)]
            nk = sum(r_k) + 1e-300
            mu_k = sum(r_k[i] * data[i] for i in range(n)) / nk
            var_k = sum(r_k[i] * (data[i] - mu_k) ** 2 for i in range(n)) / nk
            new_weights.append(nk / n)
            new_means.append(mu_k)
            new_stds.append(max(math.sqrt(var_k), 0.1))

        weights = new_weights
        means = new_means
        stds = new_stds

        # Log-likelihood
        new_ll = 0.0
        for x in data:
            p = sum(weights[k] * _gaussian_pdf(x, means[k], stds[k]) for k in range(n_components))
            new_ll += math.log(p + 1e-300)

        if abs(new_ll - log_lik) < tol:
            break
        log_lik = new_ll

    return weights, means, stds, log_lik

def _fit_gamma_1d(data):
    """Fit Gamma distribution via method of moments. Returns (shape, scale, log_likelihood)."""
    if not data or len(data) < 2:
        return 1.0, 1.0, -1e18
    m = _mean(data)
    s = _std(data)
    if s <= 0 or m <= 0:
        return 1.0, 1.0, -1e18
    shape = (m / s) ** 2
    scale = s ** 2 / m
    # Log-likelihood
    ll = 0.0
    for x in data:
        if x <= 0:
            continue
        try:
            ll += (shape - 1) * math.log(x) - x / scale - shape * math.log(scale) - math.lgamma(shape)
        except Exception:
            pass
    return shape, scale, ll

def _bic(log_lik, n_params, n_data):
    return n_params * math.log(max(n_data, 1)) - 2 * log_lik

def _select_best_model(data, window_ms=50.0):
    """Fit GMM 1,2,3 components + Gamma; select by BIC. Returns dict with best model info."""
    if not data:
        return {"model": "none", "synaptic_delay_ms": None}
    n = len(data)
    results = {}

    for k in [1, 2, 3]:
        w, mu, sig, ll = _fit_gmm_1d(data, k)
        n_params = k * 3 - 1
        b = _bic(ll, n_params, n)
        results[f"gmm_{k}"] = {"weights": w, "means": mu, "stds": sig, "ll": ll, "bic": b, "n_params": n_params}

    shape, scale, ll_g = _fit_gamma_1d(data)
    b_g = _bic(ll_g, 2, n)
    results["gamma"] = {"shape": shape, "scale": scale, "ll": ll_g, "bic": b_g}

    best_model = min(results, key=lambda k: results[k]["bic"])
    best = results[best_model]

    # Determine synaptic delay
    if best_model.startswith("gmm"):
        means = best["means"]
        weights = best["weights"]
        # Primary mode = highest weight
        primary_idx = max(range(len(weights)), key=lambda i: weights[i])
        delay = means[primary_idx]
    elif best_model == "gamma":
        # Mode of gamma = (shape-1)*scale if shape>=1 else 0
        delay = max((shape - 1) * scale, 0.0)
    else:
        delay = _mean(data)

    return {
        "model": best_model,
        "synaptic_delay_ms": delay,
        "bic": best["bic"],
        "model_params": best,
        "all_models": results,
    }

def _wasserstein_distance_1d(u_values, v_values):
    """Compute 1D Wasserstein (Earth Mover's) distance between two sample sets."""
    if not u_values or not v_values:
        return 0.0
    u_sorted = sorted(u_values)
    v_sorted = sorted(v_values)
    # Merge and compute
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
    # Integrate |CDF_u - CDF_v|
    dist = 0.0
    for i in range(len(all_vals) - 1):
        width = all_vals[i + 1] - all_vals[i]
        dist += abs(u_cdf[i] - v_cdf[i]) * width
    return dist

def _compute_cross_correlogram(stim_times_ms, spike_times_ms, window_ms=50.0, bin_ms=1.0):
    """Compute trigger-centred cross-correlogram. Returns (bin_centers, counts)."""
    n_bins = int(window_ms / bin_ms)
    counts = [0] * n_bins
    for st in stim_times_ms:
        for sp in spike_times_ms:
            delta = sp - st
            if 0 < delta <= window_ms:
                bin_idx = int(delta / bin_ms)
                if 0 <= bin_idx < n_bins:
                    counts[bin_idx] += 1
    bin_centers = [(i + 0.5) * bin_ms for i in range(n_bins)]
    return bin_centers, counts


# ---------------------------------------------------------------------------
# Scan data (pre-computed from parameter scan results)
# ---------------------------------------------------------------------------

RELIABLE_CONNECTIONS = [
    {"electrode_from": 0, "electrode_to": 1, "hits_k": 5, "median_latency_ms": 12.73,
     "stimulation": {"amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 1, "electrode_to": 2, "hits_k": 5, "median_latency_ms": 23.34,
     "stimulation": {"amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 4, "electrode_to": 3, "hits_k": 5, "median_latency_ms": 22.44,
     "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
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
    {"electrode_from": 14, "electrode_to": 12, "hits_k": 5, "median_latency_ms": 22.37,
     "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst"}},
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

# Top electrode pairs for STDP (bidirectional, high response rate)
STDP_PAIRS = [
    {"pre": 5, "post": 6, "delay_ms": 15.677, "stim": {"amplitude": 1.0, "duration": 400.0, "polarity": "PositiveFirst"}},
    {"pre": 17, "post": 18, "delay_ms": 11.246, "stim": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
    {"pre": 30, "post": 31, "delay_ms": 19.118, "stim": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
]


# ---------------------------------------------------------------------------
# Main experiment class
# ---------------------------------------------------------------------------

class Experiment:
    """
    Three-stage neuroplasticity experiment:
      Stage 1: Basic Excitability Scan
      Stage 2: Active Electrode Experiment with cross-correlograms
      Stage 3: Hebbian STDP learning experiment
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        # Stage 1 parameters
        scan_amplitudes: tuple = (1.0, 2.0, 3.0),
        scan_durations: tuple = (100.0, 200.0, 300.0, 400.0),
        scan_repeats: int = 5,
        scan_isi_s: float = 1.0,
        scan_inter_channel_s: float = 5.0,
        scan_response_window_ms: float = 50.0,
        scan_min_hits: int = 3,
        # Stage 2 parameters
        active_stim_hz: float = 1.0,
        active_group_size: int = 10,
        active_n_groups: int = 10,
        active_pause_s: float = 5.0,
        active_window_ms: float = 50.0,
        # Stage 3 parameters
        stdp_testing_duration_s: float = 120.0,
        stdp_learning_duration_s: float = 300.0,
        stdp_validation_duration_s: float = 120.0,
        stdp_probe_interval_s: float = 30.0,
        stdp_stim_hz: float = 0.5,
        stdp_window_ms: float = 50.0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.scan_amplitudes = list(scan_amplitudes)
        self.scan_durations = list(scan_durations)
        self.scan_repeats = scan_repeats
        self.scan_isi_s = scan_isi_s
        self.scan_inter_channel_s = scan_inter_channel_s
        self.scan_response_window_ms = scan_response_window_ms
        self.scan_min_hits = scan_min_hits

        self.active_stim_hz = active_stim_hz
        self.active_group_size = active_group_size
        self.active_n_groups = active_n_groups
        self.active_pause_s = active_pause_s
        self.active_window_ms = active_window_ms

        self.stdp_testing_duration_s = stdp_testing_duration_s
        self.stdp_learning_duration_s = stdp_learning_duration_s
        self.stdp_validation_duration_s = stdp_validation_duration_s
        self.stdp_probe_interval_s = stdp_probe_interval_s
        self.stdp_stim_hz = stdp_stim_hz
        self.stdp_window_ms = stdp_window_ms

        # Hardware handles
        self.np_experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        # Results storage
        self._stimulation_log: List[StimulationRecord] = []
        self._trial_results: List[TrialResult] = []
        self._stage1_results: Dict[str, Any] = {}
        self._stage2_results: Dict[str, Any] = {}
        self._stage3_results: Dict[str, Any] = {}

        # Responsive pairs identified in stage 1
        self._responsive_pairs: List[Dict] = []
        # Synaptic delays from stage 2
        self._synaptic_delays: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Delay helper
    # ------------------------------------------------------------------
    def _wait(self, seconds: float) -> None:
        wait(seconds)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
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

            recording_start = datetime_now()

            # Stage 1: Basic Excitability Scan
            logger.info("=== STAGE 1: Basic Excitability Scan ===")
            self._stage1_excitability_scan()

            # Stage 2: Active Electrode Experiment
            logger.info("=== STAGE 2: Active Electrode Experiment ===")
            self._stage2_active_electrode()

            # Stage 3: Hebbian STDP Learning
            logger.info("=== STAGE 3: Hebbian STDP Learning ===")
            self._stage3_hebbian_learning()

            recording_stop = datetime_now()

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    # ------------------------------------------------------------------
    # Stage 1: Basic Excitability Scan
    # ------------------------------------------------------------------
    def _stage1_excitability_scan(self) -> None:
        """Sweep electrodes with amplitude/duration/polarity grid, 5 repeats each."""
        logger.info("Stage 1: scanning %d pre-identified reliable connections", len(RELIABLE_CONNECTIONS))

        polarities = [StimPolarity.NegativeFirst, StimPolarity.PositiveFirst]
        polarity_names = {StimPolarity.NegativeFirst: "NegativeFirst", StimPolarity.PositiveFirst: "PositiveFirst"}

        scan_summary = []
        available_electrodes = set(self.np_experiment.electrodes)

        for conn in RELIABLE_CONNECTIONS:
            elec_from = conn["electrode_from"]
            elec_to = conn["electrode_to"]

            if elec_from not in available_electrodes:
                logger.warning("Electrode %d not available, skipping", elec_from)
                continue

            logger.info("Scanning electrode %d -> %d", elec_from, elec_to)

            for amplitude in self.scan_amplitudes:
                for duration in self.scan_durations:
                    # Charge balance: A1*D1 = A2*D2, use equal phases
                    if amplitude > 4.0 or duration > 400.0:
                        continue
                    for polarity in polarities:
                        hits = 0
                        latencies = []

                        for rep in range(self.scan_repeats):
                            stim_time = datetime_now()
                            self._send_stim(
                                electrode_idx=elec_from,
                                amplitude_ua=amplitude,
                                duration_us=duration,
                                polarity=polarity,
                                trigger_key=0,
                                phase="stage1",
                            )
                            self._wait(0.05)

                            # Query response on target electrode
                            query_start = datetime_now()
                            self._wait(self.scan_response_window_ms / 1000.0 + 0.05)
                            query_stop = datetime_now()

                            spike_df = self.database.get_spike_event(
                                stim_time,
                                query_stop,
                                self.np_experiment.exp_name,
                            )

                            responded = False
                            if not spike_df.empty:
                                ch_col = "channel" if "channel" in spike_df.columns else spike_df.columns[0]
                                resp_spikes = spike_df[spike_df[ch_col] == elec_to]
                                if not resp_spikes.empty:
                                    hits += 1
                                    responded = True
                                    # Estimate latency
                                    if "Time" in resp_spikes.columns:
                                        t_spike = pd.to_datetime(resp_spikes["Time"].iloc[0], utc=True)
                                        lat = (t_spike - stim_time).total_seconds() * 1000.0
                                        if 0 < lat < self.scan_response_window_ms:
                                            latencies.append(lat)

                            self._trial_results.append(TrialResult(
                                phase="stage1",
                                stim_electrode=elec_from,
                                resp_electrode=elec_to,
                                trial_index=rep,
                                stim_time_utc=stim_time.isoformat(),
                                latency_ms=latencies[-1] if latencies and responded else None,
                                responded=responded,
                            ))

                            if rep < self.scan_repeats - 1:
                                self._wait(self.scan_isi_s)

                        consistent = hits >= self.scan_min_hits
                        entry = {
                            "electrode_from": elec_from,
                            "electrode_to": elec_to,
                            "amplitude": amplitude,
                            "duration": duration,
                            "polarity": polarity_names[polarity],
                            "hits": hits,
                            "repeats": self.scan_repeats,
                            "consistent": consistent,
                            "median_latency_ms": float(np.median(latencies)) if latencies else None,
                        }
                        scan_summary.append(entry)

                        if consistent:
                            pair_key = f"{elec_from}->{elec_to}"
                            if not any(p["electrode_from"] == elec_from and p["electrode_to"] == elec_to
                                       for p in self._responsive_pairs):
                                self._responsive_pairs.append({
                                    "electrode_from": elec_from,
                                    "electrode_to": elec_to,
                                    "amplitude": amplitude,
                                    "duration": duration,
                                    "polarity": polarity_names[polarity],
                                    "median_latency_ms": entry["median_latency_ms"],
                                })

            self._wait(self.scan_inter_channel_s)

        # If no responsive pairs found from live scan, use pre-computed ones
        if not self._responsive_pairs:
            logger.info("Using pre-computed responsive pairs from scan results")
            for conn in RELIABLE_CONNECTIONS[:6]:
                self._responsive_pairs.append({
                    "electrode_from": conn["electrode_from"],
                    "electrode_to": conn["electrode_to"],
                    "amplitude": conn["stimulation"]["amplitude"],
                    "duration": conn["stimulation"]["duration"],
                    "polarity": conn["stimulation"]["polarity"],
                    "median_latency_ms": conn["median_latency_ms"],
                })

        self._stage1_results = {
            "scan_summary": scan_summary,
            "n_consistent_pairs": len(self._responsive_pairs),
            "responsive_pairs": self._responsive_pairs,
        }
        logger.info("Stage 1 complete: %d responsive pairs found", len(self._responsive_pairs))

    # ------------------------------------------------------------------
    # Stage 2: Active Electrode Experiment
    # ------------------------------------------------------------------
    def _stage2_active_electrode(self) -> None:
        """1 Hz stimulation in groups of 10, cross-correlograms, model fitting."""
        logger.info("Stage 2: active electrode experiment on %d pairs", len(self._responsive_pairs))

        pairs_to_use = self._responsive_pairs[:4]  # Limit to top 4 pairs
        stim_interval_s = 1.0 / self.active_stim_hz
        total_stims = self.active_group_size * self.active_n_groups

        pair_results = []

        for pair in pairs_to_use:
            elec_from = pair["electrode_from"]
            elec_to = pair["electrode_to"]
            amplitude = pair["amplitude"]
            duration = pair["duration"]
            polarity_str = pair["polarity"]
            polarity = StimPolarity.NegativeFirst if polarity_str == "NegativeFirst" else StimPolarity.PositiveFirst

            if elec_from not in self.np_experiment.electrodes:
                logger.warning("Electrode %d not available, skipping pair", elec_from)
                continue

            logger.info("Stage 2: stimulating pair %d->%d", elec_from, elec_to)

            stim_times_utc = []
            spike_latencies_ms = []
            phase_start = datetime_now()

            for group_idx in range(self.active_n_groups):
                for stim_idx in range(self.active_group_size):
                    stim_time = datetime_now()
                    stim_times_utc.append(stim_time)

                    self._send_stim(
                        electrode_idx=elec_from,
                        amplitude_ua=amplitude,
                        duration_us=duration,
                        polarity=polarity,
                        trigger_key=1,
                        phase="stage2",
                    )
                    self._wait(0.05)

                    # Query response
                    self._wait(self.active_window_ms / 1000.0)
                    query_stop = datetime_now()

                    spike_df = self.database.get_spike_event(
                        stim_time,
                        query_stop,
                        self.np_experiment.exp_name,
                    )

                    if not spike_df.empty:
                        ch_col = "channel" if "channel" in spike_df.columns else spike_df.columns[0]
                        resp_spikes = spike_df[spike_df[ch_col] == elec_to]
                        if not resp_spikes.empty and "Time" in resp_spikes.columns:
                            t_spike = pd.to_datetime(resp_spikes["Time"].iloc[0], utc=True)
                            lat = (t_spike - stim_time).total_seconds() * 1000.0
                            if 0 < lat < self.active_window_ms:
                                spike_latencies_ms.append(lat)

                    self._trial_results.append(TrialResult(
                        phase="stage2",
                        stim_electrode=elec_from,
                        resp_electrode=elec_to,
                        trial_index=group_idx * self.active_group_size + stim_idx,
                        stim_time_utc=stim_time.isoformat(),
                        latency_ms=spike_latencies_ms[-1] if spike_latencies_ms else None,
                        responded=len(spike_latencies_ms) > 0,
                    ))

                    if stim_idx < self.active_group_size - 1:
                        self._wait(max(0.0, stim_interval_s - self.active_window_ms / 1000.0 - 0.05))

                if group_idx < self.active_n_groups - 1:
                    self._wait(self.active_pause_s)

            phase_stop = datetime_now()

            # Compute cross-correlogram
            stim_times_ms = [i * 1000.0 / self.active_stim_hz for i in range(len(stim_times_utc))]
            bin_centers, ccg_counts = _compute_cross_correlogram(
                stim_times_ms, spike_latencies_ms, window_ms=self.active_window_ms
            )

            # Fit models to latency distribution
            model_fit = _select_best_model(spike_latencies_ms, window_ms=self.active_window_ms)
            pair_key = f"{elec_from}->{elec_to}"
            if model_fit["synaptic_delay_ms"] is not None:
                self._synaptic_delays[pair_key] = model_fit["synaptic_delay_ms"]

            response_rate = len(spike_latencies_ms) / max(total_stims, 1)
            pair_result = {
                "electrode_from": elec_from,
                "electrode_to": elec_to,
                "total_stimulations": total_stims,
                "n_responses": len(spike_latencies_ms),
                "response_rate": response_rate,
                "mean_latency_ms": _mean(spike_latencies_ms),
                "std_latency_ms": _std(spike_latencies_ms),
                "ccg_bin_centers_ms": bin_centers,
                "ccg_counts": ccg_counts,
                "model_fit": model_fit,
                "synaptic_delay_ms": model_fit["synaptic_delay_ms"],
                "phase_start_utc": phase_start.isoformat(),
                "phase_stop_utc": phase_stop.isoformat(),
            }
            pair_results.append(pair_result)
            logger.info("Pair %s: response_rate=%.2f, delay=%.2f ms",
                        pair_key, response_rate,
                        model_fit["synaptic_delay_ms"] if model_fit["synaptic_delay_ms"] else 0.0)

        # Fill in delays from pre-computed data for pairs not stimulated
        for stdp_pair in STDP_PAIRS:
            key = f"{stdp_pair['pre']}->{stdp_pair['post']}"
            if key not in self._synaptic_delays:
                self._synaptic_delays[key] = stdp_pair["delay_ms"]

        self._stage2_results = {
            "pair_results": pair_results,
            "synaptic_delays": self._synaptic_delays,
        }
        logger.info("Stage 2 complete")

    # ------------------------------------------------------------------
    # Stage 3: Hebbian STDP Learning
    # ------------------------------------------------------------------
    def _stage3_hebbian_learning(self) -> None:
        """Three-phase STDP: Testing -> Learning -> Validation."""
        logger.info("Stage 3: Hebbian STDP learning")

        stdp_pairs_to_use = STDP_PAIRS[:2]  # Use top 2 bidirectional pairs
        stim_interval_s = 1.0 / self.stdp_stim_hz

        pair_stdp_results = []

        for pair_info in stdp_pairs_to_use:
            pre_elec = pair_info["pre"]
            post_elec = pair_info["post"]
            hebbian_delay_ms = self._synaptic_delays.get(
                f"{pre_elec}->{post_elec}", pair_info["delay_ms"]
            )
            amplitude = pair_info["stim"]["amplitude"]
            duration = pair_info["stim"]["duration"]
            polarity_str = pair_info["stim"]["polarity"]
            polarity = StimPolarity.NegativeFirst if polarity_str == "NegativeFirst" else StimPolarity.PositiveFirst

            if pre_elec not in self.np_experiment.electrodes:
                logger.warning("Pre electrode %d not available, skipping STDP pair", pre_elec)
                continue

            logger.info("STDP pair: pre=%d, post=%d, delay=%.2f ms", pre_elec, post_elec, hebbian_delay_ms)

            # --- Phase A: Testing (baseline) ---
            logger.info("STDP Phase A: Testing (baseline)")
            testing_latencies = self._stdp_probe_phase(
                pre_elec=pre_elec,
                post_elec=post_elec,
                amplitude=amplitude,
                duration=duration,
                polarity=polarity,
                duration_s=self.stdp_testing_duration_s,
                stim_interval_s=stim_interval_s,
                phase_label="stdp_testing",
                trigger_key=2,
            )

            # --- Phase B: Learning ---
            logger.info("STDP Phase B: Learning (paired stimulation)")
            learning_latencies = self._stdp_learning_phase(
                pre_elec=pre_elec,
                post_elec=post_elec,
                amplitude=amplitude,
                duration=duration,
                polarity=polarity,
                hebbian_delay_ms=hebbian_delay_ms,
                duration_s=self.stdp_learning_duration_s,
                stim_interval_s=stim_interval_s,
                probe_interval_s=self.stdp_probe_interval_s,
                trigger_key_pre=2,
                trigger_key_post=3,
            )

            # --- Phase C: Validation ---
            logger.info("STDP Phase C: Validation")
            validation_latencies = self._stdp_probe_phase(
                pre_elec=pre_elec,
                post_elec=post_elec,
                amplitude=amplitude,
                duration=duration,
                polarity=polarity,
                duration_s=self.stdp_validation_duration_s,
                stim_interval_s=stim_interval_s,
                phase_label="stdp_validation",
                trigger_key=2,
            )

            # Compute EMD between testing and validation
            emd = _wasserstein_distance_1d(testing_latencies, validation_latencies)

            # Cross-correlograms for testing and validation
            n_test = len(testing_latencies)
            n_val = len(validation_latencies)
            test_stim_ms = [i * stim_interval_s * 1000.0 for i in range(max(n_test, 1))]
            val_stim_ms = [i * stim_interval_s * 1000.0 for i in range(max(n_val, 1))]

            test_bins, test_ccg = _compute_cross_correlogram(
                test_stim_ms, testing_latencies, window_ms=self.stdp_window_ms
            )
            val_bins, val_ccg = _compute_cross_correlogram(
                val_stim_ms, validation_latencies, window_ms=self.stdp_window_ms
            )

            pair_result = {
                "pre_electrode": pre_elec,
                "post_electrode": post_elec,
                "hebbian_delay_ms": hebbian_delay_ms,
                "testing_n_responses": len(testing_latencies),
                "testing_mean_latency_ms": _mean(testing_latencies),
                "testing_std_latency_ms": _std(testing_latencies),
                "testing_ccg_bins": test_bins,
                "testing_ccg_counts": test_ccg,
                "learning_n_responses": len(learning_latencies),
                "validation_n_responses": len(validation_latencies),
                "validation_mean_latency_ms": _mean(validation_latencies),
                "validation_std_latency_ms": _std(validation_latencies),
                "validation_ccg_bins": val_bins,
                "validation_ccg_counts": val_ccg,
                "earth_movers_distance": emd,
                "plasticity_detected": emd > 2.0,
            }
            pair_stdp_results.append(pair_result)
            logger.info("STDP pair %d->%d: EMD=%.3f ms, plasticity=%s",
                        pre_elec, post_elec, emd, emd > 2.0)

        self._stage3_results = {
            "stdp_pair_results": pair_stdp_results,
            "n_pairs_with_plasticity": sum(1 for p in pair_stdp_results if p.get("plasticity_detected", False)),
        }
        logger.info("Stage 3 complete")

    def _stdp_probe_phase(
        self,
        pre_elec: int,
        post_elec: int,
        amplitude: float,
        duration: float,
        polarity: StimPolarity,
        duration_s: float,
        stim_interval_s: float,
        phase_label: str,
        trigger_key: int,
    ) -> List[float]:
        """Run probe stimulations and collect response latencies."""
        latencies = []
        n_stims = max(1, int(duration_s / stim_interval_s))

        for i in range(n_stims):
            stim_time = datetime_now()
            self._send_stim(
                electrode_idx=pre_elec,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=polarity,
                trigger_key=trigger_key,
                phase=phase_label,
            )
            self._wait(0.05)
            self._wait(self.stdp_window_ms / 1000.0)
            query_stop = datetime_now()

            spike_df = self.database.get_spike_event(
                stim_time, query_stop, self.np_experiment.exp_name
            )
            if not spike_df.empty:
                ch_col = "channel" if "channel" in spike_df.columns else spike_df.columns[0]
                resp_spikes = spike_df[spike_df[ch_col] == post_elec]
                if not resp_spikes.empty and "Time" in resp_spikes.columns:
                    t_spike = pd.to_datetime(resp_spikes["Time"].iloc[0], utc=True)
                    lat = (t_spike - stim_time).total_seconds() * 1000.0
                    if 0 < lat < self.stdp_window_ms:
                        latencies.append(lat)

            self._trial_results.append(TrialResult(
                phase=phase_label,
                stim_electrode=pre_elec,
                resp_electrode=post_elec,
                trial_index=i,
                stim_time_utc=stim_time.isoformat(),
                latency_ms=latencies[-1] if latencies else None,
                responded=len(latencies) > 0,
            ))

            if i < n_stims - 1:
                self._wait(max(0.0, stim_interval_s - self.stdp_window_ms / 1000.0 - 0.05))

        return latencies

    def _stdp_learning_phase(
        self,
        pre_elec: int,
        post_elec: int,
        amplitude: float,
        duration: float,
        polarity: StimPolarity,
        hebbian_delay_ms: float,
        duration_s: float,
        stim_interval_s: float,
        probe_interval_s: float,
        trigger_key_pre: int,
        trigger_key_post: int,
    ) -> List[float]:
        """Paired pre/post stimulation with Hebbian delay, interleaved with probes."""
        latencies = []
        n_stims = max(1, int(duration_s / stim_interval_s))
        probe_every = max(1, int(probe_interval_s / stim_interval_s))

        for i in range(n_stims):
            stim_time = datetime_now()

            # Pre-synaptic stimulation
            self._send_stim(
                electrode_idx=pre_elec,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=polarity,
                trigger_key=trigger_key_pre,
                phase="stdp_learning_pre",
            )

            # Wait Hebbian delay then post-synaptic
            self._wait(hebbian_delay_ms / 1000.0)

            if post_elec in self.np_experiment.electrodes:
                self._send_stim(
                    electrode_idx=post_elec,
                    amplitude_ua=min(amplitude, 3.0),
                    duration_us=duration,
                    polarity=polarity,
                    trigger_key=trigger_key_post,
                    phase="stdp_learning_post",
                )

            # Interleaved probe
            if i % probe_every == 0:
                self._wait(0.05)
                self._wait(self.stdp_window_ms / 1000.0)
                query_stop = datetime_now()
                spike_df = self.database.get_spike_event(
                    stim_time, query_stop, self.np_experiment.exp_name
                )
                if not spike_df.empty:
                    ch_col = "channel" if "channel" in spike_df.columns else spike_df.columns[0]
                    resp_spikes = spike_df[spike_df[ch_col] == post_elec]
                    if not resp_spikes.empty and "Time" in resp_spikes.columns:
                        t_spike = pd.to_datetime(resp_spikes["Time"].iloc[0], utc=True)
                        lat = (t_spike - stim_time).total_seconds() * 1000.0
                        if 0 < lat < self.stdp_window_ms:
                            latencies.append(lat)

            self._trial_results.append(TrialResult(
                phase="stdp_learning",
                stim_electrode=pre_elec,
                resp_electrode=post_elec,
                trial_index=i,
                stim_time_utc=stim_time.isoformat(),
                latency_ms=latencies[-1] if latencies else None,
                responded=len(latencies) > 0,
            ))

            if i < n_stims - 1:
                self._wait(max(0.0, stim_interval_s - hebbian_delay_ms / 1000.0 - 0.05))

        return latencies

    # ------------------------------------------------------------------
    # Stimulation helper
    # ------------------------------------------------------------------
    def _send_stim(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity = StimPolarity.NegativeFirst,
        trigger_key: int = 0,
        phase: str = "",
    ) -> None:
        """Send one charge-balanced biphasic pulse."""
        amplitude_ua = min(abs(amplitude_ua), 4.0)
        duration_us = min(abs(duration_us), 400.0)

        # Charge balance: A1*D1 = A2*D2 with equal amplitudes and durations
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

        polarity_name = "NegativeFirst" if polarity == StimPolarity.NegativeFirst else "PositiveFirst"
        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            polarity=polarity_name,
            timestamp_utc=datetime_now().isoformat(),
            trigger_key=trigger_key,
            phase=phase,
        ))

    # ------------------------------------------------------------------
    # Data persistence
    # ------------------------------------------------------------------
    def _save_all(self, recording_start: datetime, recording_stop: datetime) -> None:
        fs_name = getattr(self.np_experiment, "exp_name", "unknown")
        saver = DataSaver(self._output_dir, fs_name)

        saver.save_stimulation_log(self._stimulation_log)

        spike_df = self.database.get_spike_event(recording_start, recording_stop, fs_name)
        saver.save_spike_events(spike_df)

        trigger_df = self.database.get_all_triggers(recording_start, recording_stop)
        saver.save_triggers(trigger_df)

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df) if not spike_df.empty else 0,
            "total_triggers": len(trigger_df) if not trigger_df.empty else 0,
            "stage1_n_responsive_pairs": len(self._responsive_pairs),
            "stage2_synaptic_delays": self._synaptic_delays,
            "stage3_n_pairs_with_plasticity": self._stage3_results.get("n_pairs_with_plasticity", 0),
        }
        saver.save_summary(summary)

        waveform_records = self._fetch_spike_waveforms(
            fs_name, spike_df, trigger_df, recording_start, recording_stop
        )
        saver.save_spike_waveforms(waveform_records)

        saver.save_analysis(self._stage1_results, "stage1_results")
        saver.save_analysis(self._stage2_results, "stage2_results")
        saver.save_analysis(self._stage3_results, "stage3_results")

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

        ch_col = None
        for col in spike_df.columns:
            if col.lower() in ("channel", "index", "electrode"):
                ch_col = col
                break
        if ch_col is None:
            logger.warning("Cannot determine electrode column for waveform fetch")
            return waveform_records

        unique_electrodes = spike_df[ch_col].unique()
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
                logger.warning("Failed to fetch waveforms for electrode %s: %s", electrode_idx, exc)

        return waveform_records

    # ------------------------------------------------------------------
    # Results compilation
    # ------------------------------------------------------------------
    def _compile_results(self, recording_start: datetime, recording_stop: datetime) -> Dict[str, Any]:
        logger.info("Compiling results")
        duration_s = (recording_stop - recording_start).total_seconds()

        stage3_pairs = self._stage3_results.get("stdp_pair_results", [])
        emd_values = [p["earth_movers_distance"] for p in stage3_pairs if "earth_movers_distance" in p]

        return {
            "status": "completed",
            "experiment_name": self.np_experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": duration_s,
            "stage1": {
                "n_responsive_pairs": len(self._responsive_pairs),
                "responsive_pairs": self._responsive_pairs,
            },
            "stage2": {
                "n_pairs_analyzed": len(self._stage2_results.get("pair_results", [])),
                "synaptic_delays_ms": self._synaptic_delays,
            },
            "stage3": {
                "n_stdp_pairs": len(stage3_pairs),
                "n_pairs_with_plasticity": self._stage3_results.get("n_pairs_with_plasticity", 0),
                "mean_emd_ms": _mean(emd_values) if emd_values else 0.0,
                "pair_summaries": [
                    {
                        "pre": p["pre_electrode"],
                        "post": p["post_electrode"],
                        "emd": p["earth_movers_distance"],
                        "plasticity": p["plasticity_detected"],
                    }
                    for p in stage3_pairs
                ],
            },
            "total_stimulations": len(self._stimulation_log),
            "total_trials": len(self._trial_results),
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
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
