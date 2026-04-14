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
    phase: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PairConfig:
    stim_electrode: int
    resp_electrode: int
    amplitude: float
    duration: float
    polarity: str
    median_latency_ms: float
    response_rate: float
    hits_k: int = 5
    repeats_n: int = 5


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


def _compute_wasserstein_1d(u_values: List[float], v_values: List[float]) -> float:
    if not u_values or not v_values:
        return float("nan")
    u_sorted = sorted(u_values)
    v_sorted = sorted(v_values)
    all_vals = sorted(set(u_sorted + v_sorted))
    if len(all_vals) < 2:
        return 0.0
    u_cdf = []
    v_cdf = []
    ui = 0
    vi = 0
    nu = len(u_sorted)
    nv = len(v_sorted)
    for val in all_vals:
        while ui < nu and u_sorted[ui] <= val:
            ui += 1
        while vi < nv and v_sorted[vi] <= val:
            vi += 1
        u_cdf.append(ui / nu)
        v_cdf.append(vi / nv)
    emd = 0.0
    for i in range(len(all_vals) - 1):
        dx = all_vals[i + 1] - all_vals[i]
        emd += abs(u_cdf[i] - v_cdf[i]) * dx
    return emd


def _fit_gaussian_mixture_1d(data: List[float], n_components: int, max_iter: int = 50) -> Dict[str, Any]:
    if len(data) < n_components * 2:
        return {"bic": float("inf"), "means": [], "stds": [], "weights": []}
    n = len(data)
    arr = np.array(data, dtype=float)
    dmin = float(np.min(arr))
    dmax = float(np.max(arr))
    if dmax == dmin:
        return {"bic": float("inf"), "means": [float(np.mean(arr))], "stds": [0.0], "weights": [1.0]}
    means = [dmin + (dmax - dmin) * (k + 1) / (n_components + 1) for k in range(n_components)]
    stds = [(dmax - dmin) / (2.0 * n_components)] * n_components
    weights = [1.0 / n_components] * n_components
    log_likelihood = -float("inf")
    for iteration in range(max_iter):
        resp = np.zeros((n, n_components))
        for k in range(n_components):
            if stds[k] < 1e-10:
                stds[k] = 1e-10
            diff = arr - means[k]
            resp[:, k] = weights[k] * np.exp(-0.5 * (diff / stds[k]) ** 2) / (stds[k] * math.sqrt(2 * math.pi))
        row_sums = resp.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-300)
        resp = resp / row_sums
        nk = resp.sum(axis=0)
        for k in range(n_components):
            if nk[k] < 1e-10:
                continue
            means[k] = float(np.dot(resp[:, k], arr) / nk[k])
            diff = arr - means[k]
            stds[k] = float(math.sqrt(np.dot(resp[:, k], diff ** 2) / nk[k]))
            if stds[k] < 1e-10:
                stds[k] = 1e-10
            weights[k] = float(nk[k] / n)
        new_ll = 0.0
        for i in range(n):
            p = 0.0
            for k in range(n_components):
                diff = arr[i] - means[k]
                p += weights[k] * math.exp(-0.5 * (diff / stds[k]) ** 2) / (stds[k] * math.sqrt(2 * math.pi))
            new_ll += math.log(max(p, 1e-300))
        if abs(new_ll - log_likelihood) < 1e-6:
            log_likelihood = new_ll
            break
        log_likelihood = new_ll
    num_params = 3 * n_components - 1
    bic = -2 * log_likelihood + num_params * math.log(n)
    return {"bic": bic, "means": means, "stds": stds, "weights": weights, "log_likelihood": log_likelihood}


def _fit_gamma_1d(data: List[float]) -> Dict[str, Any]:
    if len(data) < 3:
        return {"bic": float("inf"), "shape": 0, "scale": 0}
    arr = np.array(data, dtype=float)
    arr = arr[arr > 0]
    if len(arr) < 3:
        return {"bic": float("inf"), "shape": 0, "scale": 0}
    n = len(arr)
    mean_x = float(np.mean(arr))
    var_x = float(np.var(arr))
    if var_x < 1e-10:
        return {"bic": float("inf"), "shape": 0, "scale": 0}
    shape = mean_x ** 2 / var_x
    scale = var_x / mean_x
    log_likelihood = 0.0
    for x in arr:
        lp = (shape - 1) * math.log(x) - x / scale - shape * math.log(scale) - math.lgamma(shape)
        log_likelihood += lp
    bic = -2 * log_likelihood + 2 * math.log(n)
    return {"bic": bic, "shape": shape, "scale": scale, "log_likelihood": log_likelihood}


def _select_best_model(data: List[float]) -> Dict[str, Any]:
    results = {}
    for nc in [1, 2, 3]:
        key = f"gmm_{nc}"
        results[key] = _fit_gaussian_mixture_1d(data, nc)
    results["gamma"] = _fit_gamma_1d(data)
    best_name = None
    best_bic = float("inf")
    for name, res in results.items():
        if res["bic"] < best_bic:
            best_bic = res["bic"]
            best_name = name
    return {"best_model": best_name, "best_bic": best_bic, "all_models": results}


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
        scan_max_hits: int = 5,
        active_total_repeats: int = 100,
        active_group_size: int = 10,
        active_inter_stim_s: float = 1.0,
        active_inter_group_s: float = 5.0,
        active_window_ms: float = 50.0,
        hebbian_test_duration_min: float = 20.0,
        hebbian_learn_duration_min: float = 50.0,
        hebbian_valid_duration_min: float = 20.0,
        hebbian_test_rate_hz: float = 0.5,
        hebbian_learn_rate_hz: float = 1.0,
        hebbian_probe_interval_min: float = 5.0,
        hebbian_probe_count: int = 20,
        max_pairs_to_use: int = 3,
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
        self.scan_max_hits = scan_max_hits

        self.active_total_repeats = active_total_repeats
        self.active_group_size = active_group_size
        self.active_inter_stim_s = active_inter_stim_s
        self.active_inter_group_s = active_inter_group_s
        self.active_window_ms = active_window_ms

        self.hebbian_test_duration_min = hebbian_test_duration_min
        self.hebbian_learn_duration_min = hebbian_learn_duration_min
        self.hebbian_valid_duration_min = hebbian_valid_duration_min
        self.hebbian_test_rate_hz = hebbian_test_rate_hz
        self.hebbian_learn_rate_hz = hebbian_learn_rate_hz
        self.hebbian_probe_interval_min = hebbian_probe_interval_min
        self.hebbian_probe_count = hebbian_probe_count
        self.max_pairs_to_use = max_pairs_to_use

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []

        self._scan_results: Dict[str, Any] = {}
        self._responsive_pairs: List[PairConfig] = []
        self._active_results: Dict[str, Any] = {}
        self._hebbian_results: Dict[str, Any] = {}

        self._prior_pairs = self._load_prior_scan_pairs()

    def _load_prior_scan_pairs(self) -> List[PairConfig]:
        best_per_pair: Dict[Tuple[int, int], PairConfig] = {}
        deep_scan = [
            {"stim": 0, "resp": 1, "amp": 3.0, "dur": 300.0, "pol": "NegativeFirst", "lat": 24.005, "rate": 0.94},
            {"stim": 1, "resp": 2, "amp": 2.0, "dur": 400.0, "pol": "NegativeFirst", "lat": 15.795, "rate": 0.82},
            {"stim": 5, "resp": 4, "amp": 1.0, "dur": 300.0, "pol": "PositiveFirst", "lat": 16.37, "rate": 0.94},
            {"stim": 5, "resp": 6, "amp": 1.0, "dur": 200.0, "pol": "NegativeFirst", "lat": 16.98, "rate": 0.77},
            {"stim": 6, "resp": 5, "amp": 3.0, "dur": 200.0, "pol": "PositiveFirst", "lat": 15.63, "rate": 0.89},
            {"stim": 8, "resp": 9, "amp": 2.0, "dur": 200.0, "pol": "PositiveFirst", "lat": 18.71, "rate": 0.72},
            {"stim": 9, "resp": 11, "amp": 3.0, "dur": 400.0, "pol": "NegativeFirst", "lat": 21.305, "rate": 0.90},
            {"stim": 14, "resp": 15, "amp": 1.0, "dur": 300.0, "pol": "PositiveFirst", "lat": 23.54, "rate": 0.74},
            {"stim": 17, "resp": 16, "amp": 2.0, "dur": 300.0, "pol": "NegativeFirst", "lat": 15.09, "rate": 0.81},
            {"stim": 18, "resp": 17, "amp": 1.0, "dur": 200.0, "pol": "PositiveFirst", "lat": 25.015, "rate": 0.72},
            {"stim": 20, "resp": 22, "amp": 3.0, "dur": 400.0, "pol": "NegativeFirst", "lat": 22.16, "rate": 0.94},
            {"stim": 26, "resp": 27, "amp": 3.0, "dur": 400.0, "pol": "PositiveFirst", "lat": 23.61, "rate": 0.99},
            {"stim": 28, "resp": 27, "amp": 3.0, "dur": 400.0, "pol": "NegativeFirst", "lat": 22.925, "rate": 0.90},
            {"stim": 12, "resp": 11, "amp": 3.0, "dur": 400.0, "pol": "NegativeFirst", "lat": 12.71, "rate": 0.92},
            {"stim": 10, "resp": 12, "amp": 3.0, "dur": 400.0, "pol": "NegativeFirst", "lat": 18.025, "rate": 0.94},
            {"stim": 4, "resp": 3, "amp": 2.0, "dur": 400.0, "pol": "PositiveFirst", "lat": 19.64, "rate": 0.85},
        ]
        for entry in deep_scan:
            key = (entry["stim"], entry["resp"])
            existing = best_per_pair.get(key)
            if existing is None or entry["rate"] > existing.response_rate:
                best_per_pair[key] = PairConfig(
                    stim_electrode=entry["stim"],
                    resp_electrode=entry["resp"],
                    amplitude=entry["amp"],
                    duration=entry["dur"],
                    polarity=entry["pol"],
                    median_latency_ms=entry["lat"],
                    response_rate=entry["rate"],
                )
        pairs = sorted(best_per_pair.values(), key=lambda p: -p.response_rate)
        return pairs

    def _wait(self, seconds: float) -> None:
        if not self.testing:
            time.sleep(seconds)

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

            logger.info("=== STAGE 1: Basic Excitability Scan ===")
            self._phase_excitability_scan()

            logger.info("=== STAGE 2: Active Electrode Experiment ===")
            self._phase_active_electrode()

            logger.info("=== STAGE 3: Hebbian Learning Experiment ===")
            self._phase_hebbian_learning()

            recording_stop = datetime.now(timezone.utc)

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _make_stim_param(
        self,
        electrode_idx: int,
        amplitude: float,
        duration: float,
        polarity_str: str = "NegativeFirst",
        trigger_key: int = 0,
    ) -> StimParam:
        amplitude = min(abs(amplitude), 4.0)
        duration = min(abs(duration), 400.0)
        stim = StimParam()
        stim.index = electrode_idx
        stim.enable = True
        stim.trigger_key = trigger_key
        stim.trigger_delay = 0
        stim.nb_pulse = 0
        stim.pulse_train_period = 10000
        stim.post_stim_ref_period = 1000.0
        stim.stim_shape = StimShape.Biphasic
        stim.polarity = StimPolarity.NegativeFirst if polarity_str == "NegativeFirst" else StimPolarity.PositiveFirst
        stim.phase_amplitude1 = amplitude
        stim.phase_duration1 = duration
        stim.phase_amplitude2 = amplitude
        stim.phase_duration2 = duration
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

    def _stimulate_once(
        self,
        electrode_idx: int,
        amplitude: float,
        duration: float,
        polarity_str: str = "NegativeFirst",
        trigger_key: int = 0,
        phase_label: str = "",
    ) -> None:
        stim = self._make_stim_param(electrode_idx, amplitude, duration, polarity_str, trigger_key)
        self.intan.send_stimparam([stim])
        self._fire_trigger(trigger_key)
        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=min(abs(amplitude), 4.0),
            duration_us=min(abs(duration), 400.0),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            trigger_key=trigger_key,
            phase=phase_label,
        ))

    def _phase_excitability_scan(self) -> None:
        electrodes = self.experiment.electrodes
        if not electrodes:
            electrodes = list(range(32))
        scan_electrodes = electrodes[:8]
        polarities = ["NegativeFirst", "PositiveFirst"]
        scan_results = []
        trigger_key = 0

        for elec_idx in scan_electrodes:
            logger.info("Scanning electrode %d", elec_idx)
            for amp in self.scan_amplitudes:
                for dur in self.scan_durations:
                    if dur > 400.0:
                        continue
                    if amp > 4.0:
                        continue
                    for pol in polarities:
                        pre_time = datetime.now(timezone.utc)
                        hit_count = 0
                        for rep in range(self.scan_repeats):
                            self._stimulate_once(elec_idx, amp, dur, pol, trigger_key, phase_label="scan")
                            self._wait(self.scan_inter_stim_s)
                        post_time = datetime.now(timezone.utc) + timedelta(seconds=1)
                        spike_df = self.database.get_spike_event(
                            pre_time - timedelta(seconds=2),
                            post_time,
                            self.experiment.exp_name,
                        )
                        if not spike_df.empty and "channel" in spike_df.columns:
                            other_channels = spike_df[spike_df["channel"] != elec_idx]
                            hit_count = min(len(other_channels), self.scan_repeats)
                        scan_results.append({
                            "electrode": elec_idx,
                            "amplitude": amp,
                            "duration": dur,
                            "polarity": pol,
                            "hits": hit_count,
                            "repeats": self.scan_repeats,
                        })
            self._wait(self.scan_inter_channel_s)

        responsive = [
            r for r in scan_results
            if self.scan_min_hits <= r["hits"] <= self.scan_max_hits
        ]
        self._scan_results = {
            "all_results": scan_results,
            "responsive_count": len(responsive),
            "responsive": responsive,
        }
        logger.info("Excitability scan complete. %d responsive conditions found.", len(responsive))

        if self._prior_pairs:
            self._responsive_pairs = self._prior_pairs[:self.max_pairs_to_use]
        else:
            seen = set()
            for r in responsive:
                key = r["electrode"]
                if key not in seen:
                    seen.add(key)
                    self._responsive_pairs.append(PairConfig(
                        stim_electrode=r["electrode"],
                        resp_electrode=(r["electrode"] + 1) % 32,
                        amplitude=r["amplitude"],
                        duration=r["duration"],
                        polarity=r["polarity"],
                        median_latency_ms=20.0,
                        response_rate=r["hits"] / r["repeats"],
                    ))
                if len(self._responsive_pairs) >= self.max_pairs_to_use:
                    break

        logger.info("Using %d pairs for active/Hebbian phases", len(self._responsive_pairs))

    def _phase_active_electrode(self) -> None:
        if not self._responsive_pairs:
            logger.warning("No responsive pairs found. Skipping active electrode phase.")
            self._active_results = {"status": "skipped", "reason": "no_pairs"}
            return

        pair_results = []
        trigger_key = 1

        for pair_idx, pair in enumerate(self._responsive_pairs):
            logger.info(
                "Active electrode pair %d: stim=%d -> resp=%d (amp=%.1f, dur=%.0f, pol=%s)",
                pair_idx, pair.stim_electrode, pair.resp_electrode,
                pair.amplitude, pair.duration, pair.polarity,
            )
            stim_times = []
            phase_start = datetime.now(timezone.utc)

            num_groups = self.active_total_repeats // self.active_group_size
            remainder = self.active_total_repeats % self.active_group_size

            for group_idx in range(num_groups):
                for stim_idx in range(self.active_group_size):
                    t_stim = datetime.now(timezone.utc)
                    self._stimulate_once(
                        pair.stim_electrode, pair.amplitude, pair.duration,
                        pair.polarity, trigger_key, phase_label="active",
                    )
                    stim_times.append(t_stim.isoformat())
                    self._wait(self.active_inter_stim_s)
                self._wait(self.active_inter_group_s)

            for stim_idx in range(remainder):
                t_stim = datetime.now(timezone.utc)
                self._stimulate_once(
                    pair.stim_electrode, pair.amplitude, pair.duration,
                    pair.polarity, trigger_key, phase_label="active",
                )
                stim_times.append(t_stim.isoformat())
                self._wait(self.active_inter_stim_s)

            phase_end = datetime.now(timezone.utc) + timedelta(seconds=2)

            spike_df = self.database.get_spike_event(
                phase_start - timedelta(seconds=1), phase_end, self.experiment.exp_name,
            )
            trigger_df = self.database.get_all_triggers(
                phase_start - timedelta(seconds=1), phase_end,
            )

            latencies = self._compute_latencies(
                spike_df, trigger_df, pair.resp_electrode, self.active_window_ms,
            )

            model_result = _select_best_model(latencies) if latencies else {"best_model": "none", "best_bic": float("inf"), "all_models": {}}

            synaptic_delay = pair.median_latency_ms
            if latencies:
                synaptic_delay = float(np.median(latencies))

            pair_results.append({
                "pair_index": pair_idx,
                "stim_electrode": pair.stim_electrode,
                "resp_electrode": pair.resp_electrode,
                "total_stims": len(stim_times),
                "latencies_count": len(latencies),
                "median_latency_ms": synaptic_delay,
                "mean_latency_ms": float(np.mean(latencies)) if latencies else float("nan"),
                "best_model": model_result["best_model"],
                "model_details": model_result,
                "synaptic_delay_ms": synaptic_delay,
            })

            pair.median_latency_ms = synaptic_delay
            logger.info(
                "Pair %d: %d latencies, median=%.2f ms, best_model=%s",
                pair_idx, len(latencies), synaptic_delay, model_result["best_model"],
            )

            self._wait(self.scan_inter_channel_s)

        self._active_results = {"pairs": pair_results}

    def _compute_latencies(
        self,
        spike_df: pd.DataFrame,
        trigger_df: pd.DataFrame,
        resp_electrode: int,
        window_ms: float,
    ) -> List[float]:
        latencies = []
        if spike_df.empty or trigger_df.empty:
            return latencies

        resp_spikes = spike_df
        if "channel" in spike_df.columns:
            resp_spikes = spike_df[spike_df["channel"] == resp_electrode]
        if resp_spikes.empty:
            return latencies

        time_col_spike = "Time" if "Time" in resp_spikes.columns else "_time"
        time_col_trig = "_time" if "_time" in trigger_df.columns else "Time"

        if time_col_spike not in resp_spikes.columns or time_col_trig not in trigger_df.columns:
            return latencies

        spike_times = pd.to_datetime(resp_spikes[time_col_spike], utc=True)
        trig_times = pd.to_datetime(trigger_df[time_col_trig], utc=True)

        spike_ts = np.array([t.timestamp() for t in spike_times])
        trig_ts = np.array([t.timestamp() for t in trig_times])

        if len(spike_ts) == 0 or len(trig_ts) == 0:
            return latencies

        window_s = window_ms / 1000.0
        for tt in trig_ts:
            diffs = spike_ts - tt
            in_window = diffs[(diffs > 0) & (diffs <= window_s)]
            for d in in_window:
                latencies.append(d * 1000.0)

        return latencies

    def _phase_hebbian_learning(self) -> None:
        if not self._responsive_pairs:
            logger.warning("No responsive pairs. Skipping Hebbian phase.")
            self._hebbian_results = {"status": "skipped", "reason": "no_pairs"}
            return

        hebbian_results = []
        pre_trigger_key = 2
        post_trigger_key = 3

        for pair_idx, pair in enumerate(self._responsive_pairs):
            logger.info(
                "Hebbian pair %d: stim=%d -> resp=%d, delay=%.2f ms",
                pair_idx, pair.stim_electrode, pair.resp_electrode, pair.median_latency_ms,
            )

            logger.info("  Testing phase (baseline)")
            test_start = datetime.now(timezone.utc)
            test_latencies = self._run_probe_phase(
                pair, self.hebbian_test_duration_min, self.hebbian_test_rate_hz,
                pre_trigger_key, "hebbian_test",
            )
            test_end = datetime.now(timezone.utc)

            logger.info("  Learning phase (STDP conditioning)")
            learn_start = datetime.now(timezone.utc)
            learn_probe_latencies = self._run_learning_phase(
                pair, self.hebbian_learn_duration_min, self.hebbian_learn_rate_hz,
                pre_trigger_key, post_trigger_key,
                self.hebbian_probe_interval_min, self.hebbian_probe_count,
            )
            learn_end = datetime.now(timezone.utc)

            logger.info("  Validation phase")
            valid_start = datetime.now(timezone.utc)
            valid_latencies = self._run_probe_phase(
                pair, self.hebbian_valid_duration_min, self.hebbian_test_rate_hz,
                pre_trigger_key, "hebbian_valid",
            )
            valid_end = datetime.now(timezone.utc)

            emd = _compute_wasserstein_1d(test_latencies, valid_latencies)

            test_model = _select_best_model(test_latencies) if test_latencies else {}
            valid_model = _select_best_model(valid_latencies) if valid_latencies else {}

            hebbian_results.append({
                "pair_index": pair_idx,
                "stim_electrode": pair.stim_electrode,
                "resp_electrode": pair.resp_electrode,
                "synaptic_delay_ms": pair.median_latency_ms,
                "test_latencies_count": len(test_latencies),
                "test_median_ms": float(np.median(test_latencies)) if test_latencies else float("nan"),
                "valid_latencies_count": len(valid_latencies),
                "valid_median_ms": float(np.median(valid_latencies)) if valid_latencies else float("nan"),
                "earth_movers_distance": emd,
                "test_model": test_model,
                "valid_model": valid_model,
                "learning_probe_latencies": learn_probe_latencies,
                "test_start": test_start.isoformat(),
                "test_end": test_end.isoformat(),
                "learn_start": learn_start.isoformat(),
                "learn_end": learn_end.isoformat(),
                "valid_start": valid_start.isoformat(),
                "valid_end": valid_end.isoformat(),
            })

            logger.info(
                "Pair %d Hebbian result: test_n=%d, valid_n=%d, EMD=%.4f",
                pair_idx, len(test_latencies), len(valid_latencies), emd,
            )

        self._hebbian_results = {"pairs": hebbian_results}

    def _run_probe_phase(
        self,
        pair: PairConfig,
        duration_min: float,
        rate_hz: float,
        trigger_key: int,
        phase_label: str,
    ) -> List[float]:
        inter_stim_s = 1.0 / rate_hz if rate_hz > 0 else 2.0
        total_stims = int(duration_min * 60.0 * rate_hz)
        total_stims = max(total_stims, 1)

        phase_start = datetime.now(timezone.utc)
        for i in range(total_stims):
            self._stimulate_once(
                pair.stim_electrode, pair.amplitude, pair.duration,
                pair.polarity, trigger_key, phase_label=phase_label,
            )
            self._wait(inter_stim_s)
        phase_end = datetime.now(timezone.utc) + timedelta(seconds=2)

        spike_df = self.database.get_spike_event(
            phase_start - timedelta(seconds=1), phase_end, self.experiment.exp_name,
        )
        trigger_df = self.database.get_all_triggers(
            phase_start - timedelta(seconds=1), phase_end,
        )
        latencies = self._compute_latencies(
            spike_df, trigger_df, pair.resp_electrode, self.active_window_ms,
        )
        return latencies

    def _run_learning_phase(
        self,
        pair: PairConfig,
        duration_min: float,
        rate_hz: float,
        pre_trigger_key: int,
        post_trigger_key: int,
        probe_interval_min: float,
        probe_count: int,
    ) -> List[Dict[str, Any]]:
        inter_stim_s = 1.0 / rate_hz if rate_hz > 0 else 1.0
        total_stims = int(duration_min * 60.0 * rate_hz)
        total_stims = max(total_stims, 1)

        delay_ms = pair.median_latency_ms
        delay_s = delay_ms / 1000.0

        probe_every_n = int(probe_interval_min * 60.0 * rate_hz) if probe_interval_min > 0 else total_stims + 1
        probe_every_n = max(probe_every_n, 1)

        probe_results = []
        probe_phase_start = None
        probe_latencies_accum = []

        for i in range(total_stims):
            if i > 0 and i % probe_every_n == 0:
                if probe_phase_start is not None:
                    probe_phase_end = datetime.now(timezone.utc) + timedelta(seconds=1)
                    spike_df = self.database.get_spike_event(
                        probe_phase_start - timedelta(seconds=1),
                        probe_phase_end,
                        self.experiment.exp_name,
                    )
                    trigger_df = self.database.get_all_triggers(
                        probe_phase_start - timedelta(seconds=1),
                        probe_phase_end,
                    )
                    lats = self._compute_latencies(
                        spike_df, trigger_df, pair.resp_electrode, self.active_window_ms,
                    )
                    probe_results.append({
                        "stim_index": i,
                        "latencies_count": len(lats),
                        "median_latency_ms": float(np.median(lats)) if lats else float("nan"),
                    })

                logger.info("  Learning probe at stim %d/%d", i, total_stims)
                probe_phase_start = datetime.now(timezone.utc)
                for p in range(probe_count):
                    self._stimulate_once(
                        pair.stim_electrode, pair.amplitude, pair.duration,
                        pair.polarity, pre_trigger_key, phase_label="hebbian_learn_probe",
                    )
                    self._wait(inter_stim_s)
                probe_phase_end_probe = datetime.now(timezone.utc) + timedelta(seconds=1)
                spike_df_p = self.database.get_spike_event(
                    probe_phase_start - timedelta(seconds=1),
                    probe_phase_end_probe,
                    self.experiment.exp_name,
                )
                trigger_df_p = self.database.get_all_triggers(
                    probe_phase_start - timedelta(seconds=1),
                    probe_phase_end_probe,
                )
                lats_p = self._compute_latencies(
                    spike_df_p, trigger_df_p, pair.resp_electrode, self.active_window_ms,
                )
                probe_results.append({
                    "stim_index": i,
                    "type": "probe_block",
                    "latencies_count": len(lats_p),
                    "median_latency_ms": float(np.median(lats_p)) if lats_p else float("nan"),
                })
                probe_phase_start = datetime.now(timezone.utc)

            self._stimulate_once(
                pair.stim_electrode, pair.amplitude, pair.duration,
                pair.polarity, pre_trigger_key, phase_label="hebbian_learn_pre",
            )
            self._wait(delay_s)
            self._stimulate_once(
                pair.resp_electrode, pair.amplitude, pair.duration,
                pair.polarity, post_trigger_key, phase_label="hebbian_learn_post",
            )
            remaining_wait = inter_stim_s - delay_s - 0.05
            if remaining_wait > 0:
                self._wait(remaining_wait)

        return probe_results

    def _save_all(self, recording_start: datetime, recording_stop: datetime) -> None:
        fs_name = getattr(self.experiment, "exp_name", "unknown")
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
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "scan_responsive_count": self._scan_results.get("responsive_count", 0),
            "pairs_used": len(self._responsive_pairs),
        }
        saver.save_summary(summary)

        waveform_records = self._fetch_spike_waveforms(
            fs_name, spike_df, trigger_df, recording_start, recording_stop,
        )
        saver.save_spike_waveforms(waveform_records)

        analysis = {
            "scan_results": self._scan_results,
            "active_results": self._active_results,
            "hebbian_results": self._hebbian_results,
            "responsive_pairs": [asdict(p) for p in self._responsive_pairs],
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
            if col == "channel":
                electrode_col = col
                break
            if "electrode" in col.lower() or "idx" in col.lower() or col == "index":
                electrode_col = col
                break

        if electrode_col is None:
            logger.warning("Cannot determine electrode column for waveform fetch")
            return waveform_records

        unique_electrodes = spike_df[electrode_col].unique()
        max_electrodes = 10
        for electrode_idx in unique_electrodes[:max_electrodes]:
            try:
                raw_df = self.database.get_raw_spike(
                    recording_start, recording_stop, int(electrode_idx),
                )
                if not raw_df.empty:
                    waveform_records.append({
                        "electrode_idx": int(electrode_idx),
                        "num_waveforms": len(raw_df),
                        "waveform_samples": raw_df.values.tolist()[:100],
                    })
            except Exception as exc:
                logger.warning("Failed to fetch waveforms for electrode %s: %s", electrode_idx, exc)

        return waveform_records

    def _compile_results(self, recording_start: datetime, recording_stop: datetime) -> Dict[str, Any]:
        logger.info("Compiling results")
        summary = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "total_stimulations": len(self._stimulation_log),
            "scan_summary": {
                "responsive_count": self._scan_results.get("responsive_count", 0),
            },
            "active_summary": self._active_results,
            "hebbian_summary": self._hebbian_results,
            "pairs_used": [
                {
                    "stim": p.stim_electrode,
                    "resp": p.resp_electrode,
                    "amp": p.amplitude,
                    "dur": p.duration,
                    "pol": p.polarity,
                    "delay_ms": p.median_latency_ms,
                    "rate": p.response_rate,
                }
                for p in self._responsive_pairs
            ],
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
