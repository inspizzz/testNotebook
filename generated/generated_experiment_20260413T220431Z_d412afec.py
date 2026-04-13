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
class ElectrodePairConfig:
    electrode_from: int
    electrode_to: int
    amplitude: float
    duration: float
    polarity: str
    median_latency_ms: float
    hits_k: int
    repeats_n: int
    response_rate: float = 0.0
    deep_scan_median_latency_ms: float = 0.0


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

    def save_analysis(self, analysis: dict, label: str) -> Path:
        path = Path(f"{self._prefix}_{label}.json")
        path.write_text(json.dumps(analysis, indent=2, default=str))
        logger.info("Saved analysis %s -> %s", label, path)
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
        diff = abs(u_cdf[i] - v_cdf[i])
        width = all_vals[i + 1] - all_vals[i]
        emd += diff * width
    return emd


def _gaussian_pdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 0.0
    return (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _fit_single_gaussian(data: List[float]) -> Dict[str, Any]:
    if len(data) < 2:
        return {"mu": 0.0, "sigma": 1.0, "log_likelihood": float("-inf")}
    mu = sum(data) / len(data)
    sigma = math.sqrt(sum((x - mu) ** 2 for x in data) / len(data))
    if sigma < 1e-6:
        sigma = 1e-6
    ll = sum(math.log(max(_gaussian_pdf(x, mu, sigma), 1e-300)) for x in data)
    return {"mu": mu, "sigma": sigma, "log_likelihood": ll, "n_params": 2}


def _fit_gmm_em(data: List[float], k: int, max_iter: int = 50) -> Dict[str, Any]:
    n = len(data)
    if n < k * 2:
        return {"log_likelihood": float("-inf"), "n_params": 3 * k - 1}
    sorted_d = sorted(data)
    mus = [sorted_d[int(i * n / k) + n // (2 * k)] for i in range(k)]
    sigmas = [max(1.0, (max(data) - min(data)) / (2 * k))] * k
    weights = [1.0 / k] * k
    for _iteration in range(max_iter):
        resp = []
        for x in data:
            row = []
            total = 0.0
            for j in range(k):
                val = weights[j] * _gaussian_pdf(x, mus[j], sigmas[j])
                row.append(val)
                total += val
            if total < 1e-300:
                row = [1.0 / k] * k
            else:
                row = [r / total for r in row]
            resp.append(row)
        for j in range(k):
            nk = sum(resp[i][j] for i in range(n))
            if nk < 1e-10:
                continue
            weights[j] = nk / n
            mus[j] = sum(resp[i][j] * data[i] for i in range(n)) / nk
            var_j = sum(resp[i][j] * (data[i] - mus[j]) ** 2 for i in range(n)) / nk
            sigmas[j] = math.sqrt(max(var_j, 1e-6))
    ll = 0.0
    for x in data:
        p = sum(weights[j] * _gaussian_pdf(x, mus[j], sigmas[j]) for j in range(k))
        ll += math.log(max(p, 1e-300))
    return {
        "weights": weights,
        "mus": mus,
        "sigmas": sigmas,
        "log_likelihood": ll,
        "n_params": 3 * k - 1,
    }


def _fit_gamma_mom(data: List[float]) -> Dict[str, Any]:
    positive = [x for x in data if x > 0]
    if len(positive) < 3:
        return {"log_likelihood": float("-inf"), "n_params": 2}
    mean_val = sum(positive) / len(positive)
    var_val = sum((x - mean_val) ** 2 for x in positive) / len(positive)
    if var_val < 1e-10:
        var_val = 1e-10
    beta = var_val / mean_val
    alpha = mean_val / beta
    if alpha <= 0 or beta <= 0:
        return {"log_likelihood": float("-inf"), "n_params": 2}
    ll = 0.0
    for x in positive:
        log_pdf = (alpha - 1) * math.log(x) - x / beta - alpha * math.log(beta) - math.lgamma(alpha)
        ll += log_pdf
    return {"alpha": alpha, "beta": beta, "log_likelihood": ll, "n_params": 2}


def _select_best_model(data: List[float]) -> Dict[str, Any]:
    if len(data) < 3:
        return {"best_model": "insufficient_data", "delay_ms": 0.0}
    n = len(data)
    models = {}
    g1 = _fit_single_gaussian(data)
    g1["bic"] = g1["n_params"] * math.log(max(n, 1)) - 2 * g1["log_likelihood"]
    models["gaussian_1"] = g1
    g2 = _fit_gmm_em(data, 2)
    g2["bic"] = g2["n_params"] * math.log(max(n, 1)) - 2 * g2["log_likelihood"]
    models["gaussian_2"] = g2
    g3 = _fit_gmm_em(data, 3)
    g3["bic"] = g3["n_params"] * math.log(max(n, 1)) - 2 * g3["log_likelihood"]
    models["gaussian_3"] = g3
    gam = _fit_gamma_mom(data)
    gam["bic"] = gam["n_params"] * math.log(max(n, 1)) - 2 * gam["log_likelihood"]
    models["gamma"] = gam
    best_name = min(models, key=lambda k: models[k]["bic"])
    best = models[best_name]
    if best_name == "gaussian_1":
        delay = best["mu"]
    elif best_name in ("gaussian_2", "gaussian_3"):
        idx_max = 0
        max_w = 0
        for i, w in enumerate(best.get("weights", [0])):
            if w > max_w:
                max_w = w
                idx_max = i
        delay = best.get("mus", [0])[idx_max]
    elif best_name == "gamma":
        delay = best.get("alpha", 1) * best.get("beta", 1)
    else:
        delay = sum(data) / len(data)
    return {
        "best_model": best_name,
        "delay_ms": delay,
        "all_models": {k: {kk: vv for kk, vv in v.items()} for k, v in models.items()},
    }


class Experiment:
    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        scan_amplitudes: Optional[List[float]] = None,
        scan_durations: Optional[List[float]] = None,
        scan_repeats: int = 5,
        scan_min_hits: int = 3,
        scan_inter_stim_s: float = 1.0,
        scan_inter_channel_s: float = 5.0,
        active_total_repeats: int = 100,
        active_group_size: int = 10,
        active_inter_stim_s: float = 1.0,
        active_inter_group_s: float = 5.0,
        hebbian_test_duration_min: float = 20.0,
        hebbian_learn_duration_min: float = 50.0,
        hebbian_validation_duration_min: float = 20.0,
        hebbian_test_rate_hz: float = 0.25,
        max_pairs_for_hebbian: int = 3,
        response_window_ms: float = 50.0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.scan_amplitudes = scan_amplitudes or [1.0, 2.0, 3.0]
        self.scan_durations = scan_durations or [100.0, 200.0, 300.0, 400.0]
        self.scan_repeats = scan_repeats
        self.scan_min_hits = scan_min_hits
        self.scan_inter_stim_s = scan_inter_stim_s
        self.scan_inter_channel_s = scan_inter_channel_s

        self.active_total_repeats = active_total_repeats
        self.active_group_size = active_group_size
        self.active_inter_stim_s = active_inter_stim_s
        self.active_inter_group_s = active_inter_group_s

        self.hebbian_test_duration_min = hebbian_test_duration_min
        self.hebbian_learn_duration_min = hebbian_learn_duration_min
        self.hebbian_validation_duration_min = hebbian_validation_duration_min
        self.hebbian_test_rate_hz = hebbian_test_rate_hz
        self.max_pairs_for_hebbian = max_pairs_for_hebbian
        self.response_window_ms = response_window_ms

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._scan_results: List[Dict[str, Any]] = []
        self._responsive_pairs: List[ElectrodePairConfig] = []
        self._active_results: List[Dict[str, Any]] = []
        self._hebbian_results: List[Dict[str, Any]] = []
        self._model_fits: List[Dict[str, Any]] = []

        self._prior_pairs = self._load_prior_scan_data()

    def _load_prior_scan_data(self) -> List[ElectrodePairConfig]:
        best_per_pair: Dict[Tuple[int, int], ElectrodePairConfig] = {}
        deep_scan_lookup: Dict[Tuple[int, int], Dict] = {}
        deep_scan_data = [
            {"stim_electrode": 9, "resp_electrode": 10, "amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst", "response_rate": 0.95, "median_latency_ms": 25.0},
            {"stim_electrode": 10, "resp_electrode": 11, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "response_rate": 0.95, "median_latency_ms": 14.683},
            {"stim_electrode": 10, "resp_electrode": 12, "amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst", "response_rate": 1.0, "median_latency_ms": 10.6},
            {"stim_electrode": 11, "resp_electrode": 10, "amplitude": 3.0, "duration": 200.0, "polarity": "NegativeFirst", "response_rate": 0.88, "median_latency_ms": 24.117},
            {"stim_electrode": 11, "resp_electrode": 12, "amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst", "response_rate": 1.0, "median_latency_ms": 20.167},
            {"stim_electrode": 13, "resp_electrode": 12, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "response_rate": 0.99, "median_latency_ms": 15.666},
            {"stim_electrode": 27, "resp_electrode": 26, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "response_rate": 1.0, "median_latency_ms": 16.867},
            {"stim_electrode": 2, "resp_electrode": 1, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "response_rate": 1.0, "median_latency_ms": 10.3},
            {"stim_electrode": 19, "resp_electrode": 17, "amplitude": 2.0, "duration": 200.0, "polarity": "NegativeFirst", "response_rate": 0.95, "median_latency_ms": 16.2},
            {"stim_electrode": 13, "resp_electrode": 11, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "response_rate": 0.81, "median_latency_ms": 23.416},
        ]
        for ds in deep_scan_data:
            key = (ds["stim_electrode"], ds["resp_electrode"])
            if key not in deep_scan_lookup or ds["response_rate"] > deep_scan_lookup[key]["response_rate"]:
                deep_scan_lookup[key] = ds

        reliable_connections = [
            (1, 7, 5, 5, 11.933, 2.0, 400.0, "NegativeFirst"),
            (2, 1, 5, 5, 10.267, 3.0, 400.0, "PositiveFirst"),
            (9, 10, 5, 5, 25.166, 2.0, 300.0, "NegativeFirst"),
            (10, 11, 5, 5, 14.634, 3.0, 400.0, "NegativeFirst"),
            (10, 12, 5, 5, 10.4, 3.0, 300.0, "NegativeFirst"),
            (11, 12, 5, 5, 20.134, 3.0, 400.0, "NegativeFirst"),
            (12, 10, 5, 5, 17.267, 3.0, 300.0, "NegativeFirst"),
            (13, 12, 5, 5, 15.7, 3.0, 400.0, "NegativeFirst"),
            (19, 17, 5, 5, 12.934, 2.0, 400.0, "NegativeFirst"),
            (27, 26, 5, 5, 16.867, 3.0, 400.0, "NegativeFirst"),
            (9, 11, 5, 5, 15.0, 3.0, 400.0, "NegativeFirst"),
            (9, 12, 5, 5, 19.634, 3.0, 400.0, "NegativeFirst"),
            (11, 10, 5, 5, 24.534, 3.0, 200.0, "NegativeFirst"),
            (23, 22, 5, 5, 10.033, 3.0, 300.0, "NegativeFirst"),
        ]
        for ef, et, hk, rn, lat, amp, dur, pol in reliable_connections:
            key = (ef, et)
            ds_info = deep_scan_lookup.get(key, {})
            rr = ds_info.get("response_rate", hk / rn)
            ds_lat = ds_info.get("median_latency_ms", lat)
            pair = ElectrodePairConfig(
                electrode_from=ef,
                electrode_to=et,
                amplitude=amp,
                duration=dur,
                polarity=pol,
                median_latency_ms=lat,
                hits_k=hk,
                repeats_n=rn,
                response_rate=rr,
                deep_scan_median_latency_ms=ds_lat,
            )
            if key not in best_per_pair or rr > best_per_pair[key].response_rate:
                best_per_pair[key] = pair

        pairs = sorted(best_per_pair.values(), key=lambda p: -p.response_rate)
        return pairs

    def _wait(self, seconds: float) -> None:
        if not self.testing:
            time.sleep(seconds)

    def _get_polarity_enum(self, pol_str: str) -> StimPolarity:
        if pol_str == "PositiveFirst":
            return StimPolarity.PositiveFirst
        return StimPolarity.NegativeFirst

    def _make_charge_balanced_stim(
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

    def _stimulate_once(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity = StimPolarity.NegativeFirst,
        trigger_key: int = 0,
        phase: str = "",
    ) -> None:
        stim = self._make_charge_balanced_stim(
            electrode_idx, amplitude_ua, duration_us, polarity, trigger_key
        )
        self.intan.send_stimparam([stim])
        self._fire_trigger(trigger_key)
        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=min(abs(amplitude_ua), 4.0),
            duration_us=min(abs(duration_us), 400.0),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            trigger_key=trigger_key,
            phase=phase,
        ))

    def _stimulate_pair(
        self,
        electrode_from: int,
        electrode_to: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity,
        delay_ms: float,
        trigger_key_pre: int = 0,
        trigger_key_post: int = 1,
        phase: str = "",
    ) -> None:
        stim_pre = self._make_charge_balanced_stim(
            electrode_from, amplitude_ua, duration_us, polarity, trigger_key_pre
        )
        self.intan.send_stimparam([stim_pre])
        self._fire_trigger(trigger_key_pre)
        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_from,
            amplitude_ua=min(abs(amplitude_ua), 4.0),
            duration_us=min(abs(duration_us), 400.0),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            trigger_key=trigger_key_pre,
            phase=phase,
            extra={"role": "pre", "pair_to": electrode_to},
        ))
        delay_s = max(delay_ms / 1000.0, 0.001)
        self._wait(delay_s)
        stim_post = self._make_charge_balanced_stim(
            electrode_to, amplitude_ua, duration_us, polarity, trigger_key_post
        )
        self.intan.send_stimparam([stim_post])
        self._fire_trigger(trigger_key_post)
        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_to,
            amplitude_ua=min(abs(amplitude_ua), 4.0),
            duration_us=min(abs(duration_us), 400.0),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            trigger_key=trigger_key_post,
            phase=phase,
            extra={"role": "post", "pair_from": electrode_from},
        ))

    def _get_spikes_in_window(
        self,
        start: datetime,
        stop: datetime,
        electrode: int,
    ) -> pd.DataFrame:
        return self.database.get_spike_event_electrode(start, stop, electrode)

    def _phase1_excitability_scan(self) -> None:
        logger.info("=== PHASE 1: Basic Excitability Scan ===")
        electrodes = self.experiment.electrodes
        polarities = [StimPolarity.NegativeFirst, StimPolarity.PositiveFirst]
        scan_start = datetime.now(timezone.utc)
        for elec_idx in electrodes:
            logger.info("Scanning electrode %d", elec_idx)
            for amp in self.scan_amplitudes:
                for dur in self.scan_durations:
                    for pol in polarities:
                        hit_count = 0
                        for rep in range(self.scan_repeats):
                            stim_time = datetime.now(timezone.utc)
                            self._stimulate_once(
                                elec_idx, amp, dur, pol,
                                trigger_key=0, phase="scan"
                            )
                            self._wait(self.scan_inter_stim_s)
                            query_stop = datetime.now(timezone.utc)
                            query_start = stim_time - timedelta(milliseconds=100)
                            spikes = self._get_spikes_in_window(
                                query_start, query_stop, elec_idx
                            )
                            if len(spikes) > 0:
                                hit_count += 1
                        self._scan_results.append({
                            "electrode": elec_idx,
                            "amplitude": amp,
                            "duration": dur,
                            "polarity": pol.name,
                            "hits": hit_count,
                            "repeats": self.scan_repeats,
                            "responsive": self.scan_min_hits <= hit_count <= self.scan_repeats,
                        })
            self._wait(self.scan_inter_channel_s)
        scan_stop = datetime.now(timezone.utc)
        logger.info("Scan complete. %d parameter combos tested.",
                     len(self._scan_results))
        responsive = [r for r in self._scan_results if r["responsive"]]
        logger.info("Responsive combos: %d", len(responsive))
        self._build_responsive_pairs_from_scan(scan_start, scan_stop)

    def _build_responsive_pairs_from_scan(
        self, scan_start: datetime, scan_stop: datetime
    ) -> None:
        if self._prior_pairs:
            logger.info("Using %d prior-scan pairs as primary source", len(self._prior_pairs))
            self._responsive_pairs = list(self._prior_pairs)
        else:
            responsive_electrodes = set()
            for r in self._scan_results:
                if r["responsive"]:
                    responsive_electrodes.add(r["electrode"])
            electrodes_list = sorted(responsive_electrodes)
            for i, ef in enumerate(electrodes_list):
                for et in electrodes_list[i + 1:]:
                    best_r = None
                    for r in self._scan_results:
                        if r["electrode"] == ef and r["responsive"]:
                            if best_r is None or r["hits"] > best_r["hits"]:
                                best_r = r
                    if best_r:
                        self._responsive_pairs.append(ElectrodePairConfig(
                            electrode_from=ef,
                            electrode_to=et,
                            amplitude=best_r["amplitude"],
                            duration=best_r["duration"],
                            polarity=best_r["polarity"],
                            median_latency_ms=10.0,
                            hits_k=best_r["hits"],
                            repeats_n=best_r["repeats"],
                        ))
        logger.info("Total responsive pairs for active experiment: %d",
                     len(self._responsive_pairs))

    def _phase2_active_electrode_experiment(self) -> None:
        logger.info("=== PHASE 2: Active Electrode Experiment ===")
        pairs_to_test = self._responsive_pairs[:6]
        for pair_idx, pair in enumerate(pairs_to_test):
            logger.info("Active experiment pair %d: %d -> %d (amp=%.1f, dur=%.0f)",
                        pair_idx, pair.electrode_from, pair.electrode_to,
                        pair.amplitude, pair.duration)
            polarity = self._get_polarity_enum(pair.polarity)
            phase_start = datetime.now(timezone.utc)
            stim_times = []
            num_groups = self.active_total_repeats // self.active_group_size
            remainder = self.active_total_repeats % self.active_group_size
            for group_idx in range(num_groups):
                for stim_idx in range(self.active_group_size):
                    t_before = datetime.now(timezone.utc)
                    self._stimulate_once(
                        pair.electrode_from,
                        pair.amplitude,
                        pair.duration,
                        polarity,
                        trigger_key=0,
                        phase="active",
                    )
                    stim_times.append(datetime.now(timezone.utc))
                    self._wait(self.active_inter_stim_s)
                self._wait(self.active_inter_group_s)
            for stim_idx in range(remainder):
                self._stimulate_once(
                    pair.electrode_from,
                    pair.amplitude,
                    pair.duration,
                    polarity,
                    trigger_key=0,
                    phase="active",
                )
                stim_times.append(datetime.now(timezone.utc))
                self._wait(self.active_inter_stim_s)
            phase_stop = datetime.now(timezone.utc)
            spikes_resp = self.database.get_spike_event_electrode(
                phase_start, phase_stop, pair.electrode_to
            )
            latencies = self._compute_latencies(
                stim_times, spikes_resp, self.response_window_ms
            )
            model_result = _select_best_model(latencies)
            self._model_fits.append({
                "pair_from": pair.electrode_from,
                "pair_to": pair.electrode_to,
                "model": model_result,
                "latencies": latencies,
                "n_stims": len(stim_times),
                "n_spikes_in_window": len(latencies),
            })
            synaptic_delay = model_result.get("delay_ms", pair.median_latency_ms)
            if pair.deep_scan_median_latency_ms > 0:
                synaptic_delay = pair.deep_scan_median_latency_ms
            self._active_results.append({
                "pair_idx": pair_idx,
                "electrode_from": pair.electrode_from,
                "electrode_to": pair.electrode_to,
                "n_stims": len(stim_times),
                "n_latencies": len(latencies),
                "synaptic_delay_ms": synaptic_delay,
                "model_fit": model_result["best_model"],
                "phase_start": phase_start.isoformat(),
                "phase_stop": phase_stop.isoformat(),
            })
            logger.info("Pair %d->%d: %d latencies, delay=%.2f ms, model=%s",
                        pair.electrode_from, pair.electrode_to,
                        len(latencies), synaptic_delay,
                        model_result["best_model"])

    def _compute_latencies(
        self,
        stim_times: List[datetime],
        spike_df: pd.DataFrame,
        window_ms: float,
    ) -> List[float]:
        latencies = []
        if spike_df.empty or not stim_times:
            return latencies
        time_col = "Time" if "Time" in spike_df.columns else "_time"
        if time_col not in spike_df.columns:
            return latencies
        spike_times_raw = spike_df[time_col].tolist()
        spike_times = []
        for st in spike_times_raw:
            if isinstance(st, datetime):
                if st.tzinfo is None:
                    st = st.replace(tzinfo=timezone.utc)
                spike_times.append(st)
            elif isinstance(st, pd.Timestamp):
                if st.tzinfo is None:
                    st = st.tz_localize("UTC")
                spike_times.append(st.to_pydatetime())
            else:
                try:
                    spike_times.append(pd.Timestamp(st).to_pydatetime())
                except Exception:
                    pass
        for stim_t in stim_times:
            if stim_t.tzinfo is None:
                stim_t = stim_t.replace(tzinfo=timezone.utc)
            for sp_t in spike_times:
                delta_ms = (sp_t - stim_t).total_seconds() * 1000.0
                if 2.0 < delta_ms <= window_ms:
                    latencies.append(delta_ms)
        return latencies

    def _phase3_hebbian_learning(self) -> None:
        logger.info("=== PHASE 3: Two-Electrode Hebbian Learning ===")
        top_pairs = []
        for ar in self._active_results:
            if ar["n_latencies"] > 0:
                top_pairs.append(ar)
        top_pairs.sort(key=lambda x: -x["n_latencies"])
        top_pairs = top_pairs[:self.max_pairs_for_hebbian]
        if not top_pairs:
            logger.warning("No responsive pairs for Hebbian learning. Skipping.")
            return
        for pair_info in top_pairs:
            ef = pair_info["electrode_from"]
            et = pair_info["electrode_to"]
            delay_ms = pair_info["synaptic_delay_ms"]
            pair_config = None
            for p in self._responsive_pairs:
                if p.electrode_from == ef and p.electrode_to == et:
                    pair_config = p
                    break
            if pair_config is None:
                continue
            polarity = self._get_polarity_enum(pair_config.polarity)
            amplitude = pair_config.amplitude
            duration = pair_config.duration
            logger.info("Hebbian pair %d->%d, delay=%.2f ms", ef, et, delay_ms)
            test_interval_s = 1.0 / self.hebbian_test_rate_hz
            testing_n_pulses = int(self.hebbian_test_duration_min * 60 * self.hebbian_test_rate_hz)
            testing_n_pulses = max(testing_n_pulses, 10)
            learning_n_pulses = int(self.hebbian_learn_duration_min * 60 * self.hebbian_test_rate_hz)
            learning_n_pulses = max(learning_n_pulses, 20)
            validation_n_pulses = int(self.hebbian_validation_duration_min * 60 * self.hebbian_test_rate_hz)
            validation_n_pulses = max(validation_n_pulses, 10)
            testing_start = datetime.now(timezone.utc)
            testing_stim_times = []
            logger.info("  Testing phase: %d pulses", testing_n_pulses)
            for i in range(testing_n_pulses):
                t_now = datetime.now(timezone.utc)
                self._stimulate_once(
                    ef, amplitude, duration, polarity,
                    trigger_key=0, phase="hebbian_test"
                )
                testing_stim_times.append(datetime.now(timezone.utc))
                self._wait(test_interval_s)
            testing_stop = datetime.now(timezone.utc)
            learning_start = datetime.now(timezone.utc)
            learning_stim_times = []
            logger.info("  Learning phase: %d paired pulses", learning_n_pulses)
            probe_interval = max(learning_n_pulses // 10, 1)
            for i in range(learning_n_pulses):
                if i % probe_interval == 0 and i > 0:
                    self._stimulate_once(
                        ef, amplitude, duration, polarity,
                        trigger_key=0, phase="hebbian_learn_probe"
                    )
                    self._wait(test_interval_s)
                self._stimulate_pair(
                    ef, et, amplitude, duration, polarity,
                    delay_ms=delay_ms,
                    trigger_key_pre=0,
                    trigger_key_post=1,
                    phase="hebbian_learn",
                )
                learning_stim_times.append(datetime.now(timezone.utc))
                self._wait(max(test_interval_s - delay_ms / 1000.0, 0.1))
            learning_stop = datetime.now(timezone.utc)
            validation_start = datetime.now(timezone.utc)
            validation_stim_times = []
            logger.info("  Validation phase: %d pulses", validation_n_pulses)
            for i in range(validation_n_pulses):
                self._stimulate_once(
                    ef, amplitude, duration, polarity,
                    trigger_key=0, phase="hebbian_validation"
                )
                validation_stim_times.append(datetime.now(timezone.utc))
                self._wait(test_interval_s)
            validation_stop = datetime.now(timezone.utc)
            test_spikes = self.database.get_spike_event_electrode(
                testing_start, testing_stop, et
            )
            val_spikes = self.database.get_spike_event_electrode(
                validation_start, validation_stop, et
            )
            test_latencies = self._compute_latencies(
                testing_stim_times, test_spikes, self.response_window_ms
            )
            val_latencies = self._compute_latencies(
                validation_stim_times, val_spikes, self.response_window_ms
            )
            emd = _compute_wasserstein_1d(test_latencies, val_latencies)
            test_model = _select_best_model(test_latencies)
            val_model = _select_best_model(val_latencies)
            test_mean = sum(test_latencies) / len(test_latencies) if test_latencies else 0
            val_mean = sum(val_latencies) / len(val_latencies) if val_latencies else 0
            potentiation_pct = 0.0
            if test_mean > 0 and val_mean > 0:
                test_count_per_stim = len(test_latencies) / max(len(testing_stim_times), 1)
                val_count_per_stim = len(val_latencies) / max(len(validation_stim_times), 1)
                if test_count_per_stim > 0:
                    potentiation_pct = ((val_count_per_stim - test_count_per_stim) / test_count_per_stim) * 100.0
            self._hebbian_results.append({
                "electrode_from": ef,
                "electrode_to": et,
                "delay_ms": delay_ms,
                "testing_n_stims": len(testing_stim_times),
                "testing_n_latencies": len(test_latencies),
                "testing_mean_latency": test_mean,
                "testing_model": test_model["best_model"],
                "validation_n_stims": len(validation_stim_times),
                "validation_n_latencies": len(val_latencies),
                "validation_mean_latency": val_mean,
                "validation_model": val_model["best_model"],
                "emd_wasserstein": emd,
                "potentiation_pct": potentiation_pct,
                "learning_n_paired": len(learning_stim_times),
                "testing_start": testing_start.isoformat(),
                "testing_stop": testing_stop.isoformat(),
                "learning_start": learning_start.isoformat(),
                "learning_stop": learning_stop.isoformat(),
                "validation_start": validation_start.isoformat(),
                "validation_stop": validation_stop.isoformat(),
                "test_latencies": test_latencies[:200],
                "val_latencies": val_latencies[:200],
            })
            logger.info(
                "  Pair %d->%d: test=%d spikes, val=%d spikes, EMD=%.4f, potentiation=%.1f%%",
                ef, et, len(test_latencies), len(val_latencies), emd, potentiation_pct
            )

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
            "scan_results_count": len(self._scan_results),
            "responsive_pairs_count": len(self._responsive_pairs),
            "active_results_count": len(self._active_results),
            "hebbian_results_count": len(self._hebbian_results),
        }
        saver.save_summary(summary)
        waveform_records = self._fetch_spike_waveforms(
            fs_name, spike_df, trigger_df, recording_start, recording_stop
        )
        saver.save_spike_waveforms(waveform_records)
        saver.save_analysis({
            "scan_results": self._scan_results,
            "active_results": self._active_results,
            "model_fits": [{k: v for k, v in mf.items() if k != "latencies"} for mf in self._model_fits],
            "hebbian_results": self._hebbian_results,
            "responsive_pairs": [asdict(p) for p in self._responsive_pairs[:20]],
        }, "analysis")

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
        max_electrodes = 10
        for electrode_idx in unique_electrodes[:max_electrodes]:
            try:
                raw_df = self.database.get_raw_spike(
                    recording_start, recording_stop, int(electrode_idx)
                )
                if not raw_df.empty:
                    waveform_records.append({
                        "electrode_idx": int(electrode_idx),
                        "num_waveforms": len(raw_df),
                        "waveform_samples": raw_df.values.tolist()[:50],
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
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "total_stimulations": len(self._stimulation_log),
            "phase1_scan_combos": len(self._scan_results),
            "phase1_responsive": sum(1 for r in self._scan_results if r.get("responsive")),
            "responsive_pairs": len(self._responsive_pairs),
            "phase2_active_results": self._active_results,
            "phase2_model_fits": [
                {k: v for k, v in mf.items() if k != "latencies"}
                for mf in self._model_fits
            ],
            "phase3_hebbian_results": [
                {k: v for k, v in hr.items() if k not in ("test_latencies", "val_latencies")}
                for hr in self._hebbian_results
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
            self._phase1_excitability_scan()
            self._phase2_active_electrode_experiment()
            self._phase3_hebbian_learning()
            recording_stop = datetime.now(timezone.utc)
            results = self._compile_results(recording_start, recording_stop)
            self._save_all(recording_start, recording_stop)
            return results
        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()
