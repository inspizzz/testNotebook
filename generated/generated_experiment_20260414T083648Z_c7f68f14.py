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
class ElectrodePairConfig:
    electrode_from: int
    electrode_to: int
    amplitude: float
    duration: float
    polarity: str
    median_latency_ms: float
    hits_k: int
    response_rate: float = 0.0


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
        scan_min_hits: int = 3,
        scan_amplitudes: Optional[List[float]] = None,
        scan_durations: Optional[List[float]] = None,
        active_total_repeats: int = 100,
        active_group_size: int = 10,
        active_inter_stim_s: float = 1.0,
        active_inter_group_s: float = 5.0,
        stdp_test_duration_min: float = 20.0,
        stdp_learn_duration_min: float = 50.0,
        stdp_valid_duration_min: float = 20.0,
        stdp_probe_rate_hz: float = 0.1,
        ccg_window_ms: float = 50.0,
        ccg_bin_ms: float = 1.0,
        max_pairs_per_stage: int = 5,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.scan_repeats = scan_repeats
        self.scan_inter_stim_s = scan_inter_stim_s
        self.scan_inter_channel_s = scan_inter_channel_s
        self.scan_min_hits = scan_min_hits
        self.scan_amplitudes = scan_amplitudes or [1.0, 2.0, 3.0]
        self.scan_durations = scan_durations or [100.0, 200.0, 300.0, 400.0]

        self.active_total_repeats = active_total_repeats
        self.active_group_size = active_group_size
        self.active_inter_stim_s = active_inter_stim_s
        self.active_inter_group_s = active_inter_group_s

        self.stdp_test_duration_min = stdp_test_duration_min
        self.stdp_learn_duration_min = stdp_learn_duration_min
        self.stdp_valid_duration_min = stdp_valid_duration_min
        self.stdp_probe_rate_hz = stdp_probe_rate_hz

        self.ccg_window_ms = ccg_window_ms
        self.ccg_bin_ms = ccg_bin_ms
        self.max_pairs_per_stage = max_pairs_per_stage

        self.experiment_handle = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []

        self._scan_results: List[Dict[str, Any]] = []
        self._responsive_pairs: List[ElectrodePairConfig] = []
        self._active_results: List[Dict[str, Any]] = []
        self._stdp_results: List[Dict[str, Any]] = []

        self._prior_scan = self._load_prior_scan()

    def _load_prior_scan(self) -> List[ElectrodePairConfig]:
        deep_scan = [
            {"stim": 0, "resp": 1, "amp": 2.0, "dur": 400.0, "pol": "PositiveFirst", "lat": 21.53, "rate": 0.97, "hits": 5},
            {"stim": 1, "resp": 2, "amp": 1.0, "dur": 300.0, "pol": "NegativeFirst", "lat": 23.45, "rate": 0.95, "hits": 5},
            {"stim": 5, "resp": 4, "amp": 3.0, "dur": 200.0, "pol": "NegativeFirst", "lat": 24.56, "rate": 0.97, "hits": 5},
            {"stim": 6, "resp": 5, "amp": 2.0, "dur": 200.0, "pol": "PositiveFirst", "lat": 19.735, "rate": 0.87, "hits": 5},
            {"stim": 8, "resp": 9, "amp": 1.0, "dur": 400.0, "pol": "PositiveFirst", "lat": 22.855, "rate": 0.87, "hits": 5},
            {"stim": 14, "resp": 12, "amp": 1.0, "dur": 300.0, "pol": "NegativeFirst", "lat": 22.81, "rate": 0.89, "hits": 5},
            {"stim": 14, "resp": 15, "amp": 1.0, "dur": 300.0, "pol": "NegativeFirst", "lat": 13.23, "rate": 0.70, "hits": 5},
            {"stim": 17, "resp": 16, "amp": 2.0, "dur": 400.0, "pol": "NegativeFirst", "lat": 11.045, "rate": 0.94, "hits": 5},
            {"stim": 18, "resp": 17, "amp": 1.0, "dur": 400.0, "pol": "NegativeFirst", "lat": 25.025, "rate": 0.85, "hits": 5},
            {"stim": 25, "resp": 24, "amp": 3.0, "dur": 400.0, "pol": "PositiveFirst", "lat": 15.18, "rate": 0.98, "hits": 5},
            {"stim": 26, "resp": 27, "amp": 3.0, "dur": 400.0, "pol": "NegativeFirst", "lat": 15.335, "rate": 0.86, "hits": 5},
            {"stim": 27, "resp": 28, "amp": 3.0, "dur": 400.0, "pol": "NegativeFirst", "lat": 24.8, "rate": 0.98, "hits": 5},
        ]
        pairs = []
        for d in deep_scan:
            pairs.append(ElectrodePairConfig(
                electrode_from=d["stim"],
                electrode_to=d["resp"],
                amplitude=d["amp"],
                duration=d["dur"],
                polarity=d["pol"],
                median_latency_ms=d["lat"],
                hits_k=d["hits"],
                response_rate=d["rate"],
            ))
        return pairs

    def _wait(self, seconds: float) -> None:
        if not self.testing:
            time.sleep(seconds)

    def _get_polarity_enum(self, pol_str: str) -> StimPolarity:
        if pol_str == "PositiveFirst":
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

    def _stimulate_once(
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
        electrode_from: int,
        electrode_to: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity,
        trigger_key_from: int = 0,
        trigger_key_to: int = 1,
        delay_ms: float = 0.0,
        phase: str = "",
    ) -> None:
        stim_from = self._make_stim_param(electrode_from, amplitude_ua, duration_us, polarity, trigger_key_from)
        self.intan.send_stimparam([stim_from])
        self._fire_trigger(trigger_key_from)

        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_from,
            amplitude_ua=min(abs(amplitude_ua), 4.0),
            duration_us=min(abs(duration_us), 400.0),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            trigger_key=trigger_key_from,
            polarity=polarity.name,
            phase=phase,
            extra={"role": "pre", "pair_to": electrode_to},
        ))

        if delay_ms > 0:
            self._wait(delay_ms / 1000.0)

        stim_to = self._make_stim_param(electrode_to, amplitude_ua, duration_us, polarity, trigger_key_to)
        self.intan.send_stimparam([stim_to])
        self._fire_trigger(trigger_key_to)

        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_to,
            amplitude_ua=min(abs(amplitude_ua), 4.0),
            duration_us=min(abs(duration_us), 400.0),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            trigger_key=trigger_key_to,
            polarity=polarity.name,
            phase=phase,
            extra={"role": "post", "pair_from": electrode_from},
        ))

    def _compute_ccg(
        self,
        spike_times_source: np.ndarray,
        spike_times_target: np.ndarray,
        window_ms: float,
        bin_ms: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_bins = int(2 * window_ms / bin_ms)
        counts = np.zeros(n_bins, dtype=np.float64)
        edges = np.linspace(-window_ms, window_ms, n_bins + 1)
        bin_centers = (edges[:-1] + edges[1:]) / 2.0

        if len(spike_times_source) == 0 or len(spike_times_target) == 0:
            return bin_centers, counts

        for t_src in spike_times_source:
            diffs = (spike_times_target - t_src) * 1000.0
            mask = (diffs >= -window_ms) & (diffs <= window_ms)
            valid_diffs = diffs[mask]
            for d in valid_diffs:
                idx = int((d + window_ms) / bin_ms)
                if 0 <= idx < n_bins:
                    counts[idx] += 1

        return bin_centers, counts

    def _compute_wasserstein_1d(self, dist_a: np.ndarray, dist_b: np.ndarray) -> float:
        if len(dist_a) == 0 and len(dist_b) == 0:
            return 0.0
        if len(dist_a) == 0 or len(dist_b) == 0:
            return float("inf")

        a_sorted = np.sort(dist_a)
        b_sorted = np.sort(dist_b)

        n = max(len(a_sorted), len(b_sorted))
        a_interp = np.interp(
            np.linspace(0, 1, n),
            np.linspace(0, 1, len(a_sorted)),
            a_sorted,
        )
        b_interp = np.interp(
            np.linspace(0, 1, n),
            np.linspace(0, 1, len(b_sorted)),
            b_sorted,
        )
        return float(np.mean(np.abs(a_interp - b_interp)))

    def _extract_spike_times(self, spike_df: pd.DataFrame, electrode: int) -> np.ndarray:
        if spike_df.empty:
            return np.array([])

        time_col = None
        for candidate in ["Time", "_time", "time", "timestamp"]:
            if candidate in spike_df.columns:
                time_col = candidate
                break
        if time_col is None:
            cols = spike_df.columns.tolist()
            logger.warning("No time column found in spike_df. Columns: %s", cols)
            return np.array([])

        ch_col = None
        for candidate in ["channel", "electrode", "index", "ch"]:
            if candidate in spike_df.columns:
                ch_col = candidate
                break
        if ch_col is None:
            logger.warning("No channel column found in spike_df.")
            return np.array([])

        mask = spike_df[ch_col].astype(int) == electrode
        filtered = spike_df.loc[mask, time_col]

        if filtered.empty:
            return np.array([])

        times = pd.to_datetime(filtered, utc=True)
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        return ((times - epoch).dt.total_seconds()).values

    def _extract_trigger_times(self, trigger_df: pd.DataFrame, trigger_key: int) -> np.ndarray:
        if trigger_df.empty:
            return np.array([])

        time_col = None
        for candidate in ["_time", "Time", "time", "timestamp"]:
            if candidate in trigger_df.columns:
                time_col = candidate
                break
        if time_col is None:
            return np.array([])

        trig_col = None
        for candidate in ["trigger", "Trigger", "trig"]:
            if candidate in trigger_df.columns:
                trig_col = candidate
                break
        if trig_col is None:
            return np.array([])

        mask = trigger_df[trig_col].astype(int) == trigger_key
        filtered = trigger_df.loc[mask, time_col]
        if filtered.empty:
            return np.array([])

        times = pd.to_datetime(filtered, utc=True)
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        return ((times - epoch).dt.total_seconds()).values

    def _fit_gaussian_mixture(self, data: np.ndarray, n_components: int) -> Dict[str, Any]:
        if len(data) < n_components * 2:
            return {"n_components": n_components, "bic": float("inf"), "means": [], "stds": [], "weights": []}

        sorted_data = np.sort(data)
        n = len(sorted_data)
        chunk_size = max(1, n // n_components)

        means = []
        stds = []
        weights = []
        for i in range(n_components):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n)
            chunk = sorted_data[start:end]
            means.append(float(np.mean(chunk)))
            stds.append(max(float(np.std(chunk)), 0.1))
            weights.append(len(chunk) / n)

        for iteration in range(20):
            responsibilities = np.zeros((n, n_components))
            for k in range(n_components):
                if stds[k] < 1e-10:
                    stds[k] = 0.1
                log_pdf = -0.5 * ((data - means[k]) / stds[k]) ** 2 - np.log(stds[k] * math.sqrt(2 * math.pi))
                responsibilities[:, k] = np.log(max(weights[k], 1e-10)) + log_pdf

            max_resp = np.max(responsibilities, axis=1, keepdims=True)
            log_sum = max_resp.flatten() + np.log(np.sum(np.exp(responsibilities - max_resp), axis=1))

            for k in range(n_components):
                responsibilities[:, k] = np.exp(responsibilities[:, k] - log_sum)

            for k in range(n_components):
                nk = np.sum(responsibilities[:, k])
                if nk < 1e-10:
                    continue
                weights[k] = float(nk / n)
                means[k] = float(np.sum(responsibilities[:, k] * data) / nk)
                stds[k] = max(float(np.sqrt(np.sum(responsibilities[:, k] * (data - means[k]) ** 2) / nk)), 0.1)

        log_likelihood = 0.0
        for i in range(n):
            ll_i = 0.0
            for k in range(n_components):
                if stds[k] < 1e-10:
                    continue
                ll_i += weights[k] * (1.0 / (stds[k] * math.sqrt(2 * math.pi))) * math.exp(
                    -0.5 * ((data[i] - means[k]) / stds[k]) ** 2
                )
            log_likelihood += math.log(max(ll_i, 1e-300))

        n_params = n_components * 3 - 1
        bic = -2 * log_likelihood + n_params * math.log(n)

        return {
            "n_components": n_components,
            "bic": float(bic),
            "means": means,
            "stds": stds,
            "weights": weights,
            "log_likelihood": float(log_likelihood),
        }

    def _fit_gamma(self, data: np.ndarray) -> Dict[str, Any]:
        if len(data) < 3:
            return {"distribution": "gamma", "bic": float("inf"), "shape": 0, "scale": 0}

        positive_data = data[data > 0]
        if len(positive_data) < 3:
            return {"distribution": "gamma", "bic": float("inf"), "shape": 0, "scale": 0}

        mean_val = float(np.mean(positive_data))
        var_val = float(np.var(positive_data))
        if var_val < 1e-10:
            var_val = 1e-10

        shape = (mean_val ** 2) / var_val
        scale = var_val / mean_val

        if shape <= 0 or scale <= 0:
            return {"distribution": "gamma", "bic": float("inf"), "shape": 0, "scale": 0}

        log_likelihood = 0.0
        for x in positive_data:
            if x <= 0:
                continue
            ll = (shape - 1) * math.log(x) - x / scale - shape * math.log(scale) - math.lgamma(shape)
            log_likelihood += ll

        n_params = 2
        bic = -2 * log_likelihood + n_params * math.log(len(positive_data))

        return {
            "distribution": "gamma",
            "bic": float(bic),
            "shape": float(shape),
            "scale": float(scale),
            "log_likelihood": float(log_likelihood),
        }

    def _select_best_model(self, data: np.ndarray) -> Dict[str, Any]:
        models = []
        for nc in [1, 2, 3]:
            result = self._fit_gaussian_mixture(data, nc)
            result["model_name"] = f"GMM_{nc}"
            models.append(result)

        gamma_result = self._fit_gamma(data)
        gamma_result["model_name"] = "Gamma"
        models.append(gamma_result)

        best = min(models, key=lambda m: m.get("bic", float("inf")))
        return {
            "best_model": best.get("model_name", "unknown"),
            "best_bic": best.get("bic", float("inf")),
            "all_models": models,
            "synaptic_delay_ms": best.get("means", [float(np.median(data))])[0] if best.get("means") else float(np.median(data)),
        }

    def _phase1_excitability_scan(self) -> None:
        logger.info("=== PHASE 1: Basic Excitability Scan ===")
        electrodes = self.experiment_handle.electrodes
        polarities = [StimPolarity.NegativeFirst, StimPolarity.PositiveFirst]

        scan_count = 0
        max_scan_stims = 200

        for elec_idx in electrodes:
            if scan_count >= max_scan_stims:
                break
            for amp in self.scan_amplitudes:
                if scan_count >= max_scan_stims:
                    break
                for dur in self.scan_durations:
                    if scan_count >= max_scan_stims:
                        break
                    for pol in polarities:
                        if scan_count >= max_scan_stims:
                            break
                        hit_count = 0
                        for rep in range(self.scan_repeats):
                            if scan_count >= max_scan_stims:
                                break
                            self._stimulate_once(
                                electrode_idx=elec_idx,
                                amplitude_ua=amp,
                                duration_us=dur,
                                polarity=pol,
                                trigger_key=0,
                                phase="scan",
                            )
                            scan_count += 1
                            self._wait(self.scan_inter_stim_s)

                        self._scan_results.append({
                            "electrode": elec_idx,
                            "amplitude": amp,
                            "duration": dur,
                            "polarity": pol.name,
                            "repeats": self.scan_repeats,
                        })

            self._wait(self.scan_inter_channel_s)

        logger.info("Scan complete: %d stimulations delivered", scan_count)

    def _select_responsive_pairs(self) -> None:
        logger.info("Selecting responsive pairs from prior scan data")
        sorted_pairs = sorted(self._prior_scan, key=lambda p: (-p.response_rate, -p.hits_k))
        self._responsive_pairs = sorted_pairs[:self.max_pairs_per_stage]
        logger.info("Selected %d responsive pairs for active electrode experiment", len(self._responsive_pairs))
        for p in self._responsive_pairs:
            logger.info(
                "  Pair %d->%d: amp=%.1f dur=%.0f pol=%s lat=%.2fms rate=%.2f",
                p.electrode_from, p.electrode_to, p.amplitude, p.duration,
                p.polarity, p.median_latency_ms, p.response_rate,
            )

    def _phase2_active_electrode(self) -> None:
        logger.info("=== PHASE 2: Active Electrode Experiment ===")

        for pair_idx, pair in enumerate(self._responsive_pairs):
            logger.info(
                "Active electrode pair %d/%d: %d->%d",
                pair_idx + 1, len(self._responsive_pairs),
                pair.electrode_from, pair.electrode_to,
            )

            polarity = self._get_polarity_enum(pair.polarity)
            stim_start = datetime.now(timezone.utc)

            groups = self.active_total_repeats // self.active_group_size
            remainder = self.active_total_repeats % self.active_group_size

            stim_count = 0
            for g in range(groups):
                for s in range(self.active_group_size):
                    self._stimulate_once(
                        electrode_idx=pair.electrode_from,
                        amplitude_ua=pair.amplitude,
                        duration_us=pair.duration,
                        polarity=polarity,
                        trigger_key=0,
                        phase="active",
                    )
                    stim_count += 1
                    self._wait(self.active_inter_stim_s)
                self._wait(self.active_inter_group_s)

            for s in range(remainder):
                self._stimulate_once(
                    electrode_idx=pair.electrode_from,
                    amplitude_ua=pair.amplitude,
                    duration_us=pair.duration,
                    polarity=polarity,
                    trigger_key=0,
                    phase="active",
                )
                stim_count += 1
                self._wait(self.active_inter_stim_s)

            stim_stop = datetime.now(timezone.utc)

            spike_df = self.database.get_spike_event(
                stim_start, stim_stop, self.experiment_handle.exp_name
            )
            trigger_df = self.database.get_all_triggers(stim_start, stim_stop)

            trig_times = self._extract_trigger_times(trigger_df, 0)
            resp_spike_times = self._extract_spike_times(spike_df, pair.electrode_to)

            bin_centers, ccg_counts = self._compute_ccg(
                trig_times, resp_spike_times, self.ccg_window_ms, self.ccg_bin_ms
            )

            latencies = []
            for t_trig in trig_times:
                diffs_ms = (resp_spike_times - t_trig) * 1000.0
                valid = diffs_ms[(diffs_ms > 0) & (diffs_ms <= self.ccg_window_ms)]
                if len(valid) > 0:
                    latencies.append(float(np.min(valid)))

            latency_array = np.array(latencies)
            model_fit = self._select_best_model(latency_array) if len(latency_array) > 3 else {
                "best_model": "insufficient_data",
                "synaptic_delay_ms": pair.median_latency_ms,
            }

            self._active_results.append({
                "pair_index": pair_idx,
                "electrode_from": pair.electrode_from,
                "electrode_to": pair.electrode_to,
                "stimulations": stim_count,
                "total_spikes_resp": len(resp_spike_times),
                "total_triggers": len(trig_times),
                "latencies_count": len(latencies),
                "median_latency_ms": float(np.median(latency_array)) if len(latency_array) > 0 else None,
                "mean_latency_ms": float(np.mean(latency_array)) if len(latency_array) > 0 else None,
                "ccg_bin_centers": bin_centers.tolist(),
                "ccg_counts": ccg_counts.tolist(),
                "model_fit": model_fit,
                "synaptic_delay_ms": model_fit.get("synaptic_delay_ms", pair.median_latency_ms),
            })

            logger.info(
                "  Pair %d->%d: %d latencies, model=%s, delay=%.2fms",
                pair.electrode_from, pair.electrode_to,
                len(latencies),
                model_fit.get("best_model", "N/A"),
                model_fit.get("synaptic_delay_ms", 0),
            )

    def _phase3_stdp_experiment(self) -> None:
        logger.info("=== PHASE 3: Two-Electrode Hebbian Learning (STDP) ===")

        for pair_idx, pair in enumerate(self._responsive_pairs):
            active_result = None
            for ar in self._active_results:
                if ar["electrode_from"] == pair.electrode_from and ar["electrode_to"] == pair.electrode_to:
                    active_result = ar
                    break

            hebbian_delay_ms = pair.median_latency_ms
            if active_result and active_result.get("synaptic_delay_ms"):
                hebbian_delay_ms = active_result["synaptic_delay_ms"]

            hebbian_delay_ms = max(5.0, min(hebbian_delay_ms, 50.0))

            polarity = self._get_polarity_enum(pair.polarity)

            logger.info(
                "STDP pair %d/%d: %d->%d, hebbian_delay=%.2fms",
                pair_idx + 1, len(self._responsive_pairs),
                pair.electrode_from, pair.electrode_to, hebbian_delay_ms,
            )

            test_probe_interval_s = 1.0 / self.stdp_probe_rate_hz if self.stdp_probe_rate_hz > 0 else 10.0
            test_n_probes = max(1, int(self.stdp_test_duration_min * 60 * self.stdp_probe_rate_hz))
            learn_n_probes = max(1, int(self.stdp_learn_duration_min * 60 * self.stdp_probe_rate_hz))
            valid_n_probes = max(1, int(self.stdp_valid_duration_min * 60 * self.stdp_probe_rate_hz))

            test_n_probes = min(test_n_probes, 50)
            learn_n_probes = min(learn_n_probes, 100)
            valid_n_probes = min(valid_n_probes, 50)

            logger.info("  Testing phase: %d probes", test_n_probes)
            test_start = datetime.now(timezone.utc)
            for probe_i in range(test_n_probes):
                self._stimulate_once(
                    electrode_idx=pair.electrode_from,
                    amplitude_ua=pair.amplitude,
                    duration_us=pair.duration,
                    polarity=polarity,
                    trigger_key=0,
                    phase="stdp_test",
                )
                self._wait(test_probe_interval_s)
            test_stop = datetime.now(timezone.utc)

            logger.info("  Learning phase: %d paired stimulations", learn_n_probes)
            learn_start = datetime.now(timezone.utc)

            learn_stim_interval_s = max(1.0, (hebbian_delay_ms / 1000.0) + 0.5)
            probe_every = max(1, learn_n_probes // 10)

            for learn_i in range(learn_n_probes):
                self._stimulate_pair(
                    electrode_from=pair.electrode_from,
                    electrode_to=pair.electrode_to,
                    amplitude_ua=pair.amplitude,
                    duration_us=pair.duration,
                    polarity=polarity,
                    trigger_key_from=0,
                    trigger_key_to=1,
                    delay_ms=hebbian_delay_ms,
                    phase="stdp_learn",
                )
                self._wait(learn_stim_interval_s)

                if (learn_i + 1) % probe_every == 0:
                    self._stimulate_once(
                        electrode_idx=pair.electrode_from,
                        amplitude_ua=pair.amplitude,
                        duration_us=pair.duration,
                        polarity=polarity,
                        trigger_key=0,
                        phase="stdp_learn_probe",
                    )
                    self._wait(test_probe_interval_s)

            learn_stop = datetime.now(timezone.utc)

            logger.info("  Validation phase: %d probes", valid_n_probes)
            valid_start = datetime.now(timezone.utc)
            for probe_i in range(valid_n_probes):
                self._stimulate_once(
                    electrode_idx=pair.electrode_from,
                    amplitude_ua=pair.amplitude,
                    duration_us=pair.duration,
                    polarity=polarity,
                    trigger_key=0,
                    phase="stdp_valid",
                )
                self._wait(test_probe_interval_s)
            valid_stop = datetime.now(timezone.utc)

            fs_name = self.experiment_handle.exp_name

            test_spikes = self.database.get_spike_event(test_start, test_stop, fs_name)
            test_triggers = self.database.get_all_triggers(test_start, test_stop)
            valid_spikes = self.database.get_spike_event(valid_start, valid_stop, fs_name)
            valid_triggers = self.database.get_all_triggers(valid_start, valid_stop)

            test_trig_times = self._extract_trigger_times(test_triggers, 0)
            test_resp_times = self._extract_spike_times(test_spikes, pair.electrode_to)
            valid_trig_times = self._extract_trigger_times(valid_triggers, 0)
            valid_resp_times = self._extract_spike_times(valid_spikes, pair.electrode_to)

            test_bin_centers, test_ccg = self._compute_ccg(
                test_trig_times, test_resp_times, self.ccg_window_ms, self.ccg_bin_ms
            )
            valid_bin_centers, valid_ccg = self._compute_ccg(
                valid_trig_times, valid_resp_times, self.ccg_window_ms, self.ccg_bin_ms
            )

            test_latencies = []
            for t_trig in test_trig_times:
                diffs_ms = (test_resp_times - t_trig) * 1000.0
                valid_diffs = diffs_ms[(diffs_ms > 0) & (diffs_ms <= self.ccg_window_ms)]
                if len(valid_diffs) > 0:
                    test_latencies.append(float(np.min(valid_diffs)))

            valid_latencies = []
            for t_trig in valid_trig_times:
                diffs_ms = (valid_resp_times - t_trig) * 1000.0
                valid_diffs = diffs_ms[(diffs_ms > 0) & (diffs_ms <= self.ccg_window_ms)]
                if len(valid_diffs) > 0:
                    valid_latencies.append(float(np.min(valid_diffs)))

            test_lat_arr = np.array(test_latencies)
            valid_lat_arr = np.array(valid_latencies)

            emd = self._compute_wasserstein_1d(test_lat_arr, valid_lat_arr)

            test_peak = float(np.max(test_ccg)) if len(test_ccg) > 0 else 0.0
            valid_peak = float(np.max(valid_ccg)) if len(valid_ccg) > 0 else 0.0
            peak_ratio = valid_peak / test_peak if test_peak > 0 else 0.0

            self._stdp_results.append({
                "pair_index": pair_idx,
                "electrode_from": pair.electrode_from,
                "electrode_to": pair.electrode_to,
                "hebbian_delay_ms": hebbian_delay_ms,
                "test_probes": test_n_probes,
                "learn_stims": learn_n_probes,
                "valid_probes": valid_n_probes,
                "test_latencies_count": len(test_latencies),
                "valid_latencies_count": len(valid_latencies),
                "test_median_latency_ms": float(np.median(test_lat_arr)) if len(test_lat_arr) > 0 else None,
                "valid_median_latency_ms": float(np.median(valid_lat_arr)) if len(valid_lat_arr) > 0 else None,
                "test_ccg_peak": test_peak,
                "valid_ccg_peak": valid_peak,
                "ccg_peak_ratio": peak_ratio,
                "earth_movers_distance": emd,
                "test_ccg_bins": test_bin_centers.tolist(),
                "test_ccg_counts": test_ccg.tolist(),
                "valid_ccg_bins": valid_bin_centers.tolist(),
                "valid_ccg_counts": valid_ccg.tolist(),
            })

            logger.info(
                "  STDP result: test_peak=%.1f valid_peak=%.1f ratio=%.3f EMD=%.3f",
                test_peak, valid_peak, peak_ratio, emd,
            )

    def run(self) -> Dict[str, Any]:
        try:
            logger.info("Initialising hardware connections")

            from neuroplatform import Experiment as NeuroPlatformExperiment

            self.experiment_handle = NeuroPlatformExperiment(self.token)
            self.trigger_controller = TriggerController(email=self.booking_email)
            self.intan = IntanSofware()
            self.database = Database()

            logger.info("Experiment: %s", self.experiment_handle.exp_name)
            logger.info("Electrodes: %s", self.experiment_handle.electrodes)

            if not self.experiment_handle.start():
                logger.error("Failed to start experiment")
                return {"status": "failed", "error": "Could not start experiment"}

            recording_start = datetime.now(timezone.utc)

            self._phase1_excitability_scan()

            self._select_responsive_pairs()

            self._phase2_active_electrode()

            self._phase3_stdp_experiment()

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

        summary = {
            "status": "completed",
            "experiment_name": self.experiment_handle.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "total_stimulations": len(self._stimulation_log),
            "phase1_scan_configs": len(self._scan_results),
            "phase2_active_pairs": len(self._active_results),
            "phase3_stdp_pairs": len(self._stdp_results),
            "responsive_pairs": [
                {
                    "from": p.electrode_from,
                    "to": p.electrode_to,
                    "amplitude": p.amplitude,
                    "duration": p.duration,
                    "polarity": p.polarity,
                    "median_latency_ms": p.median_latency_ms,
                    "response_rate": p.response_rate,
                }
                for p in self._responsive_pairs
            ],
            "active_results": self._active_results,
            "stdp_results": self._stdp_results,
        }

        return summary

    def _save_all(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> None:
        fs_name = getattr(self.experiment_handle, "exp_name", "unknown")
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
            "phase1_scan_configs": len(self._scan_results),
            "phase2_active_pairs": len(self._active_results),
            "phase3_stdp_pairs": len(self._stdp_results),
        }
        saver.save_summary(summary)

        waveform_records = self._fetch_spike_waveforms(
            fs_name, spike_df, trigger_df, recording_start, recording_stop
        )
        saver.save_spike_waveforms(waveform_records)

        analysis = {
            "scan_results": self._scan_results,
            "active_results": self._active_results,
            "stdp_results": self._stdp_results,
        }
        saver.save_analysis(analysis, "full_analysis")

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
        for col in ["channel", "electrode", "index", "idx"]:
            if col in spike_df.columns:
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
                    electrode_idx, exc,
                )

        return waveform_records

    def _cleanup(self) -> None:
        logger.info("Cleaning up resources")

        if self.experiment_handle is not None:
            try:
                self.experiment_handle.stop()
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
