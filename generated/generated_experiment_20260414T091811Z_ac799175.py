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
    timestamp_utc: str
    trigger_key: int = 0
    polarity: str = "NegativeFirst"
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
        scan_amplitudes: List[float] = None,
        scan_durations: List[float] = None,
        scan_repeats: int = 5,
        scan_inter_stim_s: float = 1.0,
        scan_inter_channel_s: float = 5.0,
        scan_min_hits: int = 3,
        scan_max_hits: int = 5,
        active_total_repeats: int = 100,
        active_group_size: int = 10,
        active_inter_stim_s: float = 1.0,
        active_inter_group_s: float = 5.0,
        active_window_ms: float = 50.0,
        hebbian_test_duration_min: float = 20.0,
        hebbian_learn_duration_min: float = 50.0,
        hebbian_validation_duration_min: float = 20.0,
        hebbian_test_isi_s: float = 2.0,
        hebbian_learn_isi_s: float = 1.0,
        max_pairs_to_use: int = 5,
        use_scan_results: bool = True,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.scan_amplitudes = scan_amplitudes or [1.0, 2.0, 3.0]
        self.scan_durations = scan_durations or [100.0, 200.0, 300.0, 400.0]
        self.scan_repeats = scan_repeats
        self.scan_inter_stim_s = scan_inter_stim_s
        self.scan_inter_channel_s = scan_inter_channel_s
        self.scan_min_hits = scan_min_hits
        self.scan_max_hits = scan_max_hits

        self.active_total_repeats = active_total_repeats
        self.active_group_size = active_group_size
        self.active_inter_stim_s = active_inter_stim_s
        self.active_inter_group_s = active_inter_group_s
        self.active_window_ms = active_window_ms

        self.hebbian_test_duration_min = hebbian_test_duration_min
        self.hebbian_learn_duration_min = hebbian_learn_duration_min
        self.hebbian_validation_duration_min = hebbian_validation_duration_min
        self.hebbian_test_isi_s = hebbian_test_isi_s
        self.hebbian_learn_isi_s = hebbian_learn_isi_s

        self.max_pairs_to_use = max_pairs_to_use
        self.use_scan_results = use_scan_results

        self.experiment_handle = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._scan_results: List[Dict] = []
        self._responsive_pairs: List[PairConfig] = []
        self._active_results: List[Dict] = []
        self._hebbian_results: List[Dict] = []
        self._phase_timestamps: Dict[str, Any] = {}

        self._prior_deep_scan = [
            {"stim_electrode": 14, "resp_electrode": 12, "amplitude": 1.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 22.72, "response_rate": 0.94},
            {"stim_electrode": 9, "resp_electrode": 10, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 11.035, "response_rate": 0.94},
            {"stim_electrode": 22, "resp_electrode": 21, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 14.03, "response_rate": 0.93},
            {"stim_electrode": 5, "resp_electrode": 4, "amplitude": 1.0, "duration": 300.0, "polarity": "PositiveFirst", "median_latency_ms": 17.66, "response_rate": 0.93},
            {"stim_electrode": 18, "resp_electrode": 17, "amplitude": 1.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 25.075, "response_rate": 0.89},
        ]

    def _wait(self, seconds: float) -> None:
        wait(seconds)

    def _get_polarity_enum(self, polarity_str: str) -> StimPolarity:
        if polarity_str == "PositiveFirst":
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
        phase_label: str = "",
    ) -> None:
        stim = self._make_charge_balanced_stim(
            electrode_idx, amplitude_ua, duration_us, polarity, trigger_key
        )
        self.intan.send_stimparam([stim])
        self._fire_trigger(trigger_key)

        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            timestamp_utc=datetime_now().isoformat(),
            trigger_key=trigger_key,
            polarity=polarity.name,
            phase=phase_label,
        ))

    def _stimulate_pair(
        self,
        stim_electrode: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity,
        trigger_key: int = 0,
        phase_label: str = "",
    ) -> None:
        self._stimulate_once(
            stim_electrode, amplitude_ua, duration_us, polarity, trigger_key, phase_label
        )

    def _query_spikes_window(
        self, start: datetime, stop: datetime, fs_name: str
    ) -> pd.DataFrame:
        return self.database.get_spike_event(start, stop, fs_name)

    def _phase1_excitability_scan(self) -> None:
        logger.info("=== PHASE 1: Basic Excitability Scan ===")
        self._phase_timestamps["scan_start"] = datetime_now()

        electrodes = self.experiment_handle.electrodes
        if not electrodes:
            electrodes = list(range(32))

        polarities = [StimPolarity.NegativeFirst, StimPolarity.PositiveFirst]
        scan_results = []

        num_electrodes = min(len(electrodes), 8)
        scan_electrodes = electrodes[:num_electrodes]

        for ch_idx, elec in enumerate(scan_electrodes):
            logger.info("Scanning electrode %d (%d/%d)", elec, ch_idx + 1, num_electrodes)

            for amp in self.scan_amplitudes:
                for dur in self.scan_durations:
                    for pol in polarities:
                        pre_stim_time = datetime_now()
                        hit_count = 0

                        for rep in range(self.scan_repeats):
                            stim_time = datetime_now()
                            self._stimulate_once(
                                elec, amp, dur, pol, trigger_key=0,
                                phase_label="scan"
                            )
                            self._wait(self.scan_inter_stim_s)

                        post_stim_time = datetime_now()

                        spike_df = self._query_spikes_window(
                            pre_stim_time, post_stim_time,
                            self.experiment_handle.exp_name
                        )

                        if not spike_df.empty:
                            col = "channel" if "channel" in spike_df.columns else None
                            if col is None:
                                for c in spike_df.columns:
                                    if "channel" in c.lower() or "index" in c.lower():
                                        col = c
                                        break
                            if col is not None:
                                other_electrodes = [e for e in scan_electrodes if e != elec]
                                for resp_elec in other_electrodes:
                                    resp_spikes = spike_df[spike_df[col] == resp_elec]
                                    n_responses = min(len(resp_spikes), self.scan_repeats)
                                    if self.scan_min_hits <= n_responses <= self.scan_max_hits:
                                        scan_results.append({
                                            "stim_electrode": elec,
                                            "resp_electrode": resp_elec,
                                            "amplitude": amp,
                                            "duration": dur,
                                            "polarity": pol.name,
                                            "hits": n_responses,
                                            "repeats": self.scan_repeats,
                                        })

            if ch_idx < num_electrodes - 1:
                self._wait(self.scan_inter_channel_s)

        self._scan_results = scan_results
        self._phase_timestamps["scan_stop"] = datetime_now()
        logger.info("Scan complete. Found %d responsive configurations.", len(scan_results))

    def _select_best_pairs(self) -> List[PairConfig]:
        if self.use_scan_results and self._prior_deep_scan:
            logger.info("Using prior deep scan results for pair selection")
            pairs = []
            for entry in self._prior_deep_scan[:self.max_pairs_to_use]:
                pairs.append(PairConfig(
                    stim_electrode=entry["stim_electrode"],
                    resp_electrode=entry["resp_electrode"],
                    amplitude=entry["amplitude"],
                    duration=entry["duration"],
                    polarity=entry["polarity"],
                    median_latency_ms=entry["median_latency_ms"],
                    response_rate=entry["response_rate"],
                ))
            return pairs

        pair_map: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
        for r in self._scan_results:
            key = (r["stim_electrode"], r["resp_electrode"])
            pair_map[key].append(r)

        best_pairs = []
        for (stim_e, resp_e), configs in pair_map.items():
            best = max(configs, key=lambda x: x["hits"])
            best_pairs.append(PairConfig(
                stim_electrode=stim_e,
                resp_electrode=resp_e,
                amplitude=best["amplitude"],
                duration=best["duration"],
                polarity=best["polarity"],
                median_latency_ms=15.0,
                response_rate=best["hits"] / best["repeats"],
                hits_k=best["hits"],
            ))

        best_pairs.sort(key=lambda p: p.response_rate, reverse=True)
        return best_pairs[:self.max_pairs_to_use]

    def _phase2_active_electrode(self) -> None:
        logger.info("=== PHASE 2: Active Electrode Experiment ===")
        self._phase_timestamps["active_start"] = datetime_now()

        self._responsive_pairs = self._select_best_pairs()
        if not self._responsive_pairs:
            logger.warning("No responsive pairs found. Skipping Phase 2.")
            self._phase_timestamps["active_stop"] = datetime_now()
            return

        logger.info("Selected %d pairs for active electrode experiment", len(self._responsive_pairs))

        for pair_idx, pair in enumerate(self._responsive_pairs):
            logger.info(
                "Active experiment pair %d/%d: stim=%d -> resp=%d (amp=%.1f, dur=%.0f, pol=%s)",
                pair_idx + 1, len(self._responsive_pairs),
                pair.stim_electrode, pair.resp_electrode,
                pair.amplitude, pair.duration, pair.polarity
            )

            polarity_enum = self._get_polarity_enum(pair.polarity)
            pair_start = datetime_now()
            stim_times = []

            num_groups = self.active_total_repeats // self.active_group_size
            remainder = self.active_total_repeats % self.active_group_size

            for group_idx in range(num_groups):
                for stim_idx in range(self.active_group_size):
                    t_stim = datetime_now()
                    stim_times.append(t_stim.isoformat())
                    self._stimulate_pair(
                        pair.stim_electrode, pair.amplitude, pair.duration,
                        polarity_enum, trigger_key=0, phase_label="active"
                    )
                    self._wait(self.active_inter_stim_s)

                if group_idx < num_groups - 1:
                    self._wait(self.active_inter_group_s)

            if remainder > 0:
                if num_groups > 0:
                    self._wait(self.active_inter_group_s)
                for stim_idx in range(remainder):
                    t_stim = datetime_now()
                    stim_times.append(t_stim.isoformat())
                    self._stimulate_pair(
                        pair.stim_electrode, pair.amplitude, pair.duration,
                        polarity_enum, trigger_key=0, phase_label="active"
                    )
                    self._wait(self.active_inter_stim_s)

            pair_stop = datetime_now()

            spike_df = self._query_spikes_window(
                pair_start, pair_stop, self.experiment_handle.exp_name
            )

            latencies = self._compute_latencies(
                spike_df, pair.resp_electrode, pair_start, pair_stop
            )

            gmm_result = self._fit_response_distribution(latencies)

            self._active_results.append({
                "pair_index": pair_idx,
                "stim_electrode": pair.stim_electrode,
                "resp_electrode": pair.resp_electrode,
                "amplitude": pair.amplitude,
                "duration": pair.duration,
                "polarity": pair.polarity,
                "total_stims": self.active_total_repeats,
                "stim_times": stim_times,
                "n_response_spikes": len(latencies),
                "latencies_ms": latencies,
                "distribution_fit": gmm_result,
                "estimated_delay_ms": gmm_result.get("best_mode_ms", pair.median_latency_ms),
            })

            logger.info(
                "Pair %d: %d response spikes, estimated delay=%.2f ms",
                pair_idx, len(latencies),
                gmm_result.get("best_mode_ms", pair.median_latency_ms)
            )

        self._phase_timestamps["active_stop"] = datetime_now()

    def _compute_latencies(
        self, spike_df: pd.DataFrame, resp_electrode: int,
        window_start: datetime, window_stop: datetime
    ) -> List[float]:
        if spike_df.empty:
            return []

        col = "channel" if "channel" in spike_df.columns else None
        if col is None:
            for c in spike_df.columns:
                if "channel" in c.lower() or "index" in c.lower():
                    col = c
                    break
        if col is None:
            return []

        resp_spikes = spike_df[spike_df[col] == resp_electrode]
        if resp_spikes.empty:
            return []

        time_col = None
        for c in ["Time", "_time", "time", "timestamp"]:
            if c in resp_spikes.columns:
                time_col = c
                break
        if time_col is None:
            return []

        latencies = []
        try:
            times = pd.to_datetime(resp_spikes[time_col], utc=True)
            ref_time = window_start
            for t in times:
                delta_ms = (t - ref_time).total_seconds() * 1000.0
                if 0 < delta_ms < self.active_window_ms * 2:
                    latencies.append(round(delta_ms, 3))
        except Exception as e:
            logger.warning("Error computing latencies: %s", e)

        return latencies

    def _fit_response_distribution(self, latencies: List[float]) -> Dict[str, Any]:
        result = {
            "best_model": "none",
            "best_mode_ms": 0.0,
            "n_samples": len(latencies),
            "models_tested": [],
        }

        if len(latencies) < 5:
            return result

        data = np.array(latencies)
        mean_val = float(np.mean(data))
        std_val = float(np.std(data))
        median_val = float(np.median(data))

        gaussian_1_ll = self._gaussian_log_likelihood(data, [mean_val], [max(std_val, 0.1)], [1.0])
        result["models_tested"].append({
            "model": "gaussian_1",
            "log_likelihood": gaussian_1_ll,
            "bic": -2 * gaussian_1_ll + 2 * math.log(max(len(data), 1)),
            "params": {"mean": mean_val, "std": std_val},
        })

        if len(data) >= 10:
            sorted_d = np.sort(data)
            mid = len(sorted_d) // 2
            g1 = sorted_d[:mid]
            g2 = sorted_d[mid:]
            if len(g1) > 1 and len(g2) > 1:
                m1, s1 = float(np.mean(g1)), max(float(np.std(g1)), 0.1)
                m2, s2 = float(np.mean(g2)), max(float(np.std(g2)), 0.1)
                w1 = len(g1) / len(data)
                w2 = 1.0 - w1
                bi_ll = self._gaussian_log_likelihood(data, [m1, m2], [s1, s2], [w1, w2])
                result["models_tested"].append({
                    "model": "gaussian_2",
                    "log_likelihood": bi_ll,
                    "bic": -2 * bi_ll + 5 * math.log(max(len(data), 1)),
                    "params": {"means": [m1, m2], "stds": [s1, s2], "weights": [w1, w2]},
                })

        if len(data) >= 15:
            sorted_d = np.sort(data)
            t1 = len(sorted_d) // 3
            t2 = 2 * len(sorted_d) // 3
            g1 = sorted_d[:t1]
            g2 = sorted_d[t1:t2]
            g3 = sorted_d[t2:]
            if len(g1) > 1 and len(g2) > 1 and len(g3) > 1:
                means = [float(np.mean(g)) for g in [g1, g2, g3]]
                stds = [max(float(np.std(g)), 0.1) for g in [g1, g2, g3]]
                weights = [len(g) / len(data) for g in [g1, g2, g3]]
                tri_ll = self._gaussian_log_likelihood(data, means, stds, weights)
                result["models_tested"].append({
                    "model": "gaussian_3",
                    "log_likelihood": tri_ll,
                    "bic": -2 * tri_ll + 8 * math.log(max(len(data), 1)),
                    "params": {"means": means, "stds": stds, "weights": weights},
                })

        if len(data) >= 5 and mean_val > 0:
            k_shape = (mean_val / max(std_val, 0.01)) ** 2
            theta_scale = max(std_val, 0.01) ** 2 / max(mean_val, 0.01)
            gamma_ll = self._gamma_log_likelihood(data, k_shape, theta_scale)
            result["models_tested"].append({
                "model": "gamma",
                "log_likelihood": gamma_ll,
                "bic": -2 * gamma_ll + 2 * math.log(max(len(data), 1)),
                "params": {"k": k_shape, "theta": theta_scale},
            })

        if result["models_tested"]:
            best = min(result["models_tested"], key=lambda m: m["bic"])
            result["best_model"] = best["model"]

            if best["model"] == "gaussian_1":
                result["best_mode_ms"] = best["params"]["mean"]
            elif best["model"] in ("gaussian_2", "gaussian_3"):
                means = best["params"]["means"]
                weights = best["params"]["weights"]
                best_comp = max(range(len(means)), key=lambda i: weights[i])
                result["best_mode_ms"] = means[best_comp]
            elif best["model"] == "gamma":
                k = best["params"]["k"]
                theta = best["params"]["theta"]
                result["best_mode_ms"] = max((k - 1) * theta, 0.0) if k > 1 else 0.0
            else:
                result["best_mode_ms"] = median_val
        else:
            result["best_mode_ms"] = median_val

        return result

    def _gaussian_log_likelihood(
        self, data: np.ndarray, means: List[float],
        stds: List[float], weights: List[float]
    ) -> float:
        ll = 0.0
        for x in data:
            p = 0.0
            for m, s, w in zip(means, stds, weights):
                s = max(s, 0.01)
                coeff = w / (s * math.sqrt(2 * math.pi))
                exponent = -0.5 * ((x - m) / s) ** 2
                p += coeff * math.exp(max(exponent, -500))
            ll += math.log(max(p, 1e-300))
        return ll

    def _gamma_log_likelihood(self, data: np.ndarray, k: float, theta: float) -> float:
        ll = 0.0
        k = max(k, 0.01)
        theta = max(theta, 0.01)
        log_gamma_k = math.lgamma(k)
        for x in data:
            if x <= 0:
                ll += -500
                continue
            log_p = (k - 1) * math.log(x) - x / theta - k * math.log(theta) - log_gamma_k
            ll += max(log_p, -500)
        return ll

    def _phase3_hebbian_learning(self) -> None:
        logger.info("=== PHASE 3: Two-Electrode Hebbian Learning ===")
        self._phase_timestamps["hebbian_start"] = datetime_now()

        if not self._responsive_pairs:
            logger.warning("No responsive pairs. Skipping Phase 3.")
            self._phase_timestamps["hebbian_stop"] = datetime_now()
            return

        pairs_for_hebbian = self._responsive_pairs[:self.max_pairs_to_use]

        for pair_idx, pair in enumerate(pairs_for_hebbian):
            delay_from_active = pair.median_latency_ms
            for ar in self._active_results:
                if ar["stim_electrode"] == pair.stim_electrode and ar["resp_electrode"] == pair.resp_electrode:
                    delay_from_active = ar.get("estimated_delay_ms", pair.median_latency_ms)
                    break

            hebbian_delay_ms = delay_from_active
            logger.info(
                "Hebbian pair %d/%d: stim=%d -> resp=%d, delay=%.2f ms",
                pair_idx + 1, len(pairs_for_hebbian),
                pair.stim_electrode, pair.resp_electrode, hebbian_delay_ms
            )

            polarity_enum = self._get_polarity_enum(pair.polarity)

            testing_start = datetime_now()
            testing_stim_count = self._run_testing_phase(
                pair, polarity_enum, "testing"
            )
            testing_stop = datetime_now()

            learning_start = datetime_now()
            learning_stim_count = self._run_learning_phase(
                pair, polarity_enum, hebbian_delay_ms
            )
            learning_stop = datetime_now()

            validation_start = datetime_now()
            validation_stim_count = self._run_testing_phase(
                pair, polarity_enum, "validation"
            )
            validation_stop = datetime_now()

            fs_name = self.experiment_handle.exp_name

            testing_spikes = self._query_spikes_window(testing_start, testing_stop, fs_name)
            validation_spikes = self._query_spikes_window(validation_start, validation_stop, fs_name)

            testing_latencies = self._compute_latencies(
                testing_spikes, pair.resp_electrode, testing_start, testing_stop
            )
            validation_latencies = self._compute_latencies(
                validation_spikes, pair.resp_electrode, validation_start, validation_stop
            )

            emd = self._wasserstein_distance(testing_latencies, validation_latencies)

            self._hebbian_results.append({
                "pair_index": pair_idx,
                "stim_electrode": pair.stim_electrode,
                "resp_electrode": pair.resp_electrode,
                "hebbian_delay_ms": hebbian_delay_ms,
                "testing_stim_count": testing_stim_count,
                "learning_stim_count": learning_stim_count,
                "validation_stim_count": validation_stim_count,
                "testing_latencies_ms": testing_latencies,
                "validation_latencies_ms": validation_latencies,
                "testing_n_responses": len(testing_latencies),
                "validation_n_responses": len(validation_latencies),
                "wasserstein_distance": emd,
                "testing_start": testing_start.isoformat(),
                "testing_stop": testing_stop.isoformat(),
                "learning_start": learning_start.isoformat(),
                "learning_stop": learning_stop.isoformat(),
                "validation_start": validation_start.isoformat(),
                "validation_stop": validation_stop.isoformat(),
            })

            logger.info(
                "Hebbian pair %d: testing=%d spikes, validation=%d spikes, EMD=%.4f",
                pair_idx, len(testing_latencies), len(validation_latencies), emd
            )

        self._phase_timestamps["hebbian_stop"] = datetime_now()

    def _run_testing_phase(
        self, pair: PairConfig, polarity: StimPolarity, phase_label: str
    ) -> int:
        if phase_label == "testing":
            duration_min = self.hebbian_test_duration_min
        else:
            duration_min = self.hebbian_validation_duration_min

        duration_s = duration_min * 60.0
        max_stims = int(duration_s / self.hebbian_test_isi_s)
        max_stims = min(max_stims, 600)

        logger.info("Running %s phase: up to %d stims over %.1f min",
                     phase_label, max_stims, duration_min)

        stim_count = 0
        phase_start = datetime_now()

        for i in range(max_stims):
            elapsed = (datetime_now() - phase_start).total_seconds()
            if elapsed >= duration_s:
                break

            self._stimulate_pair(
                pair.stim_electrode, pair.amplitude, pair.duration,
                polarity, trigger_key=0, phase_label=phase_label
            )
            stim_count += 1
            self._wait(self.hebbian_test_isi_s)

        logger.info("%s phase complete: %d stimulations", phase_label, stim_count)
        return stim_count

    def _run_learning_phase(
        self, pair: PairConfig, polarity: StimPolarity, hebbian_delay_ms: float
    ) -> int:
        duration_s = self.hebbian_learn_duration_min * 60.0
        max_stims = int(duration_s / self.hebbian_learn_isi_s)
        max_stims = min(max_stims, 3000)

        delay_s = hebbian_delay_ms / 1000.0

        logger.info(
            "Running learning phase: up to %d paired stims, delay=%.2f ms",
            max_stims, hebbian_delay_ms
        )

        stim_count = 0
        phase_start = datetime_now()

        probe_interval = 50

        for i in range(max_stims):
            elapsed = (datetime_now() - phase_start).total_seconds()
            if elapsed >= duration_s:
                break

            self._stimulate_pair(
                pair.stim_electrode, pair.amplitude, pair.duration,
                polarity, trigger_key=0, phase_label="learning_pre"
            )

            self._wait(delay_s)

            self._stimulate_pair(
                pair.resp_electrode, pair.amplitude, pair.duration,
                polarity, trigger_key=1, phase_label="learning_post"
            )

            stim_count += 1

            if (i + 1) % probe_interval == 0 and i < max_stims - 1:
                self._wait(0.5)
                self._stimulate_pair(
                    pair.stim_electrode, pair.amplitude, pair.duration,
                    polarity, trigger_key=0, phase_label="learning_probe"
                )
                self._wait(0.5)

            remaining_wait = max(0, self.hebbian_learn_isi_s - delay_s - 0.05)
            self._wait(remaining_wait)

        logger.info("Learning phase complete: %d paired stimulations", stim_count)
        return stim_count

    def _wasserstein_distance(self, dist_a: List[float], dist_b: List[float]) -> float:
        if not dist_a or not dist_b:
            return float("inf")

        a_sorted = sorted(dist_a)
        b_sorted = sorted(dist_b)

        n = max(len(a_sorted), len(b_sorted))
        if n == 0:
            return 0.0

        a_quantiles = np.interp(
            np.linspace(0, 1, n),
            np.linspace(0, 1, len(a_sorted)),
            a_sorted
        )
        b_quantiles = np.interp(
            np.linspace(0, 1, n),
            np.linspace(0, 1, len(b_sorted)),
            b_sorted
        )

        return float(np.mean(np.abs(a_quantiles - b_quantiles)))

    def _save_all(self, recording_start: datetime, recording_stop: datetime) -> None:
        fs_name = getattr(self.experiment_handle, "exp_name", "unknown")
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
            "scan_results_count": len(self._scan_results),
            "responsive_pairs_count": len(self._responsive_pairs),
            "active_results_count": len(self._active_results),
            "hebbian_results_count": len(self._hebbian_results),
            "phase_timestamps": {k: v.isoformat() if hasattr(v, 'isoformat') else str(v)
                                 for k, v in self._phase_timestamps.items()},
        }
        saver.save_summary(summary)

        waveform_records = self._fetch_spike_waveforms(
            fs_name, spike_df, recording_start, recording_stop
        )
        saver.save_spike_waveforms(waveform_records)

        analysis = {
            "scan_results": self._scan_results,
            "responsive_pairs": [asdict(p) for p in self._responsive_pairs],
            "active_results": self._active_results,
            "hebbian_results": self._hebbian_results,
        }
        saver.save_analysis(analysis, "full_analysis")

    def _fetch_spike_waveforms(
        self, fs_name: str, spike_df: pd.DataFrame,
        recording_start: datetime, recording_stop: datetime
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
            for col in spike_df.columns:
                if "electrode" in col.lower() or "idx" in col.lower() or "channel" in col.lower():
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
                    max_waveforms = 50
                    samples = raw_df.head(max_waveforms).values.tolist()
                    waveform_records.append({
                        "electrode_idx": int(electrode_idx),
                        "num_waveforms": len(raw_df),
                        "waveform_samples": samples,
                    })
            except Exception as exc:
                logger.warning(
                    "Failed to fetch waveforms for electrode %s: %s",
                    electrode_idx, exc
                )

        return waveform_records

    def _compile_results(
        self, recording_start: datetime, recording_stop: datetime
    ) -> Dict[str, Any]:
        logger.info("Compiling results")

        summary = {
            "status": "completed",
            "experiment_name": self.experiment_handle.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "total_stimulations": len(self._stimulation_log),
            "phase1_scan_configs_found": len(self._scan_results),
            "phase2_pairs_tested": len(self._active_results),
            "phase3_hebbian_pairs": len(self._hebbian_results),
            "responsive_pairs": [
                {
                    "stim": p.stim_electrode,
                    "resp": p.resp_electrode,
                    "amp": p.amplitude,
                    "dur": p.duration,
                    "latency_ms": p.median_latency_ms,
                    "rate": p.response_rate,
                }
                for p in self._responsive_pairs
            ],
        }

        if self._active_results:
            summary["active_electrode_summary"] = []
            for ar in self._active_results:
                summary["active_electrode_summary"].append({
                    "stim": ar["stim_electrode"],
                    "resp": ar["resp_electrode"],
                    "n_responses": ar["n_response_spikes"],
                    "estimated_delay_ms": ar["estimated_delay_ms"],
                    "best_model": ar["distribution_fit"]["best_model"],
                })

        if self._hebbian_results:
            summary["hebbian_summary"] = []
            for hr in self._hebbian_results:
                summary["hebbian_summary"].append({
                    "stim": hr["stim_electrode"],
                    "resp": hr["resp_electrode"],
                    "delay_ms": hr["hebbian_delay_ms"],
                    "testing_responses": hr["testing_n_responses"],
                    "validation_responses": hr["validation_n_responses"],
                    "wasserstein_distance": hr["wasserstein_distance"],
                })

        return summary

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

            recording_start = datetime_now()

            self._phase1_excitability_scan()

            self._phase2_active_electrode()

            self._phase3_hebbian_learning()

            recording_stop = datetime_now()

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()
