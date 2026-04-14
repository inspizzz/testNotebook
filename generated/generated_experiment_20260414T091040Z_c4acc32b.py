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


SCAN_RESULTS = {
    "reliable_connections": [
        {"electrode_from": 0, "electrode_to": 1, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 20.78,
         "response_entropy": 0.155, "stimulation": {"amplitude": 2.0, "duration": 300.0, "polarity": "PositiveFirst"}},
        {"electrode_from": 1, "electrode_to": 2, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 22.83,
         "response_entropy": 0.0, "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 5, "electrode_to": 4, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 19.85,
         "response_entropy": 0.0, "stimulation": {"amplitude": 3.0, "duration": 200.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 6, "electrode_to": 5, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 19.28,
         "response_entropy": 0.0, "stimulation": {"amplitude": 2.0, "duration": 200.0, "polarity": "PositiveFirst"}},
        {"electrode_from": 8, "electrode_to": 9, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 22.77,
         "response_entropy": 0.0, "stimulation": {"amplitude": 1.0, "duration": 400.0, "polarity": "PositiveFirst"}},
        {"electrode_from": 14, "electrode_to": 15, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 12.97,
         "response_entropy": 0.0, "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 17, "electrode_to": 16, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 10.73,
         "response_entropy": 0.0, "stimulation": {"amplitude": 2.0, "duration": 400.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 18, "electrode_to": 17, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 24.94,
         "response_entropy": 0.0, "stimulation": {"amplitude": 1.0, "duration": 400.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 25, "electrode_to": 24, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 14.45,
         "response_entropy": 0.155, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
    ],
    "deep_scan_pair_summaries": [
        {"pair_index": 1, "stim_electrode": 0, "resp_electrode": 1, "amplitude": 2.0, "duration": 400.0,
         "polarity": "PositiveFirst", "response_rate": 0.97, "median_latency_ms": 21.53},
        {"pair_index": 2, "stim_electrode": 1, "resp_electrode": 2, "amplitude": 1.0, "duration": 300.0,
         "polarity": "NegativeFirst", "response_rate": 0.95, "median_latency_ms": 23.45},
        {"pair_index": 4, "stim_electrode": 5, "resp_electrode": 4, "amplitude": 3.0, "duration": 200.0,
         "polarity": "NegativeFirst", "response_rate": 0.97, "median_latency_ms": 24.56},
        {"pair_index": 5, "stim_electrode": 6, "resp_electrode": 5, "amplitude": 2.0, "duration": 200.0,
         "polarity": "PositiveFirst", "response_rate": 0.87, "median_latency_ms": 19.735},
        {"pair_index": 6, "stim_electrode": 8, "resp_electrode": 9, "amplitude": 1.0, "duration": 400.0,
         "polarity": "PositiveFirst", "response_rate": 0.87, "median_latency_ms": 22.855},
        {"pair_index": 9, "stim_electrode": 14, "resp_electrode": 15, "amplitude": 1.0, "duration": 300.0,
         "polarity": "NegativeFirst", "response_rate": 0.70, "median_latency_ms": 13.23},
        {"pair_index": 10, "stim_electrode": 17, "resp_electrode": 16, "amplitude": 2.0, "duration": 400.0,
         "polarity": "NegativeFirst", "response_rate": 0.94, "median_latency_ms": 11.045},
        {"pair_index": 14, "stim_electrode": 25, "resp_electrode": 24, "amplitude": 3.0, "duration": 400.0,
         "polarity": "PositiveFirst", "response_rate": 0.98, "median_latency_ms": 15.18},
    ],
}


def _select_best_pairs(max_pairs: int = 4) -> List[Dict[str, Any]]:
    deep = SCAN_RESULTS["deep_scan_pair_summaries"]
    sorted_pairs = sorted(deep, key=lambda p: (-p["response_rate"], p["median_latency_ms"]))
    selected = []
    seen_electrodes = set()
    for p in sorted_pairs:
        s = p["stim_electrode"]
        r = p["resp_electrode"]
        if s not in seen_electrodes and r not in seen_electrodes:
            selected.append(p)
            seen_electrodes.add(s)
            seen_electrodes.add(r)
        if len(selected) >= max_pairs:
            break
    return selected


def _polarity_enum(pol_str: str) -> StimPolarity:
    if pol_str == "PositiveFirst":
        return StimPolarity.PositiveFirst
    return StimPolarity.NegativeFirst


def _compute_wasserstein_1d(u_values: List[float], v_values: List[float]) -> float:
    if not u_values or not v_values:
        return 0.0
    u_sorted = sorted(u_values)
    v_sorted = sorted(v_values)
    all_vals = sorted(set(u_sorted + v_sorted))
    if len(all_vals) < 2:
        return abs(np.mean(u_sorted) - np.mean(v_sorted))
    u_cdf = []
    v_cdf = []
    for val in all_vals:
        u_count = sum(1 for x in u_sorted if x <= val)
        v_count = sum(1 for x in v_sorted if x <= val)
        u_cdf.append(u_count / len(u_sorted))
        v_cdf.append(v_count / len(v_sorted))
    distance = 0.0
    for i in range(len(all_vals) - 1):
        diff = abs(u_cdf[i] - v_cdf[i])
        width = all_vals[i + 1] - all_vals[i]
        distance += diff * width
    return distance


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
        hebbian_test_minutes: float = 20.0,
        hebbian_learn_minutes: float = 50.0,
        hebbian_validate_minutes: float = 20.0,
        hebbian_test_rate_hz: float = 0.1,
        hebbian_learn_rate_hz: float = 1.0,
        max_pairs: int = 4,
        response_window_ms: float = 50.0,
        artifact_blanking_ms: float = 2.5,
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

        self.hebbian_test_minutes = hebbian_test_minutes
        self.hebbian_learn_minutes = hebbian_learn_minutes
        self.hebbian_validate_minutes = hebbian_validate_minutes
        self.hebbian_test_rate_hz = hebbian_test_rate_hz
        self.hebbian_learn_rate_hz = hebbian_learn_rate_hz

        self.max_pairs = max_pairs
        self.response_window_ms = response_window_ms
        self.artifact_blanking_ms = artifact_blanking_ms

        self.experiment_handle = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []

        self._scan_results: List[Dict[str, Any]] = []
        self._active_results: List[Dict[str, Any]] = []
        self._hebbian_results: List[Dict[str, Any]] = []

        self._selected_pairs: List[Dict[str, Any]] = []
        self._synaptic_delays: Dict[str, float] = {}

    def _wait(self, seconds: float) -> None:
        wait(seconds)

    def run(self) -> Dict[str, Any]:
        try:
            logger.info("Initialising hardware connections")
            from neuroplatform import Experiment as NeuroPlatformExperiment

            self.experiment_handle = NeuroPlatformExperiment(self.token)
            self.trigger_controller = TriggerController(email=self.booking_email)
            self.intan = IntanSofware()
            self.database = Database()

            fs_name = self.experiment_handle.exp_name
            electrodes = self.experiment_handle.electrodes
            logger.info("Experiment: %s", fs_name)
            logger.info("Electrodes: %s", electrodes)

            if not self.experiment_handle.start():
                logger.error("Failed to start experiment")
                return {"status": "failed", "error": "Could not start experiment"}

            recording_start = datetime_now()

            self._phase1_excitability_scan(electrodes)

            self._select_responsive_pairs()

            self._phase2_active_electrode_experiment()

            self._compute_synaptic_delays()

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
        phase_label: str = "",
    ) -> None:
        stim = self._make_stim_param(electrode_idx, amplitude_ua, duration_us, polarity, trigger_key)
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

    def _phase1_excitability_scan(self, electrodes: list) -> None:
        logger.info("=== Phase 1: Basic Excitability Scan ===")
        scan_electrodes = electrodes[:8] if len(electrodes) > 8 else electrodes
        polarities = [StimPolarity.NegativeFirst, StimPolarity.PositiveFirst]

        for elec_idx in scan_electrodes:
            logger.info("Scanning electrode %d", elec_idx)
            for amp in self.scan_amplitudes:
                for dur in self.scan_durations:
                    for pol in polarities:
                        hits = 0
                        for rep in range(self.scan_repeats):
                            pre_time = datetime_now()
                            self._stimulate_once(
                                elec_idx, amp, dur, pol,
                                trigger_key=0, phase_label="scan"
                            )
                            self._wait(self.scan_inter_stim_s)
                            post_time = datetime_now()

                            spike_df = self.database.get_spike_event(
                                pre_time, post_time,
                                self.experiment_handle.exp_name
                            )
                            if len(spike_df) > 0:
                                hits += 1

                        self._scan_results.append({
                            "electrode": elec_idx,
                            "amplitude": amp,
                            "duration": dur,
                            "polarity": pol.name,
                            "hits": hits,
                            "repeats": self.scan_repeats,
                        })

            self._wait(self.scan_inter_channel_s)

        responsive_count = sum(1 for r in self._scan_results if r["hits"] >= self.scan_min_hits)
        logger.info("Scan complete. %d responsive parameter combos found.", responsive_count)

    def _select_responsive_pairs(self) -> None:
        logger.info("Selecting responsive electrode pairs from prior scan data")
        self._selected_pairs = _select_best_pairs(self.max_pairs)
        logger.info("Selected %d pairs for active electrode experiment", len(self._selected_pairs))
        for p in self._selected_pairs:
            logger.info("  Pair: stim=%d -> resp=%d, rate=%.2f, latency=%.2f ms",
                        p["stim_electrode"], p["resp_electrode"],
                        p["response_rate"], p["median_latency_ms"])

    def _phase2_active_electrode_experiment(self) -> None:
        logger.info("=== Phase 2: Active Electrode Experiment ===")

        for pair_info in self._selected_pairs:
            stim_elec = pair_info["stim_electrode"]
            resp_elec = pair_info["resp_electrode"]
            amp = pair_info["amplitude"]
            dur = pair_info["duration"]
            pol = _polarity_enum(pair_info["polarity"])
            pair_key = f"{stim_elec}->{resp_elec}"

            logger.info("Active experiment for pair %s", pair_key)

            stim_times = []
            spike_latencies = []

            num_groups = self.active_total_repeats // self.active_group_size
            remainder = self.active_total_repeats % self.active_group_size

            for g in range(num_groups):
                for s in range(self.active_group_size):
                    pre_time = datetime_now()
                    self._stimulate_once(
                        stim_elec, amp, dur, pol,
                        trigger_key=0, phase_label="active"
                    )
                    stim_times.append(datetime_now().isoformat())
                    self._wait(self.active_inter_stim_s)
                    post_time = datetime_now()

                    spike_df = self.database.get_spike_event_electrode(
                        pre_time, post_time, resp_elec
                    )
                    if len(spike_df) > 0:
                        time_col = "Time" if "Time" in spike_df.columns else "_time"
                        if time_col in spike_df.columns:
                            for _, row in spike_df.iterrows():
                                try:
                                    spike_t = pd.Timestamp(row[time_col])
                                    pre_t = pd.Timestamp(pre_time)
                                    if spike_t.tzinfo is None:
                                        spike_t = spike_t.tz_localize("UTC")
                                    if pre_t.tzinfo is None:
                                        pre_t = pre_t.tz_localize("UTC")
                                    latency_ms = (spike_t - pre_t).total_seconds() * 1000.0
                                    if self.artifact_blanking_ms < latency_ms < self.response_window_ms:
                                        spike_latencies.append(latency_ms)
                                except Exception:
                                    pass

                self._wait(self.active_inter_group_s)

            for s in range(remainder):
                pre_time = datetime_now()
                self._stimulate_once(
                    stim_elec, amp, dur, pol,
                    trigger_key=0, phase_label="active"
                )
                stim_times.append(datetime_now().isoformat())
                self._wait(self.active_inter_stim_s)

            self._active_results.append({
                "pair_key": pair_key,
                "stim_electrode": stim_elec,
                "resp_electrode": resp_elec,
                "stim_times": stim_times,
                "spike_latencies": spike_latencies,
                "num_stims": len(stim_times),
                "num_responses": len(spike_latencies),
            })

            logger.info("  Pair %s: %d stims, %d response latencies collected",
                        pair_key, len(stim_times), len(spike_latencies))

    def _compute_synaptic_delays(self) -> None:
        logger.info("Computing synaptic delays via distribution fitting")

        for result in self._active_results:
            pair_key = result["pair_key"]
            latencies = result["spike_latencies"]

            if len(latencies) < 3:
                pair_info = None
                for p in self._selected_pairs:
                    if f"{p['stim_electrode']}->{p['resp_electrode']}" == pair_key:
                        pair_info = p
                        break
                if pair_info:
                    self._synaptic_delays[pair_key] = pair_info["median_latency_ms"]
                else:
                    self._synaptic_delays[pair_key] = 20.0
                logger.info("  Pair %s: insufficient data, using scan median %.2f ms",
                            pair_key, self._synaptic_delays[pair_key])
                continue

            lat_arr = np.array(latencies)
            median_lat = float(np.median(lat_arr))
            mean_lat = float(np.mean(lat_arr))

            best_fit = "unimodal_gaussian"
            best_delay = median_lat

            n = len(lat_arr)
            variance = float(np.var(lat_arr))
            if variance > 0:
                log_likelihood_gauss = -0.5 * n * (1.0 + math.log(2.0 * math.pi * variance))
                k_gauss = 2
                aic_gauss = 2 * k_gauss - 2 * log_likelihood_gauss
            else:
                aic_gauss = float("inf")

            if n >= 5:
                sorted_lat = sorted(lat_arr)
                mid = n // 2
                g1 = sorted_lat[:mid]
                g2 = sorted_lat[mid:]
                if len(g1) > 1 and len(g2) > 1:
                    m1 = np.mean(g1)
                    m2 = np.mean(g2)
                    v1 = max(np.var(g1), 0.01)
                    v2 = max(np.var(g2), 0.01)
                    w1 = len(g1) / n
                    w2 = len(g2) / n
                    ll_bimodal = 0.0
                    for x in lat_arr:
                        p1 = w1 * math.exp(-0.5 * ((x - m1) ** 2) / v1) / math.sqrt(2 * math.pi * v1)
                        p2 = w2 * math.exp(-0.5 * ((x - m2) ** 2) / v2) / math.sqrt(2 * math.pi * v2)
                        ll_bimodal += math.log(max(p1 + p2, 1e-300))
                    k_bimodal = 5
                    aic_bimodal = 2 * k_bimodal - 2 * ll_bimodal

                    if aic_bimodal < aic_gauss:
                        best_fit = "bimodal_gaussian"
                        best_delay = float(m1) if w1 >= w2 else float(m2)

            if mean_lat > 0:
                shape_gamma = (mean_lat ** 2) / max(variance, 0.01)
                scale_gamma = max(variance, 0.01) / mean_lat
                if shape_gamma > 0 and scale_gamma > 0:
                    ll_gamma = 0.0
                    for x in lat_arr:
                        if x > 0:
                            log_pdf = ((shape_gamma - 1) * math.log(x)
                                       - x / scale_gamma
                                       - shape_gamma * math.log(scale_gamma)
                                       - math.lgamma(shape_gamma))
                            ll_gamma += log_pdf
                    k_gamma = 2
                    aic_gamma = 2 * k_gamma - 2 * ll_gamma

                    if aic_gamma < aic_gauss:
                        best_fit = "gamma"
                        best_delay = (shape_gamma - 1) * scale_gamma if shape_gamma > 1 else mean_lat

            self._synaptic_delays[pair_key] = best_delay
            result["best_fit_model"] = best_fit
            result["estimated_delay_ms"] = best_delay

            logger.info("  Pair %s: best_fit=%s, delay=%.2f ms (median=%.2f, mean=%.2f)",
                        pair_key, best_fit, best_delay, median_lat, mean_lat)

    def _phase3_hebbian_learning(self) -> None:
        logger.info("=== Phase 3: Two-Electrode Hebbian Learning (STDP) ===")

        for pair_info in self._selected_pairs:
            stim_elec = pair_info["stim_electrode"]
            resp_elec = pair_info["resp_electrode"]
            amp = pair_info["amplitude"]
            dur = pair_info["duration"]
            pol = _polarity_enum(pair_info["polarity"])
            pair_key = f"{stim_elec}->{resp_elec}"
            delay_ms = self._synaptic_delays.get(pair_key, 20.0)

            logger.info("Hebbian STDP for pair %s, delay=%.2f ms", pair_key, delay_ms)

            testing_latencies = self._run_testing_phase(
                stim_elec, resp_elec, amp, dur, pol,
                self.hebbian_test_minutes, "testing"
            )

            self._run_learning_phase(
                stim_elec, resp_elec, amp, dur, pol,
                delay_ms, self.hebbian_learn_minutes
            )

            validation_latencies = self._run_testing_phase(
                stim_elec, resp_elec, amp, dur, pol,
                self.hebbian_validate_minutes, "validation"
            )

            emd = _compute_wasserstein_1d(testing_latencies, validation_latencies)

            testing_median = float(np.median(testing_latencies)) if testing_latencies else 0.0
            validation_median = float(np.median(validation_latencies)) if validation_latencies else 0.0

            self._hebbian_results.append({
                "pair_key": pair_key,
                "stim_electrode": stim_elec,
                "resp_electrode": resp_elec,
                "delay_ms": delay_ms,
                "testing_latencies_count": len(testing_latencies),
                "validation_latencies_count": len(validation_latencies),
                "testing_median_ms": testing_median,
                "validation_median_ms": validation_median,
                "earth_movers_distance": emd,
                "testing_latencies": testing_latencies[:200],
                "validation_latencies": validation_latencies[:200],
            })

            logger.info("  Pair %s: EMD=%.4f, test_median=%.2f, val_median=%.2f",
                        pair_key, emd, testing_median, validation_median)

    def _run_testing_phase(
        self,
        stim_elec: int,
        resp_elec: int,
        amp: float,
        dur: float,
        pol: StimPolarity,
        duration_minutes: float,
        phase_label: str,
    ) -> List[float]:
        logger.info("  %s phase (%.1f min) for elec %d->%d",
                     phase_label, duration_minutes, stim_elec, resp_elec)

        inter_stim_s = 1.0 / self.hebbian_test_rate_hz
        max_stims = int(duration_minutes * 60.0 * self.hebbian_test_rate_hz)
        max_stims = max(max_stims, 5)

        latencies = []
        phase_start = datetime_now()

        for i in range(max_stims):
            elapsed = (datetime_now() - phase_start).total_seconds()
            if elapsed >= duration_minutes * 60.0:
                break

            pre_time = datetime_now()
            self._stimulate_once(
                stim_elec, amp, dur, pol,
                trigger_key=0, phase_label=phase_label
            )
            self._wait(0.5)
            post_time = datetime_now()

            spike_df = self.database.get_spike_event_electrode(
                pre_time, post_time, resp_elec
            )
            if len(spike_df) > 0:
                time_col = "Time" if "Time" in spike_df.columns else "_time"
                if time_col in spike_df.columns:
                    for _, row in spike_df.iterrows():
                        try:
                            spike_t = pd.Timestamp(row[time_col])
                            pre_t = pd.Timestamp(pre_time)
                            if spike_t.tzinfo is None:
                                spike_t = spike_t.tz_localize("UTC")
                            if pre_t.tzinfo is None:
                                pre_t = pre_t.tz_localize("UTC")
                            latency_ms = (spike_t - pre_t).total_seconds() * 1000.0
                            if self.artifact_blanking_ms < latency_ms < self.response_window_ms:
                                latencies.append(latency_ms)
                        except Exception:
                            pass

            remaining_wait = inter_stim_s - 0.5
            if remaining_wait > 0:
                self._wait(remaining_wait)

        logger.info("    %s: %d latencies from %d stims",
                     phase_label, len(latencies), min(i + 1, max_stims))
        return latencies

    def _run_learning_phase(
        self,
        stim_elec: int,
        resp_elec: int,
        amp: float,
        dur: float,
        pol: StimPolarity,
        delay_ms: float,
        duration_minutes: float,
    ) -> None:
        logger.info("  Learning phase (%.1f min) for elec %d->%d, delay=%.2f ms",
                     duration_minutes, stim_elec, resp_elec, delay_ms)

        inter_pair_s = 1.0 / self.hebbian_learn_rate_hz
        max_pairs_count = int(duration_minutes * 60.0 * self.hebbian_learn_rate_hz)
        max_pairs_count = max(max_pairs_count, 5)

        delay_s = delay_ms / 1000.0

        probe_interval = 50

        phase_start = datetime_now()
        pair_count = 0

        for i in range(max_pairs_count):
            elapsed = (datetime_now() - phase_start).total_seconds()
            if elapsed >= duration_minutes * 60.0:
                break

            if i > 0 and i % probe_interval == 0:
                self._stimulate_once(
                    stim_elec, amp, dur, pol,
                    trigger_key=0, phase_label="learning_probe"
                )
                self._wait(2.0)

            self._stimulate_once(
                stim_elec, amp, dur, pol,
                trigger_key=0, phase_label="learning_pre"
            )

            self._wait(delay_s)

            self._stimulate_once(
                resp_elec, amp, dur, pol,
                trigger_key=1, phase_label="learning_post"
            )

            pair_count += 1

            remaining = inter_pair_s - delay_s
            if remaining > 0:
                self._wait(remaining)

        logger.info("    Learning: %d paired stimulations delivered", pair_count)

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
            "selected_pairs": self._selected_pairs,
            "synaptic_delays": self._synaptic_delays,
            "scan_results_count": len(self._scan_results),
            "active_results_count": len(self._active_results),
            "hebbian_results_count": len(self._hebbian_results),
        }
        saver.save_summary(summary)

        waveform_records = self._fetch_spike_waveforms(
            spike_df, recording_start, recording_stop
        )
        saver.save_spike_waveforms(waveform_records)

    def _fetch_spike_waveforms(
        self,
        spike_df: pd.DataFrame,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> list:
        waveform_records = []
        if spike_df.empty:
            return waveform_records

        electrode_col = None
        for candidate in ["channel", "index", "electrode"]:
            if candidate in spike_df.columns:
                electrode_col = candidate
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

        hebbian_summary = []
        for hr in self._hebbian_results:
            hebbian_summary.append({
                "pair_key": hr["pair_key"],
                "delay_ms": hr["delay_ms"],
                "testing_median_ms": hr["testing_median_ms"],
                "validation_median_ms": hr["validation_median_ms"],
                "earth_movers_distance": hr["earth_movers_distance"],
                "testing_count": hr["testing_latencies_count"],
                "validation_count": hr["validation_latencies_count"],
            })

        active_summary = []
        for ar in self._active_results:
            active_summary.append({
                "pair_key": ar["pair_key"],
                "num_stims": ar["num_stims"],
                "num_responses": ar["num_responses"],
                "estimated_delay_ms": ar.get("estimated_delay_ms", None),
                "best_fit_model": ar.get("best_fit_model", None),
            })

        results = {
            "status": "completed",
            "experiment_name": self.experiment_handle.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "total_stimulations": len(self._stimulation_log),
            "selected_pairs": [
                {"stim": p["stim_electrode"], "resp": p["resp_electrode"],
                 "rate": p["response_rate"]}
                for p in self._selected_pairs
            ],
            "synaptic_delays": self._synaptic_delays,
            "scan_responsive_combos": sum(
                1 for r in self._scan_results if r["hits"] >= self.scan_min_hits
            ),
            "active_experiment_summary": active_summary,
            "hebbian_experiment_summary": hebbian_summary,
        }

        return results

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
