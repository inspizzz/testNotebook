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
from neuroplatform import Experiment as NeuroPlatformExperiment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StimulationRecord:
    electrode_idx: int
    amplitude_ua: float
    duration_us: float
    polarity: str
    phase: str
    timestamp_utc: str
    trigger_key: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


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

    def save_summary(self, summary: Dict[str, Any]) -> Path:
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
    Full neuronal plasticity experiment pipeline:
      Stage 1 - Basic Excitability Scan
      Stage 2 - Active Electrode Experiment (1 Hz repeated stimulation + cross-correlograms)
      Stage 3 - Two-Electrode Hebbian (STDP) Learning Experiment
    """

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
        {"electrode_from": 13, "electrode_to": 11, "hits_k": 5, "median_latency_ms": 15.95,
         "stimulation": {"amplitude": 2.0, "duration": 300.0, "polarity": "PositiveFirst"}},
        {"electrode_from": 13, "electrode_to": 14, "hits_k": 5, "median_latency_ms": 20.16,
         "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 14, "electrode_to": 12, "hits_k": 5, "median_latency_ms": 22.37,
         "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 14, "electrode_to": 15, "hits_k": 5, "median_latency_ms": 13.2,
         "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 17, "electrode_to": 16, "hits_k": 5, "median_latency_ms": 21.56,
         "stimulation": {"amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 17, "electrode_to": 18, "hits_k": 5, "median_latency_ms": 11.19,
         "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 18, "electrode_to": 17, "hits_k": 5, "median_latency_ms": 24.71,
         "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 22, "electrode_to": 21, "hits_k": 5, "median_latency_ms": 13.58,
         "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 24, "electrode_to": 25, "hits_k": 5, "median_latency_ms": 13.18,
         "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 26, "electrode_to": 27, "hits_k": 5, "median_latency_ms": 13.88,
         "stimulation": {"amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 27, "electrode_to": 28, "hits_k": 5, "median_latency_ms": 14.51,
         "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 28, "electrode_to": 29, "hits_k": 5, "median_latency_ms": 17.74,
         "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 30, "electrode_to": 31, "hits_k": 5, "median_latency_ms": 19.34,
         "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 31, "electrode_to": 30, "hits_k": 5, "median_latency_ms": 18.87,
         "stimulation": {"amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst"}},
    ]

    DEEP_SCAN_PAIRS = [
        {"stim_electrode": 1, "resp_electrode": 2, "amplitude": 2.0, "duration": 300.0,
         "polarity": "NegativeFirst", "median_latency_ms": 23.83},
        {"stim_electrode": 6, "resp_electrode": 5, "amplitude": 2.0, "duration": 400.0,
         "polarity": "PositiveFirst", "median_latency_ms": 15.245},
        {"stim_electrode": 14, "resp_electrode": 12, "amplitude": 1.0, "duration": 400.0,
         "polarity": "NegativeFirst", "median_latency_ms": 22.72},
        {"stim_electrode": 14, "resp_electrode": 15, "amplitude": 2.0, "duration": 300.0,
         "polarity": "PositiveFirst", "median_latency_ms": 12.84},
        {"stim_electrode": 17, "resp_electrode": 16, "amplitude": 3.0, "duration": 400.0,
         "polarity": "PositiveFirst", "median_latency_ms": 21.58},
        {"stim_electrode": 18, "resp_electrode": 17, "amplitude": 1.0, "duration": 400.0,
         "polarity": "PositiveFirst", "median_latency_ms": 25.075},
        {"stim_electrode": 22, "resp_electrode": 21, "amplitude": 3.0, "duration": 400.0,
         "polarity": "PositiveFirst", "median_latency_ms": 14.03},
        {"stim_electrode": 24, "resp_electrode": 25, "amplitude": 3.0, "duration": 400.0,
         "polarity": "NegativeFirst", "median_latency_ms": 13.17},
        {"stim_electrode": 30, "resp_electrode": 31, "amplitude": 3.0, "duration": 400.0,
         "polarity": "NegativeFirst", "median_latency_ms": 19.18},
        {"stim_electrode": 9, "resp_electrode": 10, "amplitude": 3.0, "duration": 400.0,
         "polarity": "NegativeFirst", "median_latency_ms": 11.035},
    ]

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
        active_total_repeats: int = 100,
        active_group_size: int = 10,
        active_group_pause_s: float = 5.0,
        active_isi_s: float = 1.0,
        stdp_testing_duration_s: float = 1200.0,
        stdp_learning_duration_s: float = 3000.0,
        stdp_validation_duration_s: float = 1200.0,
        stdp_probe_interval_s: float = 10.0,
        stdp_conditioning_interval_s: float = 2.0,
        max_electrode_pairs: int = 5,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.scan_amplitudes = scan_amplitudes if scan_amplitudes is not None else [1.0, 2.0, 3.0]
        self.scan_durations = scan_durations if scan_durations is not None else [100.0, 200.0, 300.0, 400.0]
        self.scan_repeats = scan_repeats
        self.scan_inter_stim_s = scan_inter_stim_s
        self.scan_inter_channel_s = scan_inter_channel_s

        self.active_total_repeats = active_total_repeats
        self.active_group_size = active_group_size
        self.active_group_pause_s = active_group_pause_s
        self.active_isi_s = active_isi_s

        self.stdp_testing_duration_s = stdp_testing_duration_s
        self.stdp_learning_duration_s = stdp_learning_duration_s
        self.stdp_validation_duration_s = stdp_validation_duration_s
        self.stdp_probe_interval_s = stdp_probe_interval_s
        self.stdp_conditioning_interval_s = stdp_conditioning_interval_s
        self.max_electrode_pairs = max_electrode_pairs

        self.np_experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []

        self._scan_results: Dict[str, Any] = {}
        self._responsive_pairs: List[Dict[str, Any]] = []
        self._active_electrode_results: Dict[str, Any] = {}
        self._correlograms: Dict[str, Any] = {}
        self._stdp_results: Dict[str, Any] = {}

        self._recording_start: Optional[datetime] = None
        self._recording_stop: Optional[datetime] = None

    def _wait(self, seconds: float) -> None:
        wait(seconds)

    def run(self) -> Dict[str, Any]:
        try:
            logger.info("Initialising hardware connections")

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

            logger.info("=== Stage 1: Basic Excitability Scan ===")
            self._phase_excitability_scan()

            logger.info("=== Stage 2: Active Electrode Experiment ===")
            self._phase_active_electrode_experiment()

            logger.info("=== Stage 3: Hebbian STDP Experiment ===")
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
                logger.error("Error saving data after failure: %s", save_exc)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _phase_excitability_scan(self) -> None:
        logger.info("Starting excitability scan")
        polarities = [StimPolarity.NegativeFirst, StimPolarity.PositiveFirst]
        polarity_names = ["NegativeFirst", "PositiveFirst"]

        available_electrodes = list(self.np_experiment.electrodes)
        scan_results = {}

        for electrode_idx in available_electrodes:
            electrode_responses = []
            for amplitude in self.scan_amplitudes:
                for duration in self.scan_durations:
                    for pol, pol_name in zip(polarities, polarity_names):
                        hits = 0
                        latencies = []
                        for rep in range(self.scan_repeats):
                            t_stim = datetime_now()
                            self._send_single_stim(
                                electrode_idx=electrode_idx,
                                amplitude_ua=amplitude,
                                duration_us=duration,
                                polarity=pol,
                                trigger_key=0,
                                phase_label="scan",
                            )
                            self._wait(0.05)
                            t_query_start = t_stim
                            t_query_stop = datetime_now()
                            try:
                                spike_df = self.database.get_spike_event(
                                    t_query_start, t_query_stop,
                                    self.np_experiment.exp_name
                                )
                                if not spike_df.empty:
                                    window_ms = 50.0
                                    stim_ts = t_stim.timestamp() * 1000.0
                                    for _, row in spike_df.iterrows():
                                        try:
                                            spike_ts = pd.Timestamp(row["Time"]).timestamp() * 1000.0
                                        except Exception:
                                            continue
                                        latency = spike_ts - stim_ts
                                        if 1.0 < latency < window_ms:
                                            hits += 1
                                            latencies.append(latency)
                                            break
                            except Exception as exc:
                                logger.warning("Spike query error during scan: %s", exc)
                            if rep < self.scan_repeats - 1:
                                self._wait(self.scan_inter_stim_s)

                        if hits >= 3:
                            median_lat = float(np.median(latencies)) if latencies else 0.0
                            electrode_responses.append({
                                "electrode_from": electrode_idx,
                                "amplitude": amplitude,
                                "duration": duration,
                                "polarity": pol_name,
                                "hits": hits,
                                "median_latency_ms": median_lat,
                            })

            scan_results[electrode_idx] = electrode_responses
            self._wait(self.scan_inter_channel_s)

        self._scan_results = scan_results
        logger.info("Excitability scan complete. Responsive electrodes: %d",
                    sum(1 for v in scan_results.values() if len(v) > 0))

        self._responsive_pairs = self._select_responsive_pairs_from_scan(scan_results)
        logger.info("Responsive pairs identified: %d", len(self._responsive_pairs))

    def _select_responsive_pairs_from_scan(
        self, scan_results: Dict[int, List[Dict]]
    ) -> List[Dict[str, Any]]:
        pairs = []
        seen = set()
        for conn in self.RELIABLE_CONNECTIONS:
            ef = conn["electrode_from"]
            et = conn["electrode_to"]
            key = (ef, et)
            if key in seen:
                continue
            seen.add(key)
            stim = conn["stimulation"]
            pairs.append({
                "electrode_from": ef,
                "electrode_to": et,
                "amplitude": stim["amplitude"],
                "duration": stim["duration"],
                "polarity": stim["polarity"],
                "median_latency_ms": conn["median_latency_ms"],
                "hits_k": conn["hits_k"],
            })
            if len(pairs) >= self.max_electrode_pairs * 4:
                break
        return pairs

    def _phase_active_electrode_experiment(self) -> None:
        logger.info("Starting active electrode experiment")

        pairs_to_use = self._responsive_pairs[:self.max_electrode_pairs]
        if not pairs_to_use:
            pairs_to_use = [
                {
                    "electrode_from": p["stim_electrode"],
                    "electrode_to": p["resp_electrode"],
                    "amplitude": p["amplitude"],
                    "duration": p["duration"],
                    "polarity": p["polarity"],
                    "median_latency_ms": p["median_latency_ms"],
                }
                for p in self.DEEP_SCAN_PAIRS[:self.max_electrode_pairs]
            ]

        active_results = {}
        stim_times_per_pair: Dict[str, List[float]] = {}

        for pair in pairs_to_use:
            ef = pair["electrode_from"]
            et = pair["electrode_to"]
            amplitude = pair["amplitude"]
            duration = pair["duration"]
            polarity_str = pair["polarity"]
            polarity = StimPolarity.NegativeFirst if polarity_str == "NegativeFirst" else StimPolarity.PositiveFirst
            pair_key = f"{ef}->{et}"

            logger.info("Active electrode experiment for pair %s", pair_key)
            stim_times = []
            response_count = 0
            total_stims = 0

            num_groups = self.active_total_repeats // self.active_group_size

            for group_idx in range(num_groups):
                for stim_idx in range(self.active_group_size):
                    t_stim = datetime_now()
                    self._send_single_stim(
                        electrode_idx=ef,
                        amplitude_ua=amplitude,
                        duration_us=duration,
                        polarity=polarity,
                        trigger_key=0,
                        phase_label="active",
                    )
                    stim_times.append(t_stim.timestamp())
                    total_stims += 1
                    self._wait(0.05)

                    try:
                        t_query_stop = datetime_now()
                        spike_df = self.database.get_spike_event(
                            t_stim, t_query_stop,
                            self.np_experiment.exp_name
                        )
                        if not spike_df.empty:
                            window_ms = 100.0
                            stim_ts_ms = t_stim.timestamp() * 1000.0
                            for _, row in spike_df.iterrows():
                                try:
                                    spike_ts_ms = pd.Timestamp(row["Time"]).timestamp() * 1000.0
                                    lat = spike_ts_ms - stim_ts_ms
                                    if 1.0 < lat < window_ms:
                                        response_count += 1
                                        break
                                except Exception:
                                    continue
                    except Exception as exc:
                        logger.warning("Spike query error in active experiment: %s", exc)

                    if stim_idx < self.active_group_size - 1:
                        self._wait(self.active_isi_s - 0.05)

                if group_idx < num_groups - 1:
                    self._wait(self.active_group_pause_s)

            stim_times_per_pair[pair_key] = stim_times
            active_results[pair_key] = {
                "electrode_from": ef,
                "electrode_to": et,
                "total_stims": total_stims,
                "response_count": response_count,
                "response_rate": response_count / total_stims if total_stims > 0 else 0.0,
                "stim_times_count": len(stim_times),
            }
            logger.info("Pair %s: %d/%d responses", pair_key, response_count, total_stims)

        self._active_electrode_results = active_results

        logger.info("Computing cross-correlograms")
        self._correlograms = self._compute_cross_correlograms(
            pairs_to_use, stim_times_per_pair
        )

    def _compute_cross_correlograms(
        self,
        pairs: List[Dict],
        stim_times_per_pair: Dict[str, List[float]],
    ) -> Dict[str, Any]:
        correlograms = {}
        bin_width_ms = 4.0
        window_ms = 100.0
        bins = np.arange(-window_ms, window_ms + bin_width_ms, bin_width_ms)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0

        for pair in pairs:
            ef = pair["electrode_from"]
            et = pair["electrode_to"]
            pair_key = f"{ef}->{et}"
            stim_times = stim_times_per_pair.get(pair_key, [])

            if not stim_times:
                correlograms[pair_key] = {
                    "bin_centers_ms": bin_centers.tolist(),
                    "counts": [0] * len(bin_centers),
                    "peak_latency_ms": pair.get("median_latency_ms", 20.0),
                    "note": "no_stim_times",
                }
                continue

            counts = np.zeros(len(bin_centers), dtype=int)
            t_start_query = datetime_now() - timedelta(
                seconds=self.active_total_repeats * self.active_isi_s
                + self.active_total_repeats / self.active_group_size * self.active_group_pause_s
                + 60.0
            )
            t_stop_query = datetime_now()

            try:
                spike_df = self.database.get_spike_event(
                    t_start_query, t_stop_query,
                    self.np_experiment.exp_name
                )
                if not spike_df.empty and "channel" in spike_df.columns:
                    resp_spikes = spike_df[spike_df["channel"] == et]
                    resp_spike_times = []
                    for _, row in resp_spikes.iterrows():
                        try:
                            resp_spike_times.append(pd.Timestamp(row["Time"]).timestamp() * 1000.0)
                        except Exception:
                            continue

                    for st in stim_times:
                        st_ms = st * 1000.0
                        for rsp_ms in resp_spike_times:
                            lag = rsp_ms - st_ms
                            if -window_ms <= lag <= window_ms:
                                bin_idx = int((lag + window_ms) / bin_width_ms)
                                if 0 <= bin_idx < len(counts):
                                    counts[bin_idx] += 1
            except Exception as exc:
                logger.warning("Error computing correlogram for %s: %s", pair_key, exc)

            peak_idx = int(np.argmax(counts)) if counts.sum() > 0 else 0
            peak_latency = float(bin_centers[peak_idx]) if counts.sum() > 0 else pair.get("median_latency_ms", 20.0)

            correlograms[pair_key] = {
                "bin_centers_ms": bin_centers.tolist(),
                "counts": counts.tolist(),
                "peak_latency_ms": peak_latency,
            }
            logger.info("Correlogram for %s: peak at %.2f ms", pair_key, peak_latency)

        return correlograms

    def _phase_stdp_experiment(self) -> None:
        logger.info("Starting STDP experiment")

        stdp_pairs = self._select_stdp_pairs()
        if not stdp_pairs:
            logger.warning("No STDP pairs available, skipping STDP phase")
            self._stdp_results = {"status": "skipped", "reason": "no_pairs"}
            return

        stdp_results = {}

        for pair_info in stdp_pairs[:2]:
            ef = pair_info["electrode_from"]
            et = pair_info["electrode_to"]
            amplitude = pair_info["amplitude"]
            duration = pair_info["duration"]
            polarity_str = pair_info["polarity"]
            polarity = StimPolarity.NegativeFirst if polarity_str == "NegativeFirst" else StimPolarity.PositiveFirst
            hebbian_delay_ms = pair_info.get("hebbian_delay_ms", 20.0)
            pair_key = f"{ef}->{et}"

            logger.info("STDP pair %s, Hebbian delay=%.1f ms", pair_key, hebbian_delay_ms)

            phase_results = {}

            logger.info("STDP Testing Phase (pre-conditioning) for %s", pair_key)
            pre_responses = self._stdp_probe_phase(
                stim_electrode=ef,
                resp_electrode=et,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=polarity,
                phase_duration_s=self.stdp_testing_duration_s,
                probe_interval_s=self.stdp_probe_interval_s,
                phase_label="stdp_pre",
            )
            phase_results["pre_conditioning"] = pre_responses

            logger.info("STDP Learning Phase for %s", pair_key)
            learning_results = self._stdp_learning_phase(
                stim_electrode=ef,
                resp_electrode=et,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=polarity,
                phase_duration_s=self.stdp_learning_duration_s,
                conditioning_interval_s=self.stdp_conditioning_interval_s,
                hebbian_delay_ms=hebbian_delay_ms,
                phase_label="stdp_learning",
            )
            phase_results["learning"] = learning_results

            logger.info("STDP Validation Phase (post-conditioning) for %s", pair_key)
            post_responses = self._stdp_probe_phase(
                stim_electrode=ef,
                resp_electrode=et,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=polarity,
                phase_duration_s=self.stdp_validation_duration_s,
                probe_interval_s=self.stdp_probe_interval_s,
                phase_label="stdp_post",
            )
            phase_results["post_conditioning"] = post_responses

            pre_rate = pre_responses.get("response_rate", 0.0)
            post_rate = post_responses.get("response_rate", 0.0)
            delta_rate = post_rate - pre_rate

            pre_lat = pre_responses.get("mean_latency_ms", 0.0)
            post_lat = post_responses.get("mean_latency_ms", 0.0)
            delta_lat = post_lat - pre_lat

            stdp_results[pair_key] = {
                "electrode_from": ef,
                "electrode_to": et,
                "hebbian_delay_ms": hebbian_delay_ms,
                "pre_response_rate": pre_rate,
                "post_response_rate": post_rate,
                "delta_response_rate": delta_rate,
                "pre_mean_latency_ms": pre_lat,
                "post_mean_latency_ms": post_lat,
                "delta_latency_ms": delta_lat,
                "phases": phase_results,
            }
            logger.info(
                "STDP result for %s: delta_rate=%.3f, delta_lat=%.2f ms",
                pair_key, delta_rate, delta_lat
            )

        self._stdp_results = stdp_results

    def _select_stdp_pairs(self) -> List[Dict[str, Any]]:
        stdp_pairs = []
        seen = set()

        for ds_pair in self.DEEP_SCAN_PAIRS:
            ef = ds_pair["stim_electrode"]
            et = ds_pair["resp_electrode"]
            key = (ef, et)
            if key in seen:
                continue
            seen.add(key)

            pair_key = f"{ef}->{et}"
            peak_latency = ds_pair["median_latency_ms"]
            if pair_key in self._correlograms:
                ccg_peak = self._correlograms[pair_key].get("peak_latency_ms", peak_latency)
                if ccg_peak > 1.0:
                    peak_latency = ccg_peak

            hebbian_delay_ms = max(5.0, min(peak_latency * 0.8, 25.0))

            stdp_pairs.append({
                "electrode_from": ef,
                "electrode_to": et,
                "amplitude": ds_pair["amplitude"],
                "duration": ds_pair["duration"],
                "polarity": ds_pair["polarity"],
                "median_latency_ms": peak_latency,
                "hebbian_delay_ms": hebbian_delay_ms,
            })

        if not stdp_pairs and self._responsive_pairs:
            for rp in self._responsive_pairs[:2]:
                ef = rp["electrode_from"]
                et = rp["electrode_to"]
                key = (ef, et)
                if key in seen:
                    continue
                seen.add(key)
                lat = rp.get("median_latency_ms", 20.0)
                stdp_pairs.append({
                    "electrode_from": ef,
                    "electrode_to": et,
                    "amplitude": rp["amplitude"],
                    "duration": rp["duration"],
                    "polarity": rp["polarity"],
                    "median_latency_ms": lat,
                    "hebbian_delay_ms": max(5.0, min(lat * 0.8, 25.0)),
                })

        return stdp_pairs

    def _stdp_probe_phase(
        self,
        stim_electrode: int,
        resp_electrode: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity,
        phase_duration_s: float,
        probe_interval_s: float,
        phase_label: str,
    ) -> Dict[str, Any]:
        phase_start = datetime_now()
        phase_end_target = phase_start.timestamp() + phase_duration_s

        response_count = 0
        total_probes = 0
        latencies = []

        probe_amplitude = min(amplitude_ua, 1.0)
        probe_duration = duration_us

        while datetime_now().timestamp() < phase_end_target:
            t_stim = datetime_now()
            self._send_single_stim(
                electrode_idx=stim_electrode,
                amplitude_ua=probe_amplitude,
                duration_us=probe_duration,
                polarity=polarity,
                trigger_key=0,
                phase_label=phase_label,
            )
            total_probes += 1
            self._wait(0.05)

            try:
                t_query_stop = datetime_now()
                spike_df = self.database.get_spike_event(
                    t_stim, t_query_stop,
                    self.np_experiment.exp_name
                )
                if not spike_df.empty and "channel" in spike_df.columns:
                    resp_spikes = spike_df[spike_df["channel"] == resp_electrode]
                    stim_ts_ms = t_stim.timestamp() * 1000.0
                    for _, row in resp_spikes.iterrows():
                        try:
                            spike_ts_ms = pd.Timestamp(row["Time"]).timestamp() * 1000.0
                            lat = spike_ts_ms - stim_ts_ms
                            if 1.0 < lat < 80.0:
                                response_count += 1
                                latencies.append(lat)
                                break
                        except Exception:
                            continue
            except Exception as exc:
                logger.warning("Probe spike query error: %s", exc)

            elapsed = datetime_now().timestamp() - phase_start.timestamp()
            remaining = phase_duration_s - elapsed
            if remaining <= 0:
                break
            sleep_time = min(probe_interval_s - 0.05, remaining)
            if sleep_time > 0:
                self._wait(sleep_time)

        response_rate = response_count / total_probes if total_probes > 0 else 0.0
        mean_latency = float(np.mean(latencies)) if latencies else 0.0

        return {
            "total_probes": total_probes,
            "response_count": response_count,
            "response_rate": response_rate,
            "mean_latency_ms": mean_latency,
            "latencies_ms": latencies,
        }

    def _stdp_learning_phase(
        self,
        stim_electrode: int,
        resp_electrode: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity,
        phase_duration_s: float,
        conditioning_interval_s: float,
        hebbian_delay_ms: float,
        phase_label: str,
    ) -> Dict[str, Any]:
        phase_start = datetime_now()
        phase_end_target = phase_start.timestamp() + phase_duration_s

        conditioning_count = 0
        paired_count = 0

        hebbian_delay_s = hebbian_delay_ms / 1000.0

        while datetime_now().timestamp() < phase_end_target:
            t_stim_a = datetime_now()
            self._send_single_stim(
                electrode_idx=stim_electrode,
                amplitude_ua=amplitude_ua,
                duration_us=duration_us,
                polarity=polarity,
                trigger_key=0,
                phase_label=phase_label + "_pre",
            )
            conditioning_count += 1

            self._wait(hebbian_delay_s)

            self._send_single_stim(
                electrode_idx=resp_electrode,
                amplitude_ua=amplitude_ua,
                duration_us=duration_us,
                polarity=polarity,
                trigger_key=1,
                phase_label=phase_label + "_post",
            )
            paired_count += 1

            elapsed = datetime_now().timestamp() - phase_start.timestamp()
            remaining = phase_duration_s - elapsed
            if remaining <= 0:
                break
            sleep_time = min(conditioning_interval_s - hebbian_delay_s - 0.01, remaining)
            if sleep_time > 0:
                self._wait(sleep_time)

        return {
            "conditioning_count": conditioning_count,
            "paired_count": paired_count,
            "hebbian_delay_ms": hebbian_delay_ms,
            "phase_duration_s": phase_duration_s,
        }

    def _send_single_stim(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity,
        trigger_key: int = 0,
        phase_label: str = "stim",
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
        self._wait(0.01)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

        polarity_str = "NegativeFirst" if polarity == StimPolarity.NegativeFirst else "PositiveFirst"
        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            polarity=polarity_str,
            phase=phase_label,
            timestamp_utc=datetime_now().isoformat(),
            trigger_key=trigger_key,
        ))

    def _save_all(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> None:
        fs_name = getattr(self.np_experiment, "exp_name", "unknown") if self.np_experiment else "unknown"
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

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "responsive_pairs_count": len(self._responsive_pairs),
            "active_electrode_pairs_count": len(self._active_electrode_results),
            "correlograms_count": len(self._correlograms),
            "stdp_pairs_count": len(self._stdp_results) if isinstance(self._stdp_results, dict) else 0,
        }
        saver.save_summary(summary)

        waveform_records = self._fetch_spike_waveforms(
            fs_name, spike_df, recording_start, recording_stop
        )
        saver.save_spike_waveforms(waveform_records)

    def _fetch_spike_waveforms(
        self,
        fs_name: str,
        spike_df: pd.DataFrame,
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

        stdp_summary = {}
        if isinstance(self._stdp_results, dict):
            for pk, pv in self._stdp_results.items():
                if isinstance(pv, dict) and "delta_response_rate" in pv:
                    stdp_summary[pk] = {
                        "delta_response_rate": pv.get("delta_response_rate", 0.0),
                        "delta_latency_ms": pv.get("delta_latency_ms", 0.0),
                        "hebbian_delay_ms": pv.get("hebbian_delay_ms", 0.0),
                    }

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": getattr(self.np_experiment, "exp_name", "unknown"),
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "total_stimulations": len(self._stimulation_log),
            "responsive_pairs_found": len(self._responsive_pairs),
            "active_electrode_results": self._active_electrode_results,
            "correlograms_computed": len(self._correlograms),
            "stdp_results_summary": stdp_summary,
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
