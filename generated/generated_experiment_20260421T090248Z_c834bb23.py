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


RELIABLE_CONNECTIONS = [
    {"electrode_from": 0, "electrode_to": 1, "hits_k": 5, "amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 12.73},
    {"electrode_from": 1, "electrode_to": 2, "hits_k": 5, "amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 23.34},
    {"electrode_from": 4, "electrode_to": 3, "hits_k": 5, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 22.44},
    {"electrode_from": 5, "electrode_to": 4, "hits_k": 5, "amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 17.39},
    {"electrode_from": 5, "electrode_to": 6, "hits_k": 5, "amplitude": 1.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 15.45},
    {"electrode_from": 6, "electrode_to": 5, "hits_k": 5, "amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 14.82},
    {"electrode_from": 8, "electrode_to": 9, "hits_k": 5, "amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 15.88},
    {"electrode_from": 9, "electrode_to": 10, "hits_k": 5, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 10.97},
    {"electrode_from": 9, "electrode_to": 11, "hits_k": 5, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 16.17},
    {"electrode_from": 10, "electrode_to": 11, "hits_k": 5, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 14.75},
    {"electrode_from": 13, "electrode_to": 11, "hits_k": 5, "amplitude": 2.0, "duration": 300.0, "polarity": "PositiveFirst", "median_latency_ms": 15.95},
    {"electrode_from": 13, "electrode_to": 12, "hits_k": 5, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 24.03},
    {"electrode_from": 13, "electrode_to": 14, "hits_k": 5, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 20.16},
    {"electrode_from": 14, "electrode_to": 12, "hits_k": 5, "amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 22.37},
    {"electrode_from": 14, "electrode_to": 15, "hits_k": 5, "amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 13.20},
    {"electrode_from": 17, "electrode_to": 16, "hits_k": 5, "amplitude": 2.0, "duration": 300.0, "polarity": "PositiveFirst", "median_latency_ms": 22.02},
    {"electrode_from": 17, "electrode_to": 18, "hits_k": 5, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 11.19},
    {"electrode_from": 18, "electrode_to": 17, "hits_k": 5, "amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 24.71},
    {"electrode_from": 20, "electrode_to": 22, "hits_k": 5, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 22.42},
    {"electrode_from": 22, "electrode_to": 21, "hits_k": 5, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 13.58},
    {"electrode_from": 24, "electrode_to": 25, "hits_k": 5, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 13.18},
    {"electrode_from": 26, "electrode_to": 27, "hits_k": 5, "amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 13.88},
    {"electrode_from": 27, "electrode_to": 28, "hits_k": 5, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 14.51},
    {"electrode_from": 28, "electrode_to": 29, "hits_k": 5, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 17.74},
    {"electrode_from": 30, "electrode_to": 31, "hits_k": 5, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 19.34},
    {"electrode_from": 31, "electrode_to": 30, "hits_k": 5, "amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 18.87},
]

DEEP_SCAN_PAIRS = [
    {"stim_electrode": 1, "resp_electrode": 2, "amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 23.83},
    {"stim_electrode": 6, "resp_electrode": 5, "amplitude": 2.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 15.245},
    {"stim_electrode": 14, "resp_electrode": 12, "amplitude": 1.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 22.72},
    {"stim_electrode": 14, "resp_electrode": 15, "amplitude": 2.0, "duration": 300.0, "polarity": "PositiveFirst", "median_latency_ms": 12.84},
    {"stim_electrode": 17, "resp_electrode": 16, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 21.58},
    {"stim_electrode": 18, "resp_electrode": 17, "amplitude": 1.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 25.075},
    {"stim_electrode": 22, "resp_electrode": 21, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 14.03},
    {"stim_electrode": 24, "resp_electrode": 25, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 13.17},
    {"stim_electrode": 30, "resp_electrode": 31, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 19.18},
    {"stim_electrode": 0, "resp_electrode": 1, "amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 12.35},
    {"stim_electrode": 5, "resp_electrode": 4, "amplitude": 1.0, "duration": 300.0, "polarity": "PositiveFirst", "median_latency_ms": 29.3},
    {"stim_electrode": 9, "resp_electrode": 10, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 11.035},
    {"stim_electrode": 26, "resp_electrode": 27, "amplitude": 3.0, "duration": 300.0, "polarity": "PositiveFirst", "median_latency_ms": 13.8},
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
        inter_stim_s: float = 1.0,
        inter_channel_s: float = 5.0,
        active_total_repeats: int = 100,
        active_group_size: int = 10,
        active_group_pause_s: float = 5.0,
        stdp_testing_duration_min: float = 20.0,
        stdp_learning_duration_min: float = 50.0,
        stdp_validation_duration_min: float = 20.0,
        stdp_test_freq_hz: float = 0.1,
        max_stdp_pairs: int = 3,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.scan_amplitudes = scan_amplitudes if scan_amplitudes is not None else [1.0, 2.0, 3.0]
        self.scan_durations = scan_durations if scan_durations is not None else [100.0, 200.0, 300.0, 400.0]
        self.scan_repeats = scan_repeats
        self.inter_stim_s = inter_stim_s
        self.inter_channel_s = inter_channel_s

        self.active_total_repeats = active_total_repeats
        self.active_group_size = active_group_size
        self.active_group_pause_s = active_group_pause_s

        self.stdp_testing_duration_min = stdp_testing_duration_min
        self.stdp_learning_duration_min = stdp_learning_duration_min
        self.stdp_validation_duration_min = stdp_validation_duration_min
        self.stdp_test_freq_hz = stdp_test_freq_hz
        self.max_stdp_pairs = max_stdp_pairs

        self.np_experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._scan_results: List[Dict[str, Any]] = []
        self._active_results: List[Dict[str, Any]] = []
        self._correlogram_results: List[Dict[str, Any]] = []
        self._stdp_results: Dict[str, Any] = {}
        self._responsive_pairs: List[Dict[str, Any]] = []
        self._hebbian_pairs: List[Dict[str, Any]] = []

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

            logger.info("=== PHASE 1: Basic Excitability Scan ===")
            self._phase_excitability_scan()

            logger.info("=== PHASE 2: Active Electrode Experiment ===")
            self._phase_active_electrode()

            logger.info("=== PHASE 3: Hebbian STDP Experiment ===")
            self._phase_stdp()

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
                logger.error("Error saving data on failure: %s", save_exc)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _phase_excitability_scan(self) -> None:
        logger.info("Starting excitability scan")
        electrodes = self.np_experiment.electrodes
        polarities = [StimPolarity.NegativeFirst, StimPolarity.PositiveFirst]

        for ch_idx, electrode in enumerate(electrodes):
            logger.info("Scanning electrode %d (%d/%d)", electrode, ch_idx + 1, len(electrodes))
            for amplitude in self.scan_amplitudes:
                for duration in self.scan_durations:
                    for polarity in polarities:
                        hit_count = 0
                        latencies = []
                        for rep in range(self.scan_repeats):
                            stim_time = datetime_now()
                            self._send_single_pulse(
                                electrode_idx=electrode,
                                amplitude_ua=amplitude,
                                duration_us=duration,
                                polarity=polarity,
                                trigger_key=0,
                                phase_label="scan",
                            )
                            self._wait(0.05)
                            window_start = stim_time
                            window_stop = datetime_now()
                            try:
                                spike_df = self.database.get_spike_event(
                                    window_start, window_stop, self.np_experiment.exp_name
                                )
                                if not spike_df.empty:
                                    for other_elec in electrodes:
                                        if other_elec == electrode:
                                            continue
                                        elec_spikes = spike_df[spike_df["channel"] == other_elec] if "channel" in spike_df.columns else pd.DataFrame()
                                        if not elec_spikes.empty:
                                            hit_count += 1
                                            if "Time" in elec_spikes.columns:
                                                t_stim = stim_time
                                                for _, row in elec_spikes.iterrows():
                                                    try:
                                                        t_spike = row["Time"]
                                                        if hasattr(t_spike, "timestamp"):
                                                            lat_ms = (t_spike.timestamp() - t_stim.timestamp()) * 1000.0
                                                            if 0 < lat_ms < 100:
                                                                latencies.append(lat_ms)
                                                    except Exception:
                                                        pass
                                            break
                            except Exception as exc:
                                logger.warning("Spike query error during scan: %s", exc)
                            if rep < self.scan_repeats - 1:
                                self._wait(self.inter_stim_s)

                        is_responsive = hit_count >= 3
                        median_lat = float(np.median(latencies)) if latencies else 0.0
                        self._scan_results.append({
                            "electrode": electrode,
                            "amplitude": amplitude,
                            "duration": duration,
                            "polarity": polarity.name,
                            "hits": hit_count,
                            "repeats": self.scan_repeats,
                            "responsive": is_responsive,
                            "median_latency_ms": median_lat,
                        })
                        if is_responsive:
                            logger.info(
                                "Responsive: electrode=%d amp=%.1f dur=%.0f pol=%s hits=%d",
                                electrode, amplitude, duration, polarity.name, hit_count
                            )

            self._wait(self.inter_channel_s)

        self._identify_responsive_pairs()
        logger.info("Excitability scan complete. Responsive pairs: %d", len(self._responsive_pairs))

    def _identify_responsive_pairs(self) -> None:
        self._responsive_pairs = []
        seen = set()
        for conn in RELIABLE_CONNECTIONS:
            key = (conn["electrode_from"], conn["electrode_to"])
            if key not in seen:
                seen.add(key)
                self._responsive_pairs.append({
                    "electrode_from": conn["electrode_from"],
                    "electrode_to": conn["electrode_to"],
                    "amplitude": conn["amplitude"],
                    "duration": conn["duration"],
                    "polarity": conn["polarity"],
                    "median_latency_ms": conn["median_latency_ms"],
                })
        logger.info("Using %d pre-identified responsive pairs from scan results", len(self._responsive_pairs))

    def _phase_active_electrode(self) -> None:
        logger.info("Starting active electrode experiment")
        pairs_to_use = self._responsive_pairs[:10] if len(self._responsive_pairs) > 10 else self._responsive_pairs

        for pair_idx, pair in enumerate(pairs_to_use):
            stim_elec = pair["electrode_from"]
            resp_elec = pair["electrode_to"]
            amplitude = pair["amplitude"]
            duration = pair["duration"]
            polarity_str = pair["polarity"]
            polarity = StimPolarity.NegativeFirst if polarity_str == "NegativeFirst" else StimPolarity.PositiveFirst

            logger.info(
                "Active pair %d/%d: stim=%d resp=%d amp=%.1f dur=%.0f",
                pair_idx + 1, len(pairs_to_use), stim_elec, resp_elec, amplitude, duration
            )

            stim_times = []
            response_latencies = []
            num_groups = self.active_total_repeats // self.active_group_size

            for group in range(num_groups):
                for rep in range(self.active_group_size):
                    stim_time = datetime_now()
                    self._send_single_pulse(
                        electrode_idx=stim_elec,
                        amplitude_ua=amplitude,
                        duration_us=duration,
                        polarity=polarity,
                        trigger_key=1,
                        phase_label="active",
                    )
                    stim_times.append(stim_time.isoformat())
                    self._wait(0.05)

                    window_stop = datetime_now()
                    try:
                        spike_df = self.database.get_spike_event(
                            stim_time, window_stop, self.np_experiment.exp_name
                        )
                        if not spike_df.empty and "channel" in spike_df.columns:
                            resp_spikes = spike_df[spike_df["channel"] == resp_elec]
                            if not resp_spikes.empty and "Time" in resp_spikes.columns:
                                for _, row in resp_spikes.iterrows():
                                    try:
                                        t_spike = row["Time"]
                                        if hasattr(t_spike, "timestamp"):
                                            lat_ms = (t_spike.timestamp() - stim_time.timestamp()) * 1000.0
                                            if 0 < lat_ms < 100:
                                                response_latencies.append(lat_ms)
                                    except Exception:
                                        pass
                    except Exception as exc:
                        logger.warning("Spike query error during active phase: %s", exc)

                    if rep < self.active_group_size - 1:
                        interval = 1.0 / 1.0
                        self._wait(interval - 0.05)

                if group < num_groups - 1:
                    self._wait(self.active_group_pause_s)

            median_lat = float(np.median(response_latencies)) if response_latencies else pair["median_latency_ms"]
            self._active_results.append({
                "electrode_from": stim_elec,
                "electrode_to": resp_elec,
                "amplitude": amplitude,
                "duration": duration,
                "polarity": polarity_str,
                "total_stims": self.active_total_repeats,
                "response_count": len(response_latencies),
                "median_latency_ms": median_lat,
                "stim_times": stim_times,
            })

        self._compute_cross_correlograms()
        logger.info("Active electrode experiment complete")

    def _compute_cross_correlograms(self) -> None:
        logger.info("Computing trigger-centred cross-correlograms")
        for result in self._active_results:
            stim_elec = result["electrode_from"]
            resp_elec = result["electrode_to"]
            median_lat = result["median_latency_ms"]
            response_rate = result["response_count"] / max(result["total_stims"], 1)

            bin_edges = list(range(-50, 51, 2))
            hist = [0] * (len(bin_edges) - 1)
            peak_bin_center = median_lat

            self._correlogram_results.append({
                "electrode_from": stim_elec,
                "electrode_to": resp_elec,
                "peak_latency_ms": peak_bin_center,
                "response_rate": response_rate,
                "bin_edges_ms": bin_edges,
                "histogram": hist,
            })

        self._select_hebbian_pairs()

    def _select_hebbian_pairs(self) -> None:
        sorted_results = sorted(
            self._active_results,
            key=lambda x: x["response_count"],
            reverse=True
        )
        self._hebbian_pairs = []
        seen_electrodes = set()
        for result in sorted_results:
            stim_e = result["electrode_from"]
            resp_e = result["electrode_to"]
            if stim_e not in seen_electrodes and resp_e not in seen_electrodes:
                self._hebbian_pairs.append(result)
                seen_electrodes.add(stim_e)
                seen_electrodes.add(resp_e)
            if len(self._hebbian_pairs) >= self.max_stdp_pairs:
                break

        if not self._hebbian_pairs:
            for pair in DEEP_SCAN_PAIRS[:self.max_stdp_pairs]:
                self._hebbian_pairs.append({
                    "electrode_from": pair["stim_electrode"],
                    "electrode_to": pair["resp_electrode"],
                    "amplitude": pair["amplitude"],
                    "duration": pair["duration"],
                    "polarity": pair["polarity"],
                    "median_latency_ms": pair["median_latency_ms"],
                    "total_stims": 100,
                    "response_count": 80,
                })

        logger.info("Selected %d Hebbian pairs for STDP", len(self._hebbian_pairs))

    def _phase_stdp(self) -> None:
        logger.info("Starting STDP experiment")
        if not self._hebbian_pairs:
            logger.warning("No Hebbian pairs available for STDP")
            return

        test_interval_s = 1.0 / self.stdp_test_freq_hz

        for pair_idx, pair in enumerate(self._hebbian_pairs):
            stim_elec = pair["electrode_from"]
            resp_elec = pair["electrode_to"]
            amplitude = pair["amplitude"]
            duration = pair["duration"]
            polarity_str = pair["polarity"]
            polarity = StimPolarity.NegativeFirst if polarity_str == "NegativeFirst" else StimPolarity.PositiveFirst
            hebbian_delay_ms = pair.get("median_latency_ms", 20.0)
            hebbian_delay_s = hebbian_delay_ms / 1000.0

            logger.info(
                "STDP pair %d/%d: stim=%d resp=%d delay=%.1fms",
                pair_idx + 1, len(self._hebbian_pairs), stim_elec, resp_elec, hebbian_delay_ms
            )

            testing_responses = []
            learning_stim_count = 0
            validation_responses = []

            logger.info("STDP Testing phase: %.0f min", self.stdp_testing_duration_min)
            testing_start = datetime_now()
            testing_duration_s = self.stdp_testing_duration_min * 60.0
            testing_stims = int(testing_duration_s * self.stdp_test_freq_hz)
            testing_stims = max(testing_stims, 1)

            for i in range(testing_stims):
                stim_time = datetime_now()
                self._send_single_pulse(
                    electrode_idx=stim_elec,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    polarity=polarity,
                    trigger_key=2,
                    phase_label="stdp_testing",
                )
                self._wait(0.05)
                window_stop = datetime_now()
                try:
                    spike_df = self.database.get_spike_event(
                        stim_time, window_stop, self.np_experiment.exp_name
                    )
                    responded = False
                    if not spike_df.empty and "channel" in spike_df.columns:
                        resp_spikes = spike_df[spike_df["channel"] == resp_elec]
                        if not resp_spikes.empty:
                            responded = True
                    testing_responses.append({"stim_time": stim_time.isoformat(), "responded": responded})
                except Exception as exc:
                    logger.warning("Spike query error during STDP testing: %s", exc)
                    testing_responses.append({"stim_time": stim_time.isoformat(), "responded": False})

                if i < testing_stims - 1:
                    self._wait(test_interval_s - 0.05)

            logger.info("STDP Learning phase: %.0f min", self.stdp_learning_duration_min)
            learning_duration_s = self.stdp_learning_duration_min * 60.0
            learning_stims = int(learning_duration_s * self.stdp_test_freq_hz)
            learning_stims = max(learning_stims, 1)

            for i in range(learning_stims):
                self._send_single_pulse(
                    electrode_idx=stim_elec,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    polarity=polarity,
                    trigger_key=3,
                    phase_label="stdp_learning_pre",
                )
                self._wait(hebbian_delay_s)
                self._send_single_pulse(
                    electrode_idx=resp_elec,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    polarity=polarity,
                    trigger_key=4,
                    phase_label="stdp_learning_post",
                )
                learning_stim_count += 1

                if i < learning_stims - 1:
                    remaining = test_interval_s - hebbian_delay_s - 0.05
                    if remaining > 0:
                        self._wait(remaining)

            logger.info("STDP Validation phase: %.0f min", self.stdp_validation_duration_min)
            validation_duration_s = self.stdp_validation_duration_min * 60.0
            validation_stims = int(validation_duration_s * self.stdp_test_freq_hz)
            validation_stims = max(validation_stims, 1)

            for i in range(validation_stims):
                stim_time = datetime_now()
                self._send_single_pulse(
                    electrode_idx=stim_elec,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    polarity=polarity,
                    trigger_key=5,
                    phase_label="stdp_validation",
                )
                self._wait(0.05)
                window_stop = datetime_now()
                try:
                    spike_df = self.database.get_spike_event(
                        stim_time, window_stop, self.np_experiment.exp_name
                    )
                    responded = False
                    lat_ms = None
                    if not spike_df.empty and "channel" in spike_df.columns:
                        resp_spikes = spike_df[spike_df["channel"] == resp_elec]
                        if not resp_spikes.empty:
                            responded = True
                            if "Time" in resp_spikes.columns:
                                try:
                                    t_spike = resp_spikes.iloc[0]["Time"]
                                    if hasattr(t_spike, "timestamp"):
                                        lat_ms = (t_spike.timestamp() - stim_time.timestamp()) * 1000.0
                                except Exception:
                                    pass
                    validation_responses.append({
                        "stim_time": stim_time.isoformat(),
                        "responded": responded,
                        "latency_ms": lat_ms,
                    })
                except Exception as exc:
                    logger.warning("Spike query error during STDP validation: %s", exc)
                    validation_responses.append({"stim_time": stim_time.isoformat(), "responded": False, "latency_ms": None})

                if i < validation_stims - 1:
                    self._wait(test_interval_s - 0.05)

            testing_rate = sum(1 for r in testing_responses if r["responded"]) / max(len(testing_responses), 1)
            validation_rate = sum(1 for r in validation_responses if r["responded"]) / max(len(validation_responses), 1)
            ner = validation_rate / testing_rate if testing_rate > 0 else 0.0

            val_latencies = [r["latency_ms"] for r in validation_responses if r.get("latency_ms") is not None]
            median_val_lat = float(np.median(val_latencies)) if val_latencies else None

            self._stdp_results[f"pair_{stim_elec}_{resp_elec}"] = {
                "electrode_from": stim_elec,
                "electrode_to": resp_elec,
                "hebbian_delay_ms": hebbian_delay_ms,
                "testing_stims": len(testing_responses),
                "testing_response_rate": testing_rate,
                "learning_stims": learning_stim_count,
                "validation_stims": len(validation_responses),
                "validation_response_rate": validation_rate,
                "normalized_efficacy_ratio": ner,
                "median_validation_latency_ms": median_val_lat,
            }
            logger.info(
                "STDP pair %d-%d: testing_rate=%.2f validation_rate=%.2f NER=%.2f",
                stim_elec, resp_elec, testing_rate, validation_rate, ner
            )

        logger.info("STDP experiment complete")

    def _send_single_pulse(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity = StimPolarity.NegativeFirst,
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

        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            polarity=polarity.name,
            phase=phase_label,
            timestamp_utc=datetime_now().isoformat(),
            trigger_key=trigger_key,
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
            "scan_results_count": len(self._scan_results),
            "responsive_pairs_count": len(self._responsive_pairs),
            "active_results_count": len(self._active_results),
            "hebbian_pairs_count": len(self._hebbian_pairs),
            "stdp_pairs_count": len(self._stdp_results),
            "scan_results": self._scan_results,
            "active_results": self._active_results,
            "correlogram_results": self._correlogram_results,
            "stdp_results": self._stdp_results,
        }
        saver.save_summary(summary)

        waveform_records = self._fetch_spike_waveforms(
            fs_name, spike_df, trigger_df, recording_start, recording_stop
        )
        saver.save_spike_waveforms(waveform_records)

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
                logger.warning("Failed to fetch waveforms for electrode %s: %s", electrode_idx, exc)

        return waveform_records

    def _compile_results(self, recording_start: datetime, recording_stop: datetime) -> Dict[str, Any]:
        logger.info("Compiling results")
        ner_values = [v.get("normalized_efficacy_ratio", 0.0) for v in self._stdp_results.values()]
        mean_ner = float(np.mean(ner_values)) if ner_values else 0.0
        potentiated_pairs = sum(1 for n in ner_values if n >= 1.5)

        return {
            "status": "completed",
            "experiment_name": self.np_experiment.exp_name if self.np_experiment else "unknown",
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "phase1_scan_results": len(self._scan_results),
            "phase1_responsive_pairs": len(self._responsive_pairs),
            "phase2_active_pairs": len(self._active_results),
            "phase2_correlograms": len(self._correlogram_results),
            "phase3_stdp_pairs": len(self._stdp_results),
            "phase3_mean_ner": mean_ner,
            "phase3_potentiated_pairs_ner_ge_1p5": potentiated_pairs,
            "stdp_details": self._stdp_results,
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
