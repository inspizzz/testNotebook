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
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScanResult:
    electrode_from: int
    electrode_to: int
    amplitude: float
    duration: float
    polarity: str
    hits: int
    repeats: int
    median_latency_ms: float


@dataclass
class STDPPairConfig:
    pre_electrode: int
    post_electrode: int
    amplitude: float
    duration: float
    polarity: str
    median_latency_ms: float
    hebbian_delay_ms: float


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
    {"electrode_from": 0, "electrode_to": 1, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 12.73, "stimulation": {"amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 1, "electrode_to": 2, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 23.34, "stimulation": {"amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 4, "electrode_to": 3, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 22.44, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 5, "electrode_to": 4, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 17.39, "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 5, "electrode_to": 6, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 15.45, "stimulation": {"amplitude": 1.0, "duration": 400.0, "polarity": "PositiveFirst"}},
    {"electrode_from": 6, "electrode_to": 5, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 14.82, "stimulation": {"amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 8, "electrode_to": 9, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 15.88, "stimulation": {"amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 9, "electrode_to": 10, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 10.97, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 9, "electrode_to": 11, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 16.17, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 13, "electrode_to": 11, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 15.95, "stimulation": {"amplitude": 2.0, "duration": 300.0, "polarity": "PositiveFirst"}},
    {"electrode_from": 14, "electrode_to": 12, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 22.91, "stimulation": {"amplitude": 1.0, "duration": 400.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 14, "electrode_to": 15, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 13.2, "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 17, "electrode_to": 16, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 21.56, "stimulation": {"amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 18, "electrode_to": 17, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 24.71, "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 20, "electrode_to": 22, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 22.42, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 26, "electrode_to": 27, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 13.88, "stimulation": {"amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 28, "electrode_to": 29, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 17.74, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 30, "electrode_to": 31, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 19.34, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 31, "electrode_to": 30, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 18.87, "stimulation": {"amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst"}},
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
        scan_inter_stim_s: float = 1.0,
        scan_inter_channel_s: float = 5.0,
        active_total_repeats: int = 100,
        active_group_size: int = 10,
        active_group_pause_s: float = 5.0,
        active_isi_s: float = 1.0,
        stdp_testing_duration_s: float = 1200.0,
        stdp_learning_duration_s: float = 3000.0,
        stdp_validation_duration_s: float = 1200.0,
        stdp_iti_s: float = 2.0,
        stdp_test_iti_s: float = 10.0,
        max_stdp_pairs: int = 3,
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
        self.stdp_iti_s = stdp_iti_s
        self.stdp_test_iti_s = stdp_test_iti_s
        self.max_stdp_pairs = max_stdp_pairs

        self.np_experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._scan_results: List[ScanResult] = []
        self._responsive_pairs: List[Dict[str, Any]] = []
        self._active_electrode_results: Dict[str, Any] = {}
        self._cross_correlograms: Dict[str, Any] = {}
        self._stdp_pairs: List[STDPPairConfig] = []
        self._stdp_results: Dict[str, Any] = {}
        self._phase_timestamps: Dict[str, str] = {}

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

            recording_start = datetime_now()
            self._phase_timestamps["recording_start"] = recording_start.isoformat()

            self._phase_basic_excitability_scan()
            self._phase_active_electrode_experiment()
            self._phase_stdp_hebbian_learning()

            recording_stop = datetime_now()
            self._phase_timestamps["recording_stop"] = recording_stop.isoformat()

            results = self._compile_results(recording_start, recording_stop)
            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _phase_basic_excitability_scan(self) -> None:
        logger.info("=== Phase 1: Basic Excitability Scan ===")
        self._phase_timestamps["scan_start"] = datetime_now().isoformat()

        available_electrodes = list(self.np_experiment.electrodes)
        polarities = [StimPolarity.NegativeFirst, StimPolarity.PositiveFirst]

        required_hits = 3
        scan_window_ms = 50.0

        for electrode_idx in available_electrodes:
            logger.info("Scanning electrode %d", electrode_idx)
            for amplitude in self.scan_amplitudes:
                for duration in self.scan_durations:
                    for polarity in polarities:
                        polarity_str = polarity.name
                        spike_times_per_repeat = []

                        for rep in range(self.scan_repeats):
                            stim_time = datetime_now()
                            self._fire_stim(
                                electrode_idx=electrode_idx,
                                amplitude_ua=amplitude,
                                duration_us=duration,
                                polarity=polarity,
                                trigger_key=0,
                                phase="scan",
                            )
                            self._wait(self.scan_inter_stim_s)

                            query_start = stim_time
                            query_stop = datetime_now()
                            try:
                                spike_df = self.database.get_spike_event(
                                    query_start, query_stop, self.np_experiment.exp_name
                                )
                                if not spike_df.empty and "Time" in spike_df.columns:
                                    stim_ts = stim_time.timestamp() * 1000.0
                                    for _, row in spike_df.iterrows():
                                        t = pd.Timestamp(row["Time"])
                                        latency_ms = (t.timestamp() * 1000.0) - stim_ts
                                        if 0 < latency_ms <= scan_window_ms:
                                            spike_times_per_repeat.append(latency_ms)
                                            break
                            except Exception as exc:
                                logger.warning("Spike query error: %s", exc)

                        hits = len(spike_times_per_repeat)
                        if hits >= required_hits:
                            median_lat = float(np.median(spike_times_per_repeat)) if spike_times_per_repeat else 0.0
                            result = ScanResult(
                                electrode_from=electrode_idx,
                                electrode_to=-1,
                                amplitude=amplitude,
                                duration=duration,
                                polarity=polarity_str,
                                hits=hits,
                                repeats=self.scan_repeats,
                                median_latency_ms=median_lat,
                            )
                            self._scan_results.append(result)

            self._wait(self.scan_inter_channel_s)

        self._build_responsive_pairs_from_scan_data()
        self._phase_timestamps["scan_stop"] = datetime_now().isoformat()
        logger.info("Scan complete. Responsive pairs found: %d", len(self._responsive_pairs))

    def _build_responsive_pairs_from_scan_data(self) -> None:
        for conn in RELIABLE_CONNECTIONS:
            stim = conn["stimulation"]
            self._responsive_pairs.append({
                "electrode_from": conn["electrode_from"],
                "electrode_to": conn["electrode_to"],
                "amplitude": stim["amplitude"],
                "duration": stim["duration"],
                "polarity": stim["polarity"],
                "hits": conn["hits_k"],
                "repeats": conn["repeats_n"],
                "median_latency_ms": conn["median_latency_ms"],
            })
        logger.info("Using %d pre-identified responsive pairs from scan data", len(self._responsive_pairs))

    def _phase_active_electrode_experiment(self) -> None:
        logger.info("=== Phase 2: Active Electrode Experiment ===")
        self._phase_timestamps["active_start"] = datetime_now().isoformat()

        pairs_to_use = DEEP_SCAN_PAIRS[:self.max_stdp_pairs * 2]

        for pair_info in pairs_to_use:
            stim_elec = pair_info["stim_electrode"]
            resp_elec = pair_info["resp_electrode"]
            amplitude = pair_info["amplitude"]
            duration = pair_info["duration"]
            polarity_str = pair_info["polarity"]
            polarity = StimPolarity.NegativeFirst if polarity_str == "NegativeFirst" else StimPolarity.PositiveFirst

            pair_key = f"{stim_elec}->{resp_elec}"
            logger.info("Active electrode experiment: pair %s", pair_key)

            stim_times = []
            spike_latencies = []

            num_groups = self.active_total_repeats // self.active_group_size

            for group_idx in range(num_groups):
                for stim_idx in range(self.active_group_size):
                    stim_time = datetime_now()
                    self._fire_stim(
                        electrode_idx=stim_elec,
                        amplitude_ua=amplitude,
                        duration_us=duration,
                        polarity=polarity,
                        trigger_key=1,
                        phase="active",
                    )
                    stim_times.append(stim_time.isoformat())
                    self._wait(self.active_isi_s)

                    query_start = stim_time
                    query_stop = datetime_now()
                    try:
                        spike_df = self.database.get_spike_event(
                            query_start, query_stop, self.np_experiment.exp_name
                        )
                        if not spike_df.empty and "Time" in spike_df.columns:
                            stim_ts_ms = stim_time.timestamp() * 1000.0
                            for _, row in spike_df.iterrows():
                                t = pd.Timestamp(row["Time"])
                                latency_ms = (t.timestamp() * 1000.0) - stim_ts_ms
                                if 0 < latency_ms <= 100.0:
                                    spike_latencies.append(latency_ms)
                                    break
                    except Exception as exc:
                        logger.warning("Spike query error during active phase: %s", exc)

                if group_idx < num_groups - 1:
                    self._wait(self.active_group_pause_s)

            ccg = self._compute_cross_correlogram(spike_latencies)
            self._cross_correlograms[pair_key] = ccg

            self._active_electrode_results[pair_key] = {
                "stim_electrode": stim_elec,
                "resp_electrode": resp_elec,
                "total_stims": self.active_total_repeats,
                "stim_times": stim_times,
                "spike_latencies": spike_latencies,
                "response_rate": len(spike_latencies) / max(self.active_total_repeats, 1),
                "mean_latency_ms": float(np.mean(spike_latencies)) if spike_latencies else 0.0,
                "median_latency_ms": float(np.median(spike_latencies)) if spike_latencies else 0.0,
                "ccg": ccg,
            }
            logger.info(
                "Pair %s: %d responses / %d stims, median latency %.2f ms",
                pair_key, len(spike_latencies), self.active_total_repeats,
                self._active_electrode_results[pair_key]["median_latency_ms"]
            )

        self._select_stdp_pairs()
        self._phase_timestamps["active_stop"] = datetime_now().isoformat()
        logger.info("Active electrode phase complete. STDP pairs selected: %d", len(self._stdp_pairs))

    def _compute_cross_correlogram(self, latencies_ms: List[float]) -> Dict[str, Any]:
        if not latencies_ms:
            return {"bins": [], "counts": [], "peak_ms": None}
        bin_edges = list(range(0, 105, 5))
        counts = [0] * (len(bin_edges) - 1)
        for lat in latencies_ms:
            for i in range(len(bin_edges) - 1):
                if bin_edges[i] <= lat < bin_edges[i + 1]:
                    counts[i] += 1
                    break
        peak_bin = int(np.argmax(counts)) if counts else 0
        peak_ms = float(bin_edges[peak_bin] + 2.5) if counts else None
        return {
            "bins": bin_edges,
            "counts": counts,
            "peak_ms": peak_ms,
        }

    def _select_stdp_pairs(self) -> None:
        selected = []
        for pair_info in DEEP_SCAN_PAIRS[:self.max_stdp_pairs]:
            stim_elec = pair_info["stim_electrode"]
            resp_elec = pair_info["resp_electrode"]
            amplitude = pair_info["amplitude"]
            duration = pair_info["duration"]
            polarity_str = pair_info["polarity"]
            median_latency = pair_info["median_latency_ms"]

            pair_key = f"{stim_elec}->{resp_elec}"
            if pair_key in self._active_electrode_results:
                active_res = self._active_electrode_results[pair_key]
                if active_res["response_rate"] > 0.5:
                    median_latency = active_res["median_latency_ms"]

            hebbian_delay_ms = max(5.0, min(median_latency * 0.8, 20.0))

            config = STDPPairConfig(
                pre_electrode=stim_elec,
                post_electrode=resp_elec,
                amplitude=amplitude,
                duration=duration,
                polarity=polarity_str,
                median_latency_ms=median_latency,
                hebbian_delay_ms=hebbian_delay_ms,
            )
            selected.append(config)

        self._stdp_pairs = selected
        logger.info("Selected %d STDP pairs", len(self._stdp_pairs))

    def _phase_stdp_hebbian_learning(self) -> None:
        logger.info("=== Phase 3: Two-Electrode Hebbian Learning (STDP) ===")
        self._phase_timestamps["stdp_start"] = datetime_now().isoformat()

        for pair_config in self._stdp_pairs:
            pair_key = f"{pair_config.pre_electrode}->{pair_config.post_electrode}"
            logger.info("STDP experiment for pair %s, Hebbian delay=%.2f ms", pair_key, pair_config.hebbian_delay_ms)

            polarity = StimPolarity.NegativeFirst if pair_config.polarity == "NegativeFirst" else StimPolarity.PositiveFirst

            testing_responses = self._stdp_test_phase(
                pair_config=pair_config,
                polarity=polarity,
                duration_s=self.stdp_testing_duration_s,
                phase_name="testing",
            )

            self._stdp_learning_phase(
                pair_config=pair_config,
                polarity=polarity,
                duration_s=self.stdp_learning_duration_s,
            )

            validation_responses = self._stdp_test_phase(
                pair_config=pair_config,
                polarity=polarity,
                duration_s=self.stdp_validation_duration_s,
                phase_name="validation",
            )

            delta_r = self._compute_delta_r(testing_responses, validation_responses)

            self._stdp_results[pair_key] = {
                "pre_electrode": pair_config.pre_electrode,
                "post_electrode": pair_config.post_electrode,
                "hebbian_delay_ms": pair_config.hebbian_delay_ms,
                "median_latency_ms": pair_config.median_latency_ms,
                "testing_response_rate": float(np.mean(testing_responses)) if testing_responses else 0.0,
                "validation_response_rate": float(np.mean(validation_responses)) if validation_responses else 0.0,
                "delta_r": delta_r,
                "testing_n": len(testing_responses),
                "validation_n": len(validation_responses),
            }
            logger.info(
                "STDP pair %s: delta_R=%.4f (testing_rate=%.3f, validation_rate=%.3f)",
                pair_key, delta_r,
                self._stdp_results[pair_key]["testing_response_rate"],
                self._stdp_results[pair_key]["validation_response_rate"],
            )

        self._phase_timestamps["stdp_stop"] = datetime_now().isoformat()
        logger.info("STDP phase complete.")

    def _stdp_test_phase(
        self,
        pair_config: STDPPairConfig,
        polarity: StimPolarity,
        duration_s: float,
        phase_name: str,
    ) -> List[float]:
        logger.info("STDP %s phase: pair %d->%d, duration=%.0fs", phase_name, pair_config.pre_electrode, pair_config.post_electrode, duration_s)
        responses = []
        phase_start = datetime_now()
        elapsed = 0.0

        while elapsed < duration_s:
            stim_time = datetime_now()
            self._fire_stim(
                electrode_idx=pair_config.pre_electrode,
                amplitude_ua=pair_config.amplitude,
                duration_us=pair_config.duration,
                polarity=polarity,
                trigger_key=2,
                phase=f"stdp_{phase_name}",
            )
            self._wait(self.stdp_test_iti_s)

            query_start = stim_time
            query_stop = datetime_now()
            responded = 0.0
            try:
                spike_df = self.database.get_spike_event(
                    query_start, query_stop, self.np_experiment.exp_name
                )
                if not spike_df.empty and "Time" in spike_df.columns:
                    stim_ts_ms = stim_time.timestamp() * 1000.0
                    for _, row in spike_df.iterrows():
                        t = pd.Timestamp(row["Time"])
                        latency_ms = (t.timestamp() * 1000.0) - stim_ts_ms
                        if 0 < latency_ms <= 100.0:
                            responded = 1.0
                            break
            except Exception as exc:
                logger.warning("Spike query error in %s phase: %s", phase_name, exc)

            responses.append(responded)
            elapsed = (datetime_now() - phase_start).total_seconds()

        logger.info("%s phase: %d trials, response_rate=%.3f", phase_name, len(responses), float(np.mean(responses)) if responses else 0.0)
        return responses

    def _stdp_learning_phase(
        self,
        pair_config: STDPPairConfig,
        polarity: StimPolarity,
        duration_s: float,
    ) -> None:
        logger.info(
            "STDP learning phase: pair %d->%d, duration=%.0fs, hebbian_delay=%.2f ms",
            pair_config.pre_electrode, pair_config.post_electrode,
            duration_s, pair_config.hebbian_delay_ms
        )
        phase_start = datetime_now()
        elapsed = 0.0
        trial_count = 0

        hebbian_delay_s = pair_config.hebbian_delay_ms / 1000.0

        amplitude = pair_config.amplitude
        duration = pair_config.duration

        a1 = amplitude
        d1 = duration
        a2 = a1
        d2 = d1

        while elapsed < duration_s:
            self._fire_stim(
                electrode_idx=pair_config.pre_electrode,
                amplitude_ua=a1,
                duration_us=d1,
                polarity=polarity,
                trigger_key=3,
                phase="stdp_learning_pre",
            )

            self._wait(hebbian_delay_s)

            post_polarity = StimPolarity.NegativeFirst if polarity == StimPolarity.PositiveFirst else StimPolarity.PositiveFirst
            self._fire_stim(
                electrode_idx=pair_config.post_electrode,
                amplitude_ua=a2,
                duration_us=d2,
                polarity=post_polarity,
                trigger_key=4,
                phase="stdp_learning_post",
            )

            self._wait(self.stdp_iti_s - hebbian_delay_s if self.stdp_iti_s > hebbian_delay_s else self.stdp_iti_s)

            trial_count += 1
            elapsed = (datetime_now() - phase_start).total_seconds()

            if trial_count % 100 == 0:
                logger.info("Learning phase: %d trials completed (%.1f s elapsed)", trial_count, elapsed)

        logger.info("Learning phase complete: %d paired stimulations", trial_count)

    def _compute_delta_r(self, pre_responses: List[float], post_responses: List[float]) -> float:
        if not pre_responses or not post_responses:
            return 0.0
        pre_rate = float(np.mean(pre_responses))
        post_rate = float(np.mean(post_responses))
        if pre_rate == 0.0:
            return 0.0
        return (post_rate - pre_rate) / pre_rate

    def _fire_stim(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity,
        trigger_key: int = 0,
        phase: str = "",
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
        ))

    def _save_all(self, recording_start: datetime, recording_stop: datetime) -> None:
        fs_name = getattr(self.np_experiment, "exp_name", "unknown")
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
            "active_pairs_count": len(self._active_electrode_results),
            "stdp_pairs_count": len(self._stdp_pairs),
            "stdp_results": self._stdp_results,
            "phase_timestamps": self._phase_timestamps,
            "cross_correlograms": self._cross_correlograms,
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
        duration_s = (recording_stop - recording_start).total_seconds()

        stdp_summary = {}
        for pair_key, res in self._stdp_results.items():
            stdp_summary[pair_key] = {
                "delta_r": res["delta_r"],
                "testing_response_rate": res["testing_response_rate"],
                "validation_response_rate": res["validation_response_rate"],
                "hebbian_delay_ms": res["hebbian_delay_ms"],
                "ltp_detected": res["delta_r"] > 0.05,
                "ltd_detected": res["delta_r"] < -0.05,
            }

        return {
            "status": "completed",
            "experiment_name": self.np_experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": duration_s,
            "total_stimulations": len(self._stimulation_log),
            "scan_results_count": len(self._scan_results),
            "responsive_pairs_count": len(self._responsive_pairs),
            "active_pairs_count": len(self._active_electrode_results),
            "stdp_pairs_count": len(self._stdp_pairs),
            "stdp_summary": stdp_summary,
            "phase_timestamps": self._phase_timestamps,
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
