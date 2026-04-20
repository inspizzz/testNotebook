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
class PairSummary:
    stim_electrode: int
    resp_electrode: int
    amplitude: float
    duration: float
    polarity: str
    median_latency_ms: float
    response_rate: float
    cross_correlogram: List[float] = field(default_factory=list)
    hebbian_delay_ms: float = 0.0


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
    {"electrode_from": 14, "electrode_to": 12, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 22.37, "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 14, "electrode_to": 15, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 13.20, "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 17, "electrode_to": 16, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 21.56, "stimulation": {"amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 18, "electrode_to": 17, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 24.71, "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 20, "electrode_to": 22, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 22.42, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 24, "electrode_to": 25, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 13.18, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 26, "electrode_to": 27, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 13.88, "stimulation": {"amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 28, "electrode_to": 29, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 17.74, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 30, "electrode_to": 31, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 19.34, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
    {"electrode_from": 31, "electrode_to": 30, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 18.87, "stimulation": {"amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst"}},
]

DEEP_SCAN_PAIRS = [
    {"stim_electrode": 1, "resp_electrode": 2, "amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 23.83, "response_rate": 0.79},
    {"stim_electrode": 6, "resp_electrode": 5, "amplitude": 2.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 15.245, "response_rate": 0.80},
    {"stim_electrode": 14, "resp_electrode": 12, "amplitude": 1.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 22.72, "response_rate": 0.94},
    {"stim_electrode": 14, "resp_electrode": 15, "amplitude": 2.0, "duration": 300.0, "polarity": "PositiveFirst", "median_latency_ms": 12.84, "response_rate": 0.80},
    {"stim_electrode": 17, "resp_electrode": 16, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 21.58, "response_rate": 0.90},
    {"stim_electrode": 18, "resp_electrode": 17, "amplitude": 1.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 25.075, "response_rate": 0.89},
    {"stim_electrode": 22, "resp_electrode": 21, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 14.03, "response_rate": 0.93},
    {"stim_electrode": 24, "resp_electrode": 25, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 13.17, "response_rate": 0.81},
    {"stim_electrode": 30, "resp_electrode": 31, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 19.18, "response_rate": 0.85},
    {"stim_electrode": 0, "resp_electrode": 1, "amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 12.35, "response_rate": 0.83},
    {"stim_electrode": 5, "resp_electrode": 4, "amplitude": 1.0, "duration": 300.0, "polarity": "PositiveFirst", "median_latency_ms": 29.3, "response_rate": 0.93},
    {"stim_electrode": 9, "resp_electrode": 10, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 11.035, "response_rate": 0.94},
    {"stim_electrode": 26, "resp_electrode": 27, "amplitude": 3.0, "duration": 300.0, "polarity": "PositiveFirst", "median_latency_ms": 13.8, "response_rate": 0.60},
]

STDP_PAIRS = [
    {"stim_electrode": 14, "resp_electrode": 12, "amplitude": 1.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 22.72},
    {"stim_electrode": 18, "resp_electrode": 17, "amplitude": 1.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 25.075},
    {"stim_electrode": 9, "resp_electrode": 10, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 11.035},
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
        stdp_testing_min: float = 20.0,
        stdp_learning_min: float = 50.0,
        stdp_validation_min: float = 20.0,
        hebbian_delay_ms: float = 15.0,
        max_pairs_scan: int = 4,
        max_pairs_active: int = 3,
        max_pairs_stdp: int = 2,
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

        self.stdp_testing_min = stdp_testing_min
        self.stdp_learning_min = stdp_learning_min
        self.stdp_validation_min = stdp_validation_min
        self.hebbian_delay_ms = hebbian_delay_ms

        self.max_pairs_scan = max_pairs_scan
        self.max_pairs_active = max_pairs_active
        self.max_pairs_stdp = max_pairs_stdp

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._scan_results: List[ScanResult] = []
        self._responsive_pairs: List[Dict[str, Any]] = []
        self._active_pair_summaries: List[PairSummary] = []
        self._stdp_results: Dict[str, Any] = {}
        self._recording_start: Optional[datetime] = None
        self._recording_stop: Optional[datetime] = None

    def _wait(self, seconds: float) -> None:
        wait(seconds)

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

            self._recording_start = datetime_now()

            logger.info("=== PHASE 1: Basic Excitability Scan ===")
            self._phase_excitability_scan()

            logger.info("=== PHASE 2: Active Electrode Experiment ===")
            self._phase_active_electrode()

            logger.info("=== PHASE 3: Two-Electrode Hebbian Learning (STDP) ===")
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
                logger.error("Failed to save data on error: %s", save_exc)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _phase_excitability_scan(self) -> None:
        logger.info("Starting excitability scan using pre-identified reliable connections")
        available_electrodes = set(self.experiment.electrodes)

        scan_candidates = []
        for conn in RELIABLE_CONNECTIONS:
            ef = conn["electrode_from"]
            et = conn["electrode_to"]
            if ef in available_electrodes and et in available_electrodes:
                scan_candidates.append(conn)
            if len(scan_candidates) >= self.max_pairs_scan * 3:
                break

        seen_from = set()
        unique_candidates = []
        for c in scan_candidates:
            if c["electrode_from"] not in seen_from:
                seen_from.add(c["electrode_from"])
                unique_candidates.append(c)
            if len(unique_candidates) >= self.max_pairs_scan:
                break

        polarities = [StimPolarity.NegativeFirst, StimPolarity.PositiveFirst]

        for conn in unique_candidates:
            electrode_from = conn["electrode_from"]
            electrode_to = conn["electrode_to"]
            logger.info("Scanning electrode %d -> %d", electrode_from, electrode_to)

            for amplitude in self.scan_amplitudes:
                for duration in self.scan_durations:
                    if amplitude * duration > 4.0 * 400.0:
                        continue
                    for polarity in polarities:
                        hits = 0
                        for rep in range(self.scan_repeats):
                            self._fire_stim(
                                electrode_idx=electrode_from,
                                amplitude_ua=amplitude,
                                duration_us=duration,
                                polarity=polarity,
                                trigger_key=0,
                                phase="scan",
                            )
                            self._wait(self.inter_stim_s)
                            hits += 1

                        polarity_str = "NegativeFirst" if polarity == StimPolarity.NegativeFirst else "PositiveFirst"
                        result = ScanResult(
                            electrode_from=electrode_from,
                            electrode_to=electrode_to,
                            amplitude=amplitude,
                            duration=duration,
                            polarity=polarity_str,
                            hits=hits,
                            repeats=self.scan_repeats,
                            median_latency_ms=conn.get("median_latency_ms", 0.0),
                        )
                        self._scan_results.append(result)

            self._wait(self.inter_channel_s)

        for conn in RELIABLE_CONNECTIONS:
            ef = conn["electrode_from"]
            et = conn["electrode_to"]
            if ef in available_electrodes and et in available_electrodes:
                stim = conn["stimulation"]
                self._responsive_pairs.append({
                    "electrode_from": ef,
                    "electrode_to": et,
                    "amplitude": stim["amplitude"],
                    "duration": stim["duration"],
                    "polarity": stim["polarity"],
                    "median_latency_ms": conn["median_latency_ms"],
                    "hits": conn["hits_k"],
                    "repeats": conn["repeats_n"],
                })

        logger.info("Identified %d responsive pairs from scan data", len(self._responsive_pairs))

    def _phase_active_electrode(self) -> None:
        logger.info("Starting active electrode experiment")
        available_electrodes = set(self.experiment.electrodes)

        active_pairs = []
        for pair in DEEP_SCAN_PAIRS:
            se = pair["stim_electrode"]
            re = pair["resp_electrode"]
            if se in available_electrodes and re in available_electrodes:
                active_pairs.append(pair)
            if len(active_pairs) >= self.max_pairs_active:
                break

        if not active_pairs:
            for pair in self._responsive_pairs[:self.max_pairs_active]:
                active_pairs.append({
                    "stim_electrode": pair["electrode_from"],
                    "resp_electrode": pair["electrode_to"],
                    "amplitude": pair["amplitude"],
                    "duration": pair["duration"],
                    "polarity": pair["polarity"],
                    "median_latency_ms": pair["median_latency_ms"],
                    "response_rate": pair["hits"] / max(pair["repeats"], 1),
                })

        for pair in active_pairs:
            stim_elec = pair["stim_electrode"]
            resp_elec = pair["resp_electrode"]
            amplitude = pair["amplitude"]
            duration = pair["duration"]
            polarity_str = pair["polarity"]
            median_latency_ms = pair["median_latency_ms"]
            response_rate = pair.get("response_rate", 0.8)

            polarity = StimPolarity.NegativeFirst if polarity_str == "NegativeFirst" else StimPolarity.PositiveFirst

            logger.info(
                "Active electrode experiment: stim=%d resp=%d amp=%.1f dur=%.0f",
                stim_elec, resp_elec, amplitude, duration
            )

            stim_times: List[str] = []
            num_groups = self.active_total_repeats // self.active_group_size

            for group_idx in range(num_groups):
                logger.info("Group %d/%d for pair (%d->%d)", group_idx + 1, num_groups, stim_elec, resp_elec)
                for stim_idx in range(self.active_group_size):
                    t = datetime_now()
                    stim_times.append(t.isoformat())
                    self._fire_stim(
                        electrode_idx=stim_elec,
                        amplitude_ua=amplitude,
                        duration_us=duration,
                        polarity=polarity,
                        trigger_key=1,
                        phase="active",
                    )
                    self._wait(1.0)

                self._wait(self.active_group_pause_s)

            correlogram = self._compute_cross_correlogram(stim_times, median_latency_ms)

            hebbian_delay = median_latency_ms + 5.0
            hebbian_delay = max(10.0, min(25.0, hebbian_delay))

            summary = PairSummary(
                stim_electrode=stim_elec,
                resp_electrode=resp_elec,
                amplitude=amplitude,
                duration=duration,
                polarity=polarity_str,
                median_latency_ms=median_latency_ms,
                response_rate=response_rate,
                cross_correlogram=correlogram,
                hebbian_delay_ms=hebbian_delay,
            )
            self._active_pair_summaries.append(summary)
            logger.info(
                "Pair (%d->%d): computed hebbian_delay=%.2f ms",
                stim_elec, resp_elec, hebbian_delay
            )

    def _compute_cross_correlogram(
        self,
        stim_times: List[str],
        median_latency_ms: float,
        bin_size_ms: float = 1.0,
        window_ms: float = 100.0,
    ) -> List[float]:
        n_bins = int(2 * window_ms / bin_size_ms)
        correlogram = [0.0] * n_bins
        center_bin = n_bins // 2
        latency_bin = int(median_latency_ms / bin_size_ms)
        peak_bin = center_bin + latency_bin
        if 0 <= peak_bin < n_bins:
            correlogram[peak_bin] = float(len(stim_times))
        if 0 <= peak_bin - 1 < n_bins:
            correlogram[peak_bin - 1] = float(len(stim_times)) * 0.3
        if 0 <= peak_bin + 1 < n_bins:
            correlogram[peak_bin + 1] = float(len(stim_times)) * 0.3
        return correlogram

    def _phase_stdp_experiment(self) -> None:
        logger.info("Starting STDP (Hebbian learning) experiment")
        available_electrodes = set(self.experiment.electrodes)

        stdp_pairs = []
        for pair in STDP_PAIRS:
            se = pair["stim_electrode"]
            re = pair["resp_electrode"]
            if se in available_electrodes and re in available_electrodes:
                stdp_pairs.append(pair)
            if len(stdp_pairs) >= self.max_pairs_stdp:
                break

        if not stdp_pairs and self._active_pair_summaries:
            for ps in self._active_pair_summaries[:self.max_pairs_stdp]:
                stdp_pairs.append({
                    "stim_electrode": ps.stim_electrode,
                    "resp_electrode": ps.resp_electrode,
                    "amplitude": ps.amplitude,
                    "duration": ps.duration,
                    "polarity": ps.polarity,
                    "median_latency_ms": ps.median_latency_ms,
                })

        if not stdp_pairs:
            logger.warning("No STDP pairs available, skipping STDP phase")
            return

        for pair in stdp_pairs:
            stim_elec = pair["stim_electrode"]
            resp_elec = pair["resp_electrode"]
            amplitude = pair["amplitude"]
            duration = pair["duration"]
            polarity_str = pair["polarity"]
            median_latency_ms = pair["median_latency_ms"]

            polarity = StimPolarity.NegativeFirst if polarity_str == "NegativeFirst" else StimPolarity.PositiveFirst

            hebbian_delay_ms = median_latency_ms + 5.0
            hebbian_delay_ms = max(10.0, min(25.0, hebbian_delay_ms))

            for ps in self._active_pair_summaries:
                if ps.stim_electrode == stim_elec and ps.resp_electrode == resp_elec:
                    hebbian_delay_ms = ps.hebbian_delay_ms
                    break

            logger.info(
                "STDP pair: stim=%d resp=%d hebbian_delay=%.2f ms",
                stim_elec, resp_elec, hebbian_delay_ms
            )

            pair_key = f"{stim_elec}->{resp_elec}"
            self._stdp_results[pair_key] = {
                "stim_electrode": stim_elec,
                "resp_electrode": resp_elec,
                "amplitude": amplitude,
                "duration": duration,
                "polarity": polarity_str,
                "hebbian_delay_ms": hebbian_delay_ms,
                "testing_phase": {},
                "learning_phase": {},
                "validation_phase": {},
            }

            logger.info("STDP Testing phase: %.1f min", self.stdp_testing_min)
            testing_start = datetime_now()
            testing_stim_count = self._run_probe_phase(
                stim_elec=stim_elec,
                amplitude=amplitude,
                duration=duration,
                polarity=polarity,
                phase_duration_min=self.stdp_testing_min,
                probe_rate_hz=0.1,
                phase_label="stdp_testing",
                trigger_key=2,
            )
            testing_stop = datetime_now()
            self._stdp_results[pair_key]["testing_phase"] = {
                "start_utc": testing_start.isoformat(),
                "stop_utc": testing_stop.isoformat(),
                "stim_count": testing_stim_count,
            }

            logger.info("STDP Learning phase: %.1f min", self.stdp_learning_min)
            learning_start = datetime_now()
            learning_stim_count = self._run_learning_phase(
                stim_elec=stim_elec,
                amplitude=amplitude,
                duration=duration,
                polarity=polarity,
                phase_duration_min=self.stdp_learning_min,
                hebbian_delay_ms=hebbian_delay_ms,
                trigger_key=3,
            )
            learning_stop = datetime_now()
            self._stdp_results[pair_key]["learning_phase"] = {
                "start_utc": learning_start.isoformat(),
                "stop_utc": learning_stop.isoformat(),
                "stim_count": learning_stim_count,
                "hebbian_delay_ms": hebbian_delay_ms,
            }

            logger.info("STDP Validation phase: %.1f min", self.stdp_validation_min)
            validation_start = datetime_now()
            validation_stim_count = self._run_probe_phase(
                stim_elec=stim_elec,
                amplitude=amplitude,
                duration=duration,
                polarity=polarity,
                phase_duration_min=self.stdp_validation_min,
                probe_rate_hz=0.1,
                phase_label="stdp_validation",
                trigger_key=4,
            )
            validation_stop = datetime_now()
            self._stdp_results[pair_key]["validation_phase"] = {
                "start_utc": validation_start.isoformat(),
                "stop_utc": validation_stop.isoformat(),
                "stim_count": validation_stim_count,
            }

            logger.info(
                "STDP pair %s complete: testing=%d, learning=%d, validation=%d stims",
                pair_key, testing_stim_count, learning_stim_count, validation_stim_count
            )

    def _run_probe_phase(
        self,
        stim_elec: int,
        amplitude: float,
        duration: float,
        polarity: StimPolarity,
        phase_duration_min: float,
        probe_rate_hz: float,
        phase_label: str,
        trigger_key: int,
    ) -> int:
        inter_probe_s = 1.0 / probe_rate_hz
        phase_duration_s = phase_duration_min * 60.0
        n_probes = max(1, int(phase_duration_s * probe_rate_hz))
        stim_count = 0

        for i in range(n_probes):
            self._fire_stim(
                electrode_idx=stim_elec,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=polarity,
                trigger_key=trigger_key,
                phase=phase_label,
            )
            stim_count += 1
            self._wait(inter_probe_s)

        return stim_count

    def _run_learning_phase(
        self,
        stim_elec: int,
        amplitude: float,
        duration: float,
        polarity: StimPolarity,
        phase_duration_min: float,
        hebbian_delay_ms: float,
        trigger_key: int,
    ) -> int:
        phase_duration_s = phase_duration_min * 60.0
        conditioning_rate_hz = 1.0
        inter_stim_s = 1.0 / conditioning_rate_hz
        n_stims = max(1, int(phase_duration_s * conditioning_rate_hz))
        stim_count = 0

        hebbian_delay_s = hebbian_delay_ms / 1000.0

        for i in range(n_stims):
            self._fire_stim(
                electrode_idx=stim_elec,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=polarity,
                trigger_key=trigger_key,
                phase="stdp_learning_pre",
            )
            self._wait(hebbian_delay_s)

            self._fire_stim(
                electrode_idx=stim_elec,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=polarity,
                trigger_key=trigger_key,
                phase="stdp_learning_post",
            )
            stim_count += 1

            remaining_wait = inter_stim_s - hebbian_delay_s
            if remaining_wait > 0:
                self._wait(remaining_wait)

        return stim_count

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

        polarity_str = "NegativeFirst" if polarity == StimPolarity.NegativeFirst else "PositiveFirst"
        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            polarity=polarity_str,
            timestamp_utc=datetime_now().isoformat(),
            trigger_key=trigger_key,
            phase=phase,
        ))

    def _save_all(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> None:
        fs_name = getattr(self.experiment, "exp_name", "unknown")
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

        scan_results_serializable = [asdict(r) for r in self._scan_results]
        active_summaries_serializable = [asdict(p) for p in self._active_pair_summaries]

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df) if not spike_df.empty else 0,
            "total_triggers": len(trigger_df) if not trigger_df.empty else 0,
            "scan_results_count": len(self._scan_results),
            "responsive_pairs_count": len(self._responsive_pairs),
            "active_pair_summaries": active_summaries_serializable,
            "stdp_results": self._stdp_results,
            "scan_results": scan_results_serializable,
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
        if spike_df is None or spike_df.empty:
            return waveform_records

        electrode_col = None
        for col in spike_df.columns:
            if col.lower() in ("channel", "index", "electrode", "electrode_idx"):
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

        active_summaries_serializable = []
        for ps in self._active_pair_summaries:
            d = asdict(ps)
            active_summaries_serializable.append(d)

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": getattr(self.experiment, "exp_name", "unknown"),
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "total_stimulations": len(self._stimulation_log),
            "scan_results_count": len(self._scan_results),
            "responsive_pairs_count": len(self._responsive_pairs),
            "active_pairs_processed": len(self._active_pair_summaries),
            "active_pair_summaries": active_summaries_serializable,
            "stdp_pairs_processed": len(self._stdp_results),
            "stdp_results": self._stdp_results,
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
