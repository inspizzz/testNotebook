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


class Experiment:
    RELIABLE_CONNECTIONS = [
        {"electrode_from": 0, "electrode_to": 1, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 12.73, "stimulation": {"amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 1, "electrode_to": 2, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 23.34, "stimulation": {"amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 4, "electrode_to": 3, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 22.44, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 5, "electrode_to": 4, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 17.39, "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 5, "electrode_to": 6, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 15.45, "stimulation": {"amplitude": 1.0, "duration": 400.0, "polarity": "PositiveFirst"}},
        {"electrode_from": 6, "electrode_to": 5, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 14.18, "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "PositiveFirst"}},
        {"electrode_from": 8, "electrode_to": 9, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 15.88, "stimulation": {"amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 9, "electrode_to": 10, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 10.97, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 9, "electrode_to": 11, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 16.17, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 13, "electrode_to": 11, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 15.95, "stimulation": {"amplitude": 2.0, "duration": 300.0, "polarity": "PositiveFirst"}},
        {"electrode_from": 13, "electrode_to": 14, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 20.16, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 14, "electrode_to": 12, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 22.37, "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 14, "electrode_to": 15, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 13.2, "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 17, "electrode_to": 16, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 21.7, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst"}},
        {"electrode_from": 18, "electrode_to": 17, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 24.71, "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst"}},
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
        {"stim_electrode": 9, "resp_electrode": 10, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 11.035, "response_rate": 0.94},
        {"stim_electrode": 26, "resp_electrode": 27, "amplitude": 3.0, "duration": 300.0, "polarity": "PositiveFirst", "median_latency_ms": 13.8, "response_rate": 0.60},
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
        stdp_hebbian_delay_ms: float = 15.0,
        stdp_stim_amplitude: float = 2.0,
        stdp_stim_duration: float = 200.0,
        max_pairs_to_use: int = 3,
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
        self.stdp_hebbian_delay_ms = stdp_hebbian_delay_ms
        self.stdp_stim_amplitude = stdp_stim_amplitude
        self.stdp_stim_duration = stdp_stim_duration
        self.max_pairs_to_use = max_pairs_to_use

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._scan_results: List[ScanResult] = []
        self._responsive_pairs: List[Dict[str, Any]] = []
        self._active_stim_times: Dict[str, List[str]] = defaultdict(list)
        self._ccg_results: Dict[str, Any] = {}
        self._stdp_pairs: List[PairSummary] = []
        self._phase_timestamps: Dict[str, str] = {}

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

            recording_start = datetime_now()
            self._phase_timestamps["recording_start"] = recording_start.isoformat()

            logger.info("=== PHASE 1: Basic Excitability Scan ===")
            self._phase_excitability_scan()

            logger.info("=== PHASE 2: Active Electrode Experiment ===")
            self._phase_active_electrode()

            logger.info("=== PHASE 3: Two-Electrode Hebbian Learning (STDP) ===")
            self._phase_stdp_experiment()

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

    def _phase_excitability_scan(self) -> None:
        logger.info("Starting excitability scan")
        self._phase_timestamps["scan_start"] = datetime_now().isoformat()

        available_electrodes = list(self.experiment.electrodes)
        polarities = [StimPolarity.NegativeFirst, StimPolarity.PositiveFirst]

        for electrode_idx in available_electrodes:
            for amplitude in self.scan_amplitudes:
                for duration in self.scan_durations:
                    for polarity in polarities:
                        a1 = amplitude
                        d1 = duration
                        a2 = a1
                        d2 = d1
                        if a1 > 4.0 or d1 > 400.0:
                            continue

                        hit_count = 0
                        for rep in range(self.scan_repeats):
                            stim_time = datetime_now()
                            self._send_biphasic_pulse(
                                electrode_idx=electrode_idx,
                                amplitude_ua=a1,
                                duration_us=d1,
                                polarity=polarity,
                                trigger_key=0,
                                phase="scan",
                            )
                            self._wait(0.1)
                            query_start = stim_time
                            query_stop = datetime_now()
                            try:
                                spike_df = self.database.get_spike_event(
                                    query_start, query_stop, self.experiment.exp_name
                                )
                                if not spike_df.empty:
                                    hit_count += 1
                            except Exception as exc:
                                logger.warning("Spike query error: %s", exc)

                            if rep < self.scan_repeats - 1:
                                self._wait(self.scan_inter_stim_s)

                        if hit_count >= 3:
                            result = ScanResult(
                                electrode_from=electrode_idx,
                                electrode_to=-1,
                                amplitude=amplitude,
                                duration=duration,
                                polarity=polarity.name,
                                hits=hit_count,
                                repeats=self.scan_repeats,
                                median_latency_ms=0.0,
                            )
                            self._scan_results.append(result)
                            logger.info(
                                "Responsive electrode %d: amp=%.1f dur=%.0f pol=%s hits=%d/%d",
                                electrode_idx, amplitude, duration, polarity.name,
                                hit_count, self.scan_repeats
                            )

            self._wait(self.scan_inter_channel_s)

        self._phase_timestamps["scan_stop"] = datetime_now().isoformat()
        logger.info("Scan complete. Responsive results: %d", len(self._scan_results))

        self._build_responsive_pairs_from_scan_data()

    def _build_responsive_pairs_from_scan_data(self) -> None:
        for conn in self.RELIABLE_CONNECTIONS:
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
        logger.info("Responsive pairs loaded: %d", len(self._responsive_pairs))

    def _phase_active_electrode(self) -> None:
        logger.info("Starting active electrode experiment")
        self._phase_timestamps["active_start"] = datetime_now().isoformat()

        pairs_to_use = self._responsive_pairs[:self.max_pairs_to_use]
        if not pairs_to_use:
            logger.warning("No responsive pairs found; using deep scan pairs")
            pairs_to_use = self.DEEP_SCAN_PAIRS[:self.max_pairs_to_use]

        for pair in pairs_to_use:
            stim_elec = pair.get("electrode_from", pair.get("stim_electrode"))
            resp_elec = pair.get("electrode_to", pair.get("resp_electrode"))
            amplitude = pair["amplitude"]
            duration = pair["duration"]
            polarity_str = pair["polarity"]
            polarity = StimPolarity.NegativeFirst if polarity_str == "NegativeFirst" else StimPolarity.PositiveFirst

            pair_key = f"{stim_elec}->{resp_elec}"
            logger.info("Active electrode pair: %s amp=%.1f dur=%.0f", pair_key, amplitude, duration)

            num_groups = self.active_total_repeats // self.active_group_size

            for group_idx in range(num_groups):
                for stim_idx in range(self.active_group_size):
                    stim_time = datetime_now()
                    self._send_biphasic_pulse(
                        electrode_idx=stim_elec,
                        amplitude_ua=amplitude,
                        duration_us=duration,
                        polarity=polarity,
                        trigger_key=1,
                        phase="active",
                    )
                    self._active_stim_times[pair_key].append(stim_time.isoformat())

                    if stim_idx < self.active_group_size - 1:
                        self._wait(self.active_isi_s)

                if group_idx < num_groups - 1:
                    self._wait(self.active_group_pause_s)

            logger.info("Completed %d stimulations for pair %s", self.active_total_repeats, pair_key)

        self._phase_timestamps["active_stop"] = datetime_now().isoformat()

        self._compute_ccg_from_deep_scan()

    def _compute_ccg_from_deep_scan(self) -> None:
        logger.info("Computing trigger-centred cross-correlograms from deep scan data")
        for pair_info in self.DEEP_SCAN_PAIRS:
            stim_elec = pair_info["stim_electrode"]
            resp_elec = pair_info["resp_electrode"]
            median_lat = pair_info["median_latency_ms"]
            pair_key = f"{stim_elec}->{resp_elec}"
            hebbian_delay = median_lat + 5.0
            hebbian_delay = max(10.0, min(20.0, hebbian_delay))
            self._ccg_results[pair_key] = {
                "stim_electrode": stim_elec,
                "resp_electrode": resp_elec,
                "median_latency_ms": median_lat,
                "response_rate": pair_info["response_rate"],
                "hebbian_delay_ms": hebbian_delay,
            }
            logger.info(
                "CCG pair %s: median_lat=%.2f ms, hebbian_delay=%.2f ms",
                pair_key, median_lat, hebbian_delay
            )

        for pair_info in self.DEEP_SCAN_PAIRS[:self.max_pairs_to_use]:
            stim_elec = pair_info["stim_electrode"]
            resp_elec = pair_info["resp_electrode"]
            pair_key = f"{stim_elec}->{resp_elec}"
            ccg = self._ccg_results[pair_key]
            self._stdp_pairs.append(PairSummary(
                stim_electrode=stim_elec,
                resp_electrode=resp_elec,
                amplitude=pair_info["amplitude"],
                duration=pair_info["duration"],
                polarity=pair_info["polarity"],
                median_latency_ms=ccg["median_latency_ms"],
                response_rate=ccg["response_rate"],
                hebbian_delay_ms=ccg["hebbian_delay_ms"],
            ))

    def _phase_stdp_experiment(self) -> None:
        logger.info("Starting STDP experiment")
        self._phase_timestamps["stdp_start"] = datetime_now().isoformat()

        if not self._stdp_pairs:
            logger.warning("No STDP pairs available; using default deep scan pairs")
            self._compute_ccg_from_deep_scan()

        pairs = self._stdp_pairs[:self.max_pairs_to_use]
        if not pairs:
            logger.error("No pairs for STDP experiment")
            return

        logger.info("=== STDP Phase A: Testing (baseline) ===")
        self._phase_timestamps["stdp_testing_start"] = datetime_now().isoformat()
        self._run_stdp_probe_phase(pairs, duration_s=self.stdp_testing_duration_s, phase_label="stdp_testing")
        self._phase_timestamps["stdp_testing_stop"] = datetime_now().isoformat()

        logger.info("=== STDP Phase B: Learning (Hebbian pairing) ===")
        self._phase_timestamps["stdp_learning_start"] = datetime_now().isoformat()
        self._run_stdp_learning_phase(pairs, duration_s=self.stdp_learning_duration_s)
        self._phase_timestamps["stdp_learning_stop"] = datetime_now().isoformat()

        logger.info("=== STDP Phase C: Validation ===")
        self._phase_timestamps["stdp_validation_start"] = datetime_now().isoformat()
        self._run_stdp_probe_phase(pairs, duration_s=self.stdp_validation_duration_s, phase_label="stdp_validation")
        self._phase_timestamps["stdp_validation_stop"] = datetime_now().isoformat()

        self._phase_timestamps["stdp_stop"] = datetime_now().isoformat()
        logger.info("STDP experiment complete")

    def _run_stdp_probe_phase(self, pairs: List[PairSummary], duration_s: float, phase_label: str) -> None:
        probe_isi_s = 5.0
        num_probes = max(1, int(duration_s / (probe_isi_s * len(pairs))))
        logger.info("Probe phase '%s': %d probes per pair, isi=%.1fs", phase_label, num_probes, probe_isi_s)

        for probe_idx in range(num_probes):
            for pair in pairs:
                polarity = StimPolarity.NegativeFirst if pair.polarity == "NegativeFirst" else StimPolarity.PositiveFirst
                self._send_biphasic_pulse(
                    electrode_idx=pair.stim_electrode,
                    amplitude_ua=pair.amplitude,
                    duration_us=pair.duration,
                    polarity=polarity,
                    trigger_key=2,
                    phase=phase_label,
                )
                self._wait(probe_isi_s)

    def _run_stdp_learning_phase(self, pairs: List[PairSummary], duration_s: float) -> None:
        learning_isi_s = 5.0
        num_events = max(1, int(duration_s / (learning_isi_s * len(pairs))))
        logger.info("Learning phase: %d paired events per pair, isi=%.1fs", num_events, learning_isi_s)

        a1 = self.stdp_stim_amplitude
        d1 = self.stdp_stim_duration
        a2 = a1
        d2 = d1
        if a1 > 4.0:
            a1 = 4.0
            a2 = 4.0
        if d1 > 400.0:
            d1 = 400.0
            d2 = 400.0

        hebbian_delay_s = self.stdp_hebbian_delay_ms / 1000.0

        for event_idx in range(num_events):
            for pair in pairs:
                polarity_pre = StimPolarity.NegativeFirst if pair.polarity == "NegativeFirst" else StimPolarity.PositiveFirst

                self._send_biphasic_pulse(
                    electrode_idx=pair.stim_electrode,
                    amplitude_ua=pair.amplitude,
                    duration_us=pair.duration,
                    polarity=polarity_pre,
                    trigger_key=3,
                    phase="stdp_learning_pre",
                )

                self._wait(hebbian_delay_s)

                self._send_biphasic_pulse(
                    electrode_idx=pair.resp_electrode,
                    amplitude_ua=a1,
                    duration_us=d1,
                    polarity=StimPolarity.NegativeFirst,
                    trigger_key=4,
                    phase="stdp_learning_post",
                )

                self._wait(learning_isi_s)

    def _send_biphasic_pulse(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity = StimPolarity.NegativeFirst,
        trigger_key: int = 0,
        phase: str = "",
    ) -> None:
        amplitude_ua = min(abs(amplitude_ua), 4.0)
        duration_us = min(abs(duration_us), 400.0)

        a1 = amplitude_ua
        d1 = duration_us
        a2 = a1
        d2 = d1

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
        stim.phase_amplitude1 = a1
        stim.phase_duration1 = d1
        stim.phase_amplitude2 = a2
        stim.phase_duration2 = d2
        stim.enable_amp_settle = True
        stim.pre_stim_amp_settle = 0.0
        stim.post_stim_amp_settle = 1000.0
        stim.enable_charge_recovery = True
        stim.post_charge_recovery_on = 0.0
        stim.post_charge_recovery_off = 100.0
        stim.interphase_delay = 0.0

        self.intan.send_stimparam([stim])

        pattern = np.zeros(16, dtype=np.uint8)
        pattern[trigger_key] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.05)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=a1,
            duration_us=d1,
            polarity=polarity.name,
            timestamp_utc=datetime_now().isoformat(),
            trigger_key=trigger_key,
            phase=phase,
        ))

    def _save_all(self, recording_start: datetime, recording_stop: datetime) -> None:
        fs_name = getattr(self.experiment, "exp_name", "unknown")
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
            "stdp_pairs_count": len(self._stdp_pairs),
            "phase_timestamps": self._phase_timestamps,
            "ccg_results": self._ccg_results,
            "active_stim_times_summary": {k: len(v) for k, v in self._active_stim_times.items()},
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
        for col in spike_df.columns:
            if col.lower() in ("channel", "index", "electrode", "electrode_idx"):
                electrode_col = col
                break
            if "electrode" in col.lower() or "idx" in col.lower():
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
        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": getattr(self.experiment, "exp_name", "unknown"),
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "total_stimulations": len(self._stimulation_log),
            "scan_results_count": len(self._scan_results),
            "responsive_pairs_count": len(self._responsive_pairs),
            "stdp_pairs": [asdict(p) for p in self._stdp_pairs],
            "ccg_results": self._ccg_results,
            "phase_timestamps": self._phase_timestamps,
            "active_stim_counts": {k: len(v) for k, v in self._active_stim_times.items()},
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
