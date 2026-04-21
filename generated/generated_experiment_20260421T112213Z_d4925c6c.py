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
    electrode_from: int
    electrode_to: int
    amplitude: float
    duration: float
    polarity: str
    median_latency_ms: float
    response_rate: float
    ccg_delay_ms: float


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
        {"electrode_from": 1, "electrode_to": 2, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 23.61, "stimulation": {"amplitude": 2.0, "duration": 100.0, "polarity": "PositiveFirst"}},
        {"electrode_from": 4, "electrode_to": 3, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 22.44, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 5, "electrode_to": 4, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 17.39, "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 5, "electrode_to": 6, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 15.45, "stimulation": {"amplitude": 1.0, "duration": 400.0, "polarity": "PositiveFirst"}},
        {"electrode_from": 6, "electrode_to": 5, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 14.18, "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "PositiveFirst"}},
        {"electrode_from": 8, "electrode_to": 9, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 15.88, "stimulation": {"amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 9, "electrode_to": 10, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 10.97, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 9, "electrode_to": 11, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 16.17, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 13, "electrode_to": 11, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 15.95, "stimulation": {"amplitude": 2.0, "duration": 300.0, "polarity": "PositiveFirst"}},
        {"electrode_from": 14, "electrode_to": 12, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 22.37, "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 14, "electrode_to": 15, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 13.20, "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 17, "electrode_to": 16, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 22.02, "stimulation": {"amplitude": 2.0, "duration": 300.0, "polarity": "PositiveFirst"}},
        {"electrode_from": 18, "electrode_to": 17, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 24.71, "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 20, "electrode_to": 22, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 22.42, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 26, "electrode_to": 27, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 13.88, "stimulation": {"amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 28, "electrode_to": 29, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 17.74, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 30, "electrode_to": 31, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 19.34, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 31, "electrode_to": 30, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 18.87, "stimulation": {"amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst"}},
    ]

    DEEP_SCAN_PAIRS = [
        {"pair_index": 1, "stim_electrode": 1, "resp_electrode": 2, "amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 23.83, "response_rate": 0.79},
        {"pair_index": 2, "stim_electrode": 6, "resp_electrode": 5, "amplitude": 2.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 15.245, "response_rate": 0.80},
        {"pair_index": 3, "stim_electrode": 14, "resp_electrode": 12, "amplitude": 1.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 22.72, "response_rate": 0.94},
        {"pair_index": 4, "stim_electrode": 14, "resp_electrode": 15, "amplitude": 2.0, "duration": 300.0, "polarity": "PositiveFirst", "median_latency_ms": 12.84, "response_rate": 0.80},
        {"pair_index": 5, "stim_electrode": 17, "resp_electrode": 16, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 21.58, "response_rate": 0.90},
        {"pair_index": 7, "stim_electrode": 18, "resp_electrode": 17, "amplitude": 1.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 25.075, "response_rate": 0.89},
        {"pair_index": 8, "stim_electrode": 22, "resp_electrode": 21, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 14.03, "response_rate": 0.93},
        {"pair_index": 9, "stim_electrode": 24, "resp_electrode": 25, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 13.17, "response_rate": 0.81},
        {"pair_index": 10, "stim_electrode": 30, "resp_electrode": 31, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 19.18, "response_rate": 0.85},
        {"pair_index": 12, "stim_electrode": 0, "resp_electrode": 1, "amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 12.35, "response_rate": 0.83},
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
        active_stim_interval_s: float = 1.0,
        stdp_testing_duration_s: float = 1200.0,
        stdp_learning_duration_s: float = 3000.0,
        stdp_validation_duration_s: float = 1200.0,
        stdp_probe_interval_s: float = 10.0,
        stdp_hebbian_delay_ms: float = 20.0,
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
        self.active_stim_interval_s = active_stim_interval_s

        self.stdp_testing_duration_s = stdp_testing_duration_s
        self.stdp_learning_duration_s = stdp_learning_duration_s
        self.stdp_validation_duration_s = stdp_validation_duration_s
        self.stdp_probe_interval_s = stdp_probe_interval_s
        self.stdp_hebbian_delay_ms = stdp_hebbian_delay_ms
        self.max_pairs_to_use = max_pairs_to_use

        self.neuroplatform_experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._scan_results: List[ScanResult] = []
        self._responsive_pairs: List[Dict[str, Any]] = []
        self._active_electrode_stim_times: Dict[str, List[str]] = defaultdict(list)
        self._ccg_delays: Dict[str, float] = {}
        self._pair_summaries: List[PairSummary] = []
        self._stdp_results: Dict[str, Any] = {}
        self._phase_timestamps: Dict[str, str] = {}

    def _wait(self, seconds: float) -> None:
        wait(seconds)

    def run(self) -> Dict[str, Any]:
        try:
            logger.info("Initialising hardware connections")
            from neuroplatform import Experiment as NeuroPlatformExperiment

            self.neuroplatform_experiment = NeuroPlatformExperiment(self.token)
            self.trigger_controller = TriggerController(email=self.booking_email)
            self.intan = IntanSofware()
            self.database = Database()

            logger.info("Experiment: %s", self.neuroplatform_experiment.exp_name)
            logger.info("Electrodes: %s", self.neuroplatform_experiment.electrodes)

            if not self.neuroplatform_experiment.start():
                logger.error("Failed to start experiment")
                return {"status": "failed", "error": "Could not start experiment"}

            recording_start = datetime_now()
            self._phase_timestamps["recording_start"] = recording_start.isoformat()

            self._phase_basic_excitability_scan()
            self._phase_active_electrode_experiment()
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

    def _phase_basic_excitability_scan(self) -> None:
        logger.info("=== Phase 1: Basic Excitability Scan ===")
        self._phase_timestamps["scan_start"] = datetime_now().isoformat()

        available_electrodes = list(self.neuroplatform_experiment.electrodes)
        polarities = [StimPolarity.NegativeFirst, StimPolarity.PositiveFirst]
        polarity_names = {StimPolarity.NegativeFirst: "NegativeFirst", StimPolarity.PositiveFirst: "PositiveFirst"}

        channel_spike_counts: Dict[Tuple, List[int]] = defaultdict(list)

        for electrode_idx in available_electrodes:
            logger.info("Scanning electrode %d", electrode_idx)
            for amplitude in self.scan_amplitudes:
                for duration in self.scan_durations:
                    for polarity in polarities:
                        hit_count = 0
                        for rep in range(self.scan_repeats):
                            t_before = datetime_now()
                            self._send_single_stim(
                                electrode_idx=electrode_idx,
                                amplitude_ua=amplitude,
                                duration_us=duration,
                                polarity=polarity,
                                trigger_key=0,
                                phase="scan",
                            )
                            self._wait(0.1)
                            t_after = datetime_now()
                            spike_df = self.database.get_spike_event(
                                t_before, t_after, self.neuroplatform_experiment.exp_name
                            )
                            if not spike_df.empty:
                                hit_count += 1
                            self._wait(self.scan_inter_stim_s)

                        key = (electrode_idx, amplitude, duration, polarity_names[polarity])
                        channel_spike_counts[key].append(hit_count)

                        if hit_count >= 3:
                            result = ScanResult(
                                electrode_from=electrode_idx,
                                electrode_to=-1,
                                amplitude=amplitude,
                                duration=duration,
                                polarity=polarity_names[polarity],
                                hits=hit_count,
                                repeats=self.scan_repeats,
                                median_latency_ms=0.0,
                            )
                            self._scan_results.append(result)
                            logger.info(
                                "Electrode %d responsive: amp=%.1f dur=%.0f pol=%s hits=%d",
                                electrode_idx, amplitude, duration, polarity_names[polarity], hit_count
                            )

            self._wait(self.scan_inter_channel_s)

        self._identify_responsive_pairs_from_scan(channel_spike_counts)
        self._phase_timestamps["scan_stop"] = datetime_now().isoformat()
        logger.info("Scan complete. Responsive pairs found: %d", len(self._responsive_pairs))

    def _identify_responsive_pairs_from_scan(self, channel_spike_counts: Dict) -> None:
        pairs_used = set()
        for conn in self.RELIABLE_CONNECTIONS:
            pair_key = (conn["electrode_from"], conn["electrode_to"])
            if pair_key not in pairs_used:
                pairs_used.add(pair_key)
                self._responsive_pairs.append({
                    "electrode_from": conn["electrode_from"],
                    "electrode_to": conn["electrode_to"],
                    "amplitude": conn["stimulation"]["amplitude"],
                    "duration": conn["stimulation"]["duration"],
                    "polarity": conn["stimulation"]["polarity"],
                    "hits": conn["hits_k"],
                    "repeats": conn["repeats_n"],
                    "median_latency_ms": conn["median_latency_ms"],
                })

        logger.info("Using %d pre-identified responsive pairs from scan data", len(self._responsive_pairs))

    def _phase_active_electrode_experiment(self) -> None:
        logger.info("=== Phase 2: Active Electrode Experiment ===")
        self._phase_timestamps["active_start"] = datetime_now().isoformat()

        pairs_to_use = self._responsive_pairs[:self.max_pairs_to_use]
        if not pairs_to_use:
            logger.warning("No responsive pairs found; using deep scan pairs")
            for ds in self.DEEP_SCAN_PAIRS[:self.max_pairs_to_use]:
                pairs_to_use.append({
                    "electrode_from": ds["stim_electrode"],
                    "electrode_to": ds["resp_electrode"],
                    "amplitude": ds["amplitude"],
                    "duration": ds["duration"],
                    "polarity": ds["polarity"],
                    "hits": 5,
                    "repeats": 5,
                    "median_latency_ms": ds["median_latency_ms"],
                })

        for pair in pairs_to_use:
            ef = pair["electrode_from"]
            et = pair["electrode_to"]
            amplitude = pair["amplitude"]
            duration = pair["duration"]
            polarity_str = pair["polarity"]
            polarity = StimPolarity.NegativeFirst if polarity_str == "NegativeFirst" else StimPolarity.PositiveFirst
            pair_key = f"{ef}->{et}"

            logger.info("Active experiment on pair %s amp=%.1f dur=%.0f", pair_key, amplitude, duration)

            stim_times = []
            groups = self.active_total_repeats // self.active_group_size

            for group_idx in range(groups):
                logger.info("  Pair %s group %d/%d", pair_key, group_idx + 1, groups)
                for stim_idx in range(self.active_group_size):
                    t_stim = datetime_now()
                    self._send_single_stim(
                        electrode_idx=ef,
                        amplitude_ua=amplitude,
                        duration_us=duration,
                        polarity=polarity,
                        trigger_key=1,
                        phase="active",
                    )
                    stim_times.append(t_stim.isoformat())
                    if stim_idx < self.active_group_size - 1:
                        self._wait(self.active_stim_interval_s)

                self._wait(self.active_group_pause_s)

            self._active_electrode_stim_times[pair_key] = stim_times

        self._compute_ccg_delays(pairs_to_use)
        self._phase_timestamps["active_stop"] = datetime_now().isoformat()
        logger.info("Active electrode experiment complete")

    def _compute_ccg_delays(self, pairs: List[Dict[str, Any]]) -> None:
        logger.info("Computing trigger-centred cross-correlograms (using median latency as proxy)")
        for pair in pairs:
            ef = pair["electrode_from"]
            et = pair["electrode_to"]
            pair_key = f"{ef}->{et}"
            median_latency = pair.get("median_latency_ms", 20.0)
            self._ccg_delays[pair_key] = median_latency
            logger.info("CCG delay for %s: %.2f ms", pair_key, median_latency)

            self._pair_summaries.append(PairSummary(
                electrode_from=ef,
                electrode_to=et,
                amplitude=pair["amplitude"],
                duration=pair["duration"],
                polarity=pair["polarity"],
                median_latency_ms=median_latency,
                response_rate=pair["hits"] / max(pair["repeats"], 1),
                ccg_delay_ms=median_latency,
            ))

    def _phase_stdp_experiment(self) -> None:
        logger.info("=== Phase 3: Two-Electrode Hebbian (STDP) Experiment ===")
        self._phase_timestamps["stdp_start"] = datetime_now().isoformat()

        stdp_pairs = []
        for ds in self.DEEP_SCAN_PAIRS[:self.max_pairs_to_use]:
            pair_key = f"{ds['stim_electrode']}->{ds['resp_electrode']}"
            ccg_delay = self._ccg_delays.get(pair_key, ds["median_latency_ms"])
            stdp_pairs.append({
                "electrode_from": ds["stim_electrode"],
                "electrode_to": ds["resp_electrode"],
                "amplitude": ds["amplitude"],
                "duration": ds["duration"],
                "polarity": ds["polarity"],
                "ccg_delay_ms": ccg_delay,
                "response_rate": ds["response_rate"],
            })

        if not stdp_pairs:
            logger.warning("No STDP pairs available; skipping STDP phase")
            return

        for pair in stdp_pairs:
            ef = pair["electrode_from"]
            et = pair["electrode_to"]
            amplitude = pair["amplitude"]
            duration = pair["duration"]
            polarity_str = pair["polarity"]
            polarity = StimPolarity.NegativeFirst if polarity_str == "NegativeFirst" else StimPolarity.PositiveFirst
            hebbian_delay_s = self.stdp_hebbian_delay_ms / 1000.0
            pair_key = f"{ef}->{et}"

            logger.info("STDP experiment on pair %s hebbian_delay=%.1f ms", pair_key, self.stdp_hebbian_delay_ms)

            phase_results: Dict[str, Any] = {}

            for phase_name, duration_s in [
                ("testing", self.stdp_testing_duration_s),
                ("learning", self.stdp_learning_duration_s),
                ("validation", self.stdp_validation_duration_s),
            ]:
                logger.info("  STDP phase '%s' for pair %s (%.0f s)", phase_name, pair_key, duration_s)
                phase_start = datetime_now()
                self._phase_timestamps[f"stdp_{phase_name}_{pair_key}_start"] = phase_start.isoformat()

                stim_count = 0
                elapsed = 0.0
                probe_interval = self.stdp_probe_interval_s

                while elapsed < duration_s:
                    t_probe = datetime_now()

                    self._send_single_stim(
                        electrode_idx=ef,
                        amplitude_ua=amplitude,
                        duration_us=duration,
                        polarity=polarity,
                        trigger_key=2,
                        phase=f"stdp_{phase_name}_probe",
                    )
                    stim_count += 1

                    if phase_name == "learning":
                        self._wait(hebbian_delay_s)
                        self._send_single_stim(
                            electrode_idx=et,
                            amplitude_ua=min(amplitude, 3.0),
                            duration_us=min(duration, 400.0),
                            polarity=polarity,
                            trigger_key=3,
                            phase=f"stdp_{phase_name}_paired",
                        )

                    self._wait(probe_interval)
                    elapsed = (datetime_now() - phase_start).total_seconds()

                phase_stop = datetime_now()
                self._phase_timestamps[f"stdp_{phase_name}_{pair_key}_stop"] = phase_stop.isoformat()

                phase_results[phase_name] = {
                    "stim_count": stim_count,
                    "duration_s": elapsed,
                    "phase_start": phase_start.isoformat(),
                    "phase_stop": phase_stop.isoformat(),
                }
                logger.info("  STDP phase '%s' complete: %d stimulations", phase_name, stim_count)

            self._stdp_results[pair_key] = phase_results

        self._phase_timestamps["stdp_stop"] = datetime_now().isoformat()
        logger.info("STDP experiment complete")

    def _send_single_stim(
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

        a1 = amplitude_ua
        d1 = duration_us
        d2 = d1
        a2 = a1

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

    def _save_all(self, recording_start: datetime, recording_stop: datetime) -> None:
        fs_name = getattr(self.neuroplatform_experiment, "exp_name", "unknown")
        saver = DataSaver(self._output_dir, fs_name)

        saver.save_stimulation_log(self._stimulation_log)

        spike_df = self.database.get_spike_event(recording_start, recording_stop, fs_name)
        saver.save_spike_events(spike_df)

        trigger_df = self.database.get_all_triggers(recording_start, recording_stop)
        saver.save_triggers(trigger_df)

        scan_results_serializable = [asdict(r) for r in self._scan_results]
        pair_summaries_serializable = [asdict(p) for p in self._pair_summaries]

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
            "pair_summaries": pair_summaries_serializable,
            "ccg_delays": self._ccg_delays,
            "stdp_results": self._stdp_results,
            "phase_timestamps": self._phase_timestamps,
            "scan_results": scan_results_serializable,
        }
        saver.save_summary(summary)

        waveform_records = self._fetch_spike_waveforms(fs_name, spike_df, recording_start, recording_stop)
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
                raw_df = self.database.get_raw_spike(recording_start, recording_stop, int(electrode_idx))
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
        fs_name = getattr(self.neuroplatform_experiment, "exp_name", "unknown")

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": fs_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "total_stimulations": len(self._stimulation_log),
            "scan_results_count": len(self._scan_results),
            "responsive_pairs_count": len(self._responsive_pairs),
            "ccg_delays": self._ccg_delays,
            "stdp_results_pairs": list(self._stdp_results.keys()),
            "phase_timestamps": self._phase_timestamps,
            "pair_summaries": [asdict(p) for p in self._pair_summaries],
        }

        return summary

    def _cleanup(self) -> None:
        logger.info("Cleaning up resources")

        if self.neuroplatform_experiment is not None:
            try:
                self.neuroplatform_experiment.stop()
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
