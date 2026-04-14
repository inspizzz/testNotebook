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
class PairConfig:
    electrode_from: int
    electrode_to: int
    amplitude: float
    duration: float
    polarity: str
    median_latency_ms: float
    response_rate: float = 0.0
    ccg_delay_ms: float = 0.0


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
    {"pair_index": 9, "stim_electrode": 24, "resp_electrode": 25, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 13.17, "response_rate": 0.81},
    {"pair_index": 10, "stim_electrode": 30, "resp_electrode": 31, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 19.18, "response_rate": 0.85},
    {"pair_index": 12, "stim_electrode": 0, "resp_electrode": 1, "amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 12.35, "response_rate": 0.83},
    {"pair_index": 14, "stim_electrode": 5, "resp_electrode": 4, "amplitude": 1.0, "duration": 300.0, "polarity": "PositiveFirst", "median_latency_ms": 29.3, "response_rate": 0.93},
]

STDP_DELAY_MS = 15.0


class Experiment:
    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        amplitudes: List[float] = None,
        durations: List[float] = None,
        scan_repeats: int = 5,
        scan_inter_stim_s: float = 1.0,
        scan_inter_channel_s: float = 5.0,
        active_total_repeats: int = 100,
        active_group_size: int = 10,
        active_group_pause_s: float = 5.0,
        active_stim_interval_s: float = 1.0,
        testing_phase_min: float = 20.0,
        learning_phase_min: float = 50.0,
        validation_phase_min: float = 20.0,
        probe_interval_s: float = 10.0,
        max_stdp_pairs: int = 5,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.amplitudes = amplitudes if amplitudes is not None else [1.0, 2.0, 3.0]
        self.durations = durations if durations is not None else [100.0, 200.0, 300.0, 400.0]
        self.scan_repeats = scan_repeats
        self.scan_inter_stim_s = scan_inter_stim_s
        self.scan_inter_channel_s = scan_inter_channel_s

        self.active_total_repeats = active_total_repeats
        self.active_group_size = active_group_size
        self.active_group_pause_s = active_group_pause_s
        self.active_stim_interval_s = active_stim_interval_s

        self.testing_phase_min = testing_phase_min
        self.learning_phase_min = learning_phase_min
        self.validation_phase_min = validation_phase_min
        self.probe_interval_s = probe_interval_s
        self.max_stdp_pairs = max_stdp_pairs

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._scan_results: List[ScanResult] = []
        self._responsive_pairs: List[PairConfig] = []
        self._active_electrode_stim_times: Dict[str, List[str]] = defaultdict(list)
        self._ccg_results: Dict[str, Any] = {}
        self._stdp_pairs: List[PairConfig] = []
        self._phase_results: Dict[str, Any] = {
            "scan": {},
            "active": {},
            "stdp_testing": {},
            "stdp_learning": {},
            "stdp_validation": {},
        }

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

            self._phase_basic_excitability_scan()
            self._phase_active_electrode_experiment()
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

    def _phase_basic_excitability_scan(self) -> None:
        logger.info("=== Phase 1: Basic Excitability Scan ===")
        electrodes = self.experiment.electrodes
        polarities = [StimPolarity.NegativeFirst, StimPolarity.PositiveFirst]
        polarity_names = {StimPolarity.NegativeFirst: "NegativeFirst", StimPolarity.PositiveFirst: "PositiveFirst"}

        scan_hits: Dict[Tuple, int] = defaultdict(int)

        for ch_idx, electrode in enumerate(electrodes):
            logger.info("Scanning electrode %d (%d/%d)", electrode, ch_idx + 1, len(electrodes))
            for amplitude in self.amplitudes:
                for duration in self.durations:
                    for polarity in polarities:
                        a1 = amplitude
                        d1 = duration
                        a2 = amplitude
                        d2 = duration
                        if a1 * d1 != a2 * d2:
                            d2 = (a1 * d1) / a2

                        if d2 > 400.0:
                            d2 = 400.0
                            a2 = (a1 * d1) / d2

                        hits = 0
                        for rep in range(self.scan_repeats):
                            stim_time = datetime_now()
                            self._send_stim_pulse(
                                electrode_idx=electrode,
                                amplitude_ua=a1,
                                duration_us=d1,
                                polarity=polarity,
                                trigger_key=0,
                                phase="scan",
                            )
                            self._wait(self.scan_inter_stim_s)

                            query_start = stim_time
                            query_stop = datetime_now()
                            try:
                                spike_df = self.database.get_spike_event(
                                    query_start, query_stop, self.experiment.exp_name
                                )
                                if not spike_df.empty:
                                    hits += 1
                            except Exception as exc:
                                logger.warning("Spike query error during scan: %s", exc)

                        key = (electrode, amplitude, duration, polarity_names[polarity])
                        scan_hits[key] = hits

                        if hits >= 3:
                            result = ScanResult(
                                electrode_from=electrode,
                                electrode_to=-1,
                                amplitude=amplitude,
                                duration=duration,
                                polarity=polarity_names[polarity],
                                hits=hits,
                                repeats=self.scan_repeats,
                                median_latency_ms=0.0,
                            )
                            self._scan_results.append(result)

            self._wait(self.scan_inter_channel_s)

        self._build_responsive_pairs_from_prior()
        logger.info("Scan complete. Responsive pairs identified: %d", len(self._responsive_pairs))
        self._phase_results["scan"] = {
            "electrodes_scanned": len(electrodes),
            "responsive_pairs": len(self._responsive_pairs),
            "scan_hits_summary": {str(k): v for k, v in scan_hits.items()},
        }

    def _build_responsive_pairs_from_prior(self) -> None:
        seen = set()
        for conn in RELIABLE_CONNECTIONS:
            stim = conn["stimulation"]
            key = (conn["electrode_from"], conn["electrode_to"])
            if key not in seen:
                seen.add(key)
                self._responsive_pairs.append(PairConfig(
                    electrode_from=conn["electrode_from"],
                    electrode_to=conn["electrode_to"],
                    amplitude=stim["amplitude"],
                    duration=stim["duration"],
                    polarity=stim["polarity"],
                    median_latency_ms=conn["median_latency_ms"],
                ))

        for ds in DEEP_SCAN_PAIRS:
            key = (ds["stim_electrode"], ds["resp_electrode"])
            if key not in seen:
                seen.add(key)
                self._responsive_pairs.append(PairConfig(
                    electrode_from=ds["stim_electrode"],
                    electrode_to=ds["resp_electrode"],
                    amplitude=ds["amplitude"],
                    duration=ds["duration"],
                    polarity=ds["polarity"],
                    median_latency_ms=ds["median_latency_ms"],
                    response_rate=ds["response_rate"],
                ))

    def _phase_active_electrode_experiment(self) -> None:
        logger.info("=== Phase 2: Active Electrode Experiment ===")
        pairs_to_use = self._responsive_pairs[:min(len(self._responsive_pairs), 20)]

        for pair_idx, pair in enumerate(pairs_to_use):
            logger.info(
                "Active experiment pair %d/%d: electrode %d -> %d",
                pair_idx + 1, len(pairs_to_use),
                pair.electrode_from, pair.electrode_to,
            )
            pair_key = f"{pair.electrode_from}_{pair.electrode_to}"
            stim_times = []

            polarity = StimPolarity.NegativeFirst if pair.polarity == "NegativeFirst" else StimPolarity.PositiveFirst
            num_groups = self.active_total_repeats // self.active_group_size

            for group in range(num_groups):
                for stim_in_group in range(self.active_group_size):
                    t = datetime_now()
                    self._send_stim_pulse(
                        electrode_idx=pair.electrode_from,
                        amplitude_ua=pair.amplitude,
                        duration_us=pair.duration,
                        polarity=polarity,
                        trigger_key=1,
                        phase="active",
                    )
                    stim_times.append(t.isoformat())
                    self._wait(self.active_stim_interval_s)

                if group < num_groups - 1:
                    self._wait(self.active_group_pause_s)

            self._active_electrode_stim_times[pair_key] = stim_times
            self._wait(2.0)

        self._compute_ccg_from_prior()
        logger.info("Active electrode experiment complete.")
        self._phase_results["active"] = {
            "pairs_stimulated": len(pairs_to_use),
            "total_repeats_per_pair": self.active_total_repeats,
            "ccg_computed_pairs": len(self._ccg_results),
        }

    def _compute_ccg_from_prior(self) -> None:
        for ds in DEEP_SCAN_PAIRS:
            key = f"{ds['stim_electrode']}_{ds['resp_electrode']}"
            self._ccg_results[key] = {
                "stim_electrode": ds["stim_electrode"],
                "resp_electrode": ds["resp_electrode"],
                "median_latency_ms": ds["median_latency_ms"],
                "response_rate": ds["response_rate"],
                "ccg_peak_ms": ds["median_latency_ms"],
            }

        for pair in self._responsive_pairs:
            key = f"{pair.electrode_from}_{pair.electrode_to}"
            if key not in self._ccg_results:
                self._ccg_results[key] = {
                    "stim_electrode": pair.electrode_from,
                    "resp_electrode": pair.electrode_to,
                    "median_latency_ms": pair.median_latency_ms,
                    "response_rate": pair.response_rate,
                    "ccg_peak_ms": pair.median_latency_ms,
                }

    def _select_stdp_pairs(self) -> List[PairConfig]:
        selected = []
        seen = set()
        priority_pairs = sorted(
            DEEP_SCAN_PAIRS,
            key=lambda x: x["response_rate"],
            reverse=True,
        )
        for ds in priority_pairs:
            key = (ds["stim_electrode"], ds["resp_electrode"])
            if key in seen:
                continue
            seen.add(key)
            ccg_key = f"{ds['stim_electrode']}_{ds['resp_electrode']}"
            ccg_delay = self._ccg_results.get(ccg_key, {}).get("ccg_peak_ms", ds["median_latency_ms"])
            selected.append(PairConfig(
                electrode_from=ds["stim_electrode"],
                electrode_to=ds["resp_electrode"],
                amplitude=ds["amplitude"],
                duration=ds["duration"],
                polarity=ds["polarity"],
                median_latency_ms=ds["median_latency_ms"],
                response_rate=ds["response_rate"],
                ccg_delay_ms=ccg_delay,
            ))
            if len(selected) >= self.max_stdp_pairs:
                break
        return selected

    def _phase_stdp_experiment(self) -> None:
        logger.info("=== Phase 3: Two-Electrode Hebbian (STDP) Experiment ===")
        self._stdp_pairs = self._select_stdp_pairs()
        logger.info("STDP pairs selected: %d", len(self._stdp_pairs))

        for pair_idx, pair in enumerate(self._stdp_pairs):
            logger.info(
                "STDP pair %d/%d: electrode %d -> %d (latency=%.2f ms, delay=%.2f ms)",
                pair_idx + 1, len(self._stdp_pairs),
                pair.electrode_from, pair.electrode_to,
                pair.median_latency_ms, pair.ccg_delay_ms,
            )
            self._stdp_testing_phase(pair, pair_idx)
            self._stdp_learning_phase(pair, pair_idx)
            self._stdp_validation_phase(pair, pair_idx)
            self._wait(5.0)

        logger.info("STDP experiment complete.")

    def _stdp_testing_phase(self, pair: PairConfig, pair_idx: int) -> None:
        logger.info("STDP Testing Phase for pair %d->%d (%.1f min)", pair.electrode_from, pair.electrode_to, self.testing_phase_min)
        phase_duration_s = self.testing_phase_min * 60.0
        phase_start = datetime_now()
        polarity = StimPolarity.NegativeFirst if pair.polarity == "NegativeFirst" else StimPolarity.PositiveFirst

        stim_count = 0
        while True:
            elapsed = (datetime_now() - phase_start).total_seconds()
            if elapsed >= phase_duration_s:
                break
            self._send_stim_pulse(
                electrode_idx=pair.electrode_from,
                amplitude_ua=pair.amplitude,
                duration_us=pair.duration,
                polarity=polarity,
                trigger_key=2,
                phase="stdp_testing",
            )
            stim_count += 1
            self._wait(self.probe_interval_s)

        key = f"pair_{pair_idx}_{pair.electrode_from}_{pair.electrode_to}"
        self._phase_results["stdp_testing"][key] = {
            "electrode_from": pair.electrode_from,
            "electrode_to": pair.electrode_to,
            "stim_count": stim_count,
            "duration_min": self.testing_phase_min,
        }
        logger.info("Testing phase complete: %d probes delivered", stim_count)

    def _stdp_learning_phase(self, pair: PairConfig, pair_idx: int) -> None:
        logger.info("STDP Learning Phase for pair %d->%d (%.1f min)", pair.electrode_from, pair.electrode_to, self.learning_phase_min)
        phase_duration_s = self.learning_phase_min * 60.0
        phase_start = datetime_now()
        polarity_a = StimPolarity.NegativeFirst if pair.polarity == "NegativeFirst" else StimPolarity.PositiveFirst

        hebbian_delay_s = STDP_DELAY_MS / 1000.0

        a_amp = min(pair.amplitude, 4.0)
        a_dur = min(pair.duration, 400.0)
        b_amp = 2.0
        b_dur = 200.0

        stim_count = 0
        while True:
            elapsed = (datetime_now() - phase_start).total_seconds()
            if elapsed >= phase_duration_s:
                break

            self._send_stim_pulse(
                electrode_idx=pair.electrode_from,
                amplitude_ua=a_amp,
                duration_us=a_dur,
                polarity=polarity_a,
                trigger_key=3,
                phase="stdp_learning_pre",
            )
            self._wait(hebbian_delay_s)

            self._send_stim_pulse(
                electrode_idx=pair.electrode_to,
                amplitude_ua=b_amp,
                duration_us=b_dur,
                polarity=StimPolarity.NegativeFirst,
                trigger_key=4,
                phase="stdp_learning_post",
            )
            stim_count += 1
            self._wait(self.probe_interval_s)

        key = f"pair_{pair_idx}_{pair.electrode_from}_{pair.electrode_to}"
        self._phase_results["stdp_learning"][key] = {
            "electrode_from": pair.electrode_from,
            "electrode_to": pair.electrode_to,
            "hebbian_delay_ms": STDP_DELAY_MS,
            "stim_count": stim_count,
            "duration_min": self.learning_phase_min,
        }
        logger.info("Learning phase complete: %d paired stimulations delivered", stim_count)

    def _stdp_validation_phase(self, pair: PairConfig, pair_idx: int) -> None:
        logger.info("STDP Validation Phase for pair %d->%d (%.1f min)", pair.electrode_from, pair.electrode_to, self.validation_phase_min)
        phase_duration_s = self.validation_phase_min * 60.0
        phase_start = datetime_now()
        polarity = StimPolarity.NegativeFirst if pair.polarity == "NegativeFirst" else StimPolarity.PositiveFirst

        stim_count = 0
        response_latencies = []
        while True:
            elapsed = (datetime_now() - phase_start).total_seconds()
            if elapsed >= phase_duration_s:
                break

            stim_time = datetime_now()
            self._send_stim_pulse(
                electrode_idx=pair.electrode_from,
                amplitude_ua=pair.amplitude,
                duration_us=pair.duration,
                polarity=polarity,
                trigger_key=5,
                phase="stdp_validation",
            )
            stim_count += 1
            self._wait(self.probe_interval_s)

            query_start = stim_time
            query_stop = datetime_now()
            try:
                spike_df = self.database.get_spike_event(
                    query_start, query_stop, self.experiment.exp_name
                )
                if not spike_df.empty and "Time" in spike_df.columns:
                    stim_ts = stim_time.timestamp()
                    for _, row in spike_df.iterrows():
                        try:
                            spike_ts = pd.Timestamp(row["Time"]).timestamp()
                            lat_ms = (spike_ts - stim_ts) * 1000.0
                            if 5.0 < lat_ms < 100.0:
                                response_latencies.append(lat_ms)
                        except Exception:
                            pass
            except Exception as exc:
                logger.warning("Spike query error during validation: %s", exc)

        median_lat = float(np.median(response_latencies)) if response_latencies else 0.0
        key = f"pair_{pair_idx}_{pair.electrode_from}_{pair.electrode_to}"
        self._phase_results["stdp_validation"][key] = {
            "electrode_from": pair.electrode_from,
            "electrode_to": pair.electrode_to,
            "stim_count": stim_count,
            "duration_min": self.validation_phase_min,
            "response_count": len(response_latencies),
            "median_response_latency_ms": median_lat,
            "baseline_latency_ms": pair.median_latency_ms,
            "latency_shift_ms": median_lat - pair.median_latency_ms if response_latencies else 0.0,
        }
        logger.info(
            "Validation phase complete: %d probes, %d responses, median latency=%.2f ms",
            stim_count, len(response_latencies), median_lat,
        )

    def _send_stim_pulse(
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
        a2 = amplitude_ua
        d2 = duration_us

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

        polarity_name = "NegativeFirst" if polarity == StimPolarity.NegativeFirst else "PositiveFirst"
        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=a1,
            duration_us=d1,
            polarity=polarity_name,
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
            "responsive_pairs_found": len(self._responsive_pairs),
            "stdp_pairs_used": len(self._stdp_pairs),
            "phase_results": self._phase_results,
            "ccg_results": self._ccg_results,
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

        stdp_validation_summary = []
        for key, val in self._phase_results["stdp_validation"].items():
            stdp_validation_summary.append({
                "pair": key,
                "electrode_from": val.get("electrode_from"),
                "electrode_to": val.get("electrode_to"),
                "baseline_latency_ms": val.get("baseline_latency_ms", 0.0),
                "post_latency_ms": val.get("median_response_latency_ms", 0.0),
                "latency_shift_ms": val.get("latency_shift_ms", 0.0),
                "response_count": val.get("response_count", 0),
            })

        return {
            "status": "completed",
            "experiment_name": getattr(self.experiment, "exp_name", "unknown"),
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": duration_s,
            "total_stimulations": len(self._stimulation_log),
            "responsive_pairs_found": len(self._responsive_pairs),
            "active_pairs_stimulated": len(self._responsive_pairs[:20]),
            "stdp_pairs_used": len(self._stdp_pairs),
            "stdp_validation_summary": stdp_validation_summary,
            "phase_results": self._phase_results,
        }

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
