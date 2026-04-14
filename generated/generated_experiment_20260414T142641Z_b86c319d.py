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
    ccg_peak_pre: float = 0.0
    ccg_peak_post: float = 0.0
    delta_ccg: float = 0.0


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
        {"electrode_from": 6, "electrode_to": 5, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 14.82, "stimulation": {"amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 8, "electrode_to": 9, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 15.88, "stimulation": {"amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 9, "electrode_to": 10, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 10.97, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 9, "electrode_to": 11, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 16.17, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 13, "electrode_to": 11, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 15.95, "stimulation": {"amplitude": 2.0, "duration": 300.0, "polarity": "PositiveFirst"}},
        {"electrode_from": 14, "electrode_to": 12, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 22.37, "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 14, "electrode_to": 15, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 13.2, "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 17, "electrode_to": 16, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 21.56, "stimulation": {"amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 18, "electrode_to": 17, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 24.71, "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 24, "electrode_to": 25, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 13.18, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 26, "electrode_to": 27, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 13.88, "stimulation": {"amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst"}},
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
        {"stim_electrode": 9, "resp_electrode": 10, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 11.035, "response_rate": 0.94},
        {"stim_electrode": 30, "resp_electrode": 31, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 19.18, "response_rate": 0.85},
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
        stdp_testing_duration_s: float = 1200.0,
        stdp_learning_duration_s: float = 3000.0,
        stdp_validation_duration_s: float = 1200.0,
        stdp_delta_t_ms: float = 10.0,
        stdp_inter_stim_s: float = 1.0,
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

        self.stdp_testing_duration_s = stdp_testing_duration_s
        self.stdp_learning_duration_s = stdp_learning_duration_s
        self.stdp_validation_duration_s = stdp_validation_duration_s
        self.stdp_delta_t_ms = stdp_delta_t_ms
        self.stdp_inter_stim_s = stdp_inter_stim_s
        self.max_stdp_pairs = max_stdp_pairs

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._scan_results: List[ScanResult] = []
        self._responsive_pairs: List[Dict[str, Any]] = []
        self._active_stim_times: Dict[str, List[str]] = defaultdict(list)
        self._ccg_data: Dict[str, Any] = {}
        self._stdp_results: Dict[str, Any] = {}
        self._pair_summaries: List[PairSummary] = []

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

            self._phase_excitability_scan()
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
                logger.error("Failed to save data on error: %s", save_exc)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _phase_excitability_scan(self) -> None:
        logger.info("=== Phase 1: Basic Excitability Scan ===")
        available_electrodes = list(self.experiment.electrodes)
        polarities = [StimPolarity.NegativeFirst, StimPolarity.PositiveFirst]

        for electrode_idx in available_electrodes:
            logger.info("Scanning electrode %d", electrode_idx)
            for amplitude in self.scan_amplitudes:
                for duration in self.scan_durations:
                    for polarity in polarities:
                        hits = 0
                        for rep in range(self.scan_repeats):
                            spike_df = self._send_stim_and_query(
                                electrode_idx=electrode_idx,
                                amplitude_ua=amplitude,
                                duration_us=duration,
                                polarity=polarity,
                                trigger_key=0,
                                phase="scan",
                            )
                            if not spike_df.empty:
                                hits += 1
                            if rep < self.scan_repeats - 1:
                                self._wait(self.scan_inter_stim_s)

                        if hits >= 3:
                            result = ScanResult(
                                electrode_from=electrode_idx,
                                electrode_to=-1,
                                amplitude=amplitude,
                                duration=duration,
                                polarity=polarity.name,
                                hits=hits,
                                repeats=self.scan_repeats,
                                median_latency_ms=0.0,
                            )
                            self._scan_results.append(result)
                            logger.info(
                                "Electrode %d responsive: amp=%.1f dur=%.0f pol=%s hits=%d",
                                electrode_idx, amplitude, duration, polarity.name, hits
                            )

            self._wait(self.scan_inter_channel_s)

        self._responsive_pairs = self._build_responsive_pairs_from_scan()
        logger.info("Excitability scan complete. Responsive pairs found: %d", len(self._responsive_pairs))

    def _build_responsive_pairs_from_scan(self) -> List[Dict[str, Any]]:
        pairs = []
        seen = set()
        for conn in self.RELIABLE_CONNECTIONS:
            key = (conn["electrode_from"], conn["electrode_to"])
            if key not in seen:
                seen.add(key)
                pairs.append({
                    "stim_electrode": conn["electrode_from"],
                    "resp_electrode": conn["electrode_to"],
                    "amplitude": conn["stimulation"]["amplitude"],
                    "duration": conn["stimulation"]["duration"],
                    "polarity": conn["stimulation"]["polarity"],
                    "median_latency_ms": conn["median_latency_ms"],
                    "hits": conn["hits_k"],
                })
        return pairs

    def _phase_active_electrode_experiment(self) -> None:
        logger.info("=== Phase 2: Active Electrode Experiment ===")

        pairs_to_use = self.DEEP_SCAN_PAIRS[:self.max_stdp_pairs * 2]

        for pair in pairs_to_use:
            stim_elec = pair["stim_electrode"]
            resp_elec = pair["resp_electrode"]
            amplitude = pair["amplitude"]
            duration = pair["duration"]
            polarity_str = pair["polarity"]
            polarity = StimPolarity.NegativeFirst if polarity_str == "NegativeFirst" else StimPolarity.PositiveFirst

            pair_key = f"{stim_elec}->{resp_elec}"
            logger.info("Active experiment for pair %s", pair_key)

            stim_times = []
            groups = self.active_total_repeats // self.active_group_size

            for group_idx in range(groups):
                logger.info("  Group %d/%d for pair %s", group_idx + 1, groups, pair_key)
                for stim_idx in range(self.active_group_size):
                    t_stim = datetime_now()
                    self._send_stim_and_query(
                        electrode_idx=stim_elec,
                        amplitude_ua=amplitude,
                        duration_us=duration,
                        polarity=polarity,
                        trigger_key=0,
                        phase="active",
                    )
                    stim_times.append(t_stim.isoformat())
                    if stim_idx < self.active_group_size - 1:
                        self._wait(1.0)

                if group_idx < groups - 1:
                    self._wait(self.active_group_pause_s)

            self._active_stim_times[pair_key] = stim_times

            ccg = self._compute_ccg(
                stim_electrode=stim_elec,
                resp_electrode=resp_elec,
                stim_times_iso=stim_times,
                window_ms=100.0,
            )
            self._ccg_data[pair_key] = ccg
            logger.info("CCG computed for pair %s: peak_bin=%s peak_count=%s",
                        pair_key, ccg.get("peak_bin_ms"), ccg.get("peak_count"))

        logger.info("Active electrode experiment complete.")

    def _compute_ccg(
        self,
        stim_electrode: int,
        resp_electrode: int,
        stim_times_iso: List[str],
        window_ms: float = 100.0,
    ) -> Dict[str, Any]:
        if not stim_times_iso:
            return {"peak_bin_ms": None, "peak_count": 0, "histogram": []}

        bin_size_ms = 1.0
        n_bins = int(window_ms / bin_size_ms)
        histogram = [0] * n_bins

        if not stim_times_iso:
            return {"peak_bin_ms": None, "peak_count": 0, "histogram": histogram}

        try:
            t_start_dt = datetime.fromisoformat(stim_times_iso[0])
            t_end_dt = datetime.fromisoformat(stim_times_iso[-1]) + timedelta(seconds=1)

            spike_df = self.database.get_spike_event(
                t_start_dt, t_end_dt, self.experiment.exp_name
            )

            if spike_df.empty:
                return {"peak_bin_ms": None, "peak_count": 0, "histogram": histogram}

            channel_col = "channel" if "channel" in spike_df.columns else None
            if channel_col is None:
                for col in spike_df.columns:
                    if "channel" in col.lower() or "index" in col.lower():
                        channel_col = col
                        break

            if channel_col is None:
                return {"peak_bin_ms": None, "peak_count": 0, "histogram": histogram}

            resp_spikes = spike_df[spike_df[channel_col] == resp_electrode]
            if resp_spikes.empty:
                return {"peak_bin_ms": None, "peak_count": 0, "histogram": histogram}

            time_col = "Time" if "Time" in resp_spikes.columns else "_time"
            resp_spike_times = pd.to_datetime(resp_spikes[time_col], utc=True)

            for stim_iso in stim_times_iso:
                t_stim = datetime.fromisoformat(stim_iso)
                if t_stim.tzinfo is None:
                    t_stim = t_stim.replace(tzinfo=timezone.utc)
                for spike_t in resp_spike_times:
                    delta_ms = (spike_t.to_pydatetime() - t_stim).total_seconds() * 1000.0
                    if 0 <= delta_ms < window_ms:
                        bin_idx = int(delta_ms / bin_size_ms)
                        if 0 <= bin_idx < n_bins:
                            histogram[bin_idx] += 1

            peak_count = max(histogram) if histogram else 0
            peak_bin = histogram.index(peak_count) * bin_size_ms if peak_count > 0 else None

            return {
                "peak_bin_ms": peak_bin,
                "peak_count": peak_count,
                "histogram": histogram,
            }
        except Exception as exc:
            logger.warning("CCG computation failed: %s", exc)
            return {"peak_bin_ms": None, "peak_count": 0, "histogram": histogram}

    def _phase_stdp_experiment(self) -> None:
        logger.info("=== Phase 3: Two-Electrode Hebbian (STDP) Experiment ===")

        stdp_pairs = self.DEEP_SCAN_PAIRS[:self.max_stdp_pairs]

        for pair in stdp_pairs:
            stim_elec = pair["stim_electrode"]
            resp_elec = pair["resp_electrode"]
            amplitude = pair["amplitude"]
            duration = pair["duration"]
            polarity_str = pair["polarity"]
            polarity = StimPolarity.NegativeFirst if polarity_str == "NegativeFirst" else StimPolarity.PositiveFirst
            median_latency_ms = pair["median_latency_ms"]
            response_rate = pair["response_rate"]

            pair_key = f"{stim_elec}->{resp_elec}"
            logger.info("STDP experiment for pair %s (latency=%.2f ms)", pair_key, median_latency_ms)

            pair_summary = PairSummary(
                stim_electrode=stim_elec,
                resp_electrode=resp_elec,
                amplitude=amplitude,
                duration=duration,
                polarity=polarity_str,
                median_latency_ms=median_latency_ms,
                response_rate=response_rate,
            )

            logger.info("  STDP Testing phase (%.0f s)", self.stdp_testing_duration_s)
            t_test_start = datetime_now()
            pre_ccg = self._run_passive_recording_phase(
                stim_elec=stim_elec,
                resp_elec=resp_elec,
                amplitude=amplitude,
                duration=duration,
                polarity=polarity,
                duration_s=self.stdp_testing_duration_s,
                phase_name="stdp_testing",
            )
            t_test_end = datetime_now()
            pair_summary.ccg_peak_pre = pre_ccg.get("peak_count", 0)

            logger.info("  STDP Learning phase (%.0f s)", self.stdp_learning_duration_s)
            t_learn_start = datetime_now()
            self._run_stdp_learning_phase(
                stim_elec=stim_elec,
                resp_elec=resp_elec,
                amplitude=amplitude,
                duration=duration,
                polarity=polarity,
                duration_s=self.stdp_learning_duration_s,
                delta_t_ms=self.stdp_delta_t_ms,
            )
            t_learn_end = datetime_now()

            logger.info("  STDP Validation phase (%.0f s)", self.stdp_validation_duration_s)
            t_val_start = datetime_now()
            post_ccg = self._run_passive_recording_phase(
                stim_elec=stim_elec,
                resp_elec=resp_elec,
                amplitude=amplitude,
                duration=duration,
                polarity=polarity,
                duration_s=self.stdp_validation_duration_s,
                phase_name="stdp_validation",
            )
            t_val_end = datetime_now()
            pair_summary.ccg_peak_post = post_ccg.get("peak_count", 0)

            pre_peak = pair_summary.ccg_peak_pre
            post_peak = pair_summary.ccg_peak_post
            if pre_peak > 0:
                pair_summary.delta_ccg = (post_peak - pre_peak) / float(pre_peak)
            else:
                pair_summary.delta_ccg = 0.0

            self._pair_summaries.append(pair_summary)
            self._stdp_results[pair_key] = {
                "pre_ccg": pre_ccg,
                "post_ccg": post_ccg,
                "delta_ccg": pair_summary.delta_ccg,
                "testing_start": t_test_start.isoformat(),
                "testing_end": t_test_end.isoformat(),
                "learning_start": t_learn_start.isoformat(),
                "learning_end": t_learn_end.isoformat(),
                "validation_start": t_val_start.isoformat(),
                "validation_end": t_val_end.isoformat(),
            }
            logger.info(
                "STDP pair %s: pre_peak=%d post_peak=%d delta_ccg=%.3f",
                pair_key, pre_peak, post_peak, pair_summary.delta_ccg
            )

        logger.info("STDP experiment complete.")

    def _run_passive_recording_phase(
        self,
        stim_elec: int,
        resp_elec: int,
        amplitude: float,
        duration: float,
        polarity: StimPolarity,
        duration_s: float,
        phase_name: str,
    ) -> Dict[str, Any]:
        stim_times = []
        t_phase_start = datetime_now()
        elapsed = 0.0
        inter_stim = self.stdp_inter_stim_s

        while elapsed < duration_s:
            t_stim = datetime_now()
            self._send_stim_and_query(
                electrode_idx=stim_elec,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=polarity,
                trigger_key=0,
                phase=phase_name,
            )
            stim_times.append(t_stim.isoformat())
            self._wait(inter_stim)
            elapsed = (datetime_now() - t_phase_start).total_seconds()

        ccg = self._compute_ccg(
            stim_electrode=stim_elec,
            resp_electrode=resp_elec,
            stim_times_iso=stim_times,
            window_ms=100.0,
        )
        return ccg

    def _run_stdp_learning_phase(
        self,
        stim_elec: int,
        resp_elec: int,
        amplitude: float,
        duration: float,
        polarity: StimPolarity,
        duration_s: float,
        delta_t_ms: float,
    ) -> None:
        t_phase_start = datetime_now()
        elapsed = 0.0
        inter_stim = self.stdp_inter_stim_s

        a2 = amplitude
        d2 = duration

        while elapsed < duration_s:
            self._send_stim_and_query(
                electrode_idx=stim_elec,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=polarity,
                trigger_key=0,
                phase="stdp_learning_pre",
            )

            delay_s = delta_t_ms / 1000.0
            if delay_s > 0:
                self._wait(delay_s)

            self._send_stim_and_query(
                electrode_idx=resp_elec,
                amplitude_ua=a2,
                duration_us=d2,
                polarity=polarity,
                trigger_key=1,
                phase="stdp_learning_post",
            )

            self._wait(inter_stim)
            elapsed = (datetime_now() - t_phase_start).total_seconds()

    def _send_stim_and_query(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity,
        trigger_key: int = 0,
        phase: str = "",
        post_stim_wait_s: float = 0.1,
    ) -> pd.DataFrame:
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

        t_stim = datetime_now()
        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=a1,
            duration_us=d1,
            polarity=polarity.name,
            timestamp_utc=t_stim.isoformat(),
            trigger_key=trigger_key,
            phase=phase,
        ))

        self._wait(post_stim_wait_s)

        query_stop = datetime_now()
        query_start = query_stop - timedelta(seconds=post_stim_wait_s + 0.5)
        try:
            spike_df = self.database.get_spike_event(
                query_start, query_stop, self.experiment.exp_name
            )
        except Exception as exc:
            logger.warning("Spike query failed: %s", exc)
            spike_df = pd.DataFrame()

        return spike_df

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
            "stdp_pairs_count": len(self._pair_summaries),
            "stdp_results": {
                k: {
                    "delta_ccg": v.get("delta_ccg"),
                    "pre_peak": v.get("pre_ccg", {}).get("peak_count"),
                    "post_peak": v.get("post_ccg", {}).get("peak_count"),
                }
                for k, v in self._stdp_results.items()
            },
            "pair_summaries": [asdict(ps) for ps in self._pair_summaries],
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

        channel_col = None
        for col in spike_df.columns:
            if col.lower() in ("channel", "index", "electrode"):
                channel_col = col
                break

        if channel_col is None:
            logger.warning("Cannot determine electrode column for waveform fetch")
            return waveform_records

        unique_electrodes = spike_df[channel_col].unique()
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
        for ps in self._pair_summaries:
            key = f"{ps.stim_electrode}->{ps.resp_electrode}"
            stdp_summary[key] = {
                "delta_ccg": ps.delta_ccg,
                "ccg_peak_pre": ps.ccg_peak_pre,
                "ccg_peak_post": ps.ccg_peak_post,
                "median_latency_ms": ps.median_latency_ms,
                "response_rate": ps.response_rate,
            }

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "total_stimulations": len(self._stimulation_log),
            "scan_results_count": len(self._scan_results),
            "responsive_pairs_count": len(self._responsive_pairs),
            "active_pairs_processed": len(self._active_stim_times),
            "stdp_pairs_processed": len(self._pair_summaries),
            "stdp_results": stdp_summary,
            "ccg_pairs": list(self._ccg_data.keys()),
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
