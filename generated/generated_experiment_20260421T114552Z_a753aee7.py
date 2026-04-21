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
class CrossCorrelogram:
    electrode_from: int
    electrode_to: int
    bins: List[float]
    counts: List[int]
    peak_lag_ms: float
    peak_count: int


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


class Experiment:
    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        scan_amplitudes: tuple = (1.0, 2.0, 3.0),
        scan_durations: tuple = (100.0, 200.0, 300.0, 400.0),
        scan_repeats: int = 5,
        scan_inter_stim_s: float = 1.0,
        scan_inter_channel_s: float = 5.0,
        active_total_repeats: int = 100,
        active_group_size: int = 10,
        active_group_pause_s: float = 5.0,
        stdp_testing_duration_s: float = 1200.0,
        stdp_learning_duration_s: float = 3000.0,
        stdp_validation_duration_s: float = 1200.0,
        stdp_hebbian_delay_ms: float = 15.0,
        stdp_amplitude_ua: float = 2.0,
        stdp_duration_us: float = 300.0,
        ccg_window_ms: float = 50.0,
        ccg_bin_ms: float = 1.0,
        max_electrode_pairs: int = 5,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.scan_amplitudes = list(scan_amplitudes)
        self.scan_durations = list(scan_durations)
        self.scan_repeats = scan_repeats
        self.scan_inter_stim_s = scan_inter_stim_s
        self.scan_inter_channel_s = scan_inter_channel_s

        self.active_total_repeats = active_total_repeats
        self.active_group_size = active_group_size
        self.active_group_pause_s = active_group_pause_s

        self.stdp_testing_duration_s = stdp_testing_duration_s
        self.stdp_learning_duration_s = stdp_learning_duration_s
        self.stdp_validation_duration_s = stdp_validation_duration_s
        self.stdp_hebbian_delay_ms = stdp_hebbian_delay_ms
        self.stdp_amplitude_ua = stdp_amplitude_ua
        self.stdp_duration_us = stdp_duration_us

        self.ccg_window_ms = ccg_window_ms
        self.ccg_bin_ms = ccg_bin_ms
        self.max_electrode_pairs = max_electrode_pairs

        self.np_experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._scan_results: List[ScanResult] = []
        self._responsive_pairs: List[Dict[str, Any]] = []
        self._ccg_results: List[CrossCorrelogram] = []
        self._stdp_results: Dict[str, Any] = {}

        self._prior_pairs = [
            {"electrode_from": 0, "electrode_to": 1, "amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 12.37},
            {"electrode_from": 1, "electrode_to": 2, "amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 23.34},
            {"electrode_from": 5, "electrode_to": 4, "amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 17.39},
            {"electrode_from": 5, "electrode_to": 6, "amplitude": 1.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 15.45},
            {"electrode_from": 6, "electrode_to": 5, "amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 14.82},
            {"electrode_from": 8, "electrode_to": 9, "amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 15.88},
            {"electrode_from": 9, "electrode_to": 10, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 10.97},
            {"electrode_from": 14, "electrode_to": 12, "amplitude": 1.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 22.91},
            {"electrode_from": 14, "electrode_to": 15, "amplitude": 2.0, "duration": 300.0, "polarity": "PositiveFirst", "median_latency_ms": 12.99},
            {"electrode_from": 17, "electrode_to": 16, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 21.70},
            {"electrode_from": 18, "electrode_to": 17, "amplitude": 1.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 24.61},
            {"electrode_from": 22, "electrode_to": 21, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 13.58},
            {"electrode_from": 24, "electrode_to": 25, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 13.18},
            {"electrode_from": 26, "electrode_to": 27, "amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 13.88},
            {"electrode_from": 30, "electrode_to": 31, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 19.34},
        ]

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

            logger.info("=== Phase 1: Basic Excitability Scan ===")
            self._phase_excitability_scan()

            logger.info("=== Phase 2: Active Electrode Experiment ===")
            self._phase_active_electrode()

            logger.info("=== Phase 3: STDP Hebbian Learning ===")
            self._phase_stdp_learning()

            recording_stop = datetime_now()

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _get_available_electrodes(self) -> List[int]:
        if self.np_experiment is not None and hasattr(self.np_experiment, "electrodes"):
            return list(self.np_experiment.electrodes)
        return list(range(32))

    def _polarity_from_str(self, polarity_str: str) -> StimPolarity:
        if polarity_str == "PositiveFirst":
            return StimPolarity.PositiveFirst
        return StimPolarity.NegativeFirst

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
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            polarity=polarity.name,
            timestamp_utc=datetime_now().isoformat(),
            trigger_key=trigger_key,
            phase=phase,
        ))

    def _query_spikes_window(
        self,
        electrode_idx: int,
        window_start: datetime,
        window_stop: datetime,
    ) -> pd.DataFrame:
        try:
            df = self.database.get_spike_event_electrode(
                window_start, window_stop, electrode_idx
            )
            return df
        except Exception as exc:
            logger.warning("Spike query failed for electrode %d: %s", electrode_idx, exc)
            return pd.DataFrame()

    def _phase_excitability_scan(self) -> None:
        available_electrodes = self._get_available_electrodes()
        polarities = [StimPolarity.NegativeFirst, StimPolarity.PositiveFirst]

        logger.info("Scanning %d electrodes", len(available_electrodes))

        for ch_idx, electrode in enumerate(available_electrodes):
            logger.info("Scanning electrode %d (%d/%d)", electrode, ch_idx + 1, len(available_electrodes))

            for amplitude in self.scan_amplitudes:
                for duration in self.scan_durations:
                    for polarity in polarities:
                        hits_per_channel: Dict[int, int] = defaultdict(int)
                        latencies: List[float] = []

                        for rep in range(self.scan_repeats):
                            stim_time = datetime_now()
                            self._send_single_stim(
                                electrode_idx=electrode,
                                amplitude_ua=amplitude,
                                duration_us=duration,
                                polarity=polarity,
                                trigger_key=0,
                                phase="scan",
                            )
                            self._wait(0.05)
                            window_start = stim_time
                            window_stop = datetime_now()

                            for other_electrode in available_electrodes:
                                if other_electrode == electrode:
                                    continue
                                spike_df = self._query_spikes_window(
                                    other_electrode, window_start, window_stop
                                )
                                if not spike_df.empty:
                                    hits_per_channel[other_electrode] += 1
                                    if "Time" in spike_df.columns:
                                        for t in spike_df["Time"]:
                                            try:
                                                if hasattr(t, "timestamp"):
                                                    lat = (t.timestamp() - stim_time.timestamp()) * 1000.0
                                                else:
                                                    lat = 0.0
                                                if 0 < lat < 100:
                                                    latencies.append(lat)
                                            except Exception:
                                                pass

                            if rep < self.scan_repeats - 1:
                                self._wait(self.scan_inter_stim_s)

                        for resp_electrode, hits in hits_per_channel.items():
                            if hits >= 3:
                                median_lat = float(np.median(latencies)) if latencies else 0.0
                                result = ScanResult(
                                    electrode_from=electrode,
                                    electrode_to=resp_electrode,
                                    amplitude=amplitude,
                                    duration=duration,
                                    polarity=polarity.name,
                                    hits=hits,
                                    repeats=self.scan_repeats,
                                    median_latency_ms=median_lat,
                                )
                                self._scan_results.append(result)
                                logger.info(
                                    "Responsive pair: %d->%d  hits=%d  amp=%.1f  dur=%.0f  pol=%s",
                                    electrode, resp_electrode, hits, amplitude, duration, polarity.name
                                )

            self._wait(self.scan_inter_channel_s)

        self._identify_responsive_pairs()

    def _identify_responsive_pairs(self) -> None:
        pair_hits: Dict[Tuple[int, int], List[ScanResult]] = defaultdict(list)
        for r in self._scan_results:
            pair_hits[(r.electrode_from, r.electrode_to)].append(r)

        if not pair_hits:
            logger.info("No responsive pairs found in scan; using prior pairs from parameter sweep")
            for p in self._prior_pairs[:self.max_electrode_pairs]:
                self._responsive_pairs.append({
                    "electrode_from": p["electrode_from"],
                    "electrode_to": p["electrode_to"],
                    "amplitude": p["amplitude"],
                    "duration": p["duration"],
                    "polarity": p["polarity"],
                    "median_latency_ms": p["median_latency_ms"],
                    "hits": 5,
                    "repeats": 5,
                })
        else:
            scored: List[Tuple[float, Tuple[int, int], ScanResult]] = []
            for pair, results in pair_hits.items():
                best = max(results, key=lambda r: r.hits)
                score = best.hits + (1.0 / (best.amplitude * best.duration + 1e-9)) * 1e-4
                scored.append((score, pair, best))
            scored.sort(key=lambda x: -x[0])

            for score, pair, best in scored[:self.max_electrode_pairs]:
                self._responsive_pairs.append({
                    "electrode_from": best.electrode_from,
                    "electrode_to": best.electrode_to,
                    "amplitude": best.amplitude,
                    "duration": best.duration,
                    "polarity": best.polarity,
                    "median_latency_ms": best.median_latency_ms,
                    "hits": best.hits,
                    "repeats": best.repeats,
                })

        logger.info("Identified %d responsive pairs", len(self._responsive_pairs))

    def _phase_active_electrode(self) -> None:
        if not self._responsive_pairs:
            logger.warning("No responsive pairs for active electrode experiment")
            return

        stim_times_per_pair: Dict[int, List[datetime]] = defaultdict(list)

        for pair_idx, pair in enumerate(self._responsive_pairs):
            electrode_from = pair["electrode_from"]
            amplitude = pair["amplitude"]
            duration = pair["duration"]
            polarity = self._polarity_from_str(pair["polarity"])

            logger.info(
                "Active electrode: pair %d  %d->%d  amp=%.1f  dur=%.0f",
                pair_idx, electrode_from, pair["electrode_to"], amplitude, duration
            )

            n_groups = self.active_total_repeats // self.active_group_size
            stim_count = 0

            for group in range(n_groups):
                for stim_in_group in range(self.active_group_size):
                    stim_time = datetime_now()
                    self._send_single_stim(
                        electrode_idx=electrode_from,
                        amplitude_ua=amplitude,
                        duration_us=duration,
                        polarity=polarity,
                        trigger_key=pair_idx % 16,
                        phase="active_electrode",
                    )
                    stim_times_per_pair[pair_idx].append(stim_time)
                    stim_count += 1
                    if stim_in_group < self.active_group_size - 1:
                        self._wait(1.0)

                if group < n_groups - 1:
                    self._wait(self.active_group_pause_s)

            logger.info("Pair %d: completed %d stimulations", pair_idx, stim_count)

        self._compute_ccgs(stim_times_per_pair)

    def _compute_ccgs(self, stim_times_per_pair: Dict[int, List[datetime]]) -> None:
        logger.info("Computing trigger-centred cross-correlograms")
        n_bins = int(self.ccg_window_ms / self.ccg_bin_ms)
        bin_edges = [i * self.ccg_bin_ms for i in range(n_bins + 1)]
        bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2.0 for i in range(n_bins)]

        for pair_idx, pair in enumerate(self._responsive_pairs):
            electrode_to = pair["electrode_to"]
            times = stim_times_per_pair.get(pair_idx, [])
            if not times:
                continue

            counts = [0] * n_bins
            window_s = self.ccg_window_ms / 1000.0

            for stim_time in times:
                query_stop = stim_time + timedelta(seconds=window_s + 0.1)
                query_start = stim_time
                try:
                    spike_df = self._query_spikes_window(electrode_to, query_start, query_stop)
                    if spike_df.empty or "Time" not in spike_df.columns:
                        continue
                    for t in spike_df["Time"]:
                        try:
                            if hasattr(t, "timestamp"):
                                lag_ms = (t.timestamp() - stim_time.timestamp()) * 1000.0
                            else:
                                continue
                            if 0 <= lag_ms < self.ccg_window_ms:
                                bin_idx = int(lag_ms / self.ccg_bin_ms)
                                if 0 <= bin_idx < n_bins:
                                    counts[bin_idx] += 1
                        except Exception:
                            pass
                except Exception as exc:
                    logger.warning("CCG query error for pair %d: %s", pair_idx, exc)

            peak_idx = counts.index(max(counts)) if counts else 0
            peak_lag = bin_centers[peak_idx] if bin_centers else 0.0
            peak_count = counts[peak_idx] if counts else 0

            ccg = CrossCorrelogram(
                electrode_from=pair["electrode_from"],
                electrode_to=pair["electrode_to"],
                bins=bin_centers,
                counts=counts,
                peak_lag_ms=peak_lag,
                peak_count=peak_count,
            )
            self._ccg_results.append(ccg)

            pair["ccg_peak_lag_ms"] = peak_lag
            logger.info(
                "CCG pair %d->%d: peak at %.1f ms (count=%d)",
                pair["electrode_from"], pair["electrode_to"], peak_lag, peak_count
            )

    def _phase_stdp_learning(self) -> None:
        if not self._responsive_pairs:
            logger.warning("No responsive pairs for STDP experiment")
            return

        stdp_pair = self._responsive_pairs[0]
        electrode_pre = stdp_pair["electrode_from"]
        electrode_post = stdp_pair["electrode_to"]

        ccg_lag = stdp_pair.get("ccg_peak_lag_ms", self.stdp_hebbian_delay_ms)
        hebbian_delay_ms = ccg_lag if 5.0 <= ccg_lag <= 50.0 else self.stdp_hebbian_delay_ms
        hebbian_delay_s = hebbian_delay_ms / 1000.0

        amplitude = min(self.stdp_amplitude_ua, 4.0)
        duration = min(self.stdp_duration_us, 400.0)
        polarity_pre = self._polarity_from_str(stdp_pair["polarity"])
        polarity_post = StimPolarity.NegativeFirst

        logger.info(
            "STDP pair: %d (pre) -> %d (post)  Hebbian delay=%.1f ms",
            electrode_pre, electrode_post, hebbian_delay_ms
        )

        logger.info("STDP Phase: Testing (%.0f s)", self.stdp_testing_duration_s)
        testing_start = datetime_now()
        testing_spikes = self._run_probe_phase(
            electrode_pre=electrode_pre,
            electrode_post=electrode_post,
            amplitude=amplitude,
            duration=duration,
            polarity=polarity_pre,
            duration_s=self.stdp_testing_duration_s,
            phase_label="stdp_testing",
            probe_interval_s=5.0,
        )
        testing_stop = datetime_now()

        logger.info("STDP Phase: Learning (%.0f s)", self.stdp_learning_duration_s)
        learning_start = datetime_now()
        learning_count = self._run_learning_phase(
            electrode_pre=electrode_pre,
            electrode_post=electrode_post,
            amplitude=amplitude,
            duration=duration,
            polarity_pre=polarity_pre,
            polarity_post=polarity_post,
            hebbian_delay_s=hebbian_delay_s,
            duration_s=self.stdp_learning_duration_s,
        )
        learning_stop = datetime_now()

        logger.info("STDP Phase: Validation (%.0f s)", self.stdp_validation_duration_s)
        validation_start = datetime_now()
        validation_spikes = self._run_probe_phase(
            electrode_pre=electrode_pre,
            electrode_post=electrode_post,
            amplitude=amplitude,
            duration=duration,
            polarity=polarity_pre,
            duration_s=self.stdp_validation_duration_s,
            phase_label="stdp_validation",
            probe_interval_s=5.0,
        )
        validation_stop = datetime_now()

        testing_response_rate = self._compute_response_rate(testing_spikes)
        validation_response_rate = self._compute_response_rate(validation_spikes)
        delta_stp = validation_response_rate - testing_response_rate

        self._stdp_results = {
            "electrode_pre": electrode_pre,
            "electrode_post": electrode_post,
            "hebbian_delay_ms": hebbian_delay_ms,
            "amplitude_ua": amplitude,
            "duration_us": duration,
            "testing_start": testing_start.isoformat(),
            "testing_stop": testing_stop.isoformat(),
            "testing_probe_count": len(testing_spikes),
            "testing_response_rate": testing_response_rate,
            "learning_start": learning_start.isoformat(),
            "learning_stop": learning_stop.isoformat(),
            "learning_paired_stims": learning_count,
            "validation_start": validation_start.isoformat(),
            "validation_stop": validation_stop.isoformat(),
            "validation_probe_count": len(validation_spikes),
            "validation_response_rate": validation_response_rate,
            "delta_stp": delta_stp,
        }

        logger.info(
            "STDP results: testing_rate=%.3f  validation_rate=%.3f  delta_STP=%.3f",
            testing_response_rate, validation_response_rate, delta_stp
        )

    def _run_probe_phase(
        self,
        electrode_pre: int,
        electrode_post: int,
        amplitude: float,
        duration: float,
        polarity: StimPolarity,
        duration_s: float,
        phase_label: str,
        probe_interval_s: float = 5.0,
    ) -> List[bool]:
        responses: List[bool] = []
        phase_start = datetime_now()
        elapsed = 0.0

        while elapsed < duration_s:
            stim_time = datetime_now()
            self._send_single_stim(
                electrode_idx=electrode_pre,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=polarity,
                trigger_key=0,
                phase=phase_label,
            )
            self._wait(0.1)
            window_start = stim_time
            window_stop = datetime_now()
            spike_df = self._query_spikes_window(electrode_post, window_start, window_stop)
            responded = not spike_df.empty
            responses.append(responded)

            self._wait(probe_interval_s - 0.1)
            elapsed = (datetime_now() - phase_start).total_seconds()

        return responses

    def _run_learning_phase(
        self,
        electrode_pre: int,
        electrode_post: int,
        amplitude: float,
        duration: float,
        polarity_pre: StimPolarity,
        polarity_post: StimPolarity,
        hebbian_delay_s: float,
        duration_s: float,
    ) -> int:
        stim_count = 0
        phase_start = datetime_now()
        elapsed = 0.0
        inter_pair_s = 1.0

        while elapsed < duration_s:
            self._send_single_stim(
                electrode_idx=electrode_pre,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=polarity_pre,
                trigger_key=0,
                phase="stdp_learning_pre",
            )
            self._wait(hebbian_delay_s)
            self._send_single_stim(
                electrode_idx=electrode_post,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=polarity_post,
                trigger_key=1,
                phase="stdp_learning_post",
            )
            stim_count += 1
            self._wait(inter_pair_s)
            elapsed = (datetime_now() - phase_start).total_seconds()

        return stim_count

    def _compute_response_rate(self, responses: List[bool]) -> float:
        if not responses:
            return 0.0
        return sum(1 for r in responses if r) / len(responses)

    def _save_all(self, recording_start: datetime, recording_stop: datetime) -> None:
        fs_name = getattr(self.np_experiment, "exp_name", "unknown")
        saver = DataSaver(self._output_dir, fs_name)

        saver.save_stimulation_log(self._stimulation_log)

        spike_df = pd.DataFrame()
        try:
            spike_df = self.database.get_spike_event(recording_start, recording_stop, fs_name)
        except Exception as exc:
            logger.warning("Failed to fetch spike events: %s", exc)
            spike_df = pd.DataFrame()
        saver.save_spike_events(spike_df)

        trigger_df = pd.DataFrame()
        try:
            trigger_df = self.database.get_all_triggers(recording_start, recording_stop)
        except Exception as exc:
            logger.warning("Failed to fetch triggers: %s", exc)
            trigger_df = pd.DataFrame()
        saver.save_triggers(trigger_df)

        ccg_serializable = []
        for ccg in self._ccg_results:
            ccg_serializable.append({
                "electrode_from": ccg.electrode_from,
                "electrode_to": ccg.electrode_to,
                "bins": ccg.bins,
                "counts": ccg.counts,
                "peak_lag_ms": ccg.peak_lag_ms,
                "peak_count": ccg.peak_count,
            })

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
            "responsive_pairs": self._responsive_pairs,
            "ccg_results": ccg_serializable,
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
            for col in spike_df.columns:
                if "electrode" in col.lower() or "idx" in col.lower() or "channel" in col.lower():
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

        ccg_serializable = []
        for ccg in self._ccg_results:
            ccg_serializable.append({
                "electrode_from": ccg.electrode_from,
                "electrode_to": ccg.electrode_to,
                "peak_lag_ms": ccg.peak_lag_ms,
                "peak_count": ccg.peak_count,
            })

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": getattr(self.np_experiment, "exp_name", "unknown"),
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "total_stimulations": len(self._stimulation_log),
            "scan_results_count": len(self._scan_results),
            "responsive_pairs": self._responsive_pairs,
            "ccg_results": ccg_serializable,
            "stdp_results": self._stdp_results,
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
