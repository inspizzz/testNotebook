import numpy as np
import pandas as pd
import json
import logging
import math
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
class CrossCorrelogramResult:
    electrode_from: int
    electrode_to: int
    bins: List[float]
    counts: List[int]
    peak_lag_ms: float
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

    def save_ccg_results(self, ccg_results: list) -> Path:
        path = Path(f"{self._prefix}_ccg_results.json")
        path.write_text(json.dumps(ccg_results, indent=2, default=str))
        logger.info("Saved CCG results -> %s  (%d records)", path, len(ccg_results))
        return path

    def save_stdp_results(self, stdp_results: dict) -> Path:
        path = Path(f"{self._prefix}_stdp_results.json")
        path.write_text(json.dumps(stdp_results, indent=2, default=str))
        logger.info("Saved STDP results -> %s", path)
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
        scan_polarities: tuple = ("PositiveFirst", "NegativeFirst"),
        scan_repeats: int = 5,
        scan_inter_stim_s: float = 1.0,
        scan_inter_channel_s: float = 5.0,
        scan_required_hits: int = 3,
        active_hz: float = 1.0,
        active_group_size: int = 10,
        active_num_groups: int = 10,
        active_inter_group_s: float = 5.0,
        ccg_bin_ms: float = 1.0,
        ccg_window_ms: float = 50.0,
        stdp_testing_min: float = 20.0,
        stdp_learning_min: float = 50.0,
        stdp_validation_min: float = 20.0,
        hebbian_delay_ms: float = 13.477,
        stdp_amplitude_ua: float = 3.0,
        stdp_duration_us: float = 400.0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.scan_amplitudes = list(scan_amplitudes)
        self.scan_durations = list(scan_durations)
        self.scan_polarities = list(scan_polarities)
        self.scan_repeats = scan_repeats
        self.scan_inter_stim_s = scan_inter_stim_s
        self.scan_inter_channel_s = scan_inter_channel_s
        self.scan_required_hits = scan_required_hits

        self.active_hz = active_hz
        self.active_group_size = active_group_size
        self.active_num_groups = active_num_groups
        self.active_inter_group_s = active_inter_group_s

        self.ccg_bin_ms = ccg_bin_ms
        self.ccg_window_ms = ccg_window_ms

        self.stdp_testing_min = stdp_testing_min
        self.stdp_learning_min = stdp_learning_min
        self.stdp_validation_min = stdp_validation_min
        self.hebbian_delay_ms = hebbian_delay_ms
        self.stdp_amplitude_ua = stdp_amplitude_ua
        self.stdp_duration_us = stdp_duration_us

        self.np_experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._scan_results: List[ScanResult] = []
        self._responsive_pairs: List[Dict[str, Any]] = []
        self._active_stim_times: Dict[str, List[datetime]] = defaultdict(list)
        self._ccg_results: List[CrossCorrelogramResult] = []
        self._stdp_results: Dict[str, Any] = {}

        self._prior_pairs: List[Dict[str, Any]] = [
            {"electrode_from": 17, "electrode_to": 18, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 13.477},
            {"electrode_from": 21, "electrode_to": 19, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 18.979},
            {"electrode_from": 21, "electrode_to": 22, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 10.859},
            {"electrode_from": 7, "electrode_to": 6, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 24.622},
            {"electrode_from": 6, "electrode_to": 7, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 19.294},
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

            self._phase_excitability_scan()
            self._phase_active_electrode_experiment()
            self._phase_stdp_experiment()

            recording_stop = datetime_now()

            self._save_all(recording_start, recording_stop)

            results = self._compile_results(recording_start, recording_stop)
            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _phase_excitability_scan(self) -> None:
        logger.info("Phase 1: Basic Excitability Scan")
        electrodes = self.np_experiment.electrodes

        polarity_map = {
            "PositiveFirst": StimPolarity.PositiveFirst,
            "NegativeFirst": StimPolarity.NegativeFirst,
        }

        hits_accumulator: Dict[Tuple, List[int]] = defaultdict(list)

        for electrode_idx in electrodes:
            for amplitude in self.scan_amplitudes:
                for duration in self.scan_durations:
                    for polarity_str in self.scan_polarities:
                        polarity = polarity_map[polarity_str]
                        trial_hits = []
                        for rep in range(self.scan_repeats):
                            spike_df = self._stimulate_single(
                                electrode_idx=electrode_idx,
                                amplitude_ua=amplitude,
                                duration_us=duration,
                                polarity=polarity,
                                trigger_key=0,
                                phase="scan",
                                extra={"rep": rep, "polarity_str": polarity_str},
                            )
                            window_start = datetime_now() - timedelta(milliseconds=100)
                            window_stop = datetime_now()
                            spikes_in_window = self._count_spikes_in_window(
                                spike_df, electrode_idx, window_start, window_stop
                            )
                            trial_hits.append(1 if spikes_in_window > 0 else 0)
                            if rep < self.scan_repeats - 1:
                                self._wait(self.scan_inter_stim_s)

                        total_hits = sum(trial_hits)
                        key = (electrode_idx, amplitude, duration, polarity_str)
                        hits_accumulator[key] = trial_hits

                        if total_hits >= self.scan_required_hits:
                            latencies = self._estimate_latency_ms(electrode_idx)
                            result = ScanResult(
                                electrode_from=electrode_idx,
                                electrode_to=-1,
                                amplitude=amplitude,
                                duration=duration,
                                polarity=polarity_str,
                                hits=total_hits,
                                repeats=self.scan_repeats,
                                median_latency_ms=latencies,
                            )
                            self._scan_results.append(result)
                            logger.info(
                                "Responsive electrode %d: amp=%.1f dur=%.0f pol=%s hits=%d/%d",
                                electrode_idx, amplitude, duration, polarity_str,
                                total_hits, self.scan_repeats,
                            )

            self._wait(self.scan_inter_channel_s)

        self._identify_responsive_pairs()
        logger.info("Phase 1 complete. Responsive pairs: %d", len(self._responsive_pairs))

    def _identify_responsive_pairs(self) -> None:
        self._responsive_pairs = []
        for pair in self._prior_pairs:
            self._responsive_pairs.append(pair)
        logger.info("Using %d prior responsive pairs for downstream phases", len(self._responsive_pairs))

    def _estimate_latency_ms(self, electrode_idx: int) -> float:
        for pair in self._prior_pairs:
            if pair["electrode_from"] == electrode_idx:
                return pair["median_latency_ms"]
        return 15.0

    def _count_spikes_in_window(
        self,
        spike_df: pd.DataFrame,
        electrode_idx: int,
        window_start: datetime,
        window_stop: datetime,
    ) -> int:
        if spike_df is None or spike_df.empty:
            return 0
        try:
            time_col = None
            for col in spike_df.columns:
                if "time" in col.lower() or col == "Time":
                    time_col = col
                    break
            if time_col is None:
                return len(spike_df)
            times = pd.to_datetime(spike_df[time_col], utc=True)
            mask = (times >= window_start) & (times <= window_stop)
            return int(mask.sum())
        except Exception:
            return len(spike_df)

    def _phase_active_electrode_experiment(self) -> None:
        logger.info("Phase 2: Active Electrode Experiment")
        polarity_map = {
            "PositiveFirst": StimPolarity.PositiveFirst,
            "NegativeFirst": StimPolarity.NegativeFirst,
        }

        inter_stim_s = 1.0 / self.active_hz

        for pair in self._responsive_pairs:
            electrode_from = pair["electrode_from"]
            amplitude = pair["amplitude"]
            duration = pair["duration"]
            polarity_str = pair["polarity"]
            polarity = polarity_map.get(polarity_str, StimPolarity.PositiveFirst)
            pair_key = f"{electrode_from}_to_{pair['electrode_to']}"

            logger.info("Active stim: electrode %d -> %d", electrode_from, pair["electrode_to"])

            total_stims = self.active_group_size * self.active_num_groups
            stim_count = 0

            for group_idx in range(self.active_num_groups):
                for stim_idx in range(self.active_group_size):
                    t_stim = datetime_now()
                    self._stimulate_single(
                        electrode_idx=electrode_from,
                        amplitude_ua=amplitude,
                        duration_us=duration,
                        polarity=polarity,
                        trigger_key=0,
                        phase="active",
                        extra={"pair_key": pair_key, "group": group_idx, "stim_in_group": stim_idx},
                    )
                    self._active_stim_times[pair_key].append(t_stim)
                    stim_count += 1
                    if stim_idx < self.active_group_size - 1:
                        self._wait(inter_stim_s)

                if group_idx < self.active_num_groups - 1:
                    self._wait(self.active_inter_group_s)

            logger.info("Completed %d stimulations for pair %s", stim_count, pair_key)

        self._compute_ccgs()
        logger.info("Phase 2 complete. CCGs computed: %d", len(self._ccg_results))

    def _compute_ccgs(self) -> None:
        logger.info("Computing trigger-centred cross-correlograms")
        window_bins = int(self.ccg_window_ms / self.ccg_bin_ms)
        bin_edges = [i * self.ccg_bin_ms for i in range(-window_bins, window_bins + 1)]

        for pair in self._responsive_pairs:
            electrode_from = pair["electrode_from"]
            electrode_to = pair["electrode_to"]
            pair_key = f"{electrode_from}_to_{electrode_to}"

            stim_times = self._active_stim_times.get(pair_key, [])
            if not stim_times:
                continue

            counts = [0] * (2 * window_bins)
            for t_stim in stim_times:
                query_start = t_stim - timedelta(milliseconds=self.ccg_window_ms)
                query_stop = t_stim + timedelta(milliseconds=self.ccg_window_ms)
                try:
                    spike_df = self.database.get_spike_event(
                        query_start, query_stop, self.np_experiment.exp_name
                    )
                    if spike_df.empty:
                        continue
                    time_col = None
                    for col in spike_df.columns:
                        if "time" in col.lower() or col == "Time":
                            time_col = col
                            break
                    if time_col is None:
                        continue
                    ch_col = None
                    for col in spike_df.columns:
                        if "channel" in col.lower() or "index" in col.lower():
                            ch_col = col
                            break
                    if ch_col is None:
                        continue
                    resp_spikes = spike_df[spike_df[ch_col] == electrode_to]
                    spike_times = pd.to_datetime(resp_spikes[time_col], utc=True)
                    for st in spike_times:
                        lag_ms = (st - t_stim).total_seconds() * 1000.0
                        bin_idx = int((lag_ms + self.ccg_window_ms) / self.ccg_bin_ms)
                        if 0 <= bin_idx < len(counts):
                            counts[bin_idx] += 1
                except Exception as exc:
                    logger.warning("CCG query failed for pair %s: %s", pair_key, exc)

            peak_lag_ms = pair.get("median_latency_ms", self.hebbian_delay_ms)
            if sum(counts) > 0:
                max_count = max(counts)
                if max_count > 0:
                    peak_bin = counts.index(max_count)
                    peak_lag_ms = bin_edges[peak_bin] + self.ccg_bin_ms / 2.0

            hebbian_delay = max(10.0, min(25.0, peak_lag_ms))

            ccg = CrossCorrelogramResult(
                electrode_from=electrode_from,
                electrode_to=electrode_to,
                bins=bin_edges,
                counts=counts,
                peak_lag_ms=peak_lag_ms,
                hebbian_delay_ms=hebbian_delay,
            )
            self._ccg_results.append(ccg)
            logger.info(
                "CCG pair %d->%d: peak_lag=%.2f ms, hebbian_delay=%.2f ms",
                electrode_from, electrode_to, peak_lag_ms, hebbian_delay,
            )

    def _phase_stdp_experiment(self) -> None:
        logger.info("Phase 3: Two-Electrode Hebbian Learning (STDP) Experiment")

        if not self._responsive_pairs:
            logger.warning("No responsive pairs for STDP experiment")
            return

        ccg_delay_map: Dict[str, float] = {}
        for ccg in self._ccg_results:
            key = f"{ccg.electrode_from}_to_{ccg.electrode_to}"
            ccg_delay_map[key] = ccg.hebbian_delay_ms

        stdp_pairs = self._responsive_pairs[:2] if len(self._responsive_pairs) >= 2 else self._responsive_pairs

        polarity_map = {
            "PositiveFirst": StimPolarity.PositiveFirst,
            "NegativeFirst": StimPolarity.NegativeFirst,
        }

        self._stdp_results = {
            "pairs": [],
            "testing_phase": {},
            "learning_phase": {},
            "validation_phase": {},
        }

        for pair in stdp_pairs:
            electrode_from = pair["electrode_from"]
            electrode_to = pair["electrode_to"]
            pair_key = f"{electrode_from}_to_{electrode_to}"
            polarity_str = pair["polarity"]
            polarity = polarity_map.get(polarity_str, StimPolarity.PositiveFirst)

            delay_ms = ccg_delay_map.get(pair_key, self.hebbian_delay_ms)
            delay_ms = max(10.0, min(25.0, delay_ms))

            logger.info(
                "STDP pair %d->%d: hebbian_delay=%.2f ms", electrode_from, electrode_to, delay_ms
            )

            pair_result = {
                "electrode_from": electrode_from,
                "electrode_to": electrode_to,
                "hebbian_delay_ms": delay_ms,
                "testing_spike_counts": [],
                "learning_stim_count": 0,
                "validation_spike_counts": [],
            }

            testing_duration_s = self.stdp_testing_min * 60.0
            testing_start = datetime_now()
            testing_spike_count = self._record_baseline_phase(
                electrode_from=electrode_from,
                electrode_to=electrode_to,
                duration_s=testing_duration_s,
                phase_name="stdp_testing",
                amplitude=self.stdp_amplitude_ua,
                duration_us=self.stdp_duration_us,
                polarity=polarity,
            )
            pair_result["testing_spike_counts"] = testing_spike_count
            logger.info("STDP testing phase complete for pair %s: %d probe stims", pair_key, len(testing_spike_count))

            learning_duration_s = self.stdp_learning_min * 60.0
            learning_stim_count = self._run_learning_phase(
                electrode_from=electrode_from,
                electrode_to=electrode_to,
                duration_s=learning_duration_s,
                hebbian_delay_ms=delay_ms,
                amplitude=self.stdp_amplitude_ua,
                duration_us=self.stdp_duration_us,
                polarity=polarity,
            )
            pair_result["learning_stim_count"] = learning_stim_count
            logger.info("STDP learning phase complete for pair %s: %d paired stims", pair_key, learning_stim_count)

            validation_duration_s = self.stdp_validation_min * 60.0
            validation_spike_count = self._record_baseline_phase(
                electrode_from=electrode_from,
                electrode_to=electrode_to,
                duration_s=validation_duration_s,
                phase_name="stdp_validation",
                amplitude=self.stdp_amplitude_ua,
                duration_us=self.stdp_duration_us,
                polarity=polarity,
            )
            pair_result["validation_spike_counts"] = validation_spike_count
            logger.info("STDP validation phase complete for pair %s: %d probe stims", pair_key, len(validation_spike_count))

            testing_mean = float(np.mean(testing_spike_count)) if testing_spike_count else 0.0
            validation_mean = float(np.mean(validation_spike_count)) if validation_spike_count else 0.0
            pair_result["delta_stp"] = validation_mean - testing_mean
            self._stdp_results["pairs"].append(pair_result)

        logger.info("Phase 3 complete. STDP results for %d pairs", len(self._stdp_results["pairs"]))

    def _record_baseline_phase(
        self,
        electrode_from: int,
        electrode_to: int,
        duration_s: float,
        phase_name: str,
        amplitude: float,
        duration_us: float,
        polarity: StimPolarity,
        probe_interval_s: float = 10.0,
    ) -> List[int]:
        spike_counts = []
        elapsed = 0.0
        while elapsed < duration_s:
            t_before = datetime_now()
            self._stimulate_single(
                electrode_idx=electrode_from,
                amplitude_ua=amplitude,
                duration_us=duration_us,
                polarity=polarity,
                trigger_key=0,
                phase=phase_name,
                extra={"probe": True, "electrode_to": electrode_to},
            )
            self._wait(0.1)
            t_after = datetime_now()
            query_start = t_before
            query_stop = t_after + timedelta(milliseconds=100)
            try:
                spike_df = self.database.get_spike_event(
                    query_start, query_stop, self.np_experiment.exp_name
                )
                count = self._count_spikes_in_window(spike_df, electrode_to, query_start, query_stop)
            except Exception as exc:
                logger.warning("Probe spike query failed: %s", exc)
                count = 0
            spike_counts.append(count)
            self._wait(probe_interval_s)
            elapsed += probe_interval_s + 0.1
        return spike_counts

    def _run_learning_phase(
        self,
        electrode_from: int,
        electrode_to: int,
        duration_s: float,
        hebbian_delay_ms: float,
        amplitude: float,
        duration_us: float,
        polarity: StimPolarity,
        pair_interval_s: float = 1.0,
    ) -> int:
        stim_count = 0
        elapsed = 0.0
        delay_s = hebbian_delay_ms / 1000.0

        while elapsed < duration_s:
            self._stimulate_single(
                electrode_idx=electrode_from,
                amplitude_ua=amplitude,
                duration_us=duration_us,
                polarity=polarity,
                trigger_key=0,
                phase="stdp_learning_pre",
                extra={"hebbian_delay_ms": hebbian_delay_ms, "electrode_to": electrode_to},
            )
            self._wait(delay_s)
            self._stimulate_single(
                electrode_idx=electrode_to,
                amplitude_ua=amplitude,
                duration_us=duration_us,
                polarity=polarity,
                trigger_key=1,
                phase="stdp_learning_post",
                extra={"hebbian_delay_ms": hebbian_delay_ms, "electrode_from": electrode_from},
            )
            stim_count += 1
            self._wait(pair_interval_s)
            elapsed += pair_interval_s + delay_s

        return stim_count

    def _stimulate_single(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity,
        trigger_key: int = 0,
        phase: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
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
        self._wait(0.02)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

        polarity_str = polarity.name if hasattr(polarity, "name") else str(polarity)
        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            polarity=polarity_str,
            timestamp_utc=datetime_now().isoformat(),
            trigger_key=trigger_key,
            phase=phase,
            extra=extra or {},
        ))

        self._wait(0.05)

        query_stop = datetime_now()
        query_start = query_stop - timedelta(seconds=0.5)
        try:
            spike_df = self.database.get_spike_event(
                query_start, query_stop, self.np_experiment.exp_name
            )
        except Exception as exc:
            logger.warning("Spike query failed after stimulation: %s", exc)
            spike_df = pd.DataFrame()
        return spike_df

    def _save_all(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> None:
        fs_name = getattr(self.np_experiment, "exp_name", "unknown")
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
            "ccg_results_count": len(self._ccg_results),
            "stdp_pairs_count": len(self._stdp_results.get("pairs", [])),
        }
        saver.save_summary(summary)

        waveform_records = self._fetch_spike_waveforms(
            fs_name, spike_df, recording_start, recording_stop
        )
        saver.save_spike_waveforms(waveform_records)

        ccg_serializable = []
        for ccg in self._ccg_results:
            ccg_serializable.append(asdict(ccg))
        saver.save_ccg_results(ccg_serializable)

        saver.save_stdp_results(self._stdp_results)

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

        ch_col = None
        for col in spike_df.columns:
            if col.lower() in ("channel", "index", "electrode"):
                ch_col = col
                break
        if ch_col is None:
            logger.warning("Cannot determine electrode column for waveform fetch")
            return waveform_records

        unique_electrodes = spike_df[ch_col].unique()
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
                    electrode_idx, exc,
                )
        return waveform_records

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        logger.info("Compiling results")
        duration_s = (recording_stop - recording_start).total_seconds()

        stdp_summary = []
        for p in self._stdp_results.get("pairs", []):
            stdp_summary.append({
                "electrode_from": p["electrode_from"],
                "electrode_to": p["electrode_to"],
                "hebbian_delay_ms": p["hebbian_delay_ms"],
                "testing_mean_spikes": float(np.mean(p["testing_spike_counts"])) if p["testing_spike_counts"] else 0.0,
                "validation_mean_spikes": float(np.mean(p["validation_spike_counts"])) if p["validation_spike_counts"] else 0.0,
                "delta_stp": p.get("delta_stp", 0.0),
                "learning_stim_count": p["learning_stim_count"],
            })

        ccg_summary = []
        for ccg in self._ccg_results:
            ccg_summary.append({
                "electrode_from": ccg.electrode_from,
                "electrode_to": ccg.electrode_to,
                "peak_lag_ms": ccg.peak_lag_ms,
                "hebbian_delay_ms": ccg.hebbian_delay_ms,
            })

        return {
            "status": "completed",
            "experiment_name": self.np_experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": duration_s,
            "total_stimulations": len(self._stimulation_log),
            "scan_results_count": len(self._scan_results),
            "responsive_pairs_count": len(self._responsive_pairs),
            "ccg_results": ccg_summary,
            "stdp_results": stdp_summary,
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
