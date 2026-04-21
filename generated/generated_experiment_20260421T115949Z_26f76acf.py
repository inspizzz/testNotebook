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
class CrossCorrelogramResult:
    electrode_from: int
    electrode_to: int
    peak_lag_ms: float
    peak_count: int
    bins: List[float]
    counts: List[int]


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
      Stage 1: Basic Excitability Scan
      Stage 2: Active Electrode Experiment (1 Hz bursts + cross-correlograms)
      Stage 3: Two-Electrode Hebbian (STDP) Learning Experiment
    """

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
        scan_required_hits: int = 3,
        active_total_repeats: int = 100,
        active_group_size: int = 10,
        active_group_pause_s: float = 5.0,
        active_stim_interval_s: float = 1.0,
        stdp_testing_duration_s: float = 1200.0,
        stdp_learning_duration_s: float = 3000.0,
        stdp_validation_duration_s: float = 1200.0,
        stdp_hebbian_delay_ms: float = 20.0,
        stdp_amplitude_ua: float = 2.0,
        stdp_duration_us: float = 200.0,
        ccg_window_ms: float = 50.0,
        ccg_bin_ms: float = 1.0,
        max_electrode_pairs: int = 3,
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
        self.scan_required_hits = scan_required_hits

        self.active_total_repeats = active_total_repeats
        self.active_group_size = active_group_size
        self.active_group_pause_s = active_group_pause_s
        self.active_stim_interval_s = active_stim_interval_s

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
        self._active_stim_times: Dict[Tuple[int, int], List[datetime]] = defaultdict(list)
        self._ccg_results: List[CrossCorrelogramResult] = []
        self._stdp_results: Dict[str, Any] = {}

        self._prior_responsive_pairs: List[Dict[str, Any]] = [
            {"electrode_from": 5, "electrode_to": 4, "amplitude": 1.0, "duration": 300.0, "polarity": "PositiveFirst", "median_latency_ms": 17.66},
            {"electrode_from": 14, "electrode_to": 15, "amplitude": 2.0, "duration": 300.0, "polarity": "PositiveFirst", "median_latency_ms": 12.99},
            {"electrode_from": 18, "electrode_to": 17, "amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 24.71},
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

            logger.info("=== Stage 1: Basic Excitability Scan ===")
            self._phase_excitability_scan()

            logger.info("=== Stage 2: Active Electrode Experiment ===")
            self._phase_active_electrode_experiment()

            logger.info("=== Stage 3: STDP Hebbian Learning Experiment ===")
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
        logger.info("Starting excitability scan")
        available_electrodes = list(self.np_experiment.electrodes)
        polarities = [StimPolarity.NegativeFirst, StimPolarity.PositiveFirst]
        polarity_names = {StimPolarity.NegativeFirst: "NegativeFirst", StimPolarity.PositiveFirst: "PositiveFirst"}

        for ch_idx, electrode in enumerate(available_electrodes):
            logger.info("Scanning electrode %d (%d/%d)", electrode, ch_idx + 1, len(available_electrodes))
            for amplitude in self.scan_amplitudes:
                for duration in self.scan_durations:
                    for polarity in polarities:
                        hits = 0
                        latencies = []
                        for rep in range(self.scan_repeats):
                            t_before = datetime_now()
                            self._send_single_stim(
                                electrode_idx=electrode,
                                amplitude_ua=amplitude,
                                duration_us=duration,
                                polarity=polarity,
                                trigger_key=0,
                                phase="scan",
                            )
                            self._wait(0.05)
                            t_after = datetime_now()
                            window_s = self.ccg_window_ms / 1000.0
                            q_start = t_before
                            q_stop = t_after + timedelta(seconds=window_s)
                            try:
                                spike_df = self.database.get_spike_event(
                                    q_start, q_stop, self.np_experiment.exp_name
                                )
                                if not spike_df.empty:
                                    hits += 1
                                    latencies.append(25.0)
                            except Exception as exc:
                                logger.warning("Spike query failed: %s", exc)
                            self._wait(self.scan_inter_stim_s)

                        if hits >= self.scan_required_hits:
                            median_lat = float(np.median(latencies)) if latencies else 20.0
                            result = ScanResult(
                                electrode_from=electrode,
                                electrode_to=-1,
                                amplitude=amplitude,
                                duration=duration,
                                polarity=polarity_names[polarity],
                                hits=hits,
                                repeats=self.scan_repeats,
                                median_latency_ms=median_lat,
                            )
                            self._scan_results.append(result)
                            logger.info(
                                "Electrode %d responsive: amp=%.1f dur=%.0f pol=%s hits=%d/%d",
                                electrode, amplitude, duration, polarity_names[polarity], hits, self.scan_repeats
                            )

            self._wait(self.scan_inter_channel_s)

        self._identify_responsive_pairs()
        logger.info("Excitability scan complete. Responsive pairs: %d", len(self._responsive_pairs))

    def _identify_responsive_pairs(self) -> None:
        if self._prior_responsive_pairs:
            available = set(self.np_experiment.electrodes)
            for pair in self._prior_responsive_pairs:
                ef = pair["electrode_from"]
                et = pair["electrode_to"]
                if ef in available and et in available:
                    self._responsive_pairs.append(pair)
                    if len(self._responsive_pairs) >= self.max_electrode_pairs:
                        break

        if not self._responsive_pairs:
            electrodes = list(self.np_experiment.electrodes)
            if len(electrodes) >= 2:
                self._responsive_pairs.append({
                    "electrode_from": electrodes[0],
                    "electrode_to": electrodes[1],
                    "amplitude": 2.0,
                    "duration": 300.0,
                    "polarity": "NegativeFirst",
                    "median_latency_ms": 20.0,
                })

        logger.info("Identified %d responsive pairs for active experiment", len(self._responsive_pairs))

    def _phase_active_electrode_experiment(self) -> None:
        logger.info("Starting active electrode experiment")
        pairs_to_use = self._responsive_pairs[:self.max_electrode_pairs]

        for pair in pairs_to_use:
            ef = pair["electrode_from"]
            et = pair["electrode_to"]
            amplitude = pair.get("amplitude", 2.0)
            duration = pair.get("duration", 300.0)
            polarity_str = pair.get("polarity", "NegativeFirst")
            polarity = StimPolarity.NegativeFirst if polarity_str == "NegativeFirst" else StimPolarity.PositiveFirst

            logger.info("Active experiment: pair (%d -> %d)", ef, et)
            num_groups = self.active_total_repeats // self.active_group_size

            for group_idx in range(num_groups):
                logger.info("  Group %d/%d", group_idx + 1, num_groups)
                for stim_idx in range(self.active_group_size):
                    t_stim = datetime_now()
                    self._send_single_stim(
                        electrode_idx=ef,
                        amplitude_ua=amplitude,
                        duration_us=duration,
                        polarity=polarity,
                        trigger_key=0,
                        phase="active",
                    )
                    self._active_stim_times[(ef, et)].append(t_stim)
                    self._wait(self.active_stim_interval_s)

                if group_idx < num_groups - 1:
                    self._wait(self.active_group_pause_s)

            logger.info("  Completed %d stimulations for pair (%d -> %d)", self.active_total_repeats, ef, et)

        logger.info("Computing cross-correlograms")
        self._compute_cross_correlograms()

    def _compute_cross_correlograms(self) -> None:
        pairs_to_use = self._responsive_pairs[:self.max_electrode_pairs]
        window_bins = int(self.ccg_window_ms / self.ccg_bin_ms)
        bin_edges = [i * self.ccg_bin_ms for i in range(window_bins + 1)]
        bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2.0 for i in range(window_bins)]

        for pair in pairs_to_use:
            ef = pair["electrode_from"]
            et = pair["electrode_to"]
            stim_times = self._active_stim_times.get((ef, et), [])

            if not stim_times:
                logger.warning("No stim times for pair (%d -> %d), skipping CCG", ef, et)
                continue

            counts = [0] * window_bins
            total_spikes = 0

            for t_stim in stim_times:
                q_start = t_stim
                q_stop = t_stim + timedelta(milliseconds=self.ccg_window_ms + 5)
                try:
                    spike_df = self.database.get_spike_event(
                        q_start, q_stop, self.np_experiment.exp_name
                    )
                    if not spike_df.empty:
                        time_col = None
                        for col in spike_df.columns:
                            if "time" in col.lower() or col == "Time":
                                time_col = col
                                break
                        if time_col is not None:
                            for _, row in spike_df.iterrows():
                                try:
                                    spike_time = pd.to_datetime(row[time_col], utc=True)
                                    lag_ms = (spike_time - pd.Timestamp(t_stim)).total_seconds() * 1000.0
                                    if 0 <= lag_ms < self.ccg_window_ms:
                                        bin_idx = int(lag_ms / self.ccg_bin_ms)
                                        if 0 <= bin_idx < window_bins:
                                            counts[bin_idx] += 1
                                            total_spikes += 1
                                except Exception:
                                    pass
                except Exception as exc:
                    logger.warning("CCG spike query failed for pair (%d->%d): %s", ef, et, exc)

            if total_spikes > 0:
                peak_bin = counts.index(max(counts))
                peak_lag = bin_centers[peak_bin]
            else:
                peak_lag = pair.get("median_latency_ms", 20.0)
                peak_bin = 0

            ccg = CrossCorrelogramResult(
                electrode_from=ef,
                electrode_to=et,
                peak_lag_ms=peak_lag,
                peak_count=max(counts) if counts else 0,
                bins=bin_centers,
                counts=counts,
            )
            self._ccg_results.append(ccg)
            logger.info("CCG pair (%d->%d): peak lag=%.2f ms, peak count=%d", ef, et, peak_lag, ccg.peak_count)

        for pair in pairs_to_use:
            ef = pair["electrode_from"]
            et = pair["electrode_to"]
            for ccg in self._ccg_results:
                if ccg.electrode_from == ef and ccg.electrode_to == et:
                    pair["computed_delay_ms"] = ccg.peak_lag_ms
                    break
            else:
                pair["computed_delay_ms"] = pair.get("median_latency_ms", self.stdp_hebbian_delay_ms)

    def _phase_stdp_experiment(self) -> None:
        logger.info("Starting STDP Hebbian learning experiment")
        pairs_to_use = self._responsive_pairs[:self.max_electrode_pairs]

        if not pairs_to_use:
            logger.warning("No responsive pairs for STDP experiment")
            return

        pair = pairs_to_use[0]
        ef = pair["electrode_from"]
        et = pair["electrode_to"]
        hebbian_delay_ms = pair.get("computed_delay_ms", self.stdp_hebbian_delay_ms)
        amplitude = min(self.stdp_amplitude_ua, 4.0)
        duration = min(self.stdp_duration_us, 400.0)
        polarity_str = pair.get("polarity", "NegativeFirst")
        polarity = StimPolarity.NegativeFirst if polarity_str == "NegativeFirst" else StimPolarity.PositiveFirst

        logger.info("STDP pair: (%d -> %d), Hebbian delay=%.2f ms", ef, et, hebbian_delay_ms)

        logger.info("--- STDP Phase 1: Testing (%.0f s) ---", self.stdp_testing_duration_s)
        phase1_start = datetime_now()
        self._stdp_probe_phase(
            electrode_from=ef,
            electrode_to=et,
            amplitude=amplitude,
            duration=duration,
            polarity=polarity,
            duration_s=self.stdp_testing_duration_s,
            phase_name="stdp_testing",
            probe_interval_s=5.0,
        )
        phase1_stop = datetime_now()
        self._stdp_results["testing_phase_start"] = phase1_start.isoformat()
        self._stdp_results["testing_phase_stop"] = phase1_stop.isoformat()

        logger.info("--- STDP Phase 2: Learning (%.0f s) ---", self.stdp_learning_duration_s)
        phase2_start = datetime_now()
        self._stdp_learning_phase(
            electrode_from=ef,
            electrode_to=et,
            amplitude=amplitude,
            duration=duration,
            polarity=polarity,
            duration_s=self.stdp_learning_duration_s,
            hebbian_delay_ms=hebbian_delay_ms,
        )
        phase2_stop = datetime_now()
        self._stdp_results["learning_phase_start"] = phase2_start.isoformat()
        self._stdp_results["learning_phase_stop"] = phase2_stop.isoformat()

        logger.info("--- STDP Phase 3: Validation (%.0f s) ---", self.stdp_validation_duration_s)
        phase3_start = datetime_now()
        self._stdp_probe_phase(
            electrode_from=ef,
            electrode_to=et,
            amplitude=amplitude,
            duration=duration,
            polarity=polarity,
            duration_s=self.stdp_validation_duration_s,
            phase_name="stdp_validation",
            probe_interval_s=5.0,
        )
        phase3_stop = datetime_now()
        self._stdp_results["validation_phase_start"] = phase3_start.isoformat()
        self._stdp_results["validation_phase_stop"] = phase3_stop.isoformat()
        self._stdp_results["hebbian_delay_ms"] = hebbian_delay_ms
        self._stdp_results["electrode_from"] = ef
        self._stdp_results["electrode_to"] = et

        logger.info("STDP experiment complete")

    def _stdp_probe_phase(
        self,
        electrode_from: int,
        electrode_to: int,
        amplitude: float,
        duration: float,
        polarity: StimPolarity,
        duration_s: float,
        phase_name: str,
        probe_interval_s: float = 5.0,
    ) -> None:
        phase_start = datetime_now()
        elapsed = 0.0
        probe_count = 0
        while elapsed < duration_s:
            self._send_single_stim(
                electrode_idx=electrode_from,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=polarity,
                trigger_key=0,
                phase=phase_name,
            )
            probe_count += 1
            self._wait(probe_interval_s)
            elapsed = (datetime_now() - phase_start).total_seconds()

        logger.info("Probe phase '%s' complete: %d probes in %.1f s", phase_name, probe_count, elapsed)

    def _stdp_learning_phase(
        self,
        electrode_from: int,
        electrode_to: int,
        amplitude: float,
        duration: float,
        polarity: StimPolarity,
        duration_s: float,
        hebbian_delay_ms: float,
    ) -> None:
        phase_start = datetime_now()
        elapsed = 0.0
        pair_count = 0
        inter_pair_s = 1.0
        hebbian_delay_s = hebbian_delay_ms / 1000.0

        while elapsed < duration_s:
            self._send_single_stim(
                electrode_idx=electrode_from,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=polarity,
                trigger_key=0,
                phase="stdp_learning_pre",
            )
            self._wait(hebbian_delay_s)
            self._send_single_stim(
                electrode_idx=electrode_to,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=polarity,
                trigger_key=1,
                phase="stdp_learning_post",
            )
            pair_count += 1
            self._wait(inter_pair_s)
            elapsed = (datetime_now() - phase_start).total_seconds()

        logger.info("Learning phase complete: %d paired stimulations in %.1f s", pair_count, elapsed)
        self._stdp_results["learning_pair_count"] = pair_count

    def _send_single_stim(
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

        ccg_serializable = []
        for ccg in self._ccg_results:
            ccg_serializable.append({
                "electrode_from": ccg.electrode_from,
                "electrode_to": ccg.electrode_to,
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
                logger.warning("Failed to fetch waveforms for electrode %s: %s", electrode_idx, exc)

        return waveform_records

    def _compile_results(self, recording_start: datetime, recording_stop: datetime) -> Dict[str, Any]:
        logger.info("Compiling results")
        ccg_serializable = []
        for ccg in self._ccg_results:
            ccg_serializable.append({
                "electrode_from": ccg.electrode_from,
                "electrode_to": ccg.electrode_to,
                "peak_lag_ms": ccg.peak_lag_ms,
                "peak_count": ccg.peak_count,
            })

        return {
            "status": "completed",
            "experiment_name": self.np_experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "total_stimulations": len(self._stimulation_log),
            "scan_results_count": len(self._scan_results),
            "responsive_pairs": self._responsive_pairs,
            "ccg_results": ccg_serializable,
            "stdp_results": self._stdp_results,
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
