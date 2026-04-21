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
class CrossCorrelogram:
    stim_electrode: int
    resp_electrode: int
    bins_ms: List[float]
    counts: List[int]
    peak_latency_ms: float
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
    Stage 2: Active Electrode Experiment with cross-correlograms
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
        active_stim_hz: float = 1.0,
        active_group_size: int = 10,
        active_inter_group_s: float = 5.0,
        active_total_repeats: int = 100,
        stdp_testing_duration_s: float = 1200.0,
        stdp_learning_duration_s: float = 3000.0,
        stdp_validation_duration_s: float = 1200.0,
        stdp_hebbian_delay_ms: float = 10.0,
        stdp_amplitude_ua: float = 3.0,
        stdp_duration_us: float = 400.0,
        ccg_window_ms: float = 50.0,
        ccg_bin_ms: float = 1.0,
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

        self.active_stim_hz = active_stim_hz
        self.active_group_size = active_group_size
        self.active_inter_group_s = active_inter_group_s
        self.active_total_repeats = active_total_repeats

        self.stdp_testing_duration_s = stdp_testing_duration_s
        self.stdp_learning_duration_s = stdp_learning_duration_s
        self.stdp_validation_duration_s = stdp_validation_duration_s
        self.stdp_hebbian_delay_ms = stdp_hebbian_delay_ms
        self.stdp_amplitude_ua = stdp_amplitude_ua
        self.stdp_duration_us = stdp_duration_us

        self.ccg_window_ms = ccg_window_ms
        self.ccg_bin_ms = ccg_bin_ms

        self.np_experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._scan_results: List[ScanResult] = []
        self._responsive_pairs: List[Dict[str, Any]] = []
        self._active_stim_times: Dict[Tuple[int, int], List[datetime]] = defaultdict(list)
        self._correlograms: List[CrossCorrelogram] = []
        self._stdp_results: Dict[str, Any] = {}

        self._prior_responsive_pairs = [
            {"electrode_from": 17, "electrode_to": 18, "amplitude": 3.0, "duration": 400.0,
             "polarity": "PositiveFirst", "median_latency_ms": 13.477},
            {"electrode_from": 21, "electrode_to": 19, "amplitude": 3.0, "duration": 400.0,
             "polarity": "PositiveFirst", "median_latency_ms": 18.979},
            {"electrode_from": 21, "electrode_to": 22, "amplitude": 3.0, "duration": 400.0,
             "polarity": "NegativeFirst", "median_latency_ms": 10.859},
            {"electrode_from": 7, "electrode_to": 6, "amplitude": 3.0, "duration": 400.0,
             "polarity": "PositiveFirst", "median_latency_ms": 24.622},
            {"electrode_from": 6, "electrode_to": 7, "amplitude": 3.0, "duration": 400.0,
             "polarity": "PositiveFirst", "median_latency_ms": 19.294},
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

            logger.info("=== Stage 3: Hebbian STDP Experiment ===")
            self._phase_stdp_experiment()

            recording_stop = datetime_now()

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
        electrodes = self.np_experiment.electrodes
        polarities = [StimPolarity.PositiveFirst, StimPolarity.NegativeFirst]
        polarity_names = {StimPolarity.PositiveFirst: "PositiveFirst", StimPolarity.NegativeFirst: "NegativeFirst"}

        for elec_idx in electrodes:
            for amplitude in self.scan_amplitudes:
                for duration in self.scan_durations:
                    for polarity in polarities:
                        hits = 0
                        latencies = []
                        for rep in range(self.scan_repeats):
                            stim_time = datetime_now()
                            self._send_stim_pulse(
                                electrode_idx=elec_idx,
                                amplitude_ua=amplitude,
                                duration_us=duration,
                                polarity=polarity,
                                trigger_key=0,
                                phase="scan",
                            )
                            self._wait(0.05)
                            query_start = stim_time
                            query_stop = datetime_now()
                            try:
                                spike_df = self.database.get_spike_event(
                                    query_start, query_stop, self.np_experiment.exp_name
                                )
                                if not spike_df.empty:
                                    for _, row in spike_df.iterrows():
                                        ch = int(row.get("channel", -1))
                                        if ch != elec_idx and ch in electrodes:
                                            spike_time = row.get("Time")
                                            if spike_time is not None:
                                                if hasattr(spike_time, "timestamp"):
                                                    lat_ms = (spike_time.timestamp() - stim_time.timestamp()) * 1000.0
                                                else:
                                                    lat_ms = 0.0
                                                if 0 < lat_ms < self.ccg_window_ms:
                                                    hits += 1
                                                    latencies.append(lat_ms)
                                                    break
                            except Exception as exc:
                                logger.warning("Scan spike query error: %s", exc)

                            if rep < self.scan_repeats - 1:
                                self._wait(self.scan_inter_stim_s)

                        if hits >= self.scan_required_hits:
                            median_lat = float(np.median(latencies)) if latencies else 0.0
                            result = ScanResult(
                                electrode_from=elec_idx,
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
                                "Responsive: elec %d amp=%.1f dur=%.0f pol=%s hits=%d",
                                elec_idx, amplitude, duration, polarity_names[polarity], hits
                            )

                    self._wait(self.scan_inter_channel_s)

        if self._scan_results:
            self._responsive_pairs = [
                {
                    "electrode_from": r.electrode_from,
                    "electrode_to": r.electrode_to,
                    "amplitude": r.amplitude,
                    "duration": r.duration,
                    "polarity": r.polarity,
                    "median_latency_ms": r.median_latency_ms,
                }
                for r in self._scan_results
            ]
        else:
            logger.info("No new responsive pairs found in scan; using prior scan results")
            self._responsive_pairs = list(self._prior_responsive_pairs)

        logger.info("Excitability scan complete. Responsive pairs: %d", len(self._responsive_pairs))

    def _phase_active_electrode_experiment(self) -> None:
        logger.info("Starting active electrode experiment")
        pairs_to_use = self._prior_responsive_pairs

        inter_stim_s = 1.0 / max(self.active_stim_hz, 0.001)
        n_groups = self.active_total_repeats // self.active_group_size

        for pair in pairs_to_use:
            elec_from = pair["electrode_from"]
            amplitude = min(pair["amplitude"], 4.0)
            duration = min(pair["duration"], 400.0)
            polarity_str = pair.get("polarity", "PositiveFirst")
            polarity = StimPolarity.PositiveFirst if polarity_str == "PositiveFirst" else StimPolarity.NegativeFirst
            pair_key = (elec_from, pair.get("electrode_to", -1))

            logger.info("Active experiment for electrode %d -> %d", elec_from, pair.get("electrode_to", -1))

            for group_idx in range(n_groups):
                for stim_idx in range(self.active_group_size):
                    stim_time = datetime_now()
                    self._send_stim_pulse(
                        electrode_idx=elec_from,
                        amplitude_ua=amplitude,
                        duration_us=duration,
                        polarity=polarity,
                        trigger_key=0,
                        phase="active",
                    )
                    self._active_stim_times[pair_key].append(stim_time)
                    if stim_idx < self.active_group_size - 1:
                        self._wait(inter_stim_s)

                if group_idx < n_groups - 1:
                    self._wait(self.active_inter_group_s)

            logger.info("Completed %d stimulations for pair %s", self.active_total_repeats, pair_key)

        logger.info("Computing cross-correlograms")
        self._compute_correlograms(pairs_to_use)

    def _compute_correlograms(self, pairs: List[Dict[str, Any]]) -> None:
        n_bins = int(self.ccg_window_ms / self.ccg_bin_ms)
        bins_ms = [i * self.ccg_bin_ms for i in range(n_bins)]

        for pair in pairs:
            elec_from = pair["electrode_from"]
            elec_to = pair.get("electrode_to", -1)
            pair_key = (elec_from, elec_to)
            stim_times = self._active_stim_times.get(pair_key, [])

            if not stim_times:
                continue

            counts = [0] * n_bins

            for stim_time in stim_times:
                query_start = stim_time
                query_stop = stim_time + timedelta(milliseconds=self.ccg_window_ms + 10)
                try:
                    spike_df = self.database.get_spike_event(
                        query_start, query_stop, self.np_experiment.exp_name
                    )
                    if not spike_df.empty:
                        for _, row in spike_df.iterrows():
                            ch = int(row.get("channel", -1))
                            if ch == elec_to:
                                spike_time = row.get("Time")
                                if spike_time is not None:
                                    if hasattr(spike_time, "timestamp"):
                                        lat_ms = (spike_time.timestamp() - stim_time.timestamp()) * 1000.0
                                    else:
                                        lat_ms = 0.0
                                    bin_idx = int(lat_ms / self.ccg_bin_ms)
                                    if 0 <= bin_idx < n_bins:
                                        counts[bin_idx] += 1
                except Exception as exc:
                    logger.warning("CCG query error for pair %s: %s", pair_key, exc)

            if max(counts) > 0:
                peak_bin = int(np.argmax(counts))
                peak_latency = bins_ms[peak_bin]
                peak_count = counts[peak_bin]
            else:
                peak_latency = pair.get("median_latency_ms", 0.0)
                peak_count = 0

            ccg = CrossCorrelogram(
                stim_electrode=elec_from,
                resp_electrode=elec_to,
                bins_ms=bins_ms,
                counts=counts,
                peak_latency_ms=peak_latency,
                peak_count=peak_count,
            )
            self._correlograms.append(ccg)
            logger.info(
                "CCG pair (%d->%d): peak at %.1f ms (count=%d)",
                elec_from, elec_to, peak_latency, peak_count
            )

    def _phase_stdp_experiment(self) -> None:
        logger.info("Starting STDP experiment")
        pairs_to_use = self._prior_responsive_pairs

        if not pairs_to_use:
            logger.warning("No pairs available for STDP experiment")
            return

        primary_pair = pairs_to_use[0]
        elec_pre = primary_pair["electrode_from"]
        elec_post = primary_pair.get("electrode_to", -1)
        amplitude = min(self.stdp_amplitude_ua, 4.0)
        duration = min(self.stdp_duration_us, 400.0)
        polarity_str = primary_pair.get("polarity", "PositiveFirst")
        polarity = StimPolarity.PositiveFirst if polarity_str == "PositiveFirst" else StimPolarity.NegativeFirst

        ccg_for_pair = None
        for ccg in self._correlograms:
            if ccg.stim_electrode == elec_pre and ccg.resp_electrode == elec_post:
                ccg_for_pair = ccg
                break

        if ccg_for_pair is not None and ccg_for_pair.peak_latency_ms > 0:
            hebbian_delay_ms = ccg_for_pair.peak_latency_ms
        else:
            hebbian_delay_ms = self.stdp_hebbian_delay_ms

        logger.info(
            "STDP pair: pre=%d post=%d hebbian_delay=%.1f ms",
            elec_pre, elec_post, hebbian_delay_ms
        )

        logger.info("STDP Phase 1: Testing (%.0f s)", self.stdp_testing_duration_s)
        testing_start = datetime_now()
        pre_test_spikes = self._run_passive_recording_phase(
            elec_pre, elec_post, self.stdp_testing_duration_s, "stdp_testing"
        )
        testing_stop = datetime_now()

        logger.info("STDP Phase 2: Learning (%.0f s)", self.stdp_learning_duration_s)
        learning_start = datetime_now()
        learning_stim_count = self._run_learning_phase(
            elec_pre, elec_post, amplitude, duration, polarity,
            hebbian_delay_ms, self.stdp_learning_duration_s
        )
        learning_stop = datetime_now()

        logger.info("STDP Phase 3: Validation (%.0f s)", self.stdp_validation_duration_s)
        validation_start = datetime_now()
        post_test_spikes = self._run_passive_recording_phase(
            elec_pre, elec_post, self.stdp_validation_duration_s, "stdp_validation"
        )
        validation_stop = datetime_now()

        pre_ccg = self._compute_phase_ccg(elec_pre, elec_post, testing_start, testing_stop)
        post_ccg = self._compute_phase_ccg(elec_pre, elec_post, validation_start, validation_stop)

        delta_ccg = {}
        for bin_ms in pre_ccg:
            delta_ccg[bin_ms] = post_ccg.get(bin_ms, 0) - pre_ccg.get(bin_ms, 0)

        self._stdp_results = {
            "pre_electrode": elec_pre,
            "post_electrode": elec_post,
            "hebbian_delay_ms": hebbian_delay_ms,
            "learning_stim_count": learning_stim_count,
            "pre_test_spike_count": pre_test_spikes,
            "post_test_spike_count": post_test_spikes,
            "pre_ccg_peak_bin_ms": max(pre_ccg, key=pre_ccg.get) if pre_ccg else None,
            "post_ccg_peak_bin_ms": max(post_ccg, key=post_ccg.get) if post_ccg else None,
            "delta_ccg_summary": {
                "max_increase_bin_ms": max(delta_ccg, key=delta_ccg.get) if delta_ccg else None,
                "max_increase_count": max(delta_ccg.values()) if delta_ccg else 0,
            },
            "testing_start_utc": testing_start.isoformat(),
            "testing_stop_utc": testing_stop.isoformat(),
            "learning_start_utc": learning_start.isoformat(),
            "learning_stop_utc": learning_stop.isoformat(),
            "validation_start_utc": validation_start.isoformat(),
            "validation_stop_utc": validation_stop.isoformat(),
        }
        logger.info("STDP experiment complete: %s", self._stdp_results)

    def _run_passive_recording_phase(
        self,
        elec_pre: int,
        elec_post: int,
        duration_s: float,
        phase_label: str,
    ) -> int:
        phase_start = datetime_now()
        probe_interval_s = 30.0
        n_probes = max(1, int(duration_s / probe_interval_s))
        total_spikes = 0

        for probe_idx in range(n_probes):
            elapsed = (datetime_now() - phase_start).total_seconds()
            remaining = duration_s - elapsed
            if remaining <= 0:
                break
            sleep_time = min(probe_interval_s, remaining)
            self._wait(sleep_time)

            query_stop = datetime_now()
            query_start = query_stop - timedelta(seconds=sleep_time)
            try:
                spike_df = self.database.get_spike_event(
                    query_start, query_stop, self.np_experiment.exp_name
                )
                if not spike_df.empty:
                    total_spikes += len(spike_df)
            except Exception as exc:
                logger.warning("Passive recording query error (%s): %s", phase_label, exc)

        logger.info("Phase %s: total spikes observed = %d", phase_label, total_spikes)
        return total_spikes

    def _run_learning_phase(
        self,
        elec_pre: int,
        elec_post: int,
        amplitude: float,
        duration: float,
        polarity: StimPolarity,
        hebbian_delay_ms: float,
        total_duration_s: float,
    ) -> int:
        phase_start = datetime_now()
        stim_count = 0
        inter_stim_s = 10.0
        hebbian_delay_s = hebbian_delay_ms / 1000.0

        while True:
            elapsed = (datetime_now() - phase_start).total_seconds()
            if elapsed >= total_duration_s:
                break

            self._send_stim_pulse(
                electrode_idx=elec_pre,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=polarity,
                trigger_key=0,
                phase="stdp_learning_pre",
            )
            stim_count += 1

            self._wait(hebbian_delay_s)

            self._send_stim_pulse(
                electrode_idx=elec_post,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=polarity,
                trigger_key=1,
                phase="stdp_learning_post",
            )
            stim_count += 1

            remaining = total_duration_s - (datetime_now() - phase_start).total_seconds()
            if remaining <= 0:
                break
            self._wait(min(inter_stim_s, remaining))

        logger.info("Learning phase complete: %d stimulations delivered", stim_count)
        return stim_count

    def _compute_phase_ccg(
        self,
        elec_pre: int,
        elec_post: int,
        phase_start: datetime,
        phase_stop: datetime,
    ) -> Dict[float, int]:
        n_bins = int(self.ccg_window_ms / self.ccg_bin_ms)
        bins_ms = [i * self.ccg_bin_ms for i in range(n_bins)]
        counts = {b: 0 for b in bins_ms}

        try:
            trigger_df = self.database.get_all_triggers(phase_start, phase_stop)
            spike_df = self.database.get_spike_event(
                phase_start, phase_stop, self.np_experiment.exp_name
            )

            if trigger_df.empty or spike_df.empty:
                return counts

            post_spikes = spike_df[spike_df["channel"] == elec_post] if "channel" in spike_df.columns else pd.DataFrame()
            if post_spikes.empty:
                return counts

            for _, trig_row in trigger_df.iterrows():
                trig_time = trig_row.get("_time")
                if trig_time is None:
                    continue
                if hasattr(trig_time, "timestamp"):
                    trig_ts = trig_time.timestamp()
                else:
                    continue

                for _, spike_row in post_spikes.iterrows():
                    spike_time = spike_row.get("Time")
                    if spike_time is None:
                        continue
                    if hasattr(spike_time, "timestamp"):
                        spike_ts = spike_time.timestamp()
                    else:
                        continue
                    lat_ms = (spike_ts - trig_ts) * 1000.0
                    bin_idx = int(lat_ms / self.ccg_bin_ms)
                    if 0 <= bin_idx < n_bins:
                        counts[bins_ms[bin_idx]] += 1

        except Exception as exc:
            logger.warning("Phase CCG computation error: %s", exc)

        return counts

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

        polarity_str = "PositiveFirst" if polarity == StimPolarity.PositiveFirst else "NegativeFirst"
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
            "correlograms_count": len(self._correlograms),
            "stdp_results": self._stdp_results,
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

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        logger.info("Compiling results")
        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.np_experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "total_stimulations": len(self._stimulation_log),
            "scan_results": [asdict(r) for r in self._scan_results],
            "responsive_pairs": self._responsive_pairs,
            "correlograms": [
                {
                    "stim_electrode": c.stim_electrode,
                    "resp_electrode": c.resp_electrode,
                    "peak_latency_ms": c.peak_latency_ms,
                    "peak_count": c.peak_count,
                }
                for c in self._correlograms
            ],
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
