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
    bin_edges_ms: List[float]
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
        active_stim_hz: float = 1.0,
        active_group_size: int = 10,
        active_inter_group_s: float = 5.0,
        active_total_repeats: int = 100,
        ccg_bin_ms: float = 4.0,
        ccg_window_ms: float = 100.0,
        testing_phase_min: float = 20.0,
        learning_phase_min: float = 50.0,
        validation_phase_min: float = 20.0,
        probe_amplitude_ua: float = 1.0,
        probe_duration_us: float = 200.0,
        probe_hz: float = 0.1,
        conditioning_amplitude_ua: float = 2.0,
        conditioning_duration_us: float = 200.0,
        hebbian_delay_ms: float = 20.0,
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

        self.active_stim_hz = active_stim_hz
        self.active_group_size = active_group_size
        self.active_inter_group_s = active_inter_group_s
        self.active_total_repeats = active_total_repeats

        self.ccg_bin_ms = ccg_bin_ms
        self.ccg_window_ms = ccg_window_ms

        self.testing_phase_min = testing_phase_min
        self.learning_phase_min = learning_phase_min
        self.validation_phase_min = validation_phase_min

        self.probe_amplitude_ua = probe_amplitude_ua
        self.probe_duration_us = probe_duration_us
        self.probe_hz = probe_hz

        self.conditioning_amplitude_ua = conditioning_amplitude_ua
        self.conditioning_duration_us = conditioning_duration_us
        self.hebbian_delay_ms = hebbian_delay_ms

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._scan_results: List[ScanResult] = []
        self._responsive_pairs: List[Dict[str, Any]] = []
        self._ccg_results: List[CrossCorrelogram] = []
        self._phase_boundaries: Dict[str, str] = {}

        self._prior_reliable_connections = [
            {"electrode_from": 5, "electrode_to": 4, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 14.79},
            {"electrode_from": 6, "electrode_to": 7, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 20.195},
            {"electrode_from": 7, "electrode_to": 6, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 24.74},
            {"electrode_from": 17, "electrode_to": 18, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 13.17},
            {"electrode_from": 21, "electrode_to": 19, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 19.3},
            {"electrode_from": 21, "electrode_to": 22, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 11.34},
        ]

        self._prior_deep_scan = [
            {"stim_electrode": 17, "resp_electrode": 18, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 13.7, "response_rate": 0.92},
            {"stim_electrode": 21, "resp_electrode": 19, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 28.3, "response_rate": 0.92},
            {"stim_electrode": 21, "resp_electrode": 22, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 10.97, "response_rate": 0.84},
            {"stim_electrode": 7, "resp_electrode": 6, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 25.22, "response_rate": 0.87},
            {"stim_electrode": 6, "resp_electrode": 7, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 19.815, "response_rate": 0.46},
        ]

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

            logger.info("=== Phase 1: Basic Excitability Scan ===")
            self._phase_excitability_scan()

            logger.info("=== Phase 2: Active Electrode Experiment ===")
            self._phase_active_electrode()

            logger.info("=== Phase 3: Hebbian STDP Experiment ===")
            self._phase_hebbian_stdp()

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
        self._phase_boundaries["scan_start"] = datetime_now().isoformat()

        electrodes = self.experiment.electrodes
        polarities = [StimPolarity.PositiveFirst, StimPolarity.NegativeFirst]
        polarity_names = {StimPolarity.PositiveFirst: "PositiveFirst", StimPolarity.NegativeFirst: "NegativeFirst"}

        for electrode_idx in electrodes:
            for amplitude in self.scan_amplitudes:
                for duration in self.scan_durations:
                    for polarity in polarities:
                        hits = 0
                        latencies = []
                        for rep in range(self.scan_repeats):
                            stim_time = datetime_now()
                            self._send_stim(
                                electrode_idx=electrode_idx,
                                amplitude_ua=amplitude,
                                duration_us=duration,
                                polarity=polarity,
                                trigger_key=0,
                                phase="scan",
                            )
                            self._wait(0.05)
                            query_start = stim_time
                            query_stop = datetime_now()
                            spike_df = self.database.get_spike_event(
                                query_start, query_stop, self.experiment.exp_name
                            )
                            if not spike_df.empty:
                                resp_spikes = spike_df[spike_df["channel"] != electrode_idx]
                                if not resp_spikes.empty:
                                    hits += 1
                                    if "Time" in resp_spikes.columns:
                                        t_stim = stim_time.timestamp()
                                        for _, row in resp_spikes.iterrows():
                                            t_spike = pd.Timestamp(row["Time"]).timestamp()
                                            lat_ms = (t_spike - t_stim) * 1000.0
                                            if 0 < lat_ms < 100:
                                                latencies.append(lat_ms)
                            if rep < self.scan_repeats - 1:
                                self._wait(self.scan_inter_stim_s)

                        if hits >= 3:
                            median_lat = float(np.median(latencies)) if latencies else 0.0
                            result = ScanResult(
                                electrode_from=electrode_idx,
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
                                "Responsive electrode %d: amp=%.1f dur=%.0f pol=%s hits=%d",
                                electrode_idx, amplitude, duration, polarity_names[polarity], hits
                            )

                    self._wait(self.scan_inter_channel_s)

        self._phase_boundaries["scan_stop"] = datetime_now().isoformat()
        logger.info("Excitability scan complete. Responsive configs: %d", len(self._scan_results))

        self._build_responsive_pairs()

    def _build_responsive_pairs(self) -> None:
        if self._scan_results:
            seen = set()
            for r in self._scan_results:
                key = (r.electrode_from, r.electrode_to)
                if key not in seen:
                    seen.add(key)
                    self._responsive_pairs.append({
                        "electrode_from": r.electrode_from,
                        "electrode_to": r.electrode_to,
                        "amplitude": r.amplitude,
                        "duration": r.duration,
                        "polarity": r.polarity,
                        "median_latency_ms": r.median_latency_ms,
                    })
        else:
            logger.info("No new scan results; using prior reliable connections")
            for conn in self._prior_reliable_connections:
                self._responsive_pairs.append({
                    "electrode_from": conn["electrode_from"],
                    "electrode_to": conn["electrode_to"],
                    "amplitude": conn["amplitude"],
                    "duration": conn["duration"],
                    "polarity": conn["polarity"],
                    "median_latency_ms": conn["median_latency_ms"],
                })

        unique_pairs = {}
        for p in self._responsive_pairs:
            key = (p["electrode_from"], p["electrode_to"])
            if key not in unique_pairs:
                unique_pairs[key] = p
        self._responsive_pairs = list(unique_pairs.values())
        logger.info("Responsive pairs identified: %d", len(self._responsive_pairs))

    def _phase_active_electrode(self) -> None:
        logger.info("Starting active electrode experiment")
        self._phase_boundaries["active_start"] = datetime_now().isoformat()

        pairs_to_use = self._responsive_pairs if self._responsive_pairs else [
            {
                "electrode_from": p["stim_electrode"],
                "electrode_to": p["resp_electrode"],
                "amplitude": p["amplitude"],
                "duration": p["duration"],
                "polarity": p["polarity"],
                "median_latency_ms": p["median_latency_ms"],
            }
            for p in self._prior_deep_scan
        ]

        stim_interval_s = 1.0 / self.active_stim_hz
        n_groups = self.active_total_repeats // self.active_group_size

        for pair in pairs_to_use:
            stim_elec = pair["electrode_from"]
            resp_elec = pair["electrode_to"]
            amplitude = min(pair["amplitude"], 4.0)
            duration = min(pair["duration"], 400.0)
            polarity_str = pair.get("polarity", "PositiveFirst")
            polarity = StimPolarity.PositiveFirst if polarity_str == "PositiveFirst" else StimPolarity.NegativeFirst

            logger.info("Active stim pair: %d -> %d", stim_elec, resp_elec)

            pair_stim_times = []
            pair_spike_latencies = []

            for group_idx in range(n_groups):
                for stim_idx in range(self.active_group_size):
                    stim_time = datetime_now()
                    self._send_stim(
                        electrode_idx=stim_elec,
                        amplitude_ua=amplitude,
                        duration_us=duration,
                        polarity=polarity,
                        trigger_key=1,
                        phase="active",
                    )
                    pair_stim_times.append(stim_time.isoformat())
                    self._wait(0.05)

                    query_start = stim_time
                    query_stop = datetime_now()
                    spike_df = self.database.get_spike_event(
                        query_start, query_stop, self.experiment.exp_name
                    )
                    if not spike_df.empty and "channel" in spike_df.columns:
                        resp_spikes = spike_df[spike_df["channel"] == resp_elec]
                        if not resp_spikes.empty and "Time" in resp_spikes.columns:
                            t_stim_ts = stim_time.timestamp()
                            for _, row in resp_spikes.iterrows():
                                t_spike = pd.Timestamp(row["Time"]).timestamp()
                                lat_ms = (t_spike - t_stim_ts) * 1000.0
                                if 0 < lat_ms < 200:
                                    pair_spike_latencies.append(lat_ms)

                    if stim_idx < self.active_group_size - 1:
                        self._wait(stim_interval_s - 0.05)

                if group_idx < n_groups - 1:
                    self._wait(self.active_inter_group_s)

            ccg = self._compute_ccg(
                stim_electrode=stim_elec,
                resp_electrode=resp_elec,
                latencies_ms=pair_spike_latencies,
            )
            self._ccg_results.append(ccg)
            logger.info(
                "CCG pair %d->%d: peak=%.1f ms count=%d",
                stim_elec, resp_elec, ccg.peak_latency_ms, ccg.peak_count
            )

        self._phase_boundaries["active_stop"] = datetime_now().isoformat()
        logger.info("Active electrode experiment complete")

    def _compute_ccg(
        self,
        stim_electrode: int,
        resp_electrode: int,
        latencies_ms: List[float],
    ) -> CrossCorrelogram:
        n_bins = int(self.ccg_window_ms / self.ccg_bin_ms)
        bin_edges = [i * self.ccg_bin_ms for i in range(n_bins + 1)]
        counts = [0] * n_bins

        for lat in latencies_ms:
            if 0 <= lat < self.ccg_window_ms:
                bin_idx = int(lat / self.ccg_bin_ms)
                if bin_idx < n_bins:
                    counts[bin_idx] += 1

        peak_count = max(counts) if counts else 0
        peak_bin = counts.index(peak_count) if peak_count > 0 else 0
        peak_latency = (peak_bin + 0.5) * self.ccg_bin_ms

        return CrossCorrelogram(
            stim_electrode=stim_electrode,
            resp_electrode=resp_electrode,
            bin_edges_ms=bin_edges,
            counts=counts,
            peak_latency_ms=peak_latency,
            peak_count=peak_count,
        )

    def _phase_hebbian_stdp(self) -> None:
        logger.info("Starting Hebbian STDP experiment")
        self._phase_boundaries["stdp_start"] = datetime_now().isoformat()

        pairs_for_stdp = []
        if self._ccg_results:
            for ccg in self._ccg_results:
                if ccg.peak_count > 0:
                    pairs_for_stdp.append({
                        "stim_electrode": ccg.stim_electrode,
                        "resp_electrode": ccg.resp_electrode,
                        "hebbian_delay_ms": ccg.peak_latency_ms,
                    })

        if not pairs_for_stdp:
            for p in self._prior_deep_scan:
                pairs_for_stdp.append({
                    "stim_electrode": p["stim_electrode"],
                    "resp_electrode": p["resp_electrode"],
                    "hebbian_delay_ms": p["median_latency_ms"],
                })

        if not pairs_for_stdp:
            logger.warning("No pairs for STDP; skipping")
            return

        primary_pair = pairs_for_stdp[0]
        stim_elec = primary_pair["stim_electrode"]
        resp_elec = primary_pair["resp_electrode"]
        delay_ms = primary_pair.get("hebbian_delay_ms", self.hebbian_delay_ms)
        delay_ms = max(10.0, min(delay_ms, 50.0))

        logger.info("STDP pair: %d -> %d, delay=%.1f ms", stim_elec, resp_elec, delay_ms)

        probe_interval_s = 1.0 / self.probe_hz

        logger.info("STDP Testing phase: %.0f min", self.testing_phase_min)
        self._phase_boundaries["stdp_testing_start"] = datetime_now().isoformat()
        self._run_probe_phase(
            stim_elec=stim_elec,
            resp_elec=resp_elec,
            duration_min=self.testing_phase_min,
            probe_interval_s=probe_interval_s,
            phase_label="stdp_testing",
        )
        self._phase_boundaries["stdp_testing_stop"] = datetime_now().isoformat()

        logger.info("STDP Learning phase: %.0f min", self.learning_phase_min)
        self._phase_boundaries["stdp_learning_start"] = datetime_now().isoformat()
        self._run_learning_phase(
            stim_elec=stim_elec,
            resp_elec=resp_elec,
            duration_min=self.learning_phase_min,
            hebbian_delay_ms=delay_ms,
        )
        self._phase_boundaries["stdp_learning_stop"] = datetime_now().isoformat()

        logger.info("STDP Validation phase: %.0f min", self.validation_phase_min)
        self._phase_boundaries["stdp_validation_start"] = datetime_now().isoformat()
        self._run_probe_phase(
            stim_elec=stim_elec,
            resp_elec=resp_elec,
            duration_min=self.validation_phase_min,
            probe_interval_s=probe_interval_s,
            phase_label="stdp_validation",
        )
        self._phase_boundaries["stdp_validation_stop"] = datetime_now().isoformat()

        self._phase_boundaries["stdp_stop"] = datetime_now().isoformat()
        logger.info("Hebbian STDP experiment complete")

    def _run_probe_phase(
        self,
        stim_elec: int,
        resp_elec: int,
        duration_min: float,
        probe_interval_s: float,
        phase_label: str,
    ) -> None:
        duration_s = duration_min * 60.0
        phase_start = datetime_now()
        probe_count = 0

        while True:
            elapsed = (datetime_now() - phase_start).total_seconds()
            if elapsed >= duration_s:
                break

            stim_time = datetime_now()
            self._send_stim(
                electrode_idx=stim_elec,
                amplitude_ua=self.probe_amplitude_ua,
                duration_us=self.probe_duration_us,
                polarity=StimPolarity.PositiveFirst,
                trigger_key=2,
                phase=phase_label,
            )
            probe_count += 1
            self._wait(0.05)

            remaining = duration_s - (datetime_now() - phase_start).total_seconds()
            sleep_time = min(probe_interval_s - 0.05, remaining)
            if sleep_time > 0:
                self._wait(sleep_time)

        logger.info("Probe phase '%s' complete: %d probes", phase_label, probe_count)

    def _run_learning_phase(
        self,
        stim_elec: int,
        resp_elec: int,
        duration_min: float,
        hebbian_delay_ms: float,
    ) -> None:
        duration_s = duration_min * 60.0
        phase_start = datetime_now()
        conditioning_interval_s = 1.0 / 0.5
        stim_count = 0

        cond_amplitude = min(self.conditioning_amplitude_ua, 4.0)
        cond_duration = min(self.conditioning_duration_us, 400.0)
        delay_s = hebbian_delay_ms / 1000.0

        while True:
            elapsed = (datetime_now() - phase_start).total_seconds()
            if elapsed >= duration_s:
                break

            self._send_stim(
                electrode_idx=stim_elec,
                amplitude_ua=cond_amplitude,
                duration_us=cond_duration,
                polarity=StimPolarity.PositiveFirst,
                trigger_key=3,
                phase="stdp_learning_pre",
            )
            self._wait(delay_s)

            self._send_stim(
                electrode_idx=resp_elec,
                amplitude_ua=cond_amplitude,
                duration_us=cond_duration,
                polarity=StimPolarity.PositiveFirst,
                trigger_key=4,
                phase="stdp_learning_post",
            )
            stim_count += 1
            self._wait(0.05)

            remaining = duration_s - (datetime_now() - phase_start).total_seconds()
            sleep_time = min(conditioning_interval_s - delay_s - 0.05, remaining)
            if sleep_time > 0:
                self._wait(sleep_time)

        logger.info("Learning phase complete: %d paired stimulations", stim_count)

    def _send_stim(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity = StimPolarity.PositiveFirst,
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
        self._wait(0.01)
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

    def _save_all(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> None:
        fs_name = getattr(self.experiment, "exp_name", "unknown")
        saver = DataSaver(self._output_dir, fs_name)

        saver.save_stimulation_log(self._stimulation_log)

        spike_df = self.database.get_spike_event(
            recording_start, recording_stop, fs_name
        )
        saver.save_spike_events(spike_df)

        trigger_df = self.database.get_all_triggers(
            recording_start, recording_stop
        )
        saver.save_triggers(trigger_df)

        ccg_serializable = []
        for ccg in self._ccg_results:
            ccg_serializable.append({
                "stim_electrode": ccg.stim_electrode,
                "resp_electrode": ccg.resp_electrode,
                "bin_edges_ms": ccg.bin_edges_ms,
                "counts": ccg.counts,
                "peak_latency_ms": ccg.peak_latency_ms,
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
            "ccg_results": ccg_serializable,
            "phase_boundaries": self._phase_boundaries,
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

        ccg_summary = []
        for ccg in self._ccg_results:
            ccg_summary.append({
                "stim_electrode": ccg.stim_electrode,
                "resp_electrode": ccg.resp_electrode,
                "peak_latency_ms": ccg.peak_latency_ms,
                "peak_count": ccg.peak_count,
            })

        scan_summary = []
        for r in self._scan_results:
            scan_summary.append({
                "electrode_from": r.electrode_from,
                "amplitude": r.amplitude,
                "duration": r.duration,
                "polarity": r.polarity,
                "hits": r.hits,
                "median_latency_ms": r.median_latency_ms,
            })

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "total_stimulations": len(self._stimulation_log),
            "scan_responsive_configs": len(self._scan_results),
            "responsive_pairs": len(self._responsive_pairs),
            "ccg_results": ccg_summary,
            "scan_summary": scan_summary,
            "phase_boundaries": self._phase_boundaries,
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
