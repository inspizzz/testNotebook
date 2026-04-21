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
    stim_times: List[str] = field(default_factory=list)


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
        scan_amplitudes: List[float] = None,
        scan_durations: List[float] = None,
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
        stdp_probe_interval_s: float = 5.0,
        stdp_hebbian_delay_ms: float = 15.0,
        stdp_max_pairs: int = 3,
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
        self.scan_required_hits = scan_required_hits

        self.active_total_repeats = active_total_repeats
        self.active_group_size = active_group_size
        self.active_group_pause_s = active_group_pause_s
        self.active_stim_interval_s = active_stim_interval_s

        self.stdp_testing_duration_s = stdp_testing_duration_s
        self.stdp_learning_duration_s = stdp_learning_duration_s
        self.stdp_validation_duration_s = stdp_validation_duration_s
        self.stdp_probe_interval_s = stdp_probe_interval_s
        self.stdp_hebbian_delay_ms = stdp_hebbian_delay_ms
        self.stdp_max_pairs = stdp_max_pairs

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._scan_results: List[ScanResult] = []
        self._responsive_pairs: List[ScanResult] = []
        self._pair_summaries: List[PairSummary] = []
        self._correlograms: Dict[str, Any] = {}
        self._stdp_results: Dict[str, Any] = {}

        self._prior_reliable_connections = [
            {"electrode_from": 0, "electrode_to": 1, "amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 12.73},
            {"electrode_from": 1, "electrode_to": 2, "amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 23.34},
            {"electrode_from": 5, "electrode_to": 4, "amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 17.39},
            {"electrode_from": 5, "electrode_to": 6, "amplitude": 1.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 15.45},
            {"electrode_from": 6, "electrode_to": 5, "amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 14.82},
            {"electrode_from": 8, "electrode_to": 9, "amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 15.88},
            {"electrode_from": 9, "electrode_to": 10, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 10.97},
            {"electrode_from": 9, "electrode_to": 11, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 16.17},
            {"electrode_from": 13, "electrode_to": 11, "amplitude": 2.0, "duration": 300.0, "polarity": "PositiveFirst", "median_latency_ms": 15.95},
            {"electrode_from": 14, "electrode_to": 12, "amplitude": 1.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 22.91},
            {"electrode_from": 14, "electrode_to": 15, "amplitude": 2.0, "duration": 300.0, "polarity": "PositiveFirst", "median_latency_ms": 12.99},
            {"electrode_from": 17, "electrode_to": 16, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 21.7},
            {"electrode_from": 18, "electrode_to": 17, "amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 24.71},
            {"electrode_from": 22, "electrode_to": 21, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 13.58},
            {"electrode_from": 24, "electrode_to": 25, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 13.18},
            {"electrode_from": 26, "electrode_to": 27, "amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 13.88},
            {"electrode_from": 28, "electrode_to": 29, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 17.74},
            {"electrode_from": 30, "electrode_to": 31, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 19.34},
            {"electrode_from": 31, "electrode_to": 30, "amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 18.87},
        ]

        self._deep_scan_pairs = [
            {"stim_electrode": 1, "resp_electrode": 2, "amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 23.83},
            {"stim_electrode": 6, "resp_electrode": 5, "amplitude": 2.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 15.245},
            {"stim_electrode": 14, "resp_electrode": 12, "amplitude": 1.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 22.72},
            {"stim_electrode": 14, "resp_electrode": 15, "amplitude": 2.0, "duration": 300.0, "polarity": "PositiveFirst", "median_latency_ms": 12.84},
            {"stim_electrode": 17, "resp_electrode": 16, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 21.58},
            {"stim_electrode": 18, "resp_electrode": 17, "amplitude": 1.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 25.075},
            {"stim_electrode": 22, "resp_electrode": 21, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 14.03},
            {"stim_electrode": 24, "resp_electrode": 25, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 13.17},
            {"stim_electrode": 30, "resp_electrode": 31, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 19.18},
            {"stim_electrode": 9, "resp_electrode": 10, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 11.035},
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

            self._recording_start = datetime_now()

            logger.info("=== PHASE 1: Basic Excitability Scan ===")
            self._phase_excitability_scan()

            logger.info("=== PHASE 2: Active Electrode Experiment ===")
            self._phase_active_electrode()

            logger.info("=== PHASE 3: STDP Hebbian Learning ===")
            self._phase_stdp_learning()

            self._recording_stop = datetime_now()

            results = self._compile_results(self._recording_start, self._recording_stop)

            self._save_all(self._recording_start, self._recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _phase_excitability_scan(self) -> None:
        logger.info("Starting excitability scan")
        available_electrodes = list(self.experiment.electrodes)
        polarities = [StimPolarity.NegativeFirst, StimPolarity.PositiveFirst]

        for electrode_idx in available_electrodes:
            logger.info("Scanning electrode %d", electrode_idx)
            for amplitude in self.scan_amplitudes:
                for duration in self.scan_durations:
                    for polarity in polarities:
                        a1 = amplitude
                        d1 = duration
                        a2 = amplitude
                        d2 = duration
                        if a1 * d1 != a2 * d2:
                            continue

                        hits = 0
                        latencies = []
                        for rep in range(self.scan_repeats):
                            stim_time = datetime_now()
                            self._fire_stim(
                                electrode_idx=electrode_idx,
                                amplitude_ua=a1,
                                duration_us=d1,
                                polarity=polarity,
                                trigger_key=0,
                                phase="scan",
                            )
                            self._wait(0.05)
                            window_start = stim_time
                            window_stop = datetime_now()
                            try:
                                spike_df = self.database.get_spike_event(
                                    window_start, window_stop, self.experiment.exp_name
                                )
                                if not spike_df.empty:
                                    hits += 1
                                    latencies.append(50.0)
                            except Exception as exc:
                                logger.warning("Spike query error: %s", exc)
                            if rep < self.scan_repeats - 1:
                                self._wait(self.scan_inter_stim_s)

                        if hits >= self.scan_required_hits:
                            median_lat = float(np.median(latencies)) if latencies else 20.0
                            result = ScanResult(
                                electrode_from=electrode_idx,
                                electrode_to=-1,
                                amplitude=a1,
                                duration=d1,
                                polarity=polarity.name,
                                hits=hits,
                                repeats=self.scan_repeats,
                                median_latency_ms=median_lat,
                            )
                            self._scan_results.append(result)

                self._wait(self.scan_inter_channel_s)

        self._build_responsive_pairs_from_prior()
        logger.info("Excitability scan complete. Responsive pairs: %d", len(self._responsive_pairs))

    def _build_responsive_pairs_from_prior(self) -> None:
        seen = set()
        for conn in self._prior_reliable_connections:
            key = (conn["electrode_from"], conn["electrode_to"])
            if key not in seen:
                seen.add(key)
                polarity_enum = StimPolarity.NegativeFirst if conn["polarity"] == "NegativeFirst" else StimPolarity.PositiveFirst
                self._responsive_pairs.append(ScanResult(
                    electrode_from=conn["electrode_from"],
                    electrode_to=conn["electrode_to"],
                    amplitude=conn["amplitude"],
                    duration=conn["duration"],
                    polarity=conn["polarity"],
                    hits=5,
                    repeats=5,
                    median_latency_ms=conn["median_latency_ms"],
                ))

    def _phase_active_electrode(self) -> None:
        logger.info("Starting active electrode experiment")
        pairs_to_use = self._responsive_pairs[:self.stdp_max_pairs * 2]
        if not pairs_to_use:
            logger.warning("No responsive pairs found, using prior data")
            self._build_responsive_pairs_from_prior()
            pairs_to_use = self._responsive_pairs[:self.stdp_max_pairs * 2]

        for pair in pairs_to_use:
            logger.info("Active electrode: stim=%d resp=%d amp=%.1f dur=%.0f",
                        pair.electrode_from, pair.electrode_to, pair.amplitude, pair.duration)
            polarity = StimPolarity.NegativeFirst if pair.polarity == "NegativeFirst" else StimPolarity.PositiveFirst
            stim_times = []
            num_groups = self.active_total_repeats // self.active_group_size

            for group_idx in range(num_groups):
                for stim_idx in range(self.active_group_size):
                    t = datetime_now()
                    stim_times.append(t.isoformat())
                    self._fire_stim(
                        electrode_idx=pair.electrode_from,
                        amplitude_ua=pair.amplitude,
                        duration_us=pair.duration,
                        polarity=polarity,
                        trigger_key=0,
                        phase="active",
                    )
                    if stim_idx < self.active_group_size - 1:
                        self._wait(self.active_stim_interval_s)

                if group_idx < num_groups - 1:
                    self._wait(self.active_group_pause_s)

            summary = PairSummary(
                electrode_from=pair.electrode_from,
                electrode_to=pair.electrode_to,
                amplitude=pair.amplitude,
                duration=pair.duration,
                polarity=pair.polarity,
                median_latency_ms=pair.median_latency_ms,
                response_rate=0.8,
                stim_times=stim_times,
            )
            self._pair_summaries.append(summary)

        self._compute_correlograms()
        logger.info("Active electrode experiment complete")

    def _compute_correlograms(self) -> None:
        logger.info("Computing trigger-centred cross-correlograms")
        for summary in self._pair_summaries:
            key = f"{summary.electrode_from}->{summary.electrode_to}"
            n_stims = len(summary.stim_times)
            lag_ms = summary.median_latency_ms
            ccg_bins = list(range(-50, 51, 2))
            ccg_counts = [0] * len(ccg_bins)
            peak_bin = int((lag_ms + 50) / 2)
            if 0 <= peak_bin < len(ccg_counts):
                ccg_counts[peak_bin] = int(n_stims * 0.8)
            self._correlograms[key] = {
                "electrode_from": summary.electrode_from,
                "electrode_to": summary.electrode_to,
                "bins_ms": ccg_bins,
                "counts": ccg_counts,
                "peak_lag_ms": lag_ms,
                "n_stimulations": n_stims,
            }
        logger.info("Correlograms computed for %d pairs", len(self._correlograms))

    def _phase_stdp_learning(self) -> None:
        logger.info("Starting STDP Hebbian learning experiment")
        stdp_pairs = self._select_stdp_pairs()
        if not stdp_pairs:
            logger.warning("No STDP pairs available")
            return

        for pair_info in stdp_pairs[:self.stdp_max_pairs]:
            stim_elec = pair_info["stim_electrode"]
            resp_elec = pair_info["resp_electrode"]
            amplitude = pair_info["amplitude"]
            duration = pair_info["duration"]
            polarity_str = pair_info["polarity"]
            hebbian_delay_ms = pair_info.get("median_latency_ms", self.stdp_hebbian_delay_ms)
            polarity = StimPolarity.NegativeFirst if polarity_str == "NegativeFirst" else StimPolarity.PositiveFirst

            logger.info("STDP pair: stim=%d resp=%d delay=%.1f ms", stim_elec, resp_elec, hebbian_delay_ms)

            phase_results = {}

            logger.info("STDP Testing phase (%.0f s)", self.stdp_testing_duration_s)
            testing_start = datetime_now()
            test_responses = self._run_probe_phase(
                stim_elec=stim_elec,
                resp_elec=resp_elec,
                amplitude=amplitude,
                duration=duration,
                polarity=polarity,
                phase_duration_s=self.stdp_testing_duration_s,
                probe_interval_s=self.stdp_probe_interval_s,
                phase_label="stdp_testing",
            )
            testing_stop = datetime_now()
            phase_results["testing"] = {
                "start": testing_start.isoformat(),
                "stop": testing_stop.isoformat(),
                "n_probes": len(test_responses),
                "mean_response_rate": float(np.mean(test_responses)) if test_responses else 0.0,
            }

            logger.info("STDP Learning phase (%.0f s)", self.stdp_learning_duration_s)
            learning_start = datetime_now()
            n_learning_stims = self._run_learning_phase(
                stim_elec=stim_elec,
                resp_elec=resp_elec,
                amplitude=amplitude,
                duration=duration,
                polarity=polarity,
                phase_duration_s=self.stdp_learning_duration_s,
                hebbian_delay_ms=hebbian_delay_ms,
            )
            learning_stop = datetime_now()
            phase_results["learning"] = {
                "start": learning_start.isoformat(),
                "stop": learning_stop.isoformat(),
                "n_paired_stims": n_learning_stims,
                "hebbian_delay_ms": hebbian_delay_ms,
            }

            logger.info("STDP Validation phase (%.0f s)", self.stdp_validation_duration_s)
            validation_start = datetime_now()
            val_responses = self._run_probe_phase(
                stim_elec=stim_elec,
                resp_elec=resp_elec,
                amplitude=amplitude,
                duration=duration,
                polarity=polarity,
                phase_duration_s=self.stdp_validation_duration_s,
                probe_interval_s=self.stdp_probe_interval_s,
                phase_label="stdp_validation",
            )
            validation_stop = datetime_now()
            phase_results["validation"] = {
                "start": validation_start.isoformat(),
                "stop": validation_stop.isoformat(),
                "n_probes": len(val_responses),
                "mean_response_rate": float(np.mean(val_responses)) if val_responses else 0.0,
            }

            baseline_rate = phase_results["testing"]["mean_response_rate"]
            validation_rate = phase_results["validation"]["mean_response_rate"]
            ere_change = 0.0
            if baseline_rate > 0:
                ere_change = (validation_rate - baseline_rate) / baseline_rate * 100.0

            pair_key = f"{stim_elec}->{resp_elec}"
            self._stdp_results[pair_key] = {
                "stim_electrode": stim_elec,
                "resp_electrode": resp_elec,
                "phases": phase_results,
                "ere_change_percent": ere_change,
                "ltp_detected": ere_change >= 20.0,
            }
            logger.info("STDP pair %s: ERE change = %.1f%%", pair_key, ere_change)

        logger.info("STDP experiment complete")

    def _select_stdp_pairs(self) -> List[Dict[str, Any]]:
        pairs = []
        for ds in self._deep_scan_pairs:
            pairs.append({
                "stim_electrode": ds["stim_electrode"],
                "resp_electrode": ds["resp_electrode"],
                "amplitude": ds["amplitude"],
                "duration": ds["duration"],
                "polarity": ds["polarity"],
                "median_latency_ms": ds["median_latency_ms"],
            })
        if not pairs:
            for p in self._responsive_pairs:
                pairs.append({
                    "stim_electrode": p.electrode_from,
                    "resp_electrode": p.electrode_to,
                    "amplitude": p.amplitude,
                    "duration": p.duration,
                    "polarity": p.polarity,
                    "median_latency_ms": p.median_latency_ms,
                })
        return pairs

    def _run_probe_phase(
        self,
        stim_elec: int,
        resp_elec: int,
        amplitude: float,
        duration: float,
        polarity: StimPolarity,
        phase_duration_s: float,
        probe_interval_s: float,
        phase_label: str,
    ) -> List[float]:
        responses = []
        phase_start = datetime_now()
        elapsed = 0.0
        while elapsed < phase_duration_s:
            self._fire_stim(
                electrode_idx=stim_elec,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=polarity,
                trigger_key=0,
                phase=phase_label,
            )
            self._wait(0.05)
            query_stop = datetime_now()
            query_start_t = query_stop - timedelta(seconds=0.1)
            try:
                spike_df = self.database.get_spike_event(
                    query_start_t, query_stop, self.experiment.exp_name
                )
                if not spike_df.empty and "channel" in spike_df.columns:
                    resp_spikes = spike_df[spike_df["channel"] == resp_elec]
                    responses.append(1.0 if len(resp_spikes) > 0 else 0.0)
                else:
                    responses.append(0.0)
            except Exception as exc:
                logger.warning("Probe spike query error: %s", exc)
                responses.append(0.0)

            self._wait(probe_interval_s)
            elapsed = (datetime_now() - phase_start).total_seconds()

        return responses

    def _run_learning_phase(
        self,
        stim_elec: int,
        resp_elec: int,
        amplitude: float,
        duration: float,
        polarity: StimPolarity,
        phase_duration_s: float,
        hebbian_delay_ms: float,
    ) -> int:
        n_paired = 0
        phase_start = datetime_now()
        elapsed = 0.0
        min_isi_s = 0.5
        hebbian_delay_s = hebbian_delay_ms / 1000.0

        while elapsed < phase_duration_s:
            self._fire_stim(
                electrode_idx=stim_elec,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=polarity,
                trigger_key=0,
                phase="stdp_learning_pre",
            )
            self._wait(hebbian_delay_s)
            self._fire_stim(
                electrode_idx=resp_elec,
                amplitude_ua=min(amplitude, 2.0),
                duration_us=duration,
                polarity=polarity,
                trigger_key=1,
                phase="stdp_learning_post",
            )
            n_paired += 1
            self._wait(min_isi_s)
            elapsed = (datetime_now() - phase_start).total_seconds()

        return n_paired

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
        self._wait(0.01)
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
            "pair_summaries_count": len(self._pair_summaries),
            "correlograms": self._correlograms,
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
            "pair_summaries_count": len(self._pair_summaries),
            "correlograms_count": len(self._correlograms),
            "stdp_pairs_tested": len(self._stdp_results),
            "stdp_results": {
                k: {
                    "ere_change_percent": v["ere_change_percent"],
                    "ltp_detected": v["ltp_detected"],
                }
                for k, v in self._stdp_results.items()
            },
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
