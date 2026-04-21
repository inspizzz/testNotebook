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
    peak_lag_ms: float
    peak_count: int
    bin_edges_ms: List[float]
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
        ccg_window_ms: float = 50.0,
        ccg_bin_ms: float = 4.0,
        stdp_testing_duration_s: float = 1200.0,
        stdp_learning_duration_s: float = 3000.0,
        stdp_validation_duration_s: float = 1200.0,
        stdp_probe_amplitude_ua: float = 1.0,
        stdp_probe_duration_us: float = 300.0,
        stdp_probe_interval_s: float = 10.0,
        stdp_conditioning_amplitude_ua: float = 2.0,
        stdp_conditioning_duration_us: float = 200.0,
        hebbian_delay_ms: float = 20.0,
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

        self.active_stim_hz = active_stim_hz
        self.active_group_size = active_group_size
        self.active_inter_group_s = active_inter_group_s
        self.active_total_repeats = active_total_repeats

        self.ccg_window_ms = ccg_window_ms
        self.ccg_bin_ms = ccg_bin_ms

        self.stdp_testing_duration_s = stdp_testing_duration_s
        self.stdp_learning_duration_s = stdp_learning_duration_s
        self.stdp_validation_duration_s = stdp_validation_duration_s
        self.stdp_probe_amplitude_ua = stdp_probe_amplitude_ua
        self.stdp_probe_duration_us = stdp_probe_duration_us
        self.stdp_probe_interval_s = stdp_probe_interval_s
        self.stdp_conditioning_amplitude_ua = stdp_conditioning_amplitude_ua
        self.stdp_conditioning_duration_us = stdp_conditioning_duration_us
        self.hebbian_delay_ms = hebbian_delay_ms
        self.max_electrode_pairs = max_electrode_pairs

        self.neuroplatform_experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._scan_results: List[ScanResult] = []
        self._responsive_pairs: List[Dict[str, Any]] = []
        self._active_stim_times: Dict[str, List[str]] = {}
        self._ccg_results: List[CrossCorrelogramResult] = []
        self._stdp_results: Dict[str, Any] = {}

        self._prior_reliable_connections = [
            {"electrode_from": 5, "electrode_to": 4, "amplitude": 1.0, "duration": 300.0, "polarity": "PositiveFirst", "median_latency_ms": 17.66},
            {"electrode_from": 14, "electrode_to": 15, "amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 13.2},
            {"electrode_from": 18, "electrode_to": 17, "amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 24.71},
            {"electrode_from": 6, "electrode_to": 5, "amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 14.82},
            {"electrode_from": 9, "electrode_to": 10, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 10.97},
        ]

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

            logger.info("=== Phase 1: Basic Excitability Scan ===")
            self._phase_excitability_scan()

            logger.info("=== Phase 2: Active Electrode Experiment ===")
            self._phase_active_electrode_experiment()

            logger.info("=== Phase 3: Two-Electrode Hebbian Learning (STDP) ===")
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
        electrodes = list(self.neuroplatform_experiment.electrodes)
        polarities = [StimPolarity.NegativeFirst, StimPolarity.PositiveFirst]
        polarity_names = {StimPolarity.NegativeFirst: "NegativeFirst", StimPolarity.PositiveFirst: "PositiveFirst"}

        for ch_idx, electrode in enumerate(electrodes):
            logger.info("Scanning electrode %d (%d/%d)", electrode, ch_idx + 1, len(electrodes))
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
                            window_start = t_before
                            window_stop = t_after + timedelta(milliseconds=self.ccg_window_ms)
                            self._wait(max(0.0, self.ccg_window_ms / 1000.0 - 0.05))
                            spike_df = self.database.get_spike_event(
                                window_start,
                                window_stop,
                                self.neuroplatform_experiment.exp_name,
                            )
                            if not spike_df.empty:
                                responding = spike_df[spike_df["channel"] != electrode]
                                if not responding.empty:
                                    hits += 1
                                    if "Time" in responding.columns:
                                        for _, row in responding.iterrows():
                                            try:
                                                t_spike = pd.to_datetime(row["Time"], utc=True)
                                                t_stim = pd.to_datetime(t_before, utc=True)
                                                lat = (t_spike - t_stim).total_seconds() * 1000.0
                                                if 0 < lat < self.ccg_window_ms:
                                                    latencies.append(lat)
                                            except Exception:
                                                pass
                            if rep < self.scan_repeats - 1:
                                self._wait(self.scan_inter_stim_s)

                        if hits >= 3:
                            median_lat = float(np.median(latencies)) if latencies else 0.0
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
                                "Electrode %d responsive: amp=%.1f dur=%.0f pol=%s hits=%d",
                                electrode, amplitude, duration, polarity_names[polarity], hits,
                            )

            self._wait(self.scan_inter_channel_s)

        self._identify_responsive_pairs()
        logger.info("Excitability scan complete. Responsive pairs: %d", len(self._responsive_pairs))

    def _identify_responsive_pairs(self) -> None:
        prior_pairs = []
        for conn in self._prior_reliable_connections[:self.max_electrode_pairs]:
            prior_pairs.append({
                "electrode_from": conn["electrode_from"],
                "electrode_to": conn["electrode_to"],
                "amplitude": conn["amplitude"],
                "duration": conn["duration"],
                "polarity": conn["polarity"],
                "median_latency_ms": conn["median_latency_ms"],
                "hits": 5,
                "source": "prior_scan",
            })

        responsive_electrodes = set()
        for r in self._scan_results:
            if r.hits >= 3:
                responsive_electrodes.add(r.electrode_from)

        scan_pairs = []
        for conn in self._prior_reliable_connections:
            ef = conn["electrode_from"]
            et = conn["electrode_to"]
            if ef in responsive_electrodes or et in responsive_electrodes:
                scan_pairs.append({
                    "electrode_from": ef,
                    "electrode_to": et,
                    "amplitude": conn["amplitude"],
                    "duration": conn["duration"],
                    "polarity": conn["polarity"],
                    "median_latency_ms": conn["median_latency_ms"],
                    "hits": 5,
                    "source": "scan_confirmed",
                })

        combined = prior_pairs if prior_pairs else scan_pairs
        seen = set()
        for p in combined:
            key = (p["electrode_from"], p["electrode_to"])
            if key not in seen:
                seen.add(key)
                self._responsive_pairs.append(p)
            if len(self._responsive_pairs) >= self.max_electrode_pairs:
                break

        if not self._responsive_pairs:
            for conn in self._prior_reliable_connections[:self.max_electrode_pairs]:
                self._responsive_pairs.append({
                    "electrode_from": conn["electrode_from"],
                    "electrode_to": conn["electrode_to"],
                    "amplitude": conn["amplitude"],
                    "duration": conn["duration"],
                    "polarity": conn["polarity"],
                    "median_latency_ms": conn["median_latency_ms"],
                    "hits": 5,
                    "source": "prior_fallback",
                })

        logger.info("Identified %d responsive pairs for active experiment", len(self._responsive_pairs))

    def _phase_active_electrode_experiment(self) -> None:
        logger.info("Starting active electrode experiment")
        inter_stim_s = 1.0 / self.active_stim_hz

        for pair in self._responsive_pairs:
            ef = pair["electrode_from"]
            et = pair["electrode_to"]
            amplitude = min(pair["amplitude"], 4.0)
            duration = min(pair["duration"], 400.0)
            polarity_str = pair["polarity"]
            polarity = StimPolarity.NegativeFirst if polarity_str == "NegativeFirst" else StimPolarity.PositiveFirst

            pair_key = f"{ef}_to_{et}"
            self._active_stim_times[pair_key] = []

            logger.info("Active experiment: pair %d->%d, amp=%.1f, dur=%.0f", ef, et, amplitude, duration)

            num_groups = self.active_total_repeats // self.active_group_size
            stim_count = 0

            for group_idx in range(num_groups):
                logger.info("  Group %d/%d for pair %d->%d", group_idx + 1, num_groups, ef, et)
                for stim_in_group in range(self.active_group_size):
                    t_stim = datetime_now()
                    self._send_single_stim(
                        electrode_idx=ef,
                        amplitude_ua=amplitude,
                        duration_us=duration,
                        polarity=polarity,
                        trigger_key=0,
                        phase="active",
                    )
                    self._active_stim_times[pair_key].append(t_stim.isoformat())
                    stim_count += 1
                    if stim_in_group < self.active_group_size - 1:
                        self._wait(inter_stim_s)

                if group_idx < num_groups - 1:
                    self._wait(self.active_inter_group_s)

            logger.info("  Completed %d stimulations for pair %d->%d", stim_count, ef, et)

        logger.info("Computing cross-correlograms")
        self._compute_cross_correlograms()
        logger.info("Active electrode experiment complete")

    def _compute_cross_correlograms(self) -> None:
        if not self._active_stim_times:
            return

        all_times = []
        for times_list in self._active_stim_times.values():
            all_times.extend(times_list)

        if not all_times:
            return

        try:
            t_start = datetime.fromisoformat(min(all_times)).replace(tzinfo=timezone.utc)
            t_stop = datetime_now()

            spike_df = self.database.get_spike_event(
                t_start,
                t_stop,
                self.neuroplatform_experiment.exp_name,
            )
        except Exception as exc:
            logger.warning("Failed to fetch spikes for CCG: %s", exc)
            return

        if spike_df.empty:
            logger.info("No spikes found for CCG computation")
            return

        for pair in self._responsive_pairs:
            ef = pair["electrode_from"]
            et = pair["electrode_to"]
            pair_key = f"{ef}_to_{et}"

            stim_times_iso = self._active_stim_times.get(pair_key, [])
            if not stim_times_iso:
                continue

            stim_times_dt = []
            for t_iso in stim_times_iso:
                try:
                    stim_times_dt.append(datetime.fromisoformat(t_iso).replace(tzinfo=timezone.utc))
                except Exception:
                    pass

            if not stim_times_dt:
                continue

            resp_spikes = spike_df[spike_df["channel"] == et]
            if resp_spikes.empty:
                continue

            n_bins = int(self.ccg_window_ms / self.ccg_bin_ms)
            bin_edges = [i * self.ccg_bin_ms for i in range(n_bins + 1)]
            counts = [0] * n_bins

            for _, spike_row in resp_spikes.iterrows():
                try:
                    t_spike = pd.to_datetime(spike_row["Time"], utc=True).to_pydatetime()
                    for t_stim in stim_times_dt:
                        lag_ms = (t_spike - t_stim).total_seconds() * 1000.0
                        if 0.0 <= lag_ms < self.ccg_window_ms:
                            bin_idx = int(lag_ms / self.ccg_bin_ms)
                            if 0 <= bin_idx < n_bins:
                                counts[bin_idx] += 1
                except Exception:
                    pass

            if sum(counts) > 0:
                peak_bin = int(np.argmax(counts))
                peak_lag = bin_edges[peak_bin] + self.ccg_bin_ms / 2.0
                ccg = CrossCorrelogramResult(
                    electrode_from=ef,
                    electrode_to=et,
                    peak_lag_ms=peak_lag,
                    peak_count=counts[peak_bin],
                    bin_edges_ms=bin_edges,
                    counts=counts,
                )
                self._ccg_results.append(ccg)
                logger.info("CCG pair %d->%d: peak at %.1f ms (count=%d)", ef, et, peak_lag, counts[peak_bin])

    def _phase_stdp_experiment(self) -> None:
        logger.info("Starting STDP experiment")

        if not self._responsive_pairs:
            logger.warning("No responsive pairs for STDP experiment")
            return

        stdp_pair = self._responsive_pairs[0]
        ef = stdp_pair["electrode_from"]
        et = stdp_pair["electrode_to"]

        ccg_delay_ms = self.hebbian_delay_ms
        for ccg in self._ccg_results:
            if ccg.electrode_from == ef and ccg.electrode_to == et:
                ccg_delay_ms = ccg.peak_lag_ms
                break

        logger.info("STDP pair: %d->%d, Hebbian delay=%.1f ms", ef, et, ccg_delay_ms)

        probe_amp = min(self.stdp_probe_amplitude_ua, 4.0)
        probe_dur = min(self.stdp_probe_duration_us, 400.0)
        cond_amp = min(self.stdp_conditioning_amplitude_ua, 4.0)
        cond_dur = min(self.stdp_conditioning_duration_us, 400.0)

        self._stdp_results = {
            "electrode_from": ef,
            "electrode_to": et,
            "hebbian_delay_ms": ccg_delay_ms,
            "probe_amplitude_ua": probe_amp,
            "probe_duration_us": probe_dur,
            "conditioning_amplitude_ua": cond_amp,
            "conditioning_duration_us": cond_dur,
            "testing_phase": {},
            "learning_phase": {},
            "validation_phase": {},
        }

        logger.info("STDP Phase 1: Testing (%.0f s)", self.stdp_testing_duration_s)
        testing_responses = self._stdp_probe_phase(
            ef, et, probe_amp, probe_dur,
            duration_s=self.stdp_testing_duration_s,
            phase_name="testing",
        )
        self._stdp_results["testing_phase"] = {
            "n_probes": len(testing_responses),
            "response_times": testing_responses,
        }

        logger.info("STDP Phase 2: Learning (%.0f s)", self.stdp_learning_duration_s)
        learning_stats = self._stdp_learning_phase(
            ef, et,
            probe_amp=probe_amp,
            probe_dur=probe_dur,
            cond_amp=cond_amp,
            cond_dur=cond_dur,
            hebbian_delay_ms=ccg_delay_ms,
            duration_s=self.stdp_learning_duration_s,
        )
        self._stdp_results["learning_phase"] = learning_stats

        logger.info("STDP Phase 3: Validation (%.0f s)", self.stdp_validation_duration_s)
        validation_responses = self._stdp_probe_phase(
            ef, et, probe_amp, probe_dur,
            duration_s=self.stdp_validation_duration_s,
            phase_name="validation",
        )
        self._stdp_results["validation_phase"] = {
            "n_probes": len(validation_responses),
            "response_times": validation_responses,
        }

        self._analyze_stdp_results()
        logger.info("STDP experiment complete")

    def _stdp_probe_phase(
        self,
        electrode_from: int,
        electrode_to: int,
        probe_amp: float,
        probe_dur: float,
        duration_s: float,
        phase_name: str,
    ) -> List[str]:
        polarity = StimPolarity.NegativeFirst
        response_times = []
        phase_start = datetime_now()
        probe_count = 0

        while True:
            elapsed = (datetime_now() - phase_start).total_seconds()
            if elapsed >= duration_s:
                break

            t_probe = datetime_now()
            self._send_single_stim(
                electrode_idx=electrode_from,
                amplitude_ua=probe_amp,
                duration_us=probe_dur,
                polarity=polarity,
                trigger_key=0,
                phase=f"stdp_{phase_name}_probe",
            )
            probe_count += 1

            self._wait(0.05)
            window_stop = datetime_now() + timedelta(milliseconds=self.ccg_window_ms)
            self._wait(self.ccg_window_ms / 1000.0)

            try:
                spike_df = self.database.get_spike_event(
                    t_probe,
                    window_stop,
                    self.neuroplatform_experiment.exp_name,
                )
                if not spike_df.empty:
                    resp = spike_df[spike_df["channel"] == electrode_to]
                    if not resp.empty and "Time" in resp.columns:
                        response_times.append(str(resp.iloc[0]["Time"]))
            except Exception as exc:
                logger.warning("Probe spike query failed: %s", exc)

            remaining = duration_s - (datetime_now() - phase_start).total_seconds()
            if remaining <= 0:
                break
            self._wait(min(self.stdp_probe_interval_s, remaining))

        logger.info("  %s phase: %d probes, %d responses", phase_name, probe_count, len(response_times))
        return response_times

    def _stdp_learning_phase(
        self,
        electrode_from: int,
        electrode_to: int,
        probe_amp: float,
        probe_dur: float,
        cond_amp: float,
        cond_dur: float,
        hebbian_delay_ms: float,
        duration_s: float,
    ) -> Dict[str, Any]:
        polarity_probe = StimPolarity.NegativeFirst
        polarity_cond = StimPolarity.NegativeFirst

        phase_start = datetime_now()
        n_paired = 0
        n_probes = 0
        probe_responses = []
        refractory_s = 0.5

        while True:
            elapsed = (datetime_now() - phase_start).total_seconds()
            if elapsed >= duration_s:
                break

            t_probe = datetime_now()
            self._send_single_stim(
                electrode_idx=electrode_from,
                amplitude_ua=probe_amp,
                duration_us=probe_dur,
                polarity=polarity_probe,
                trigger_key=0,
                phase="stdp_learning_probe",
            )
            n_probes += 1

            delay_s = hebbian_delay_ms / 1000.0
            self._wait(delay_s)

            self._send_single_stim(
                electrode_idx=electrode_to,
                amplitude_ua=cond_amp,
                duration_us=cond_dur,
                polarity=polarity_cond,
                trigger_key=1,
                phase="stdp_learning_conditioning",
            )
            n_paired += 1

            self._wait(0.05)
            window_stop = datetime_now() + timedelta(milliseconds=self.ccg_window_ms)
            self._wait(self.ccg_window_ms / 1000.0)

            try:
                spike_df = self.database.get_spike_event(
                    t_probe,
                    window_stop,
                    self.neuroplatform_experiment.exp_name,
                )
                if not spike_df.empty:
                    resp = spike_df[spike_df["channel"] == electrode_to]
                    if not resp.empty and "Time" in resp.columns:
                        probe_responses.append(str(resp.iloc[0]["Time"]))
            except Exception as exc:
                logger.warning("Learning phase spike query failed: %s", exc)

            remaining = duration_s - (datetime_now() - phase_start).total_seconds()
            if remaining <= 0:
                break
            self._wait(min(self.stdp_probe_interval_s, remaining))

        logger.info("  Learning phase: %d paired stimulations, %d probe responses", n_paired, len(probe_responses))
        return {
            "n_paired_stimulations": n_paired,
            "n_probes": n_probes,
            "n_probe_responses": len(probe_responses),
            "probe_response_times": probe_responses,
        }

    def _analyze_stdp_results(self) -> None:
        testing = self._stdp_results.get("testing_phase", {})
        validation = self._stdp_results.get("validation_phase", {})

        n_test_probes = testing.get("n_probes", 0)
        n_test_resp = len(testing.get("response_times", []))
        n_val_probes = validation.get("n_probes", 0)
        n_val_resp = len(validation.get("response_times", []))

        test_rate = n_test_resp / n_test_probes if n_test_probes > 0 else 0.0
        val_rate = n_val_resp / n_val_probes if n_val_probes > 0 else 0.0
        delta_rate = val_rate - test_rate

        self._stdp_results["analysis"] = {
            "testing_response_rate": test_rate,
            "validation_response_rate": val_rate,
            "delta_response_rate": delta_rate,
            "potentiation_detected": delta_rate > 0.1,
        }
        logger.info(
            "STDP analysis: test_rate=%.3f val_rate=%.3f delta=%.3f potentiation=%s",
            test_rate, val_rate, delta_rate,
            self._stdp_results["analysis"]["potentiation_detected"],
        )

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
        self._wait(0.02)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

        polarity_name = "NegativeFirst" if polarity == StimPolarity.NegativeFirst else "PositiveFirst"
        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            polarity=polarity_name,
            timestamp_utc=datetime_now().isoformat(),
            trigger_key=trigger_key,
            phase=phase,
        ))

    def _save_all(self, recording_start: datetime, recording_stop: datetime) -> None:
        fs_name = getattr(self.neuroplatform_experiment, "exp_name", "unknown")
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
            "ccg_results_count": len(self._ccg_results),
            "stdp_results": self._stdp_results,
            "responsive_pairs": self._responsive_pairs,
            "ccg_results": [asdict(c) for c in self._ccg_results],
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
        fs_name = getattr(self.neuroplatform_experiment, "exp_name", "unknown")

        scan_summary = defaultdict(int)
        for r in self._scan_results:
            scan_summary["total_responsive_conditions"] += 1
            if r.hits == 5:
                scan_summary["perfect_response_conditions"] += 1

        stdp_analysis = self._stdp_results.get("analysis", {})

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": fs_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "phase1_scan": {
                "total_stimulations": sum(1 for s in self._stimulation_log if s.phase == "scan"),
                "responsive_conditions": scan_summary["total_responsive_conditions"],
                "perfect_response_conditions": scan_summary["perfect_response_conditions"],
                "responsive_pairs_identified": len(self._responsive_pairs),
            },
            "phase2_active": {
                "total_stimulations": sum(1 for s in self._stimulation_log if s.phase == "active"),
                "pairs_tested": len(self._active_stim_times),
                "ccg_results": [
                    {
                        "electrode_from": c.electrode_from,
                        "electrode_to": c.electrode_to,
                        "peak_lag_ms": c.peak_lag_ms,
                        "peak_count": c.peak_count,
                    }
                    for c in self._ccg_results
                ],
            },
            "phase3_stdp": {
                "electrode_from": self._stdp_results.get("electrode_from"),
                "electrode_to": self._stdp_results.get("electrode_to"),
                "hebbian_delay_ms": self._stdp_results.get("hebbian_delay_ms"),
                "testing_response_rate": stdp_analysis.get("testing_response_rate"),
                "validation_response_rate": stdp_analysis.get("validation_response_rate"),
                "delta_response_rate": stdp_analysis.get("delta_response_rate"),
                "potentiation_detected": stdp_analysis.get("potentiation_detected"),
            },
            "total_stimulations": len(self._stimulation_log),
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
