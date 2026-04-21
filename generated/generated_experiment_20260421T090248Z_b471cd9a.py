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
    phase: str
    timestamp_utc: str
    trigger_key: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialResult:
    phase: str
    electrode_from: int
    electrode_to: int
    trial_index: int
    stim_time_utc: str
    spike_count: int
    latency_ms: Optional[float]


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
        inter_stim_s: float = 1.0,
        inter_channel_s: float = 5.0,
        active_total_repeats: int = 100,
        active_group_size: int = 10,
        active_group_pause_s: float = 5.0,
        stdp_testing_duration_s: float = 1200.0,
        stdp_learning_duration_s: float = 3000.0,
        stdp_validation_duration_s: float = 1200.0,
        stdp_probe_interval_s: float = 10.0,
        hebbian_delay_ms: float = 20.0,
        stim_amplitude_ua: float = 2.0,
        stim_duration_us: float = 200.0,
        max_electrode_pairs: int = 5,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.scan_amplitudes = scan_amplitudes if scan_amplitudes is not None else [1.0, 2.0, 3.0]
        self.scan_durations = scan_durations if scan_durations is not None else [100.0, 200.0, 300.0, 400.0]
        self.scan_repeats = scan_repeats
        self.inter_stim_s = inter_stim_s
        self.inter_channel_s = inter_channel_s
        self.active_total_repeats = active_total_repeats
        self.active_group_size = active_group_size
        self.active_group_pause_s = active_group_pause_s
        self.stdp_testing_duration_s = stdp_testing_duration_s
        self.stdp_learning_duration_s = stdp_learning_duration_s
        self.stdp_validation_duration_s = stdp_validation_duration_s
        self.stdp_probe_interval_s = stdp_probe_interval_s
        self.hebbian_delay_ms = hebbian_delay_ms
        self.stim_amplitude_ua = min(stim_amplitude_ua, 4.0)
        self.stim_duration_us = min(stim_duration_us, 400.0)
        self.max_electrode_pairs = max_electrode_pairs

        self.np_experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._trial_results: List[TrialResult] = []

        self._scan_responsive_pairs: List[Dict[str, Any]] = []
        self._active_stim_times: Dict[Tuple[int, int], List[str]] = defaultdict(list)
        self._cross_correlograms: Dict[str, Any] = {}
        self._stdp_results: Dict[str, Any] = {}

        self._known_pairs = [
            {"electrode_from": 0, "electrode_to": 1, "amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 12.73},
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
            self._phase_stdp_hebbian()

            recording_stop = datetime_now()

            self._save_all(recording_start, recording_stop)

            results = self._compile_results(recording_start, recording_stop)
            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _make_stim_param(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity,
        trigger_key: int = 0,
    ) -> StimParam:
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
        return stim

    def _fire_trigger(self, trigger_key: int = 0) -> None:
        pattern = np.zeros(16, dtype=np.uint8)
        pattern[trigger_key] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.02)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

    def _stimulate(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity,
        trigger_key: int = 0,
        phase: str = "scan",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        stim = self._make_stim_param(electrode_idx, amplitude_ua, duration_us, polarity, trigger_key)
        self.intan.send_stimparam([stim])
        ts = datetime_now().isoformat()
        self._fire_trigger(trigger_key)
        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            polarity=polarity.name,
            phase=phase,
            timestamp_utc=ts,
            trigger_key=trigger_key,
            extra=extra or {},
        ))

    def _query_spikes_window(
        self,
        start: datetime,
        stop: datetime,
        fs_name: str,
    ) -> pd.DataFrame:
        try:
            return self.database.get_spike_event(start, stop, fs_name)
        except Exception as exc:
            logger.warning("Spike query failed: %s", exc)
            return pd.DataFrame()

    def _phase_excitability_scan(self) -> None:
        logger.info("Starting excitability scan")
        fs_name = self.np_experiment.exp_name
        available_electrodes = list(self.np_experiment.electrodes)

        polarities = [StimPolarity.NegativeFirst, StimPolarity.PositiveFirst]
        responsive_pairs: Dict[Tuple[int, int], List[float]] = defaultdict(list)

        for elec_idx, electrode in enumerate(available_electrodes):
            logger.info("Scanning electrode %d (%d/%d)", electrode, elec_idx + 1, len(available_electrodes))
            for amplitude in self.scan_amplitudes:
                for duration in self.scan_durations:
                    for polarity in polarities:
                        hit_count = 0
                        latencies = []
                        for rep in range(self.scan_repeats):
                            stim_time = datetime_now()
                            self._stimulate(
                                electrode_idx=electrode,
                                amplitude_ua=amplitude,
                                duration_us=duration,
                                polarity=polarity,
                                trigger_key=0,
                                phase="scan",
                                extra={
                                    "rep": rep,
                                    "amplitude": amplitude,
                                    "duration": duration,
                                    "polarity": polarity.name,
                                },
                            )
                            self._wait(0.05)
                            window_start = stim_time
                            window_stop = datetime_now()
                            spike_df = self._query_spikes_window(window_start, window_stop, fs_name)
                            if not spike_df.empty:
                                other_electrodes = [e for e in available_electrodes if e != electrode]
                                for other_elec in other_electrodes:
                                    ch_col = self._get_channel_col(spike_df)
                                    if ch_col:
                                        elec_spikes = spike_df[spike_df[ch_col] == other_elec]
                                        if not elec_spikes.empty:
                                            hit_count += 1
                                            if "Time" in elec_spikes.columns:
                                                spike_time = pd.to_datetime(elec_spikes["Time"].iloc[0])
                                                lat_ms = (spike_time.timestamp() - stim_time.timestamp()) * 1000.0
                                                if 1.0 < lat_ms < 100.0:
                                                    latencies.append(lat_ms)
                                            break
                            self._wait(self.inter_stim_s)

                        if hit_count >= 3:
                            for other_elec in available_electrodes:
                                if other_elec != electrode:
                                    key = (electrode, other_elec)
                                    median_lat = float(np.median(latencies)) if latencies else 20.0
                                    responsive_pairs[key].append(median_lat)

            self._wait(self.inter_channel_s)

        for (ef, et), lats in responsive_pairs.items():
            self._scan_responsive_pairs.append({
                "electrode_from": ef,
                "electrode_to": et,
                "median_latency_ms": float(np.median(lats)),
                "hit_count": len(lats),
            })

        if not self._scan_responsive_pairs:
            logger.info("No responsive pairs found in scan; using known pairs from parameter scan")
            for p in self._known_pairs[:self.max_electrode_pairs]:
                self._scan_responsive_pairs.append({
                    "electrode_from": p["electrode_from"],
                    "electrode_to": p["electrode_to"],
                    "median_latency_ms": p["median_latency_ms"],
                    "hit_count": 5,
                })

        logger.info("Excitability scan complete. Responsive pairs: %d", len(self._scan_responsive_pairs))

    def _get_channel_col(self, df: pd.DataFrame) -> Optional[str]:
        for col in ["channel", "index", "electrode"]:
            if col in df.columns:
                return col
        return None

    def _phase_active_electrode(self) -> None:
        logger.info("Starting active electrode experiment")
        fs_name = self.np_experiment.exp_name

        pairs_to_use = self._scan_responsive_pairs[:self.max_electrode_pairs]
        if not pairs_to_use:
            pairs_to_use = [
                {"electrode_from": p["electrode_from"], "electrode_to": p["electrode_to"],
                 "median_latency_ms": p["median_latency_ms"]}
                for p in self._known_pairs[:self.max_electrode_pairs]
            ]

        for pair in pairs_to_use:
            ef = pair["electrode_from"]
            et = pair["electrode_to"]
            logger.info("Active electrode stimulation: %d -> %d", ef, et)

            known = self._get_known_pair_params(ef, et)
            amplitude = known["amplitude"] if known else self.stim_amplitude_ua
            duration = known["duration"] if known else self.stim_duration_us
            polarity_str = known["polarity"] if known else "NegativeFirst"
            polarity = StimPolarity.NegativeFirst if polarity_str == "NegativeFirst" else StimPolarity.PositiveFirst

            stim_times = []
            num_groups = self.active_total_repeats // self.active_group_size

            for group_idx in range(num_groups):
                for stim_idx in range(self.active_group_size):
                    ts = datetime_now().isoformat()
                    self._stimulate(
                        electrode_idx=ef,
                        amplitude_ua=amplitude,
                        duration_us=duration,
                        polarity=polarity,
                        trigger_key=0,
                        phase="active",
                        extra={"pair": f"{ef}->{et}", "group": group_idx, "stim_in_group": stim_idx},
                    )
                    stim_times.append(ts)
                    self._wait(1.0)

                if group_idx < num_groups - 1:
                    self._wait(self.active_group_pause_s)

            self._active_stim_times[(ef, et)] = stim_times
            logger.info("Completed %d stimulations for pair %d->%d", len(stim_times), ef, et)

        logger.info("Computing cross-correlograms")
        self._compute_cross_correlograms(pairs_to_use, fs_name)

    def _get_known_pair_params(self, ef: int, et: int) -> Optional[Dict[str, Any]]:
        for p in self._known_pairs:
            if p["electrode_from"] == ef and p["electrode_to"] == et:
                return p
        return None

    def _compute_cross_correlograms(self, pairs: List[Dict], fs_name: str) -> None:
        for pair in pairs:
            ef = pair["electrode_from"]
            et = pair["electrode_to"]
            stim_times = self._active_stim_times.get((ef, et), [])
            if not stim_times:
                continue

            try:
                t_start = datetime.fromisoformat(stim_times[0])
                t_stop = datetime.fromisoformat(stim_times[-1]) + timedelta(seconds=2)
                spike_df = self._query_spikes_window(t_start, t_stop, fs_name)
            except Exception as exc:
                logger.warning("Failed to fetch spikes for correlogram %d->%d: %s", ef, et, exc)
                spike_df = pd.DataFrame()

            if spike_df.empty:
                self._cross_correlograms[f"{ef}->{et}"] = {
                    "bins": [],
                    "counts": [],
                    "peak_latency_ms": pair.get("median_latency_ms", 20.0),
                }
                continue

            ch_col = self._get_channel_col(spike_df)
            if ch_col is None:
                self._cross_correlograms[f"{ef}->{et}"] = {
                    "bins": [],
                    "counts": [],
                    "peak_latency_ms": pair.get("median_latency_ms", 20.0),
                }
                continue

            resp_spikes = spike_df[spike_df[ch_col] == et]
            if resp_spikes.empty or "Time" not in resp_spikes.columns:
                self._cross_correlograms[f"{ef}->{et}"] = {
                    "bins": [],
                    "counts": [],
                    "peak_latency_ms": pair.get("median_latency_ms", 20.0),
                }
                continue

            resp_times_s = [pd.to_datetime(t).timestamp() for t in resp_spikes["Time"].values]
            stim_times_s = []
            for ts_str in stim_times:
                try:
                    stim_times_s.append(datetime.fromisoformat(ts_str).timestamp())
                except Exception:
                    pass

            bin_edges = [i * 0.001 for i in range(101)]
            counts = [0] * 100

            for st in stim_times_s:
                for rt in resp_times_s:
                    diff_ms = (rt - st) * 1000.0
                    if 0.0 <= diff_ms < 100.0:
                        bin_idx = int(diff_ms)
                        if 0 <= bin_idx < 100:
                            counts[bin_idx] += 1

            peak_bin = int(np.argmax(counts)) if any(c > 0 for c in counts) else 20
            peak_latency_ms = float(peak_bin) + 0.5

            self._cross_correlograms[f"{ef}->{et}"] = {
                "bins": bin_edges,
                "counts": counts,
                "peak_latency_ms": peak_latency_ms,
            }
            logger.info("Correlogram %d->%d: peak latency %.1f ms", ef, et, peak_latency_ms)

    def _phase_stdp_hebbian(self) -> None:
        logger.info("Starting STDP Hebbian learning experiment")

        pairs_to_use = self._scan_responsive_pairs[:self.max_electrode_pairs]
        if not pairs_to_use:
            pairs_to_use = [
                {"electrode_from": p["electrode_from"], "electrode_to": p["electrode_to"],
                 "median_latency_ms": p["median_latency_ms"]}
                for p in self._known_pairs[:self.max_electrode_pairs]
            ]

        if not pairs_to_use:
            logger.warning("No electrode pairs available for STDP experiment")
            return

        primary_pair = pairs_to_use[0]
        ef = primary_pair["electrode_from"]
        et = primary_pair["electrode_to"]

        corr_key = f"{ef}->{et}"
        if corr_key in self._cross_correlograms:
            hebbian_delay = self._cross_correlograms[corr_key]["peak_latency_ms"]
        else:
            hebbian_delay = primary_pair.get("median_latency_ms", self.hebbian_delay_ms)

        hebbian_delay = max(10.0, min(30.0, hebbian_delay))
        logger.info("STDP pair: %d->%d, Hebbian delay: %.1f ms", ef, et, hebbian_delay)

        known = self._get_known_pair_params(ef, et)
        probe_amplitude = known["amplitude"] if known else self.stim_amplitude_ua
        probe_duration = known["duration"] if known else self.stim_duration_us
        polarity_str = known["polarity"] if known else "NegativeFirst"
        probe_polarity = StimPolarity.NegativeFirst if polarity_str == "NegativeFirst" else StimPolarity.PositiveFirst

        learn_amplitude = min(probe_amplitude, 3.0)
        learn_duration = probe_duration

        self._stdp_results = {
            "pair": f"{ef}->{et}",
            "hebbian_delay_ms": hebbian_delay,
            "testing_phase": {},
            "learning_phase": {},
            "validation_phase": {},
        }

        logger.info("STDP Testing Phase (%.0f s)", self.stdp_testing_duration_s)
        testing_results = self._stdp_probe_phase(
            ef, et, probe_amplitude, probe_duration, probe_polarity,
            self.stdp_testing_duration_s, "stdp_testing"
        )
        self._stdp_results["testing_phase"] = testing_results

        logger.info("STDP Learning Phase (%.0f s)", self.stdp_learning_duration_s)
        learning_results = self._stdp_learning_phase(
            ef, et, probe_amplitude, probe_duration, probe_polarity,
            learn_amplitude, learn_duration,
            hebbian_delay, self.stdp_learning_duration_s
        )
        self._stdp_results["learning_phase"] = learning_results

        logger.info("STDP Validation Phase (%.0f s)", self.stdp_validation_duration_s)
        validation_results = self._stdp_probe_phase(
            ef, et, probe_amplitude, probe_duration, probe_polarity,
            self.stdp_validation_duration_s, "stdp_validation"
        )
        self._stdp_results["validation_phase"] = validation_results

        logger.info("STDP experiment complete")

    def _stdp_probe_phase(
        self,
        ef: int,
        et: int,
        amplitude: float,
        duration: float,
        polarity: StimPolarity,
        phase_duration_s: float,
        phase_name: str,
    ) -> Dict[str, Any]:
        fs_name = self.np_experiment.exp_name
        phase_start = datetime_now()
        phase_end_target = phase_start.timestamp() + phase_duration_s

        probe_count = 0
        response_count = 0
        latencies = []

        while datetime_now().timestamp() < phase_end_target:
            stim_time = datetime_now()
            self._stimulate(
                electrode_idx=ef,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=polarity,
                trigger_key=0,
                phase=phase_name,
                extra={"probe_count": probe_count, "pair": f"{ef}->{et}"},
            )
            probe_count += 1

            self._wait(0.1)
            window_start = stim_time
            window_stop = datetime_now()
            spike_df = self._query_spikes_window(window_start, window_stop, fs_name)

            if not spike_df.empty:
                ch_col = self._get_channel_col(spike_df)
                if ch_col:
                    resp_spikes = spike_df[spike_df[ch_col] == et]
                    if not resp_spikes.empty:
                        response_count += 1
                        if "Time" in resp_spikes.columns:
                            spike_time = pd.to_datetime(resp_spikes["Time"].iloc[0])
                            lat_ms = (spike_time.timestamp() - stim_time.timestamp()) * 1000.0
                            if 1.0 < lat_ms < 200.0:
                                latencies.append(lat_ms)

            self._trial_results.append(TrialResult(
                phase=phase_name,
                electrode_from=ef,
                electrode_to=et,
                trial_index=probe_count,
                stim_time_utc=stim_time.isoformat(),
                spike_count=1 if (not spike_df.empty) else 0,
                latency_ms=latencies[-1] if latencies else None,
            ))

            remaining = phase_end_target - datetime_now().timestamp()
            if remaining <= 0:
                break
            sleep_time = min(self.stdp_probe_interval_s, remaining)
            self._wait(sleep_time)

        response_rate = response_count / probe_count if probe_count > 0 else 0.0
        median_latency = float(np.median(latencies)) if latencies else None

        return {
            "probe_count": probe_count,
            "response_count": response_count,
            "response_rate": response_rate,
            "median_latency_ms": median_latency,
            "latencies": latencies,
        }

    def _stdp_learning_phase(
        self,
        ef: int,
        et: int,
        probe_amplitude: float,
        probe_duration: float,
        probe_polarity: StimPolarity,
        learn_amplitude: float,
        learn_duration: float,
        hebbian_delay_ms: float,
        phase_duration_s: float,
    ) -> Dict[str, Any]:
        fs_name = self.np_experiment.exp_name
        phase_start = datetime_now()
        phase_end_target = phase_start.timestamp() + phase_duration_s

        conditioning_count = 0
        hebbian_delay_s = hebbian_delay_ms / 1000.0

        learn_polarity = StimPolarity.NegativeFirst

        while datetime_now().timestamp() < phase_end_target:
            stim_time = datetime_now()

            self._stimulate(
                electrode_idx=ef,
                amplitude_ua=probe_amplitude,
                duration_us=probe_duration,
                polarity=probe_polarity,
                trigger_key=0,
                phase="stdp_learning_probe",
                extra={"conditioning_count": conditioning_count, "pair": f"{ef}->{et}"},
            )

            self._wait(hebbian_delay_s)

            self._stimulate(
                electrode_idx=et,
                amplitude_ua=learn_amplitude,
                duration_us=learn_duration,
                polarity=learn_polarity,
                trigger_key=1,
                phase="stdp_learning_conditioning",
                extra={"conditioning_count": conditioning_count, "pair": f"{ef}->{et}", "delay_ms": hebbian_delay_ms},
            )

            conditioning_count += 1

            remaining = phase_end_target - datetime_now().timestamp()
            if remaining <= 0:
                break

            inter_event_interval = max(0.5, min(1.0, remaining))
            self._wait(inter_event_interval)

        return {
            "conditioning_count": conditioning_count,
            "hebbian_delay_ms": hebbian_delay_ms,
            "phase_duration_s": phase_duration_s,
        }

    def _save_all(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> None:
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

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "scan_responsive_pairs": len(self._scan_responsive_pairs),
            "stdp_results": self._stdp_results,
            "cross_correlograms_keys": list(self._cross_correlograms.keys()),
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

        ch_col = self._get_channel_col(spike_df)
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
                logger.warning("Failed to fetch waveforms for electrode %s: %s", electrode_idx, exc)

        return waveform_records

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        logger.info("Compiling results")

        testing_rate = self._stdp_results.get("testing_phase", {}).get("response_rate", None)
        validation_rate = self._stdp_results.get("validation_phase", {}).get("response_rate", None)

        potentiation_ratio = None
        if testing_rate is not None and validation_rate is not None and testing_rate > 0:
            potentiation_ratio = validation_rate / testing_rate

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.np_experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "scan_responsive_pairs_count": len(self._scan_responsive_pairs),
            "scan_responsive_pairs": self._scan_responsive_pairs,
            "active_electrode_pairs": list(self._active_stim_times.keys()),
            "cross_correlograms": {
                k: {
                    "peak_latency_ms": v["peak_latency_ms"],
                    "total_counts": sum(v["counts"]) if v["counts"] else 0,
                }
                for k, v in self._cross_correlograms.items()
            },
            "stdp_results": self._stdp_results,
            "potentiation_ratio": potentiation_ratio,
            "total_stimulations": len(self._stimulation_log),
            "total_trials": len(self._trial_results),
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
