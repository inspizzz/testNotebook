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
    bin_edges_ms: List[float]
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
        scan_required_hits: int = 3,
        active_stim_amplitude: float = 3.0,
        active_stim_duration: float = 400.0,
        active_total_repeats: int = 100,
        active_group_size: int = 10,
        active_inter_group_s: float = 5.0,
        active_isi_s: float = 1.0,
        ccg_window_ms: float = 50.0,
        ccg_bin_ms: float = 4.0,
        testing_phase_min: float = 20.0,
        learning_phase_min: float = 50.0,
        validation_phase_min: float = 20.0,
        probe_amplitude: float = 1.0,
        probe_duration: float = 200.0,
        probe_interval_s: float = 10.0,
        hebbian_delay_ms: float = 20.0,
        conditioning_amplitude: float = 3.0,
        conditioning_duration: float = 400.0,
        conditioning_interval_s: float = 1.0,
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

        self.active_stim_amplitude = active_stim_amplitude
        self.active_stim_duration = active_stim_duration
        self.active_total_repeats = active_total_repeats
        self.active_group_size = active_group_size
        self.active_inter_group_s = active_inter_group_s
        self.active_isi_s = active_isi_s

        self.ccg_window_ms = ccg_window_ms
        self.ccg_bin_ms = ccg_bin_ms

        self.testing_phase_min = testing_phase_min
        self.learning_phase_min = learning_phase_min
        self.validation_phase_min = validation_phase_min

        self.probe_amplitude = probe_amplitude
        self.probe_duration = probe_duration
        self.probe_interval_s = probe_interval_s

        self.hebbian_delay_ms = hebbian_delay_ms
        self.conditioning_amplitude = conditioning_amplitude
        self.conditioning_duration = conditioning_duration
        self.conditioning_interval_s = conditioning_interval_s

        self.np_experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._scan_results: List[ScanResult] = []
        self._responsive_pairs: List[Dict[str, Any]] = []
        self._active_stim_times: Dict[str, List[str]] = defaultdict(list)
        self._ccg_results: List[CrossCorrelogramResult] = []
        self._phase_boundaries: Dict[str, str] = {}
        self._stdp_results: Dict[str, Any] = {}

        self._known_responsive_pairs = [
            {"electrode_from": 7, "electrode_to": 6, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 24.622},
            {"electrode_from": 17, "electrode_to": 18, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 13.477},
            {"electrode_from": 21, "electrode_to": 19, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 18.979},
            {"electrode_from": 21, "electrode_to": 22, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 10.859},
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

    def _get_polarity_enum(self, polarity_str: str) -> StimPolarity:
        if polarity_str == "PositiveFirst":
            return StimPolarity.PositiveFirst
        return StimPolarity.NegativeFirst

    def _send_biphasic_pulse(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity,
        trigger_key: int = 0,
        phase_label: str = "",
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
            phase=phase_label,
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
        electrodes = list(self.np_experiment.electrodes)
        polarities = [StimPolarity.PositiveFirst, StimPolarity.NegativeFirst]
        polarity_names = {StimPolarity.PositiveFirst: "PositiveFirst", StimPolarity.NegativeFirst: "NegativeFirst"}

        self._phase_boundaries["scan_start"] = datetime_now().isoformat()

        for elec_from in electrodes:
            for amplitude in self.scan_amplitudes:
                for duration in self.scan_durations:
                    for polarity in polarities:
                        hits = 0
                        latencies = []
                        for rep in range(self.scan_repeats):
                            pre_stim_time = datetime_now()
                            self._send_biphasic_pulse(
                                electrode_idx=elec_from,
                                amplitude_ua=amplitude,
                                duration_us=duration,
                                polarity=polarity,
                                trigger_key=0,
                                phase_label="scan",
                            )
                            self._wait(self.scan_inter_stim_s)
                            post_stim_time = datetime_now()

                            window_ms = 50.0
                            query_start = pre_stim_time
                            query_stop = post_stim_time

                            for elec_to in electrodes:
                                if elec_to == elec_from:
                                    continue
                                spike_df = self._query_spikes_window(elec_to, query_start, query_stop)
                                if not spike_df.empty:
                                    hits += 1
                                    if "Time" in spike_df.columns:
                                        for t in spike_df["Time"]:
                                            try:
                                                if hasattr(t, "timestamp"):
                                                    lat = (t.timestamp() - pre_stim_time.timestamp()) * 1000.0
                                                else:
                                                    lat = 0.0
                                                if 0 < lat < window_ms:
                                                    latencies.append(lat)
                                            except Exception:
                                                pass

                        median_lat = float(np.median(latencies)) if latencies else 0.0

                        if hits >= self.scan_required_hits:
                            for elec_to in electrodes:
                                if elec_to == elec_from:
                                    continue
                                result = ScanResult(
                                    electrode_from=elec_from,
                                    electrode_to=elec_to,
                                    amplitude=amplitude,
                                    duration=duration,
                                    polarity=polarity_names[polarity],
                                    hits=hits,
                                    repeats=self.scan_repeats,
                                    median_latency_ms=median_lat,
                                )
                                self._scan_results.append(result)

            self._wait(self.scan_inter_channel_s)

        self._phase_boundaries["scan_stop"] = datetime_now().isoformat()

        self._build_responsive_pairs()
        logger.info("Scan complete. Responsive pairs found: %d", len(self._responsive_pairs))

    def _build_responsive_pairs(self) -> None:
        seen = set()
        for pair_info in self._known_responsive_pairs:
            key = (pair_info["electrode_from"], pair_info["electrode_to"])
            if key not in seen:
                seen.add(key)
                self._responsive_pairs.append(pair_info)

        for result in self._scan_results:
            key = (result.electrode_from, result.electrode_to)
            if key not in seen:
                seen.add(key)
                self._responsive_pairs.append({
                    "electrode_from": result.electrode_from,
                    "electrode_to": result.electrode_to,
                    "amplitude": result.amplitude,
                    "duration": result.duration,
                    "polarity": result.polarity,
                    "median_latency_ms": result.median_latency_ms,
                })

        if not self._responsive_pairs:
            self._responsive_pairs = list(self._known_responsive_pairs)

    def _phase_active_electrode(self) -> None:
        self._phase_boundaries["active_start"] = datetime_now().isoformat()

        pairs_to_use = self._responsive_pairs[:4]

        for pair in pairs_to_use:
            elec_from = pair["electrode_from"]
            elec_to = pair["electrode_to"]
            amplitude = min(pair.get("amplitude", self.active_stim_amplitude), 4.0)
            duration = min(pair.get("duration", self.active_stim_duration), 400.0)
            polarity = self._get_polarity_enum(pair.get("polarity", "PositiveFirst"))
            pair_key = f"{elec_from}->{elec_to}"

            logger.info("Active electrode experiment: pair %s", pair_key)

            stim_times = []
            n_groups = self.active_total_repeats // self.active_group_size

            for group_idx in range(n_groups):
                for stim_idx in range(self.active_group_size):
                    t_stim = datetime_now()
                    self._send_biphasic_pulse(
                        electrode_idx=elec_from,
                        amplitude_ua=amplitude,
                        duration_us=duration,
                        polarity=polarity,
                        trigger_key=1,
                        phase_label="active",
                    )
                    stim_times.append(t_stim.isoformat())
                    if stim_idx < self.active_group_size - 1:
                        self._wait(self.active_isi_s)

                if group_idx < n_groups - 1:
                    self._wait(self.active_inter_group_s)

            self._active_stim_times[pair_key] = stim_times

        self._phase_boundaries["active_stop"] = datetime_now().isoformat()

        self._compute_ccg_for_pairs(pairs_to_use)

    def _compute_ccg_for_pairs(self, pairs: List[Dict[str, Any]]) -> None:
        logger.info("Computing cross-correlograms for %d pairs", len(pairs))
        window_ms = self.ccg_window_ms
        bin_ms = self.ccg_bin_ms
        n_bins = int(2 * window_ms / bin_ms)
        bin_edges = [(-window_ms + i * bin_ms) for i in range(n_bins + 1)]

        active_start_str = self._phase_boundaries.get("active_start")
        active_stop_str = self._phase_boundaries.get("active_stop")

        if not active_start_str or not active_stop_str:
            return

        try:
            active_start = datetime.fromisoformat(active_start_str)
            active_stop = datetime.fromisoformat(active_stop_str)
        except Exception:
            return

        for pair in pairs:
            elec_from = pair["electrode_from"]
            elec_to = pair["electrode_to"]

            try:
                spikes_from = self.database.get_spike_event_electrode(
                    active_start, active_stop, elec_from
                )
                spikes_to = self.database.get_spike_event_electrode(
                    active_start, active_stop, elec_to
                )
            except Exception as exc:
                logger.warning("CCG spike fetch failed for pair %d->%d: %s", elec_from, elec_to, exc)
                continue

            counts = [0] * n_bins

            if not spikes_from.empty and not spikes_to.empty and "Time" in spikes_from.columns and "Time" in spikes_to.columns:
                times_from = []
                times_to = []
                for t in spikes_from["Time"]:
                    try:
                        times_from.append(t.timestamp() if hasattr(t, "timestamp") else float(t))
                    except Exception:
                        pass
                for t in spikes_to["Time"]:
                    try:
                        times_to.append(t.timestamp() if hasattr(t, "timestamp") else float(t))
                    except Exception:
                        pass

                for tf in times_from:
                    for tt in times_to:
                        lag_ms = (tt - tf) * 1000.0
                        if -window_ms <= lag_ms < window_ms:
                            bin_idx = int((lag_ms + window_ms) / bin_ms)
                            if 0 <= bin_idx < n_bins:
                                counts[bin_idx] += 1

            peak_idx = int(np.argmax(counts)) if any(c > 0 for c in counts) else 0
            peak_lag = bin_edges[peak_idx] + bin_ms / 2.0
            peak_count = counts[peak_idx]

            ccg = CrossCorrelogramResult(
                electrode_from=elec_from,
                electrode_to=elec_to,
                bin_edges_ms=bin_edges,
                counts=counts,
                peak_lag_ms=peak_lag,
                peak_count=peak_count,
            )
            self._ccg_results.append(ccg)
            logger.info("CCG pair %d->%d: peak at %.1f ms (count=%d)", elec_from, elec_to, peak_lag, peak_count)

    def _get_hebbian_delay_for_pair(self, elec_from: int, elec_to: int) -> float:
        for ccg in self._ccg_results:
            if ccg.electrode_from == elec_from and ccg.electrode_to == elec_to:
                if 10.0 <= ccg.peak_lag_ms <= 50.0:
                    return ccg.peak_lag_ms
        for pair in self._known_responsive_pairs:
            if pair["electrode_from"] == elec_from and pair["electrode_to"] == elec_to:
                lat = pair.get("median_latency_ms", self.hebbian_delay_ms)
                return max(10.0, min(lat, 50.0))
        return self.hebbian_delay_ms

    def _phase_hebbian_stdp(self) -> None:
        pairs_to_use = self._responsive_pairs[:2]
        if not pairs_to_use:
            pairs_to_use = self._known_responsive_pairs[:2]

        self._stdp_results = {
            "pairs": [],
            "testing_phase": {},
            "learning_phase": {},
            "validation_phase": {},
        }

        for pair in pairs_to_use:
            elec_from = pair["electrode_from"]
            elec_to = pair["electrode_to"]
            pair_key = f"{elec_from}->{elec_to}"
            hebbian_delay = self._get_hebbian_delay_for_pair(elec_from, elec_to)

            logger.info("STDP pair %s, Hebbian delay=%.1f ms", pair_key, hebbian_delay)

            pair_result: Dict[str, Any] = {
                "pair_key": pair_key,
                "electrode_from": elec_from,
                "electrode_to": elec_to,
                "hebbian_delay_ms": hebbian_delay,
            }

            logger.info("--- Testing phase (%.0f min) ---", self.testing_phase_min)
            self._phase_boundaries[f"testing_start_{pair_key}"] = datetime_now().isoformat()
            testing_responses = self._run_probe_phase(
                elec_from=elec_from,
                elec_to=elec_to,
                duration_min=self.testing_phase_min,
                phase_label="testing",
            )
            self._phase_boundaries[f"testing_stop_{pair_key}"] = datetime_now().isoformat()
            pair_result["testing_probe_count"] = len(testing_responses)
            pair_result["testing_mean_response"] = float(np.mean(testing_responses)) if testing_responses else 0.0

            logger.info("--- Learning phase (%.0f min) ---", self.learning_phase_min)
            self._phase_boundaries[f"learning_start_{pair_key}"] = datetime_now().isoformat()
            self._run_learning_phase(
                elec_from=elec_from,
                elec_to=elec_to,
                duration_min=self.learning_phase_min,
                hebbian_delay_ms=hebbian_delay,
                pair_key=pair_key,
            )
            self._phase_boundaries[f"learning_stop_{pair_key}"] = datetime_now().isoformat()

            logger.info("--- Validation phase (%.0f min) ---", self.validation_phase_min)
            self._phase_boundaries[f"validation_start_{pair_key}"] = datetime_now().isoformat()
            validation_responses = self._run_probe_phase(
                elec_from=elec_from,
                elec_to=elec_to,
                duration_min=self.validation_phase_min,
                phase_label="validation",
            )
            self._phase_boundaries[f"validation_stop_{pair_key}"] = datetime_now().isoformat()
            pair_result["validation_probe_count"] = len(validation_responses)
            pair_result["validation_mean_response"] = float(np.mean(validation_responses)) if validation_responses else 0.0

            if testing_responses and validation_responses:
                delta = pair_result["validation_mean_response"] - pair_result["testing_mean_response"]
                pair_result["response_delta"] = delta
            else:
                pair_result["response_delta"] = 0.0

            self._stdp_results["pairs"].append(pair_result)
            logger.info("STDP pair %s complete. Delta=%.3f", pair_key, pair_result.get("response_delta", 0.0))

    def _run_probe_phase(
        self,
        elec_from: int,
        elec_to: int,
        duration_min: float,
        phase_label: str,
    ) -> List[float]:
        duration_s = duration_min * 60.0
        phase_start = datetime_now()
        responses = []

        probe_amp = min(self.probe_amplitude, 4.0)
        probe_dur = min(self.probe_duration, 400.0)

        while True:
            elapsed = (datetime_now() - phase_start).total_seconds()
            if elapsed >= duration_s:
                break

            pre_stim = datetime_now()
            self._send_biphasic_pulse(
                electrode_idx=elec_from,
                amplitude_ua=probe_amp,
                duration_us=probe_dur,
                polarity=StimPolarity.PositiveFirst,
                trigger_key=2,
                phase_label=phase_label,
            )
            self._wait(0.1)
            post_stim = datetime_now()

            window_s = 0.08
            query_start = pre_stim
            query_stop = datetime_now()
            spike_df = self._query_spikes_window(elec_to, query_start, query_stop)

            if not spike_df.empty and "Time" in spike_df.columns:
                valid_spikes = 0
                for t in spike_df["Time"]:
                    try:
                        if hasattr(t, "timestamp"):
                            lat_ms = (t.timestamp() - pre_stim.timestamp()) * 1000.0
                        else:
                            lat_ms = 0.0
                        if 5.0 <= lat_ms <= 80.0:
                            valid_spikes += 1
                    except Exception:
                        pass
                responses.append(float(valid_spikes))
            else:
                responses.append(0.0)

            remaining = duration_s - (datetime_now() - phase_start).total_seconds()
            if remaining <= 0:
                break
            sleep_time = min(self.probe_interval_s, remaining)
            self._wait(sleep_time)

        return responses

    def _run_learning_phase(
        self,
        elec_from: int,
        elec_to: int,
        duration_min: float,
        hebbian_delay_ms: float,
        pair_key: str,
    ) -> None:
        duration_s = duration_min * 60.0
        phase_start = datetime_now()

        cond_amp = min(self.conditioning_amplitude, 4.0)
        cond_dur = min(self.conditioning_duration, 400.0)
        delay_s = hebbian_delay_ms / 1000.0

        stim_count = 0
        while True:
            elapsed = (datetime_now() - phase_start).total_seconds()
            if elapsed >= duration_s:
                break

            self._send_biphasic_pulse(
                electrode_idx=elec_from,
                amplitude_ua=cond_amp,
                duration_us=cond_dur,
                polarity=StimPolarity.PositiveFirst,
                trigger_key=3,
                phase_label="learning_pre",
            )

            self._wait(delay_s)

            self._send_biphasic_pulse(
                electrode_idx=elec_to,
                amplitude_ua=cond_amp,
                duration_us=cond_dur,
                polarity=StimPolarity.PositiveFirst,
                trigger_key=4,
                phase_label="learning_post",
            )

            stim_count += 1

            remaining = duration_s - (datetime_now() - phase_start).total_seconds()
            if remaining <= 0:
                break
            sleep_time = min(self.conditioning_interval_s, remaining)
            self._wait(sleep_time)

        logger.info("Learning phase for %s: %d paired stimulations delivered", pair_key, stim_count)

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

        ccg_serializable = []
        for ccg in self._ccg_results:
            ccg_serializable.append({
                "electrode_from": ccg.electrode_from,
                "electrode_to": ccg.electrode_to,
                "bin_edges_ms": ccg.bin_edges_ms,
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
            "ccg_results": ccg_serializable,
            "stdp_results": self._stdp_results,
            "phase_boundaries": self._phase_boundaries,
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
            "experiment_name": self.np_experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "scan_results_count": len(self._scan_results),
            "responsive_pairs_count": len(self._responsive_pairs),
            "responsive_pairs": self._responsive_pairs,
            "ccg_results": ccg_serializable,
            "stdp_results": self._stdp_results,
            "total_stimulations": len(self._stimulation_log),
            "phase_boundaries": self._phase_boundaries,
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
