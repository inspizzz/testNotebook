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
        active_stim_hz: float = 1.0,
        active_group_size: int = 10,
        active_inter_group_s: float = 5.0,
        active_total_repeats: int = 100,
        stdp_testing_duration_s: float = 1200.0,
        stdp_learning_duration_s: float = 3000.0,
        stdp_validation_duration_s: float = 1200.0,
        stdp_test_interval_s: float = 5.0,
        stdp_amplitude_ua: float = 3.0,
        stdp_duration_us: float = 400.0,
        correlogram_bin_ms: float = 1.0,
        correlogram_window_ms: float = 100.0,
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
        self.stdp_test_interval_s = stdp_test_interval_s
        self.stdp_amplitude_ua = min(stdp_amplitude_ua, 4.0)
        self.stdp_duration_us = min(stdp_duration_us, 400.0)

        self.correlogram_bin_ms = correlogram_bin_ms
        self.correlogram_window_ms = correlogram_window_ms

        self.np_experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._scan_results: List[ScanResult] = []
        self._responsive_pairs: List[Dict[str, Any]] = []
        self._active_stim_times: Dict[str, List[datetime]] = defaultdict(list)
        self._correlogram_results: List[CrossCorrelogramResult] = []
        self._stdp_results: Dict[str, Any] = {}

        self._prior_reliable_connections = [
            {"electrode_from": 5, "electrode_to": 4, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "hits": 3, "median_latency_ms": 14.79},
            {"electrode_from": 6, "electrode_to": 7, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "hits": 4, "median_latency_ms": 20.195},
            {"electrode_from": 7, "electrode_to": 6, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "hits": 5, "median_latency_ms": 24.74},
            {"electrode_from": 13, "electrode_to": 14, "amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst", "hits": 3, "median_latency_ms": 11.85},
            {"electrode_from": 17, "electrode_to": 18, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "hits": 5, "median_latency_ms": 13.17},
            {"electrode_from": 21, "electrode_to": 19, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "hits": 5, "median_latency_ms": 19.3},
            {"electrode_from": 21, "electrode_to": 22, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "hits": 5, "median_latency_ms": 11.34},
        ]

        self._stdp_pairs = [
            {"stim": 7, "resp": 6, "latency_ms": 24.74, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst"},
            {"stim": 17, "resp": 18, "latency_ms": 13.477, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst"},
            {"stim": 21, "resp": 22, "latency_ms": 10.859, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"},
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

            logger.info("=== PHASE 1: Basic Excitability Scan ===")
            self._phase_excitability_scan()

            logger.info("=== PHASE 2: Active Electrode Experiment ===")
            self._phase_active_electrode()

            logger.info("=== PHASE 3: Hebbian STDP Experiment ===")
            self._phase_stdp()

            recording_stop = datetime_now()

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _build_stim_param(
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
        self._wait(0.05)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

    def _stimulate_single(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity,
        trigger_key: int = 0,
        phase: str = "",
        post_wait_s: float = 0.9,
    ) -> None:
        stim = self._build_stim_param(electrode_idx, amplitude_ua, duration_us, polarity, trigger_key)
        self.intan.send_stimparam([stim])
        stim_time = datetime_now()
        self._fire_trigger(trigger_key)
        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            polarity=polarity.name,
            timestamp_utc=stim_time.isoformat(),
            trigger_key=trigger_key,
            phase=phase,
        ))
        self._wait(post_wait_s)

    def _count_spikes_in_window(
        self,
        electrode_idx: int,
        stim_time: datetime,
        window_ms: float = 50.0,
    ) -> int:
        window_start = stim_time
        window_end = stim_time + timedelta(milliseconds=window_ms + 100)
        try:
            df = self.database.get_spike_event_electrode(window_start, window_end, electrode_idx)
            if df.empty:
                return 0
            time_col = "Time" if "Time" in df.columns else df.columns[0]
            spike_times = pd.to_datetime(df[time_col], utc=True)
            stim_dt = stim_time
            if stim_dt.tzinfo is None:
                stim_dt = stim_dt.replace(tzinfo=timezone.utc)
            mask = (spike_times >= stim_dt) & (spike_times <= stim_dt + timedelta(milliseconds=window_ms))
            return int(mask.sum())
        except Exception as exc:
            logger.warning("Spike count query failed for electrode %d: %s", electrode_idx, exc)
            return 0

    def _phase_excitability_scan(self) -> None:
        electrodes = list(self.np_experiment.electrodes)
        polarities = [StimPolarity.PositiveFirst, StimPolarity.NegativeFirst]

        logger.info("Scanning %d electrodes", len(electrodes))

        for elec_idx in electrodes:
            logger.info("Scanning electrode %d", elec_idx)
            for amplitude in self.scan_amplitudes:
                for duration in self.scan_durations:
                    for polarity in polarities:
                        hits = 0
                        latencies = []
                        for rep in range(self.scan_repeats):
                            stim = self._build_stim_param(elec_idx, amplitude, duration, polarity, trigger_key=0)
                            self.intan.send_stimparam([stim])
                            stim_time = datetime_now()
                            self._fire_trigger(0)
                            self._stimulation_log.append(StimulationRecord(
                                electrode_idx=elec_idx,
                                amplitude_ua=amplitude,
                                duration_us=duration,
                                polarity=polarity.name,
                                timestamp_utc=stim_time.isoformat(),
                                trigger_key=0,
                                phase="scan",
                            ))
                            self._wait(self.scan_inter_stim_s)

                            window_end = datetime_now()
                            window_start = stim_time
                            try:
                                df = self.database.get_spike_event_electrode(
                                    window_start,
                                    window_end + timedelta(milliseconds=100),
                                    elec_idx,
                                )
                                if not df.empty:
                                    time_col = "Time" if "Time" in df.columns else df.columns[0]
                                    spike_times = pd.to_datetime(df[time_col], utc=True)
                                    st = stim_time
                                    if st.tzinfo is None:
                                        st = st.replace(tzinfo=timezone.utc)
                                    mask = (spike_times >= st) & (spike_times <= st + timedelta(milliseconds=50))
                                    n_spikes = int(mask.sum())
                                    if n_spikes > 0:
                                        hits += 1
                                        first_spike = spike_times[mask].min()
                                        lat_ms = (first_spike - st).total_seconds() * 1000.0
                                        latencies.append(lat_ms)
                            except Exception as exc:
                                logger.warning("Scan query error electrode %d: %s", elec_idx, exc)

                        if hits >= self.scan_required_hits:
                            median_lat = float(np.median(latencies)) if latencies else 0.0
                            result = ScanResult(
                                electrode_from=elec_idx,
                                electrode_to=-1,
                                amplitude=amplitude,
                                duration=duration,
                                polarity=polarity.name,
                                hits=hits,
                                repeats=self.scan_repeats,
                                median_latency_ms=median_lat,
                            )
                            self._scan_results.append(result)
                            logger.info(
                                "Responsive: electrode %d amp=%.1f dur=%.0f pol=%s hits=%d",
                                elec_idx, amplitude, duration, polarity.name, hits,
                            )

            self._wait(self.scan_inter_channel_s)

        self._build_responsive_pairs()
        logger.info("Scan complete. Responsive pairs found: %d", len(self._responsive_pairs))

    def _build_responsive_pairs(self) -> None:
        seen = set()
        for conn in self._prior_reliable_connections:
            key = (conn["electrode_from"], conn["electrode_to"])
            if key not in seen:
                seen.add(key)
                self._responsive_pairs.append({
                    "stim": conn["electrode_from"],
                    "resp": conn["electrode_to"],
                    "amplitude": conn["amplitude"],
                    "duration": conn["duration"],
                    "polarity": conn["polarity"],
                    "hits": conn["hits"],
                    "median_latency_ms": conn["median_latency_ms"],
                })

        for sr in self._scan_results:
            if sr.electrode_to >= 0:
                key = (sr.electrode_from, sr.electrode_to)
                if key not in seen:
                    seen.add(key)
                    self._responsive_pairs.append({
                        "stim": sr.electrode_from,
                        "resp": sr.electrode_to,
                        "amplitude": sr.amplitude,
                        "duration": sr.duration,
                        "polarity": sr.polarity,
                        "hits": sr.hits,
                        "median_latency_ms": sr.median_latency_ms,
                    })

        logger.info("Total responsive pairs for active experiment: %d", len(self._responsive_pairs))

    def _phase_active_electrode(self) -> None:
        if not self._responsive_pairs:
            logger.warning("No responsive pairs found; using prior connections")
            self._build_responsive_pairs()

        inter_stim_s = 1.0 / self.active_stim_hz
        n_groups = self.active_total_repeats // self.active_group_size

        for pair in self._responsive_pairs:
            stim_elec = pair["stim"]
            resp_elec = pair["resp"]
            amplitude = min(pair["amplitude"], 4.0)
            duration = min(pair["duration"], 400.0)
            polarity_str = pair["polarity"]
            polarity = StimPolarity.PositiveFirst if polarity_str == "PositiveFirst" else StimPolarity.NegativeFirst
            pair_key = f"{stim_elec}->{resp_elec}"

            logger.info("Active experiment: pair %s", pair_key)

            stim = self._build_stim_param(stim_elec, amplitude, duration, polarity, trigger_key=0)
            self.intan.send_stimparam([stim])

            for group_idx in range(n_groups):
                for pulse_idx in range(self.active_group_size):
                    stim_time = datetime_now()
                    self._fire_trigger(0)
                    self._stimulation_log.append(StimulationRecord(
                        electrode_idx=stim_elec,
                        amplitude_ua=amplitude,
                        duration_us=duration,
                        polarity=polarity.name,
                        timestamp_utc=stim_time.isoformat(),
                        trigger_key=0,
                        phase="active",
                        extra={"resp_electrode": resp_elec, "group": group_idx, "pulse": pulse_idx},
                    ))
                    self._active_stim_times[pair_key].append(stim_time)
                    if pulse_idx < self.active_group_size - 1:
                        self._wait(inter_stim_s)

                if group_idx < n_groups - 1:
                    self._wait(self.active_inter_group_s)

            logger.info("Active experiment complete for pair %s (%d stimulations)", pair_key, len(self._active_stim_times[pair_key]))

        self._compute_cross_correlograms()

    def _compute_cross_correlograms(self) -> None:
        logger.info("Computing trigger-centred cross-correlograms")
        bin_s = self.correlogram_bin_ms / 1000.0
        window_s = self.correlogram_window_ms / 1000.0
        n_bins = int(self.correlogram_window_ms / self.correlogram_bin_ms)
        bin_edges = [i * self.correlogram_bin_ms for i in range(n_bins + 1)]

        for pair in self._responsive_pairs:
            stim_elec = pair["stim"]
            resp_elec = pair["resp"]
            pair_key = f"{stim_elec}->{resp_elec}"
            stim_times = self._active_stim_times.get(pair_key, [])
            if not stim_times:
                continue

            counts = [0] * n_bins
            if stim_times:
                query_start = min(stim_times) - timedelta(seconds=1)
                query_end = max(stim_times) + timedelta(seconds=window_s + 1)
                try:
                    spike_df = self.database.get_spike_event_electrode(query_start, query_end, resp_elec)
                    if not spike_df.empty:
                        time_col = "Time" if "Time" in spike_df.columns else spike_df.columns[0]
                        spike_times_dt = pd.to_datetime(spike_df[time_col], utc=True)
                        for st in stim_times:
                            if st.tzinfo is None:
                                st = st.replace(tzinfo=timezone.utc)
                            for spike_t in spike_times_dt:
                                lag_ms = (spike_t - st).total_seconds() * 1000.0
                                if 0 <= lag_ms < self.correlogram_window_ms:
                                    bin_idx = int(lag_ms / self.correlogram_bin_ms)
                                    if bin_idx < n_bins:
                                        counts[bin_idx] += 1
                except Exception as exc:
                    logger.warning("Correlogram query failed for pair %s: %s", pair_key, exc)

            peak_lag_ms = 0.0
            if any(c > 0 for c in counts):
                peak_bin = counts.index(max(counts))
                peak_lag_ms = (peak_bin + 0.5) * self.correlogram_bin_ms

            ccg = CrossCorrelogramResult(
                electrode_from=stim_elec,
                electrode_to=resp_elec,
                bins=bin_edges,
                counts=counts,
                peak_lag_ms=peak_lag_ms,
            )
            self._correlogram_results.append(ccg)
            logger.info("Correlogram pair %s: peak lag = %.2f ms", pair_key, peak_lag_ms)

    def _phase_stdp(self) -> None:
        stdp_pairs = self._stdp_pairs

        for pair_info in stdp_pairs:
            stim_elec = pair_info["stim"]
            resp_elec = pair_info["resp"]
            latency_ms = pair_info["latency_ms"]
            amplitude = min(pair_info["amplitude"], 4.0)
            duration = min(pair_info["duration"], 400.0)
            polarity_str = pair_info["polarity"]
            polarity = StimPolarity.PositiveFirst if polarity_str == "PositiveFirst" else StimPolarity.NegativeFirst

            hebbian_delay_s = latency_ms / 1000.0

            logger.info(
                "STDP pair %d->%d: latency=%.2f ms, Hebbian delay=%.4f s",
                stim_elec, resp_elec, latency_ms, hebbian_delay_s,
            )

            pair_key = f"stdp_{stim_elec}->{resp_elec}"
            self._stdp_results[pair_key] = {
                "stim_electrode": stim_elec,
                "resp_electrode": resp_elec,
                "latency_ms": latency_ms,
                "hebbian_delay_s": hebbian_delay_s,
                "testing_spikes": [],
                "learning_stim_count": 0,
                "validation_spikes": [],
            }

            stim = self._build_stim_param(stim_elec, amplitude, duration, polarity, trigger_key=0)
            self.intan.send_stimparam([stim])

            logger.info("STDP Testing phase: %.0f s", self.stdp_testing_duration_s)
            phase_start = datetime_now()
            phase_end_target = phase_start + timedelta(seconds=self.stdp_testing_duration_s)
            testing_spike_counts = []
            while datetime_now() < phase_end_target:
                stim_time = datetime_now()
                self._fire_trigger(0)
                self._stimulation_log.append(StimulationRecord(
                    electrode_idx=stim_elec,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    polarity=polarity.name,
                    timestamp_utc=stim_time.isoformat(),
                    trigger_key=0,
                    phase="stdp_testing",
                    extra={"resp_electrode": resp_elec},
                ))
                self._wait(self.stdp_test_interval_s)
                query_end = datetime_now()
                query_start = stim_time
                try:
                    df = self.database.get_spike_event_electrode(
                        query_start,
                        query_end + timedelta(milliseconds=200),
                        resp_elec,
                    )
                    if not df.empty:
                        time_col = "Time" if "Time" in df.columns else df.columns[0]
                        spike_times = pd.to_datetime(df[time_col], utc=True)
                        st = stim_time
                        if st.tzinfo is None:
                            st = st.replace(tzinfo=timezone.utc)
                        mask = (spike_times >= st) & (spike_times <= st + timedelta(milliseconds=100))
                        n = int(mask.sum())
                        testing_spike_counts.append(n)
                except Exception as exc:
                    logger.warning("STDP testing query error: %s", exc)

            self._stdp_results[pair_key]["testing_spikes"] = testing_spike_counts
            logger.info(
                "STDP Testing phase complete: %d probes, mean spikes=%.2f",
                len(testing_spike_counts),
                float(np.mean(testing_spike_counts)) if testing_spike_counts else 0.0,
            )

            logger.info("STDP Learning phase: %.0f s", self.stdp_learning_duration_s)
            phase_start = datetime_now()
            phase_end_target = phase_start + timedelta(seconds=self.stdp_learning_duration_s)
            learning_count = 0

            resp_amplitude = min(amplitude, 4.0)
            resp_duration = min(duration, 400.0)
            resp_polarity = polarity

            stim_pre = self._build_stim_param(stim_elec, amplitude, duration, polarity, trigger_key=0)
            stim_post = self._build_stim_param(resp_elec, resp_amplitude, resp_duration, resp_polarity, trigger_key=1)
            self.intan.send_stimparam([stim_pre, stim_post])

            while datetime_now() < phase_end_target:
                stim_time = datetime_now()
                pattern_pre = np.zeros(16, dtype=np.uint8)
                pattern_pre[0] = 1
                self.trigger_controller.send(pattern_pre)
                self._stimulation_log.append(StimulationRecord(
                    electrode_idx=stim_elec,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    polarity=polarity.name,
                    timestamp_utc=stim_time.isoformat(),
                    trigger_key=0,
                    phase="stdp_learning_pre",
                    extra={"resp_electrode": resp_elec, "hebbian_delay_s": hebbian_delay_s},
                ))
                self._wait(0.01)
                pattern_pre[0] = 0
                self.trigger_controller.send(pattern_pre)

                self._wait(hebbian_delay_s)

                post_time = datetime_now()
                pattern_post = np.zeros(16, dtype=np.uint8)
                pattern_post[1] = 1
                self.trigger_controller.send(pattern_post)
                self._stimulation_log.append(StimulationRecord(
                    electrode_idx=resp_elec,
                    amplitude_ua=resp_amplitude,
                    duration_us=resp_duration,
                    polarity=resp_polarity.name,
                    timestamp_utc=post_time.isoformat(),
                    trigger_key=1,
                    phase="stdp_learning_post",
                    extra={"stim_electrode": stim_elec},
                ))
                self._wait(0.01)
                pattern_post[1] = 0
                self.trigger_controller.send(pattern_post)

                learning_count += 1
                self._wait(self.stdp_test_interval_s)

            self._stdp_results[pair_key]["learning_stim_count"] = learning_count
            logger.info("STDP Learning phase complete: %d paired stimulations", learning_count)

            stim_val = self._build_stim_param(stim_elec, amplitude, duration, polarity, trigger_key=0)
            self.intan.send_stimparam([stim_val])

            logger.info("STDP Validation phase: %.0f s", self.stdp_validation_duration_s)
            phase_start = datetime_now()
            phase_end_target = phase_start + timedelta(seconds=self.stdp_validation_duration_s)
            validation_spike_counts = []
            while datetime_now() < phase_end_target:
                stim_time = datetime_now()
                self._fire_trigger(0)
                self._stimulation_log.append(StimulationRecord(
                    electrode_idx=stim_elec,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    polarity=polarity.name,
                    timestamp_utc=stim_time.isoformat(),
                    trigger_key=0,
                    phase="stdp_validation",
                    extra={"resp_electrode": resp_elec},
                ))
                self._wait(self.stdp_test_interval_s)
                query_end = datetime_now()
                try:
                    df = self.database.get_spike_event_electrode(
                        stim_time,
                        query_end + timedelta(milliseconds=200),
                        resp_elec,
                    )
                    if not df.empty:
                        time_col = "Time" if "Time" in df.columns else df.columns[0]
                        spike_times = pd.to_datetime(df[time_col], utc=True)
                        st = stim_time
                        if st.tzinfo is None:
                            st = st.replace(tzinfo=timezone.utc)
                        mask = (spike_times >= st) & (spike_times <= st + timedelta(milliseconds=100))
                        n = int(mask.sum())
                        validation_spike_counts.append(n)
                except Exception as exc:
                    logger.warning("STDP validation query error: %s", exc)

            self._stdp_results[pair_key]["validation_spikes"] = validation_spike_counts

            testing_mean = float(np.mean(testing_spike_counts)) if testing_spike_counts else 0.0
            validation_mean = float(np.mean(validation_spike_counts)) if validation_spike_counts else 0.0
            ner = validation_mean / testing_mean if testing_mean > 0 else float("nan")
            self._stdp_results[pair_key]["testing_mean_spikes"] = testing_mean
            self._stdp_results[pair_key]["validation_mean_spikes"] = validation_mean
            self._stdp_results[pair_key]["normalized_efficacy_ratio"] = ner

            logger.info(
                "STDP Validation complete for pair %d->%d: testing_mean=%.3f validation_mean=%.3f NER=%.3f",
                stim_elec, resp_elec, testing_mean, validation_mean, ner,
            )

    def _save_all(self, recording_start: datetime, recording_stop: datetime) -> None:
        fs_name = getattr(self.np_experiment, "exp_name", "unknown")
        saver = DataSaver(self._output_dir, fs_name)

        saver.save_stimulation_log(self._stimulation_log)

        spike_df = self.database.get_spike_event(recording_start, recording_stop, fs_name)
        saver.save_spike_events(spike_df)

        trigger_df = self.database.get_all_triggers(recording_start, recording_stop)
        saver.save_triggers(trigger_df)

        correlogram_data = []
        for ccg in self._correlogram_results:
            correlogram_data.append({
                "electrode_from": ccg.electrode_from,
                "electrode_to": ccg.electrode_to,
                "bins": ccg.bins,
                "counts": ccg.counts,
                "peak_lag_ms": ccg.peak_lag_ms,
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
            "correlogram_results": correlogram_data,
            "stdp_results": self._stdp_results,
        }
        saver.save_summary(summary)

        waveform_records = self._fetch_spike_waveforms(fs_name, spike_df, recording_start, recording_stop)
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
                raw_df = self.database.get_raw_spike(recording_start, recording_stop, int(electrode_idx))
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
        duration_s = (recording_stop - recording_start).total_seconds()

        stdp_summary = {}
        for pair_key, data in self._stdp_results.items():
            stdp_summary[pair_key] = {
                "stim_electrode": data.get("stim_electrode"),
                "resp_electrode": data.get("resp_electrode"),
                "latency_ms": data.get("latency_ms"),
                "testing_mean_spikes": data.get("testing_mean_spikes", 0.0),
                "validation_mean_spikes": data.get("validation_mean_spikes", 0.0),
                "normalized_efficacy_ratio": data.get("normalized_efficacy_ratio", float("nan")),
                "learning_stim_count": data.get("learning_stim_count", 0),
            }

        correlogram_summary = []
        for ccg in self._correlogram_results:
            correlogram_summary.append({
                "electrode_from": ccg.electrode_from,
                "electrode_to": ccg.electrode_to,
                "peak_lag_ms": ccg.peak_lag_ms,
            })

        return {
            "status": "completed",
            "experiment_name": getattr(self.np_experiment, "exp_name", "unknown"),
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": duration_s,
            "total_stimulations": len(self._stimulation_log),
            "scan_results_count": len(self._scan_results),
            "responsive_pairs_count": len(self._responsive_pairs),
            "correlogram_summary": correlogram_summary,
            "stdp_summary": stdp_summary,
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
