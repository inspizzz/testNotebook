import numpy as np
import pandas as pd
import json
import time
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
    stim_electrode: int
    resp_electrode: int
    bins_ms: List[float]
    counts: List[int]
    peak_latency_ms: float


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
        active_total_repeats: int = 100,
        active_inter_group_s: float = 5.0,
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
        self.active_total_repeats = active_total_repeats
        self.active_inter_group_s = active_inter_group_s

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
        self._active_stim_times: Dict[str, List[str]] = defaultdict(list)
        self._correlograms: List[CrossCorrelogram] = []
        self._stdp_results: Dict[str, Any] = {}

        self._known_responsive_pairs = [
            {"electrode_from": 7, "electrode_to": 6, "amplitude": 3.0, "duration": 400.0,
             "polarity": "PositiveFirst", "median_latency_ms": 24.622},
            {"electrode_from": 17, "electrode_to": 18, "amplitude": 3.0, "duration": 400.0,
             "polarity": "PositiveFirst", "median_latency_ms": 13.477},
            {"electrode_from": 21, "electrode_to": 19, "amplitude": 3.0, "duration": 400.0,
             "polarity": "PositiveFirst", "median_latency_ms": 18.979},
            {"electrode_from": 21, "electrode_to": 22, "amplitude": 3.0, "duration": 400.0,
             "polarity": "NegativeFirst", "median_latency_ms": 10.859},
            {"electrode_from": 6, "electrode_to": 7, "amplitude": 3.0, "duration": 400.0,
             "polarity": "PositiveFirst", "median_latency_ms": 19.294},
        ]

        self._recording_start: Optional[datetime] = None
        self._recording_stop: Optional[datetime] = None

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

            self._recording_start = datetime_now()

            logger.info("=== Phase 1: Basic Excitability Scan ===")
            self._phase_excitability_scan()

            logger.info("=== Phase 2: Active Electrode Experiment ===")
            self._phase_active_electrode()

            logger.info("=== Phase 3: Hebbian STDP Experiment ===")
            self._phase_stdp()

            self._recording_stop = datetime_now()

            results = self._compile_results(self._recording_start, self._recording_stop)

            self._save_all(self._recording_start, self._recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            if self._recording_stop is None:
                self._recording_stop = datetime_now()
            try:
                if self._recording_start is not None and self._recording_stop is not None:
                    self._save_all(self._recording_start, self._recording_stop)
            except Exception as save_exc:
                logger.error("Error saving data after failure: %s", save_exc)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _phase_excitability_scan(self) -> None:
        logger.info("Starting excitability scan across all available electrodes")
        electrodes = self.np_experiment.electrodes
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
                                query_start, query_stop, self.np_experiment.exp_name
                            )
                            if not spike_df.empty:
                                resp_spikes = spike_df[spike_df.get("channel", pd.Series(dtype=int)) != electrode_idx] if "channel" in spike_df.columns else spike_df
                                if len(resp_spikes) > 0:
                                    hits += 1
                                    if "Time" in resp_spikes.columns:
                                        t_stim = stim_time.timestamp()
                                        for _, row in resp_spikes.iterrows():
                                            try:
                                                t_spike = pd.Timestamp(row["Time"]).timestamp()
                                                lat = (t_spike - t_stim) * 1000.0
                                                if 0 < lat < 200:
                                                    latencies.append(lat)
                                            except Exception:
                                                pass
                            if rep < self.scan_repeats - 1:
                                self._wait(self.scan_inter_stim_s)

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

                        if hits >= self.scan_required_hits:
                            logger.info(
                                "Responsive electrode %d: amp=%.1f dur=%.0f pol=%s hits=%d/%d",
                                electrode_idx, amplitude, duration,
                                polarity_names[polarity], hits, self.scan_repeats
                            )

                self._wait(self.scan_inter_channel_s)

        self._identify_responsive_pairs()
        logger.info("Excitability scan complete. Responsive pairs: %d", len(self._responsive_pairs))

    def _identify_responsive_pairs(self) -> None:
        self._responsive_pairs = list(self._known_responsive_pairs)
        logger.info("Using %d pre-identified responsive pairs from parameter scan", len(self._responsive_pairs))

    def _phase_active_electrode(self) -> None:
        logger.info("Starting active electrode experiment with %d pairs", len(self._responsive_pairs))
        stim_interval_s = 1.0 / self.active_stim_hz

        for pair in self._responsive_pairs:
            stim_elec = pair["electrode_from"]
            resp_elec = pair["electrode_to"]
            amplitude = min(pair["amplitude"], 4.0)
            duration = min(pair["duration"], 400.0)
            polarity_str = pair["polarity"]
            polarity = StimPolarity.PositiveFirst if polarity_str == "PositiveFirst" else StimPolarity.NegativeFirst
            pair_key = f"{stim_elec}_to_{resp_elec}"

            logger.info("Active stim: electrode %d -> %d, amp=%.1f, dur=%.0f",
                        stim_elec, resp_elec, amplitude, duration)

            n_groups = self.active_total_repeats // self.active_group_size
            for group_idx in range(n_groups):
                for pulse_idx in range(self.active_group_size):
                    stim_time = datetime_now()
                    self._send_stim(
                        electrode_idx=stim_elec,
                        amplitude_ua=amplitude,
                        duration_us=duration,
                        polarity=polarity,
                        trigger_key=0,
                        phase="active",
                    )
                    self._active_stim_times[pair_key].append(stim_time.isoformat())
                    if pulse_idx < self.active_group_size - 1:
                        self._wait(stim_interval_s)

                logger.info("Group %d/%d complete for pair %s", group_idx + 1, n_groups, pair_key)
                if group_idx < n_groups - 1:
                    self._wait(self.active_inter_group_s)

            self._wait(self.scan_inter_channel_s)

        logger.info("Active electrode experiment complete. Computing correlograms.")
        self._compute_correlograms()

    def _compute_correlograms(self) -> None:
        logger.info("Computing trigger-centred cross-correlograms")
        window_ms = self.correlogram_window_ms
        bin_ms = self.correlogram_bin_ms
        n_bins = int(window_ms / bin_ms)
        bins_ms = [i * bin_ms for i in range(n_bins)]

        for pair in self._responsive_pairs:
            stim_elec = pair["electrode_from"]
            resp_elec = pair["electrode_to"]
            pair_key = f"{stim_elec}_to_{resp_elec}"
            stim_times_iso = self._active_stim_times.get(pair_key, [])
            if not stim_times_iso:
                continue

            counts = [0] * n_bins

            query_start = self._recording_start if self._recording_start else datetime_now() - timedelta(hours=2)
            query_stop = datetime_now()

            try:
                spike_df = self.database.get_spike_event(
                    query_start, query_stop, self.np_experiment.exp_name
                )
            except Exception as exc:
                logger.warning("Failed to fetch spikes for correlogram: %s", exc)
                spike_df = pd.DataFrame()

            if not spike_df.empty and "channel" in spike_df.columns and "Time" in spike_df.columns:
                resp_spikes = spike_df[spike_df["channel"] == resp_elec]
                resp_times = []
                for _, row in resp_spikes.iterrows():
                    try:
                        resp_times.append(pd.Timestamp(row["Time"]).timestamp())
                    except Exception:
                        pass

                for stim_iso in stim_times_iso:
                    try:
                        t_stim = datetime.fromisoformat(stim_iso).timestamp()
                    except Exception:
                        continue
                    for t_spike in resp_times:
                        lag_ms = (t_spike - t_stim) * 1000.0
                        if 0 <= lag_ms < window_ms:
                            bin_idx = int(lag_ms / bin_ms)
                            if bin_idx < n_bins:
                                counts[bin_idx] += 1

            peak_bin = int(np.argmax(counts)) if any(c > 0 for c in counts) else 0
            peak_latency_ms = bins_ms[peak_bin] if counts[peak_bin] > 0 else pair.get("median_latency_ms", 20.0)

            ccg = CrossCorrelogram(
                stim_electrode=stim_elec,
                resp_electrode=resp_elec,
                bins_ms=bins_ms,
                counts=counts,
                peak_latency_ms=peak_latency_ms,
            )
            self._correlograms.append(ccg)
            logger.info("Correlogram for %d->%d: peak latency=%.2f ms",
                        stim_elec, resp_elec, peak_latency_ms)

    def _get_hebbian_delay_ms(self, stim_elec: int, resp_elec: int) -> float:
        for ccg in self._correlograms:
            if ccg.stim_electrode == stim_elec and ccg.resp_electrode == resp_elec:
                return ccg.peak_latency_ms
        for pair in self._responsive_pairs:
            if pair["electrode_from"] == stim_elec and pair["electrode_to"] == resp_elec:
                return pair.get("median_latency_ms", 20.0)
        return 20.0

    def _phase_stdp(self) -> None:
        logger.info("Starting STDP Hebbian learning experiment")
        if not self._responsive_pairs:
            logger.warning("No responsive pairs found; skipping STDP phase")
            return

        primary_pair = self._responsive_pairs[0]
        stim_elec = primary_pair["electrode_from"]
        resp_elec = primary_pair["electrode_to"]
        amplitude = min(self.stdp_amplitude_ua, 4.0)
        duration = min(self.stdp_duration_us, 400.0)
        polarity_str = primary_pair.get("polarity", "PositiveFirst")
        polarity = StimPolarity.PositiveFirst if polarity_str == "PositiveFirst" else StimPolarity.NegativeFirst

        hebbian_delay_ms = self._get_hebbian_delay_ms(stim_elec, resp_elec)
        logger.info("STDP pair: %d -> %d, Hebbian delay=%.2f ms", stim_elec, resp_elec, hebbian_delay_ms)

        logger.info("STDP Phase A: Testing (%.0f s)", self.stdp_testing_duration_s)
        testing_responses = self._stdp_test_phase(
            stim_elec=stim_elec,
            resp_elec=resp_elec,
            amplitude=amplitude,
            duration=duration,
            polarity=polarity,
            phase_duration_s=self.stdp_testing_duration_s,
            phase_label="stdp_testing",
        )

        logger.info("STDP Phase B: Learning (%.0f s)", self.stdp_learning_duration_s)
        learning_responses = self._stdp_learning_phase(
            stim_elec=stim_elec,
            resp_elec=resp_elec,
            amplitude=amplitude,
            duration=duration,
            polarity=polarity,
            phase_duration_s=self.stdp_learning_duration_s,
            hebbian_delay_ms=hebbian_delay_ms,
        )

        logger.info("STDP Phase C: Validation (%.0f s)", self.stdp_validation_duration_s)
        validation_responses = self._stdp_test_phase(
            stim_elec=stim_elec,
            resp_elec=resp_elec,
            amplitude=amplitude,
            duration=duration,
            polarity=polarity,
            phase_duration_s=self.stdp_validation_duration_s,
            phase_label="stdp_validation",
        )

        baseline_rate = float(np.mean(testing_responses)) if testing_responses else 0.0
        validation_rate = float(np.mean(validation_responses)) if validation_responses else 0.0
        ner = validation_rate / baseline_rate if baseline_rate > 0 else 0.0

        self._stdp_results = {
            "stim_electrode": stim_elec,
            "resp_electrode": resp_elec,
            "hebbian_delay_ms": hebbian_delay_ms,
            "testing_n_trials": len(testing_responses),
            "learning_n_trials": len(learning_responses),
            "validation_n_trials": len(validation_responses),
            "baseline_response_rate": baseline_rate,
            "validation_response_rate": validation_rate,
            "normalized_efficacy_ratio": ner,
            "testing_responses": testing_responses,
            "validation_responses": validation_responses,
        }
        logger.info("STDP complete. NER=%.3f (baseline=%.3f, validation=%.3f)",
                    ner, baseline_rate, validation_rate)

    def _stdp_test_phase(
        self,
        stim_elec: int,
        resp_elec: int,
        amplitude: float,
        duration: float,
        polarity: StimPolarity,
        phase_duration_s: float,
        phase_label: str,
    ) -> List[float]:
        responses = []
        phase_start = datetime_now()
        trial_count = 0
        while True:
            elapsed = (datetime_now() - phase_start).total_seconds()
            if elapsed >= phase_duration_s:
                break
            stim_time = datetime_now()
            self._send_stim(
                electrode_idx=stim_elec,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=polarity,
                trigger_key=0,
                phase=phase_label,
            )
            self._wait(0.05)
            query_start = stim_time
            query_stop = datetime_now()
            try:
                spike_df = self.database.get_spike_event(
                    query_start, query_stop, self.np_experiment.exp_name
                )
                if not spike_df.empty and "channel" in spike_df.columns:
                    resp_spikes = spike_df[spike_df["channel"] == resp_elec]
                    responses.append(float(len(resp_spikes)))
                else:
                    responses.append(0.0)
            except Exception as exc:
                logger.warning("Spike query failed in %s: %s", phase_label, exc)
                responses.append(0.0)
            trial_count += 1
            remaining = phase_duration_s - (datetime_now() - phase_start).total_seconds()
            if remaining <= 0:
                break
            self._wait(min(self.stdp_test_interval_s, remaining))
        logger.info("%s: %d trials completed", phase_label, trial_count)
        return responses

    def _stdp_learning_phase(
        self,
        stim_elec: int,
        resp_elec: int,
        amplitude: float,
        duration: float,
        polarity: StimPolarity,
        phase_duration_s: float,
        hebbian_delay_ms: float,
    ) -> List[float]:
        responses = []
        phase_start = datetime_now()
        trial_count = 0
        hebbian_delay_s = hebbian_delay_ms / 1000.0

        while True:
            elapsed = (datetime_now() - phase_start).total_seconds()
            if elapsed >= phase_duration_s:
                break

            stim_time = datetime_now()
            self._send_stim(
                electrode_idx=stim_elec,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=polarity,
                trigger_key=0,
                phase="stdp_learning_pre",
            )

            self._wait(hebbian_delay_s)

            self._send_stim(
                electrode_idx=resp_elec,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=polarity,
                trigger_key=1,
                phase="stdp_learning_post",
            )

            self._wait(0.05)
            query_start = stim_time
            query_stop = datetime_now()
            try:
                spike_df = self.database.get_spike_event(
                    query_start, query_stop, self.np_experiment.exp_name
                )
                if not spike_df.empty and "channel" in spike_df.columns:
                    resp_spikes = spike_df[spike_df["channel"] == resp_elec]
                    responses.append(float(len(resp_spikes)))
                else:
                    responses.append(0.0)
            except Exception as exc:
                logger.warning("Spike query failed in learning phase: %s", exc)
                responses.append(0.0)

            trial_count += 1
            remaining = phase_duration_s - (datetime_now() - phase_start).total_seconds()
            if remaining <= 0:
                break
            self._wait(min(self.stdp_test_interval_s, remaining))

        logger.info("stdp_learning: %d paired trials completed", trial_count)
        return responses

    def _send_stim(
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

    def _save_all(self, recording_start: datetime, recording_stop: datetime) -> None:
        fs_name = getattr(self.np_experiment, "exp_name", "unknown") if self.np_experiment else "unknown"
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

        correlograms_serializable = []
        for ccg in self._correlograms:
            correlograms_serializable.append({
                "stim_electrode": ccg.stim_electrode,
                "resp_electrode": ccg.resp_electrode,
                "bins_ms": ccg.bins_ms,
                "counts": ccg.counts,
                "peak_latency_ms": ccg.peak_latency_ms,
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
            "correlograms_count": len(self._correlograms),
            "correlograms": correlograms_serializable,
            "stdp_results": self._stdp_results,
            "active_stim_times": dict(self._active_stim_times),
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

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        logger.info("Compiling results")
        fs_name = getattr(self.np_experiment, "exp_name", "unknown") if self.np_experiment else "unknown"

        scan_summary = []
        for r in self._scan_results:
            if r.hits >= self.scan_required_hits:
                scan_summary.append({
                    "electrode_from": r.electrode_from,
                    "amplitude": r.amplitude,
                    "duration": r.duration,
                    "polarity": r.polarity,
                    "hits": r.hits,
                    "repeats": r.repeats,
                    "median_latency_ms": r.median_latency_ms,
                })

        correlograms_out = []
        for ccg in self._correlograms:
            correlograms_out.append({
                "stim_electrode": ccg.stim_electrode,
                "resp_electrode": ccg.resp_electrode,
                "peak_latency_ms": ccg.peak_latency_ms,
            })

        summary = {
            "status": "completed",
            "experiment_name": fs_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "total_stimulations": len(self._stimulation_log),
            "responsive_scan_results": scan_summary,
            "responsive_pairs_used": self._responsive_pairs,
            "correlograms": correlograms_out,
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
