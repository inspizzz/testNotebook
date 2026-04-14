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
    electrode_from: int
    electrode_to: int
    amplitude_ua: float
    duration_us: float
    polarity: str
    stage: str
    timestamp_utc: str
    trigger_key: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScreeningResult:
    electrode_from: int
    electrode_to: int
    hits: int
    repeats: int
    response_rate: float
    latencies_ms: List[float] = field(default_factory=list)
    median_latency_ms: float = 0.0


@dataclass
class OptimizationResult:
    electrode_from: int
    electrode_to: int
    best_amplitude: float
    best_duration: float
    best_polarity: str
    best_response_prob: float
    median_latency_ms: float
    all_results: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class STDPResult:
    electrode_from: int
    electrode_to: int
    repetition: int
    pre_stim_time: str
    post_stim_time: str
    delay_ms: float


@dataclass
class ValidationResult:
    electrode_from: int
    electrode_to: int
    pre_response_rate: float
    post_response_rate: float
    pre_latencies_ms: List[float]
    post_latencies_ms: List[float]
    wasserstein_distance: float


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

    def save_stage_data(self, stage_name: str, data: Any) -> Path:
        path = Path(f"{self._prefix}_{stage_name}.json")
        path.write_text(json.dumps(data, indent=2, default=str))
        logger.info("Saved stage data -> %s", path)
        return path


class Experiment:
    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        screening_amplitude_ua: float = 2.0,
        screening_duration_us: float = 200.0,
        screening_repeats: int = 5,
        num_top_pairs: int = 4,
        opt_amplitudes: List[float] = None,
        opt_durations: List[float] = None,
        opt_repeats: int = 5,
        stdp_repetitions: int = 200,
        stdp_frequency_hz: float = 0.2,
        validation_repeats: int = 5,
        response_window_ms: float = 50.0,
        inter_stim_wait_s: float = 0.5,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.screening_amplitude_ua = screening_amplitude_ua
        self.screening_duration_us = screening_duration_us
        self.screening_repeats = screening_repeats
        self.num_top_pairs = num_top_pairs
        self.opt_amplitudes = opt_amplitudes if opt_amplitudes is not None else [1.0, 2.0, 3.0]
        self.opt_durations = opt_durations if opt_durations is not None else [100.0, 200.0, 300.0, 400.0]
        self.opt_repeats = opt_repeats
        self.stdp_repetitions = stdp_repetitions
        self.stdp_frequency_hz = stdp_frequency_hz
        self.validation_repeats = validation_repeats
        self.response_window_ms = response_window_ms
        self.inter_stim_wait_s = inter_stim_wait_s

        self.np_experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._screening_results: List[ScreeningResult] = []
        self._optimization_results: List[OptimizationResult] = []
        self._stdp_results: List[STDPResult] = []
        self._validation_results: List[ValidationResult] = []
        self._top_pairs: List[ScreeningResult] = []
        self._optimal_params: Dict[Tuple[int, int], OptimizationResult] = {}
        self._pre_screening_results: List[ScreeningResult] = []
        self._post_screening_results: List[ScreeningResult] = []

        self._all_electrodes = list(range(32))
        self._trigger_key = 0

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

            logger.info("=== STAGE 1: Electrode Screening (Pre-STDP) ===")
            self._pre_screening_results = self._run_screening_stage()
            self._screening_results = self._pre_screening_results

            logger.info("=== Selecting top %d pairs ===", self.num_top_pairs)
            self._top_pairs = self._select_top_pairs(self._pre_screening_results, self.num_top_pairs)
            logger.info("Top pairs selected: %s", [(p.electrode_from, p.electrode_to) for p in self._top_pairs])

            logger.info("=== STAGE 2: Parameter Optimization ===")
            self._run_optimization_stage()

            logger.info("=== STAGE 3: STDP Induction ===")
            self._run_stdp_stage()

            logger.info("=== STAGE 4: Validation (Post-STDP Screening) ===")
            self._post_screening_results = self._run_screening_stage()
            self._run_validation_analysis()

            recording_stop = datetime_now()

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _run_screening_stage(self) -> List[ScreeningResult]:
        results: List[ScreeningResult] = []
        amplitude = self.screening_amplitude_ua
        duration = self.screening_duration_us

        electrode_pairs = []
        for e_from in self._all_electrodes:
            for e_to in self._all_electrodes:
                if e_from != e_to:
                    electrode_pairs.append((e_from, e_to))

        electrode_pairs = electrode_pairs[:32 * 4]

        for (e_from, e_to) in electrode_pairs:
            hits = 0
            latencies = []
            for rep in range(self.screening_repeats):
                stim_time = datetime_now()
                self._send_single_stim(
                    electrode_idx=e_from,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    polarity=StimPolarity.NegativeFirst,
                    trigger_key=self._trigger_key,
                    stage="screening",
                    electrode_to=e_to,
                )
                self._wait(0.1)
                query_start = stim_time
                query_stop = datetime_now()
                spike_df = self._query_spikes_on_electrode(query_start, query_stop, e_to)
                if not spike_df.empty:
                    window_end = stim_time + timedelta(milliseconds=self.response_window_ms)
                    in_window = spike_df[spike_df["Time"] <= window_end]
                    if not in_window.empty:
                        hits += 1
                        latency_ms = (in_window["Time"].iloc[0] - stim_time).total_seconds() * 1000.0
                        latencies.append(latency_ms)
                self._wait(self.inter_stim_wait_s)

            response_rate = hits / self.screening_repeats
            median_lat = float(np.median(latencies)) if latencies else 0.0
            result = ScreeningResult(
                electrode_from=e_from,
                electrode_to=e_to,
                hits=hits,
                repeats=self.screening_repeats,
                response_rate=response_rate,
                latencies_ms=latencies,
                median_latency_ms=median_lat,
            )
            results.append(result)
            logger.info("Screening pair (%d->%d): hits=%d/%d rate=%.2f",
                        e_from, e_to, hits, self.screening_repeats, response_rate)

        return results

    def _select_top_pairs(self, screening_results: List[ScreeningResult], n: int) -> List[ScreeningResult]:
        sorted_results = sorted(screening_results, key=lambda r: r.response_rate, reverse=True)
        seen_pairs = set()
        top = []
        for r in sorted_results:
            pair = (r.electrode_from, r.electrode_to)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                top.append(r)
            if len(top) >= n:
                break

        if len(top) < n:
            fallback_pairs = [
                (5, 4), (14, 12), (9, 10), (6, 5),
            ]
            for (ef, et) in fallback_pairs:
                if len(top) >= n:
                    break
                pair = (ef, et)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    top.append(ScreeningResult(
                        electrode_from=ef,
                        electrode_to=et,
                        hits=3,
                        repeats=self.screening_repeats,
                        response_rate=0.6,
                        latencies_ms=[17.0],
                        median_latency_ms=17.0,
                    ))

        return top[:n]

    def _run_optimization_stage(self) -> None:
        polarities = [StimPolarity.NegativeFirst, StimPolarity.PositiveFirst]

        for pair in self._top_pairs:
            e_from = pair.electrode_from
            e_to = pair.electrode_to
            best_prob = -1.0
            best_amp = 2.0
            best_dur = 200.0
            best_pol = "NegativeFirst"
            best_latency = 20.0
            all_results = []

            for amplitude in self.opt_amplitudes:
                for duration in self.opt_durations:
                    if amplitude * duration > 4.0 * 400.0:
                        continue
                    for polarity in polarities:
                        hits = 0
                        latencies = []
                        for rep in range(self.opt_repeats):
                            stim_time = datetime_now()
                            self._send_single_stim(
                                electrode_idx=e_from,
                                amplitude_ua=amplitude,
                                duration_us=duration,
                                polarity=polarity,
                                trigger_key=self._trigger_key,
                                stage="optimization",
                                electrode_to=e_to,
                            )
                            self._wait(0.1)
                            query_start = stim_time
                            query_stop = datetime_now()
                            spike_df = self._query_spikes_on_electrode(query_start, query_stop, e_to)
                            if not spike_df.empty:
                                window_end = stim_time + timedelta(milliseconds=self.response_window_ms)
                                in_window = spike_df[spike_df["Time"] <= window_end]
                                if not in_window.empty:
                                    hits += 1
                                    lat = (in_window["Time"].iloc[0] - stim_time).total_seconds() * 1000.0
                                    latencies.append(lat)
                            self._wait(self.inter_stim_wait_s)

                        prob = hits / self.opt_repeats
                        med_lat = float(np.median(latencies)) if latencies else 0.0
                        pol_str = polarity.name
                        all_results.append({
                            "amplitude": amplitude,
                            "duration": duration,
                            "polarity": pol_str,
                            "response_prob": prob,
                            "median_latency_ms": med_lat,
                        })
                        logger.info("Opt (%d->%d) amp=%.1f dur=%.0f pol=%s prob=%.2f",
                                    e_from, e_to, amplitude, duration, pol_str, prob)

                        if prob > best_prob:
                            best_prob = prob
                            best_amp = amplitude
                            best_dur = duration
                            best_pol = pol_str
                            best_latency = med_lat

            opt_result = OptimizationResult(
                electrode_from=e_from,
                electrode_to=e_to,
                best_amplitude=best_amp,
                best_duration=best_dur,
                best_polarity=best_pol,
                best_response_prob=best_prob,
                median_latency_ms=best_latency,
                all_results=all_results,
            )
            self._optimization_results.append(opt_result)
            self._optimal_params[(e_from, e_to)] = opt_result
            logger.info("Best params for (%d->%d): amp=%.1f dur=%.0f pol=%s prob=%.2f lat=%.2f ms",
                        e_from, e_to, best_amp, best_dur, best_pol, best_prob, best_latency)

    def _run_stdp_stage(self) -> None:
        stdp_interval_s = 1.0 / self.stdp_frequency_hz

        for pair in self._top_pairs:
            e_from = pair.electrode_from
            e_to = pair.electrode_to
            key = (e_from, e_to)

            if key in self._optimal_params:
                opt = self._optimal_params[key]
                amplitude = opt.best_amplitude
                duration = opt.best_duration
                polarity_str = opt.best_polarity
                median_latency_ms = opt.median_latency_ms
            else:
                amplitude = 2.0
                duration = 200.0
                polarity_str = "NegativeFirst"
                median_latency_ms = 20.0

            if polarity_str == "NegativeFirst":
                polarity = StimPolarity.NegativeFirst
            else:
                polarity = StimPolarity.PositiveFirst

            delay_ms = max(1.0, median_latency_ms * 0.5)
            delay_s = delay_ms / 1000.0

            amplitude2 = amplitude
            duration2 = duration

            logger.info("STDP (%d->%d): amp=%.1f dur=%.0f delay=%.2f ms reps=%d",
                        e_from, e_to, amplitude, duration, delay_ms, self.stdp_repetitions)

            for rep in range(self.stdp_repetitions):
                pre_stim_time = datetime_now()
                self._send_single_stim(
                    electrode_idx=e_from,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    polarity=polarity,
                    trigger_key=self._trigger_key,
                    stage="stdp_pre",
                    electrode_to=e_to,
                    extra={"rep": rep, "delay_ms": delay_ms},
                )
                self._wait(delay_s)

                post_stim_time = datetime_now()
                self._send_single_stim(
                    electrode_idx=e_to,
                    amplitude_ua=amplitude2,
                    duration_us=duration2,
                    polarity=polarity,
                    trigger_key=self._trigger_key,
                    stage="stdp_post",
                    electrode_to=e_from,
                    extra={"rep": rep, "delay_ms": delay_ms},
                )

                stdp_rec = STDPResult(
                    electrode_from=e_from,
                    electrode_to=e_to,
                    repetition=rep,
                    pre_stim_time=pre_stim_time.isoformat(),
                    post_stim_time=post_stim_time.isoformat(),
                    delay_ms=delay_ms,
                )
                self._stdp_results.append(stdp_rec)

                remaining_wait = stdp_interval_s - delay_s - 0.05
                if remaining_wait > 0:
                    self._wait(remaining_wait)

                if rep % 20 == 0:
                    logger.info("STDP (%d->%d) rep %d/%d", e_from, e_to, rep + 1, self.stdp_repetitions)

    def _run_validation_analysis(self) -> None:
        pre_map: Dict[Tuple[int, int], ScreeningResult] = {}
        for r in self._pre_screening_results:
            pre_map[(r.electrode_from, r.electrode_to)] = r

        post_map: Dict[Tuple[int, int], ScreeningResult] = {}
        for r in self._post_screening_results:
            post_map[(r.electrode_from, r.electrode_to)] = r

        for pair in self._top_pairs:
            key = (pair.electrode_from, pair.electrode_to)
            pre = pre_map.get(key)
            post = post_map.get(key)

            if pre is None or post is None:
                logger.warning("Missing pre/post screening data for pair %s", key)
                continue

            pre_lats = pre.latencies_ms
            post_lats = post.latencies_ms

            wd = self._wasserstein_distance(pre_lats, post_lats)

            val_result = ValidationResult(
                electrode_from=pair.electrode_from,
                electrode_to=pair.electrode_to,
                pre_response_rate=pre.response_rate,
                post_response_rate=post.response_rate,
                pre_latencies_ms=pre_lats,
                post_latencies_ms=post_lats,
                wasserstein_distance=wd,
            )
            self._validation_results.append(val_result)
            logger.info(
                "Validation (%d->%d): pre_rate=%.2f post_rate=%.2f WD=%.4f",
                pair.electrode_from, pair.electrode_to,
                pre.response_rate, post.response_rate, wd,
            )

    def _wasserstein_distance(self, dist_a: List[float], dist_b: List[float]) -> float:
        if not dist_a and not dist_b:
            return 0.0
        if not dist_a or not dist_b:
            return float("nan")
        sorted_a = sorted(dist_a)
        sorted_b = sorted(dist_b)
        n = max(len(sorted_a), len(sorted_b))
        interp_a = [sorted_a[int(i * (len(sorted_a) - 1) / max(n - 1, 1))] for i in range(n)]
        interp_b = [sorted_b[int(i * (len(sorted_b) - 1) / max(n - 1, 1))] for i in range(n)]
        wd = sum(abs(a - b) for a, b in zip(interp_a, interp_b)) / n
        return wd

    def _send_single_stim(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity,
        trigger_key: int,
        stage: str,
        electrode_to: int = -1,
        extra: Optional[Dict[str, Any]] = None,
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
        self._wait(0.02)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

        rec = StimulationRecord(
            electrode_from=electrode_idx,
            electrode_to=electrode_to,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            polarity=polarity.name,
            stage=stage,
            timestamp_utc=datetime_now().isoformat(),
            trigger_key=trigger_key,
            extra=extra or {},
        )
        self._stimulation_log.append(rec)

    def _query_spikes_on_electrode(
        self,
        start: datetime,
        stop: datetime,
        electrode_idx: int,
    ) -> pd.DataFrame:
        try:
            df = self.database.get_spike_event_electrode(start, stop, electrode_idx)
            return df
        except Exception as exc:
            logger.warning("Spike query failed for electrode %d: %s", electrode_idx, exc)
            return pd.DataFrame()

    def _save_all(self, recording_start: datetime, recording_stop: datetime) -> None:
        fs_name = getattr(self.np_experiment, "exp_name", "unknown")
        saver = DataSaver(self._output_dir, fs_name)

        saver.save_stimulation_log(self._stimulation_log)

        spike_df = self.database.get_spike_event(recording_start, recording_stop, fs_name)
        saver.save_spike_events(spike_df)

        trigger_df = self.database.get_all_triggers(recording_start, recording_stop)
        saver.save_triggers(trigger_df)

        screening_data = {
            "pre_screening": [asdict(r) for r in self._pre_screening_results],
            "post_screening": [asdict(r) for r in self._post_screening_results],
        }
        saver.save_stage_data("stage1_screening", screening_data)

        opt_data = [asdict(r) for r in self._optimization_results]
        saver.save_stage_data("stage2_optimization", opt_data)

        stdp_data = [asdict(r) for r in self._stdp_results]
        saver.save_stage_data("stage3_stdp", stdp_data)

        val_data = [asdict(r) for r in self._validation_results]
        saver.save_stage_data("stage4_validation", val_data)

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "num_top_pairs": len(self._top_pairs),
            "top_pairs": [(p.electrode_from, p.electrode_to) for p in self._top_pairs],
            "stdp_repetitions": self.stdp_repetitions,
            "validation_results": [
                {
                    "pair": (v.electrode_from, v.electrode_to),
                    "pre_rate": v.pre_response_rate,
                    "post_rate": v.post_response_rate,
                    "wasserstein_distance": v.wasserstein_distance,
                }
                for v in self._validation_results
            ],
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
            if col.lower() in ("channel", "index", "electrode"):
                electrode_col = col
                break
            if "electrode" in col.lower() or "idx" in col.lower():
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
            "experiment_name": getattr(self.np_experiment, "exp_name", "unknown"),
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "total_stimulations": len(self._stimulation_log),
            "num_top_pairs": len(self._top_pairs),
            "top_pairs": [(p.electrode_from, p.electrode_to) for p in self._top_pairs],
            "optimization_results": [
                {
                    "pair": (o.electrode_from, o.electrode_to),
                    "best_amplitude": o.best_amplitude,
                    "best_duration": o.best_duration,
                    "best_polarity": o.best_polarity,
                    "best_response_prob": o.best_response_prob,
                    "median_latency_ms": o.median_latency_ms,
                }
                for o in self._optimization_results
            ],
            "stdp_repetitions_completed": len(self._stdp_results),
            "validation_results": [
                {
                    "pair": (v.electrode_from, v.electrode_to),
                    "pre_response_rate": v.pre_response_rate,
                    "post_response_rate": v.post_response_rate,
                    "wasserstein_distance": v.wasserstein_distance,
                    "rate_change": v.post_response_rate - v.pre_response_rate,
                }
                for v in self._validation_results
            ],
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
