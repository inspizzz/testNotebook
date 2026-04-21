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
    stage: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialResult:
    stage: str
    stim_electrode: int
    resp_electrode: int
    amplitude_ua: float
    duration_us: float
    polarity: str
    trial_index: int
    responded: bool
    latency_ms: float
    timestamp_utc: str


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
        top_pairs_n: int = 4,
        opt_amplitudes: Tuple[float, ...] = (1.0, 2.0, 3.0),
        opt_durations: Tuple[float, ...] = (100.0, 200.0, 400.0),
        opt_repeats: int = 5,
        stdp_repetitions: int = 200,
        stdp_frequency_hz: float = 0.2,
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
        self.top_pairs_n = top_pairs_n
        self.opt_amplitudes = list(opt_amplitudes)
        self.opt_durations = list(opt_durations)
        self.opt_repeats = opt_repeats
        self.stdp_repetitions = stdp_repetitions
        self.stdp_frequency_hz = stdp_frequency_hz
        self.response_window_ms = response_window_ms
        self.inter_stim_wait_s = inter_stim_wait_s

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._trial_results: List[TrialResult] = []

        self._stage1_results: Dict[str, Any] = {}
        self._stage2_results: Dict[str, Any] = {}
        self._stage3_results: Dict[str, Any] = {}
        self._stage4_results: Dict[str, Any] = {}

        self._selected_pairs: List[Dict[str, Any]] = []
        self._optimal_params: Dict[str, Any] = {}
        self._median_latencies: Dict[str, float] = {}

        self._recording_start: Optional[datetime] = None
        self._recording_stop: Optional[datetime] = None

        self._all_electrodes = list(range(32))

        self._prior_pairs = [
            {"stim": 17, "resp": 18, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "response_rate": 0.92},
            {"stim": 21, "resp": 19, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "response_rate": 0.92},
            {"stim": 21, "resp": 22, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "response_rate": 0.84},
            {"stim": 7,  "resp": 6,  "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "response_rate": 0.87},
            {"stim": 6,  "resp": 7,  "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "response_rate": 0.46},
            {"stim": 5,  "resp": 4,  "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "response_rate": 0.16},
            {"stim": 13, "resp": 14, "amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst", "response_rate": 0.13},
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

            logger.info("=== STAGE 1: Electrode Screening ===")
            self._stage1_electrode_screening()

            logger.info("=== STAGE 2: Parameter Optimization ===")
            self._stage2_parameter_optimization()

            logger.info("=== STAGE 3: STDP Induction ===")
            self._stage3_stdp_induction()

            logger.info("=== STAGE 4: Validation ===")
            self._stage4_validation()

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

    def _stage1_electrode_screening(self) -> None:
        logger.info("Stage 1: Scanning all electrode pairs at %.1f uA, %.0f us, %d repeats",
                    self.screening_amplitude_ua, self.screening_duration_us, self.screening_repeats)

        amplitude = self.screening_amplitude_ua
        duration = self.screening_duration_us

        available_electrodes = self.experiment.electrodes
        if not available_electrodes:
            available_electrodes = list(range(32))

        pair_scores: Dict[Tuple[int, int], Dict[str, Any]] = {}

        for stim_elec in available_electrodes:
            for resp_elec in available_electrodes:
                if stim_elec == resp_elec:
                    continue

                hits = 0
                latencies = []

                for rep in range(self.screening_repeats):
                    polarity = StimPolarity.PositiveFirst
                    stim_time = datetime_now()
                    self._send_single_stim(
                        electrode_idx=stim_elec,
                        amplitude_ua=amplitude,
                        duration_us=duration,
                        polarity=polarity,
                        trigger_key=0,
                        stage="stage1",
                    )
                    self._wait(self.inter_stim_wait_s)

                    query_start = stim_time
                    query_stop = datetime_now()
                    spike_df = self.database.get_spike_event(
                        query_start, query_stop, self.experiment.exp_name
                    )

                    responded, latency_ms = self._check_response(
                        spike_df, resp_elec, stim_time, self.response_window_ms
                    )
                    if responded:
                        hits += 1
                        latencies.append(latency_ms)

                    self._trial_results.append(TrialResult(
                        stage="stage1",
                        stim_electrode=stim_elec,
                        resp_electrode=resp_elec,
                        amplitude_ua=amplitude,
                        duration_us=duration,
                        polarity="PositiveFirst",
                        trial_index=rep,
                        responded=responded,
                        latency_ms=latency_ms if responded else float('nan'),
                        timestamp_utc=stim_time.isoformat(),
                    ))

                response_rate = hits / self.screening_repeats
                median_latency = float(np.median(latencies)) if latencies else float('nan')

                pair_key = (stim_elec, resp_elec)
                pair_scores[pair_key] = {
                    "stim_electrode": stim_elec,
                    "resp_electrode": resp_elec,
                    "hits": hits,
                    "repeats": self.screening_repeats,
                    "response_rate": response_rate,
                    "median_latency_ms": median_latency,
                    "latencies": latencies,
                }

                logger.info("Pair (%d->%d): hits=%d/%d rate=%.2f",
                            stim_elec, resp_elec, hits, self.screening_repeats, response_rate)

        sorted_pairs = sorted(
            pair_scores.values(),
            key=lambda x: (x["response_rate"], -x["median_latency_ms"] if not math.isnan(x["median_latency_ms"]) else float('inf')),
            reverse=True
        )

        self._selected_pairs = sorted_pairs[:self.top_pairs_n]

        if len(self._selected_pairs) < self.top_pairs_n:
            logger.warning("Only %d responsive pairs found, supplementing from prior scan data",
                           len(self._selected_pairs))
            existing_keys = {(p["stim_electrode"], p["resp_electrode"]) for p in self._selected_pairs}
            for prior in self._prior_pairs:
                if len(self._selected_pairs) >= self.top_pairs_n:
                    break
                key = (prior["stim"], prior["resp"])
                if key not in existing_keys:
                    self._selected_pairs.append({
                        "stim_electrode": prior["stim"],
                        "resp_electrode": prior["resp"],
                        "hits": int(prior["response_rate"] * self.screening_repeats),
                        "repeats": self.screening_repeats,
                        "response_rate": prior["response_rate"],
                        "median_latency_ms": 20.0,
                        "latencies": [],
                    })
                    existing_keys.add(key)

        logger.info("Selected top %d pairs:", len(self._selected_pairs))
        for p in self._selected_pairs:
            logger.info("  (%d->%d) rate=%.2f", p["stim_electrode"], p["resp_electrode"], p["response_rate"])

        self._stage1_results = {
            "all_pair_scores": [
                {
                    "stim_electrode": v["stim_electrode"],
                    "resp_electrode": v["resp_electrode"],
                    "response_rate": v["response_rate"],
                    "median_latency_ms": v["median_latency_ms"],
                }
                for v in sorted_pairs
            ],
            "selected_pairs": [
                {
                    "stim_electrode": p["stim_electrode"],
                    "resp_electrode": p["resp_electrode"],
                    "response_rate": p["response_rate"],
                    "median_latency_ms": p["median_latency_ms"],
                }
                for p in self._selected_pairs
            ],
        }

    def _stage2_parameter_optimization(self) -> None:
        logger.info("Stage 2: Parameter optimization for %d selected pairs", len(self._selected_pairs))

        for pair in self._selected_pairs:
            stim_elec = pair["stim_electrode"]
            resp_elec = pair["resp_electrode"]
            pair_key = f"{stim_elec}_{resp_elec}"

            best_rate = -1.0
            best_amplitude = self.screening_amplitude_ua
            best_duration = self.screening_duration_us
            best_polarity_str = "PositiveFirst"
            best_latencies: List[float] = []

            param_results = []

            for amplitude in self.opt_amplitudes:
                for duration in self.opt_durations:
                    if amplitude * duration > 4.0 * 400.0:
                        continue

                    for polarity_str, polarity_enum in [("PositiveFirst", StimPolarity.PositiveFirst),
                                                         ("NegativeFirst", StimPolarity.NegativeFirst)]:
                        hits = 0
                        latencies = []

                        for rep in range(self.opt_repeats):
                            stim_time = datetime_now()
                            self._send_single_stim(
                                electrode_idx=stim_elec,
                                amplitude_ua=amplitude,
                                duration_us=duration,
                                polarity=polarity_enum,
                                trigger_key=0,
                                stage="stage2",
                            )
                            self._wait(self.inter_stim_wait_s)

                            query_start = stim_time
                            query_stop = datetime_now()
                            spike_df = self.database.get_spike_event(
                                query_start, query_stop, self.experiment.exp_name
                            )

                            responded, latency_ms = self._check_response(
                                spike_df, resp_elec, stim_time, self.response_window_ms
                            )
                            if responded:
                                hits += 1
                                latencies.append(latency_ms)

                            self._trial_results.append(TrialResult(
                                stage="stage2",
                                stim_electrode=stim_elec,
                                resp_electrode=resp_elec,
                                amplitude_ua=amplitude,
                                duration_us=duration,
                                polarity=polarity_str,
                                trial_index=rep,
                                responded=responded,
                                latency_ms=latency_ms if responded else float('nan'),
                                timestamp_utc=stim_time.isoformat(),
                            ))

                        response_rate = hits / self.opt_repeats
                        median_latency = float(np.median(latencies)) if latencies else float('nan')

                        param_results.append({
                            "amplitude_ua": amplitude,
                            "duration_us": duration,
                            "polarity": polarity_str,
                            "response_rate": response_rate,
                            "median_latency_ms": median_latency,
                        })

                        if response_rate > best_rate:
                            best_rate = response_rate
                            best_amplitude = amplitude
                            best_duration = duration
                            best_polarity_str = polarity_str
                            best_latencies = latencies

                        logger.info("Pair (%d->%d) amp=%.1f dur=%.0f pol=%s: rate=%.2f",
                                    stim_elec, resp_elec, amplitude, duration, polarity_str, response_rate)

            median_lat = float(np.median(best_latencies)) if best_latencies else 20.0
            self._optimal_params[pair_key] = {
                "stim_electrode": stim_elec,
                "resp_electrode": resp_elec,
                "amplitude_ua": best_amplitude,
                "duration_us": best_duration,
                "polarity": best_polarity_str,
                "response_rate": best_rate,
                "median_latency_ms": median_lat,
            }
            self._median_latencies[pair_key] = median_lat

            self._stage2_results[pair_key] = {
                "stim_electrode": stim_elec,
                "resp_electrode": resp_elec,
                "best_amplitude_ua": best_amplitude,
                "best_duration_us": best_duration,
                "best_polarity": best_polarity_str,
                "best_response_rate": best_rate,
                "best_median_latency_ms": median_lat,
                "all_param_results": param_results,
            }

            logger.info("Pair (%d->%d) best: amp=%.1f dur=%.0f pol=%s rate=%.2f lat=%.1f ms",
                        stim_elec, resp_elec, best_amplitude, best_duration,
                        best_polarity_str, best_rate, median_lat)

    def _stage3_stdp_induction(self) -> None:
        logger.info("Stage 3: STDP induction for %d pairs, %d repetitions at %.2f Hz",
                    len(self._selected_pairs), self.stdp_repetitions, self.stdp_frequency_hz)

        inter_stim_interval_s = 1.0 / self.stdp_frequency_hz

        for pair in self._selected_pairs:
            stim_elec = pair["stim_electrode"]
            resp_elec = pair["resp_electrode"]
            pair_key = f"{stim_elec}_{resp_elec}"

            opt = self._optimal_params.get(pair_key, {})
            amplitude = opt.get("amplitude_ua", self.screening_amplitude_ua)
            duration = opt.get("duration_us", self.screening_duration_us)
            polarity_str = opt.get("polarity", "PositiveFirst")
            polarity_enum = StimPolarity.PositiveFirst if polarity_str == "PositiveFirst" else StimPolarity.NegativeFirst

            median_latency_ms = self._median_latencies.get(pair_key, 20.0)
            if math.isnan(median_latency_ms) or median_latency_ms <= 0:
                median_latency_ms = 20.0

            paired_delay_s = median_latency_ms / 1000.0

            logger.info("STDP pair (%d->%d): amp=%.1f dur=%.0f pol=%s delay=%.1f ms",
                        stim_elec, resp_elec, amplitude, duration, polarity_str, median_latency_ms)

            pair_stdp_results = []

            for rep in range(self.stdp_repetitions):
                stim_time = datetime_now()

                self._send_single_stim(
                    electrode_idx=stim_elec,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    polarity=polarity_enum,
                    trigger_key=0,
                    stage="stage3_pre",
                )

                self._wait(paired_delay_s)

                self._send_single_stim(
                    electrode_idx=resp_elec,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    polarity=polarity_enum,
                    trigger_key=1,
                    stage="stage3_post",
                )

                elapsed_s = (datetime_now() - stim_time).total_seconds()
                remaining_s = inter_stim_interval_s - elapsed_s - 0.05
                if remaining_s > 0:
                    self._wait(remaining_s)

                query_start = stim_time
                query_stop = datetime_now()
                spike_df = self.database.get_spike_event(
                    query_start, query_stop, self.experiment.exp_name
                )

                responded, latency_ms = self._check_response(
                    spike_df, resp_elec, stim_time, self.response_window_ms
                )

                pair_stdp_results.append({
                    "rep": rep,
                    "responded": responded,
                    "latency_ms": latency_ms if responded else float('nan'),
                    "timestamp_utc": stim_time.isoformat(),
                })

                self._trial_results.append(TrialResult(
                    stage="stage3",
                    stim_electrode=stim_elec,
                    resp_electrode=resp_elec,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    polarity=polarity_str,
                    trial_index=rep,
                    responded=responded,
                    latency_ms=latency_ms if responded else float('nan'),
                    timestamp_utc=stim_time.isoformat(),
                ))

                if (rep + 1) % 20 == 0:
                    logger.info("STDP pair (%d->%d): rep %d/%d",
                                stim_elec, resp_elec, rep + 1, self.stdp_repetitions)

            total_responses = sum(1 for r in pair_stdp_results if r["responded"])
            self._stage3_results[pair_key] = {
                "stim_electrode": stim_elec,
                "resp_electrode": resp_elec,
                "amplitude_ua": amplitude,
                "duration_us": duration,
                "polarity": polarity_str,
                "paired_delay_ms": median_latency_ms,
                "total_repetitions": self.stdp_repetitions,
                "total_responses": total_responses,
                "response_rate": total_responses / self.stdp_repetitions,
                "trial_results": pair_stdp_results,
            }

    def _stage4_validation(self) -> None:
        logger.info("Stage 4: Validation screening (repeat Stage 1 protocol)")

        amplitude = self.screening_amplitude_ua
        duration = self.screening_duration_us

        validation_pair_scores: Dict[str, Dict[str, Any]] = {}

        for pair in self._selected_pairs:
            stim_elec = pair["stim_electrode"]
            resp_elec = pair["resp_electrode"]
            pair_key = f"{stim_elec}_{resp_elec}"

            hits = 0
            latencies = []

            for rep in range(self.screening_repeats):
                polarity = StimPolarity.PositiveFirst
                stim_time = datetime_now()
                self._send_single_stim(
                    electrode_idx=stim_elec,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    polarity=polarity,
                    trigger_key=0,
                    stage="stage4",
                )
                self._wait(self.inter_stim_wait_s)

                query_start = stim_time
                query_stop = datetime_now()
                spike_df = self.database.get_spike_event(
                    query_start, query_stop, self.experiment.exp_name
                )

                responded, latency_ms = self._check_response(
                    spike_df, resp_elec, stim_time, self.response_window_ms
                )
                if responded:
                    hits += 1
                    latencies.append(latency_ms)

                self._trial_results.append(TrialResult(
                    stage="stage4",
                    stim_electrode=stim_elec,
                    resp_electrode=resp_elec,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    polarity="PositiveFirst",
                    trial_index=rep,
                    responded=responded,
                    latency_ms=latency_ms if responded else float('nan'),
                    timestamp_utc=stim_time.isoformat(),
                ))

            response_rate_post = hits / self.screening_repeats
            median_latency_post = float(np.median(latencies)) if latencies else float('nan')

            stage1_pair = next(
                (p for p in self._stage1_results.get("selected_pairs", [])
                 if p["stim_electrode"] == stim_elec and p["resp_electrode"] == resp_elec),
                None
            )
            response_rate_pre = stage1_pair["response_rate"] if stage1_pair else float('nan')
            median_latency_pre = stage1_pair["median_latency_ms"] if stage1_pair else float('nan')

            pre_latencies = []
            for tr in self._trial_results:
                if (tr.stage == "stage1" and tr.stim_electrode == stim_elec
                        and tr.resp_electrode == resp_elec and tr.responded):
                    pre_latencies.append(tr.latency_ms)

            wasserstein_dist = self._wasserstein_distance(pre_latencies, latencies)

            rate_change = response_rate_post - response_rate_pre if not math.isnan(response_rate_pre) else float('nan')

            validation_pair_scores[pair_key] = {
                "stim_electrode": stim_elec,
                "resp_electrode": resp_elec,
                "response_rate_pre": response_rate_pre,
                "response_rate_post": response_rate_post,
                "response_rate_change": rate_change,
                "median_latency_pre_ms": median_latency_pre,
                "median_latency_post_ms": median_latency_post,
                "wasserstein_distance": wasserstein_dist,
                "pre_latencies": pre_latencies,
                "post_latencies": latencies,
            }

            logger.info("Validation pair (%d->%d): pre_rate=%.2f post_rate=%.2f change=%.2f wasserstein=%.3f",
                        stim_elec, resp_elec, response_rate_pre, response_rate_post,
                        rate_change if not math.isnan(rate_change) else 0.0, wasserstein_dist)

        self._stage4_results = {
            "pair_comparisons": validation_pair_scores,
            "summary": {
                "n_pairs": len(validation_pair_scores),
                "mean_rate_change": float(np.mean([
                    v["response_rate_change"]
                    for v in validation_pair_scores.values()
                    if not math.isnan(v["response_rate_change"])
                ])) if validation_pair_scores else float('nan'),
                "mean_wasserstein": float(np.mean([
                    v["wasserstein_distance"]
                    for v in validation_pair_scores.values()
                ])) if validation_pair_scores else float('nan'),
            }
        }

    def _wasserstein_distance(self, dist_a: List[float], dist_b: List[float]) -> float:
        if not dist_a or not dist_b:
            return float('nan')
        a = sorted([x for x in dist_a if not math.isnan(x)])
        b = sorted([x for x in dist_b if not math.isnan(x)])
        if not a or not b:
            return float('nan')
        n = len(a)
        m = len(b)
        all_vals = sorted(set(a + b))
        cdf_a = []
        cdf_b = []
        for v in all_vals:
            cdf_a.append(sum(1 for x in a if x <= v) / n)
            cdf_b.append(sum(1 for x in b if x <= v) / m)
        if len(all_vals) < 2:
            return abs(cdf_a[0] - cdf_b[0]) if all_vals else 0.0
        dist = 0.0
        for i in range(len(all_vals) - 1):
            width = all_vals[i + 1] - all_vals[i]
            dist += abs(cdf_a[i] - cdf_b[i]) * width
        return float(dist)

    def _send_single_stim(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity,
        trigger_key: int = 0,
        stage: str = "",
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

        polarity_str = "PositiveFirst" if polarity == StimPolarity.PositiveFirst else "NegativeFirst"
        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            polarity=polarity_str,
            timestamp_utc=datetime_now().isoformat(),
            trigger_key=trigger_key,
            stage=stage,
        ))

    def _check_response(
        self,
        spike_df: pd.DataFrame,
        resp_electrode: int,
        stim_time: datetime,
        window_ms: float,
    ) -> Tuple[bool, float]:
        if spike_df is None or spike_df.empty:
            return False, float('nan')

        channel_col = None
        for col in ["channel", "index", "electrode"]:
            if col in spike_df.columns:
                channel_col = col
                break

        if channel_col is None:
            return False, float('nan')

        time_col = None
        for col in ["Time", "time", "_time"]:
            if col in spike_df.columns:
                time_col = col
                break

        if time_col is None:
            return False, float('nan')

        resp_spikes = spike_df[spike_df[channel_col] == resp_electrode]
        if resp_spikes.empty:
            return False, float('nan')

        window_end = stim_time + timedelta(milliseconds=window_ms)
        artifact_end = stim_time + timedelta(milliseconds=2.0)

        for _, row in resp_spikes.iterrows():
            spike_time = row[time_col]
            if hasattr(spike_time, 'tzinfo') and spike_time.tzinfo is None:
                spike_time = spike_time.replace(tzinfo=timezone.utc)
            if artifact_end <= spike_time <= window_end:
                latency_ms = (spike_time - stim_time).total_seconds() * 1000.0
                return True, latency_ms

        return False, float('nan')

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

        trial_records = [asdict(t) for t in self._trial_results]

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "total_trials": len(self._trial_results),
            "stage1_selected_pairs": self._stage1_results.get("selected_pairs", []),
            "stage2_optimal_params": {k: {
                "amplitude_ua": v["amplitude_ua"],
                "duration_us": v["duration_us"],
                "polarity": v["polarity"],
                "response_rate": v["response_rate"],
            } for k, v in self._optimal_params.items()},
            "stage3_summary": {k: {
                "response_rate": v["response_rate"],
                "total_responses": v["total_responses"],
            } for k, v in self._stage3_results.items()},
            "stage4_summary": self._stage4_results.get("summary", {}),
        }
        saver.save_summary(summary)

        saver.save_stage_data("stage1_results", self._stage1_results)
        saver.save_stage_data("stage2_results", self._stage2_results)
        saver.save_stage_data("stage3_results", {
            k: {
                "stim_electrode": v["stim_electrode"],
                "resp_electrode": v["resp_electrode"],
                "amplitude_ua": v["amplitude_ua"],
                "duration_us": v["duration_us"],
                "polarity": v["polarity"],
                "paired_delay_ms": v["paired_delay_ms"],
                "total_repetitions": v["total_repetitions"],
                "total_responses": v["total_responses"],
                "response_rate": v["response_rate"],
            }
            for k, v in self._stage3_results.items()
        })
        saver.save_stage_data("stage4_results", self._stage4_results)
        saver.save_stage_data("trial_results", trial_records)

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
        if spike_df is None or spike_df.empty:
            return waveform_records

        channel_col = None
        for col in ["channel", "index", "electrode"]:
            if col in spike_df.columns:
                channel_col = col
                break

        if channel_col is None:
            logger.warning("Cannot determine electrode column for waveform fetch")
            return waveform_records

        unique_electrodes = spike_df[channel_col].unique()
        for electrode_idx in unique_electrodes:
            try:
                raw_df = self.database.get_raw_spike(
                    recording_start, recording_stop, int(electrode_idx)
                )
                if raw_df is not None and not raw_df.empty:
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

        stage4_summary = self._stage4_results.get("summary", {})
        pair_comparisons = self._stage4_results.get("pair_comparisons", {})

        potentiated = [
            k for k, v in pair_comparisons.items()
            if not math.isnan(v.get("response_rate_change", float('nan')))
            and v["response_rate_change"] > 0
        ]
        depressed = [
            k for k, v in pair_comparisons.items()
            if not math.isnan(v.get("response_rate_change", float('nan')))
            and v["response_rate_change"] < 0
        ]

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "stage1": {
                "n_selected_pairs": len(self._selected_pairs),
                "selected_pairs": [
                    {"stim": p["stim_electrode"], "resp": p["resp_electrode"],
                     "rate": p["response_rate"]}
                    for p in self._selected_pairs
                ],
            },
            "stage2": {
                "n_pairs_optimized": len(self._optimal_params),
                "optimal_params": {
                    k: {
                        "amplitude_ua": v["amplitude_ua"],
                        "duration_us": v["duration_us"],
                        "polarity": v["polarity"],
                        "response_rate": v["response_rate"],
                        "median_latency_ms": v["median_latency_ms"],
                    }
                    for k, v in self._optimal_params.items()
                },
            },
            "stage3": {
                "stdp_repetitions": self.stdp_repetitions,
                "frequency_hz": self.stdp_frequency_hz,
                "pair_response_rates": {
                    k: v["response_rate"]
                    for k, v in self._stage3_results.items()
                },
            },
            "stage4": {
                "n_pairs_validated": len(pair_comparisons),
                "mean_rate_change": stage4_summary.get("mean_rate_change", float('nan')),
                "mean_wasserstein_distance": stage4_summary.get("mean_wasserstein", float('nan')),
                "n_potentiated": len(potentiated),
                "n_depressed": len(depressed),
                "pair_comparisons": {
                    k: {
                        "response_rate_pre": v["response_rate_pre"],
                        "response_rate_post": v["response_rate_post"],
                        "response_rate_change": v["response_rate_change"],
                        "wasserstein_distance": v["wasserstein_distance"],
                    }
                    for k, v in pair_comparisons.items()
                },
            },
            "total_stimulations": len(self._stimulation_log),
            "total_trials": len(self._trial_results),
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
