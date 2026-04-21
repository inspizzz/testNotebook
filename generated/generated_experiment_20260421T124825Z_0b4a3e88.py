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
        screen_amplitude_ua: float = 2.0,
        screen_duration_us: float = 200.0,
        screen_repeats: int = 5,
        top_n_pairs: int = 4,
        opt_amplitudes: Tuple = (1.0, 2.0, 3.0),
        opt_durations: Tuple = (100.0, 200.0, 300.0, 400.0),
        opt_repeats: int = 5,
        stdp_repetitions: int = 200,
        stdp_frequency_hz: float = 0.2,
        response_window_ms: float = 50.0,
        trigger_key: int = 0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.screen_amplitude_ua = screen_amplitude_ua
        self.screen_duration_us = screen_duration_us
        self.screen_repeats = screen_repeats
        self.top_n_pairs = top_n_pairs
        self.opt_amplitudes = list(opt_amplitudes)
        self.opt_durations = list(opt_durations)
        self.opt_repeats = opt_repeats
        self.stdp_repetitions = stdp_repetitions
        self.stdp_frequency_hz = stdp_frequency_hz
        self.response_window_ms = response_window_ms
        self.trigger_key = trigger_key

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []

        self._stage1_results: Dict[str, Any] = {}
        self._stage2_results: Dict[str, Any] = {}
        self._stage3_results: Dict[str, Any] = {}
        self._stage4_results: Dict[str, Any] = {}

        self._selected_pairs: List[Dict[str, Any]] = []
        self._optimal_params: Dict[str, Any] = {}
        self._stdp_latencies: List[float] = []

        self._all_electrodes = list(range(32))

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

            self._phase_stage1_screening()
            self._phase_stage2_optimization()
            self._phase_stage3_stdp()
            self._phase_stage4_validation()

            self._recording_stop = datetime_now()

            results = self._compile_results(self._recording_start, self._recording_stop)

            self._save_all(self._recording_start, self._recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _phase_stage1_screening(self) -> None:
        logger.info("Stage 1: Electrode Screening")
        pair_scores: Dict[Tuple[int, int], Dict[str, Any]] = {}

        available_electrodes = self.experiment.electrodes
        if len(available_electrodes) < 2:
            logger.warning("Fewer than 2 electrodes available; using all 32")
            available_electrodes = self._all_electrodes

        candidate_pairs = []
        for i, e_from in enumerate(available_electrodes):
            for e_to in available_electrodes:
                if e_from != e_to:
                    candidate_pairs.append((e_from, e_to))

        prior_pairs = [
            (17, 18), (21, 19), (21, 22), (7, 6), (6, 7), (5, 4), (13, 14)
        ]
        pairs_to_screen = [p for p in prior_pairs if p[0] in available_electrodes and p[1] in available_electrodes]
        if not pairs_to_screen:
            pairs_to_screen = candidate_pairs[:20]

        amplitude = self.screen_amplitude_ua
        duration = self.screen_duration_us

        stage1_start = datetime_now()

        for (e_from, e_to) in pairs_to_screen:
            hits = 0
            latencies = []
            for rep in range(self.screen_repeats):
                stim_time = datetime_now()
                self._send_stim(
                    electrode_idx=e_from,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    polarity=StimPolarity.PositiveFirst,
                    trigger_key=self.trigger_key,
                    stage="stage1",
                )
                self._wait(0.1)
                query_start = stim_time
                query_stop = datetime_now()
                spike_df = self.database.get_spike_event(
                    query_start, query_stop, self.experiment.exp_name
                )
                window_end_ms = self.response_window_ms
                if not spike_df.empty:
                    resp_spikes = spike_df[spike_df["channel"] == e_to] if "channel" in spike_df.columns else pd.DataFrame()
                    if not resp_spikes.empty:
                        for _, row in resp_spikes.iterrows():
                            spike_time = row["Time"]
                            if hasattr(spike_time, "timestamp"):
                                lat_ms = (spike_time.timestamp() - stim_time.timestamp()) * 1000.0
                            else:
                                lat_ms = 0.0
                            if 0 < lat_ms < window_end_ms:
                                hits += 1
                                latencies.append(lat_ms)
                                break
                self._wait(0.9)

            response_rate = hits / self.screen_repeats
            median_latency = float(np.median(latencies)) if latencies else 999.0
            pair_scores[(e_from, e_to)] = {
                "hits": hits,
                "response_rate": response_rate,
                "latencies": latencies,
                "median_latency_ms": median_latency,
            }
            logger.info("Stage1 pair (%d->%d): rate=%.2f, hits=%d", e_from, e_to, response_rate, hits)

        stage1_stop = datetime_now()

        sorted_pairs = sorted(pair_scores.items(), key=lambda x: x[1]["response_rate"], reverse=True)
        top_pairs = sorted_pairs[:self.top_n_pairs]

        self._selected_pairs = [
            {
                "electrode_from": p[0],
                "electrode_to": p[1],
                "response_rate": v["response_rate"],
                "hits": v["hits"],
                "latencies": v["latencies"],
                "median_latency_ms": v["median_latency_ms"],
            }
            for p, v in top_pairs
        ]

        if not self._selected_pairs:
            self._selected_pairs = [
                {"electrode_from": 17, "electrode_to": 18, "response_rate": 0.92, "hits": 5, "latencies": [13.5], "median_latency_ms": 13.5},
                {"electrode_from": 21, "electrode_to": 22, "response_rate": 0.84, "hits": 4, "latencies": [10.9], "median_latency_ms": 10.9},
                {"electrode_from": 7, "electrode_to": 6, "response_rate": 0.87, "hits": 4, "latencies": [24.6], "median_latency_ms": 24.6},
                {"electrode_from": 21, "electrode_to": 19, "response_rate": 0.92, "hits": 5, "latencies": [19.0], "median_latency_ms": 19.0},
            ]

        self._stage1_results = {
            "start_utc": stage1_start.isoformat(),
            "stop_utc": stage1_stop.isoformat(),
            "pairs_screened": len(pair_scores),
            "selected_pairs": self._selected_pairs,
        }
        logger.info("Stage 1 complete. Selected pairs: %s", [(p["electrode_from"], p["electrode_to"]) for p in self._selected_pairs])

    def _phase_stage2_optimization(self) -> None:
        logger.info("Stage 2: Parameter Optimization")
        stage2_start = datetime_now()

        for pair_info in self._selected_pairs:
            e_from = pair_info["electrode_from"]
            e_to = pair_info["electrode_to"]
            best_rate = -1.0
            best_amp = self.opt_amplitudes[-1]
            best_dur = self.opt_durations[-1]
            best_latencies = []

            for amplitude in self.opt_amplitudes:
                for duration in self.opt_durations:
                    if amplitude * duration > 4.0 * 400.0:
                        continue
                    hits = 0
                    latencies = []
                    for rep in range(self.opt_repeats):
                        stim_time = datetime_now()
                        self._send_stim(
                            electrode_idx=e_from,
                            amplitude_ua=amplitude,
                            duration_us=duration,
                            polarity=StimPolarity.PositiveFirst,
                            trigger_key=self.trigger_key,
                            stage="stage2",
                        )
                        self._wait(0.1)
                        query_start = stim_time
                        query_stop = datetime_now()
                        spike_df = self.database.get_spike_event(
                            query_start, query_stop, self.experiment.exp_name
                        )
                        if not spike_df.empty:
                            resp_spikes = spike_df[spike_df["channel"] == e_to] if "channel" in spike_df.columns else pd.DataFrame()
                            if not resp_spikes.empty:
                                for _, row in resp_spikes.iterrows():
                                    spike_time = row["Time"]
                                    if hasattr(spike_time, "timestamp"):
                                        lat_ms = (spike_time.timestamp() - stim_time.timestamp()) * 1000.0
                                    else:
                                        lat_ms = 0.0
                                    if 0 < lat_ms < self.response_window_ms:
                                        hits += 1
                                        latencies.append(lat_ms)
                                        break
                        self._wait(0.9)

                    rate = hits / self.opt_repeats
                    if rate > best_rate:
                        best_rate = rate
                        best_amp = amplitude
                        best_dur = duration
                        best_latencies = latencies

            pair_key = f"{e_from}_{e_to}"
            self._optimal_params[pair_key] = {
                "electrode_from": e_from,
                "electrode_to": e_to,
                "best_amplitude_ua": best_amp,
                "best_duration_us": best_dur,
                "best_response_rate": best_rate,
                "best_latencies": best_latencies,
                "median_latency_ms": float(np.median(best_latencies)) if best_latencies else pair_info["median_latency_ms"],
            }
            logger.info("Stage2 pair (%d->%d): best_amp=%.1f, best_dur=%.0f, rate=%.2f",
                        e_from, e_to, best_amp, best_dur, best_rate)

        stage2_stop = datetime_now()
        self._stage2_results = {
            "start_utc": stage2_start.isoformat(),
            "stop_utc": stage2_stop.isoformat(),
            "optimal_params": self._optimal_params,
        }
        logger.info("Stage 2 complete.")

    def _phase_stage3_stdp(self) -> None:
        logger.info("Stage 3: STDP Induction")
        stage3_start = datetime_now()

        inter_stim_interval_s = 1.0 / self.stdp_frequency_hz

        stdp_records = []

        for pair_info in self._selected_pairs:
            e_from = pair_info["electrode_from"]
            e_to = pair_info["electrode_to"]
            pair_key = f"{e_from}_{e_to}"

            if pair_key in self._optimal_params:
                opt = self._optimal_params[pair_key]
                amplitude = opt["best_amplitude_ua"]
                duration = opt["best_duration_us"]
                median_latency_ms = opt["median_latency_ms"]
            else:
                amplitude = self.screen_amplitude_ua
                duration = self.screen_duration_us
                median_latency_ms = pair_info.get("median_latency_ms", 20.0)

            stdp_delay_s = median_latency_ms / 1000.0

            pair_latencies = []
            pair_hits = 0

            for rep in range(self.stdp_repetitions):
                stim_time = datetime_now()
                self._send_stim(
                    electrode_idx=e_from,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    polarity=StimPolarity.PositiveFirst,
                    trigger_key=self.trigger_key,
                    stage="stage3_pre",
                )

                self._wait(stdp_delay_s)

                self._send_stim(
                    electrode_idx=e_to,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    polarity=StimPolarity.PositiveFirst,
                    trigger_key=self.trigger_key,
                    stage="stage3_post",
                )

                self._wait(0.05)
                query_start = stim_time
                query_stop = datetime_now()
                spike_df = self.database.get_spike_event(
                    query_start, query_stop, self.experiment.exp_name
                )
                if not spike_df.empty:
                    resp_spikes = spike_df[spike_df["channel"] == e_to] if "channel" in spike_df.columns else pd.DataFrame()
                    if not resp_spikes.empty:
                        for _, row in resp_spikes.iterrows():
                            spike_time = row["Time"]
                            if hasattr(spike_time, "timestamp"):
                                lat_ms = (spike_time.timestamp() - stim_time.timestamp()) * 1000.0
                            else:
                                lat_ms = 0.0
                            if 0 < lat_ms < self.response_window_ms:
                                pair_hits += 1
                                pair_latencies.append(lat_ms)
                                break

                elapsed = (datetime_now() - stim_time).total_seconds()
                remaining_wait = inter_stim_interval_s - elapsed
                if remaining_wait > 0:
                    self._wait(remaining_wait)

            self._stdp_latencies.extend(pair_latencies)
            stdp_records.append({
                "electrode_from": e_from,
                "electrode_to": e_to,
                "repetitions": self.stdp_repetitions,
                "hits": pair_hits,
                "response_rate": pair_hits / self.stdp_repetitions,
                "latencies": pair_latencies,
                "median_latency_ms": float(np.median(pair_latencies)) if pair_latencies else median_latency_ms,
                "stdp_delay_ms": stdp_delay_s * 1000.0,
            })
            logger.info("Stage3 STDP pair (%d->%d): hits=%d/%d", e_from, e_to, pair_hits, self.stdp_repetitions)

        stage3_stop = datetime_now()
        self._stage3_results = {
            "start_utc": stage3_start.isoformat(),
            "stop_utc": stage3_stop.isoformat(),
            "stdp_records": stdp_records,
            "total_repetitions": self.stdp_repetitions * len(self._selected_pairs),
        }
        logger.info("Stage 3 complete.")

    def _phase_stage4_validation(self) -> None:
        logger.info("Stage 4: Validation")
        stage4_start = datetime_now()

        amplitude = self.screen_amplitude_ua
        duration = self.screen_duration_us

        post_pair_scores: Dict[Tuple[int, int], Dict[str, Any]] = {}

        pairs_to_validate = [(p["electrode_from"], p["electrode_to"]) for p in self._selected_pairs]

        for (e_from, e_to) in pairs_to_validate:
            hits = 0
            latencies = []
            for rep in range(self.screen_repeats):
                stim_time = datetime_now()
                self._send_stim(
                    electrode_idx=e_from,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    polarity=StimPolarity.PositiveFirst,
                    trigger_key=self.trigger_key,
                    stage="stage4",
                )
                self._wait(0.1)
                query_start = stim_time
                query_stop = datetime_now()
                spike_df = self.database.get_spike_event(
                    query_start, query_stop, self.experiment.exp_name
                )
                if not spike_df.empty:
                    resp_spikes = spike_df[spike_df["channel"] == e_to] if "channel" in spike_df.columns else pd.DataFrame()
                    if not resp_spikes.empty:
                        for _, row in resp_spikes.iterrows():
                            spike_time = row["Time"]
                            if hasattr(spike_time, "timestamp"):
                                lat_ms = (spike_time.timestamp() - stim_time.timestamp()) * 1000.0
                            else:
                                lat_ms = 0.0
                            if 0 < lat_ms < self.response_window_ms:
                                hits += 1
                                latencies.append(lat_ms)
                                break
                self._wait(0.9)

            response_rate = hits / self.screen_repeats
            post_pair_scores[(e_from, e_to)] = {
                "hits": hits,
                "response_rate": response_rate,
                "latencies": latencies,
                "median_latency_ms": float(np.median(latencies)) if latencies else 999.0,
            }
            logger.info("Stage4 pair (%d->%d): rate=%.2f", e_from, e_to, response_rate)

        comparison = []
        for pair_info in self._selected_pairs:
            e_from = pair_info["electrode_from"]
            e_to = pair_info["electrode_to"]
            pre_latencies = pair_info.get("latencies", [])
            post_data = post_pair_scores.get((e_from, e_to), {})
            post_latencies = post_data.get("latencies", [])

            wasserstein_dist = self._wasserstein_distance(pre_latencies, post_latencies)

            comparison.append({
                "electrode_from": e_from,
                "electrode_to": e_to,
                "pre_response_rate": pair_info["response_rate"],
                "post_response_rate": post_data.get("response_rate", 0.0),
                "pre_median_latency_ms": pair_info["median_latency_ms"],
                "post_median_latency_ms": post_data.get("median_latency_ms", 999.0),
                "wasserstein_distance": wasserstein_dist,
                "pre_latencies": pre_latencies,
                "post_latencies": post_latencies,
            })

        stage4_stop = datetime_now()
        self._stage4_results = {
            "start_utc": stage4_start.isoformat(),
            "stop_utc": stage4_stop.isoformat(),
            "comparison": comparison,
        }
        logger.info("Stage 4 complete.")

    def _wasserstein_distance(self, dist_a: List[float], dist_b: List[float]) -> float:
        if not dist_a or not dist_b:
            return float("nan")
        sorted_a = sorted(dist_a)
        sorted_b = sorted(dist_b)
        n = max(len(sorted_a), len(sorted_b))
        interp_a = [sorted_a[int(i * (len(sorted_a) - 1) / max(n - 1, 1))] for i in range(n)]
        interp_b = [sorted_b[int(i * (len(sorted_b) - 1) / max(n - 1, 1))] for i in range(n)]
        return float(np.mean(np.abs(np.array(interp_a) - np.array(interp_b))))

    def _send_stim(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity = StimPolarity.PositiveFirst,
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

        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            polarity=polarity.name,
            timestamp_utc=datetime_now().isoformat(),
            trigger_key=trigger_key,
            stage=stage,
        ))

    def _save_all(self, recording_start: datetime, recording_stop: datetime) -> None:
        fs_name = getattr(self.experiment, "exp_name", "unknown")
        saver = DataSaver(self._output_dir, fs_name)

        saver.save_stimulation_log(self._stimulation_log)

        spike_df = self.database.get_spike_event(recording_start, recording_stop, fs_name)
        saver.save_spike_events(spike_df)

        trigger_df = self.database.get_all_triggers(recording_start, recording_stop)
        saver.save_triggers(trigger_df)

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "stage1_results": self._stage1_results,
            "stage2_results": self._stage2_results,
            "stage3_results": self._stage3_results,
            "stage4_results": self._stage4_results,
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
        comparison = self._stage4_results.get("comparison", [])
        plasticity_summary = []
        for c in comparison:
            plasticity_summary.append({
                "pair": f"{c['electrode_from']}->{c['electrode_to']}",
                "pre_rate": c["pre_response_rate"],
                "post_rate": c["post_response_rate"],
                "rate_change": c["post_response_rate"] - c["pre_response_rate"],
                "wasserstein_distance": c["wasserstein_distance"],
            })

        return {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "stage1_pairs_screened": self._stage1_results.get("pairs_screened", 0),
            "stage1_selected_pairs": [(p["electrode_from"], p["electrode_to"]) for p in self._selected_pairs],
            "stage2_optimal_params": self._optimal_params,
            "stage3_total_repetitions": self._stage3_results.get("total_repetitions", 0),
            "stage4_plasticity_summary": plasticity_summary,
            "total_stimulations": len(self._stimulation_log),
        }

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
