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
    timestamp_utc: str
    trigger_key: int = 0
    trial_index: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialResult:
    electrode_idx: int
    trial_index: int
    amplitude_ua: float
    duration_us: float
    spike_count: int
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

    def save_analysis(self, analysis: Dict[str, Any]) -> Path:
        path = Path(f"{self._prefix}_analysis.json")
        path.write_text(json.dumps(analysis, indent=2, default=str))
        logger.info("Saved analysis -> %s", path)
        return path


class Experiment:
    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        scan_electrodes: tuple = (5, 6, 7, 13, 17, 21, 22, 19),
        stim_amplitude_ua: float = 2.0,
        stim_duration_us: float = 200.0,
        num_repetitions: int = 20,
        isi_seconds: float = 1.0,
        response_window_ms: float = 50.0,
        trigger_key: int = 0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.scan_electrodes = list(scan_electrodes)
        self.stim_amplitude_ua = min(abs(stim_amplitude_ua), 4.0)
        self.stim_duration_us = min(abs(stim_duration_us), 400.0)
        self.num_repetitions = num_repetitions
        self.isi_seconds = isi_seconds
        self.response_window_ms = response_window_ms
        self.trigger_key = trigger_key

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._trial_results: List[TrialResult] = []
        self._electrode_spike_counts: Dict[int, List[int]] = defaultdict(list)
        self._electrode_latencies: Dict[int, List[float]] = defaultdict(list)
        self._analysis_results: Dict[str, Any] = {}

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

            self._phase_excitability_scan()

            self._phase_analysis()

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
        logger.info("Phase: excitability scan on %d electrodes, %d reps at %.1f uA, %.0f us, %.1f Hz",
                    len(self.scan_electrodes), self.num_repetitions,
                    self.stim_amplitude_ua, self.stim_duration_us,
                    1.0 / self.isi_seconds)

        a1 = self.stim_amplitude_ua
        d1 = self.stim_duration_us
        d2 = d1
        a2 = a1

        for electrode_idx in self.scan_electrodes:
            logger.info("Scanning electrode %d", electrode_idx)
            stim = StimParam()
            stim.index = electrode_idx
            stim.enable = True
            stim.trigger_key = self.trigger_key
            stim.trigger_delay = 0
            stim.nb_pulse = 0
            stim.pulse_train_period = 10000
            stim.post_stim_ref_period = 1000.0
            stim.stim_shape = StimShape.Biphasic
            stim.polarity = StimPolarity.PositiveFirst
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

            self.intan.send_stimparam([stim])

            for trial_idx in range(self.num_repetitions):
                stim_time = datetime_now()

                pattern = np.zeros(16, dtype=np.uint8)
                pattern[self.trigger_key] = 1
                self.trigger_controller.send(pattern)
                self._wait(0.05)
                pattern[self.trigger_key] = 0
                self.trigger_controller.send(pattern)

                self._stimulation_log.append(StimulationRecord(
                    electrode_idx=electrode_idx,
                    amplitude_ua=a1,
                    duration_us=d1,
                    timestamp_utc=stim_time.isoformat(),
                    trigger_key=self.trigger_key,
                    trial_index=trial_idx,
                ))

                self._wait(self.isi_seconds - 0.05)

                query_start = stim_time
                query_stop = datetime_now()
                try:
                    spike_df = self.database.get_spike_event_electrode(
                        query_start, query_stop, electrode_idx
                    )
                except Exception as exc:
                    logger.warning("Spike query failed for electrode %d trial %d: %s",
                                   electrode_idx, trial_idx, exc)
                    spike_df = pd.DataFrame()

                spike_count = 0
                responded = False
                latency_ms = float('nan')

                if not spike_df.empty:
                    window_end = stim_time + timedelta(milliseconds=self.response_window_ms)
                    if 'Time' in spike_df.columns:
                        in_window = spike_df[spike_df['Time'] <= window_end]
                        spike_count = len(in_window)
                        if spike_count > 0:
                            responded = True
                            earliest = in_window['Time'].min()
                            delta_s = (earliest - stim_time).total_seconds()
                            latency_ms = delta_s * 1000.0

                self._electrode_spike_counts[electrode_idx].append(spike_count)
                if not math.isnan(latency_ms):
                    self._electrode_latencies[electrode_idx].append(latency_ms)

                self._trial_results.append(TrialResult(
                    electrode_idx=electrode_idx,
                    trial_index=trial_idx,
                    amplitude_ua=a1,
                    duration_us=d1,
                    spike_count=spike_count,
                    responded=responded,
                    latency_ms=latency_ms if not math.isnan(latency_ms) else -1.0,
                    timestamp_utc=stim_time.isoformat(),
                ))

                logger.info("  Electrode %d trial %d/%d: spikes=%d responded=%s latency=%.2f ms",
                            electrode_idx, trial_idx + 1, self.num_repetitions,
                            spike_count, responded, latency_ms if not math.isnan(latency_ms) else -1.0)

            self._wait(1.0)

    def _phase_analysis(self) -> None:
        logger.info("Phase: analysis")

        electrode_summaries = {}
        all_spike_counts_by_electrode: Dict[int, List[int]] = {}

        for electrode_idx in self.scan_electrodes:
            counts = self._electrode_spike_counts[electrode_idx]
            latencies = self._electrode_latencies[electrode_idx]
            n = len(counts)
            if n == 0:
                continue

            total_spikes = sum(counts)
            responded_trials = sum(1 for c in counts if c > 0)
            response_rate = responded_trials / n if n > 0 else 0.0

            mean_latency = float(np.mean(latencies)) if latencies else float('nan')
            std_latency = float(np.std(latencies)) if len(latencies) > 1 else float('nan')
            median_latency = float(np.median(latencies)) if latencies else float('nan')

            all_spike_counts_by_electrode[electrode_idx] = counts

            electrode_summaries[electrode_idx] = {
                "electrode_idx": electrode_idx,
                "n_trials": n,
                "total_spikes": total_spikes,
                "responded_trials": responded_trials,
                "response_rate": response_rate,
                "mean_latency_ms": mean_latency,
                "std_latency_ms": std_latency,
                "median_latency_ms": median_latency,
                "mean_spikes_per_trial": total_spikes / n,
            }

        mw_results = self._run_mann_whitney_comparisons(all_spike_counts_by_electrode)

        waveform_cluster_results = self._run_waveform_clustering()

        raster_data = self._build_raster_data()

        self._analysis_results = {
            "electrode_summaries": electrode_summaries,
            "mann_whitney_comparisons": mw_results,
            "waveform_clusters": waveform_cluster_results,
            "raster_data": raster_data,
        }

        logger.info("Analysis complete. Electrode summaries: %d", len(electrode_summaries))

    def _run_mann_whitney_comparisons(
        self, spike_counts_by_electrode: Dict[int, List[int]]
    ) -> List[Dict[str, Any]]:
        results = []
        electrodes = list(spike_counts_by_electrode.keys())
        for i in range(len(electrodes)):
            for j in range(i + 1, len(electrodes)):
                e1 = electrodes[i]
                e2 = electrodes[j]
                x = spike_counts_by_electrode[e1]
                y = spike_counts_by_electrode[e2]
                if len(x) < 2 or len(y) < 2:
                    continue
                u_stat, p_value = self._mann_whitney_u(x, y)
                results.append({
                    "electrode_a": e1,
                    "electrode_b": e2,
                    "u_statistic": u_stat,
                    "p_value": p_value,
                    "significant_p05": p_value < 0.05,
                    "n_a": len(x),
                    "n_b": len(y),
                })
        return results

    def _mann_whitney_u(self, x: List[float], y: List[float]) -> Tuple[float, float]:
        nx = len(x)
        ny = len(y)
        u1 = 0.0
        for xi in x:
            for yi in y:
                if xi > yi:
                    u1 += 1.0
                elif xi == yi:
                    u1 += 0.5
        u2 = nx * ny - u1
        u_stat = min(u1, u2)

        mean_u = nx * ny / 2.0
        std_u = math.sqrt(nx * ny * (nx + ny + 1) / 12.0) if (nx + ny + 1) > 0 else 1.0
        if std_u == 0:
            p_value = 1.0
        else:
            z = (u_stat - mean_u) / std_u
            p_value = 2.0 * self._norm_sf(abs(z))

        return u_stat, p_value

    def _norm_sf(self, z: float) -> float:
        return 0.5 * math.erfc(z / math.sqrt(2.0))

    def _run_waveform_clustering(self) -> Dict[str, Any]:
        results = {}
        for trial in self._trial_results:
            e = trial.electrode_idx
            if e not in results:
                results[e] = {"spike_counts": [], "latencies": []}
            results[e]["spike_counts"].append(trial.spike_count)
            if trial.latency_ms > 0:
                results[e]["latencies"].append(trial.latency_ms)

        cluster_results = {}
        for electrode_idx, data in results.items():
            latencies = data["latencies"]
            if len(latencies) < 3:
                cluster_results[electrode_idx] = {
                    "n_points": len(latencies),
                    "n_clusters": 0,
                    "cluster_assignments": [],
                    "cluster_centers": [],
                    "note": "insufficient data for clustering",
                }
                continue

            k = min(3, len(latencies))
            assignments, centers = self._kmeans_1d(latencies, k=k, max_iter=50)
            cluster_results[electrode_idx] = {
                "n_points": len(latencies),
                "n_clusters": k,
                "cluster_assignments": assignments,
                "cluster_centers": [float(c) for c in centers],
            }

        return cluster_results

    def _kmeans_1d(
        self, data: List[float], k: int = 3, max_iter: int = 50
    ) -> Tuple[List[int], List[float]]:
        if len(data) == 0 or k == 0:
            return [], []

        arr = sorted(data)
        n = len(arr)
        step = max(1, n // k)
        centers = [float(arr[min(i * step, n - 1)]) for i in range(k)]

        assignments = [0] * n
        for _ in range(max_iter):
            new_assignments = []
            for val in data:
                dists = [abs(val - c) for c in centers]
                new_assignments.append(dists.index(min(dists)))

            if new_assignments == assignments:
                break
            assignments = new_assignments

            new_centers = []
            for ci in range(k):
                cluster_vals = [data[i] for i in range(len(data)) if assignments[i] == ci]
                if cluster_vals:
                    new_centers.append(float(np.mean(cluster_vals)))
                else:
                    new_centers.append(centers[ci])
            centers = new_centers

        return assignments, centers

    def _build_raster_data(self) -> Dict[str, Any]:
        raster: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for trial in self._trial_results:
            raster[trial.electrode_idx].append({
                "trial_index": trial.trial_index,
                "spike_count": trial.spike_count,
                "responded": trial.responded,
                "latency_ms": trial.latency_ms,
                "timestamp_utc": trial.timestamp_utc,
            })
        return {str(k): v for k, v in raster.items()}

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

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "scan_electrodes": self.scan_electrodes,
            "stim_amplitude_ua": self.stim_amplitude_ua,
            "stim_duration_us": self.stim_duration_us,
            "num_repetitions": self.num_repetitions,
            "isi_seconds": self.isi_seconds,
            "total_trial_results": len(self._trial_results),
        }
        saver.save_summary(summary)

        waveform_records = self._fetch_spike_waveforms(
            fs_name, spike_df, recording_start, recording_stop
        )
        saver.save_spike_waveforms(waveform_records)

        saver.save_analysis(self._analysis_results)

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

        electrode_response_rates = {}
        for electrode_idx in self.scan_electrodes:
            counts = self._electrode_spike_counts[electrode_idx]
            if counts:
                responded = sum(1 for c in counts if c > 0)
                electrode_response_rates[electrode_idx] = responded / len(counts)
            else:
                electrode_response_rates[electrode_idx] = 0.0

        most_responsive = max(electrode_response_rates, key=electrode_response_rates.get) \
            if electrode_response_rates else None
        least_responsive = min(electrode_response_rates, key=electrode_response_rates.get) \
            if electrode_response_rates else None

        mw_significant = [
            r for r in self._analysis_results.get("mann_whitney_comparisons", [])
            if r.get("significant_p05", False)
        ]

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": getattr(self.experiment, "exp_name", "unknown"),
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "scan_electrodes": self.scan_electrodes,
            "stim_amplitude_ua": self.stim_amplitude_ua,
            "stim_duration_us": self.stim_duration_us,
            "num_repetitions": self.num_repetitions,
            "total_stimulations": len(self._stimulation_log),
            "total_trial_results": len(self._trial_results),
            "electrode_response_rates": electrode_response_rates,
            "most_responsive_electrode": most_responsive,
            "least_responsive_electrode": least_responsive,
            "mann_whitney_significant_pairs": len(mw_significant),
            "mann_whitney_total_pairs": len(self._analysis_results.get("mann_whitney_comparisons", [])),
            "waveform_cluster_electrodes": list(self._analysis_results.get("waveform_clusters", {}).keys()),
            "analysis_summary": {
                "electrode_summaries_count": len(
                    self._analysis_results.get("electrode_summaries", {})
                ),
                "mw_comparisons_count": len(
                    self._analysis_results.get("mann_whitney_comparisons", [])
                ),
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
