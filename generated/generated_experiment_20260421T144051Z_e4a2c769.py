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
    stim_time_utc: str
    spike_count: int
    responded: bool
    latencies_ms: List[float] = field(default_factory=list)
    amplitudes_uv: List[float] = field(default_factory=list)


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
        stim_amplitude_ua: float = 2.0,
        stim_duration_us: float = 200.0,
        num_repetitions: int = 20,
        isi_seconds: float = 1.0,
        response_window_ms: float = 50.0,
        electrodes: Tuple = (4, 5, 6, 7, 13, 17, 21, 22),
        trigger_key: int = 0,
        n_waveform_clusters: int = 3,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.stim_amplitude_ua = float(stim_amplitude_ua)
        self.stim_duration_us = float(stim_duration_us)
        self.num_repetitions = int(num_repetitions)
        self.isi_seconds = float(isi_seconds)
        self.response_window_ms = float(response_window_ms)
        self.electrodes = list(electrodes)
        self.trigger_key = int(trigger_key)
        self.n_waveform_clusters = int(n_waveform_clusters)

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._trial_results: List[TrialResult] = []
        self._electrode_trial_results: Dict[int, List[TrialResult]] = defaultdict(list)
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
        logger.info("Phase: excitability scan on %d electrodes, %d repetitions at %.1f Hz",
                    len(self.electrodes), self.num_repetitions, 1.0 / self.isi_seconds)

        amplitude = min(self.stim_amplitude_ua, 4.0)
        duration = min(self.stim_duration_us, 400.0)

        for electrode_idx in self.electrodes:
            logger.info("Scanning electrode %d", electrode_idx)
            for trial_idx in range(self.num_repetitions):
                stim_time = datetime_now()
                spike_df = self._stimulate_and_record(
                    electrode_idx=electrode_idx,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    polarity=StimPolarity.PositiveFirst,
                    trigger_key=self.trigger_key,
                    post_stim_wait_s=self.response_window_ms / 1000.0 + 0.05,
                    recording_window_s=self.response_window_ms / 1000.0 + 0.1,
                    trial_index=trial_idx,
                )

                latencies_ms = []
                amplitudes_uv = []
                stim_time_ms = stim_time.timestamp() * 1000.0

                if not spike_df.empty:
                    for _, row in spike_df.iterrows():
                        try:
                            spike_t = row["Time"]
                            if hasattr(spike_t, "timestamp"):
                                spike_t_ms = spike_t.timestamp() * 1000.0
                            else:
                                spike_t_ms = float(spike_t) * 1000.0
                            lat = spike_t_ms - stim_time_ms
                            if 0.0 < lat <= self.response_window_ms:
                                latencies_ms.append(lat)
                                if "Amplitude" in row:
                                    amplitudes_uv.append(float(row["Amplitude"]))
                        except Exception:
                            pass

                responded = len(latencies_ms) > 0
                trial = TrialResult(
                    electrode_idx=electrode_idx,
                    trial_index=trial_idx,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    stim_time_utc=stim_time.isoformat(),
                    spike_count=len(latencies_ms),
                    responded=responded,
                    latencies_ms=latencies_ms,
                    amplitudes_uv=amplitudes_uv,
                )
                self._trial_results.append(trial)
                self._electrode_trial_results[electrode_idx].append(trial)

                remaining = self.isi_seconds - (self.response_window_ms / 1000.0 + 0.05)
                if remaining > 0:
                    self._wait(remaining)

            logger.info("Electrode %d: %d/%d trials responded",
                        electrode_idx,
                        sum(1 for t in self._electrode_trial_results[electrode_idx] if t.responded),
                        self.num_repetitions)

    def _phase_analysis(self) -> None:
        logger.info("Phase: analysis")
        self._analysis_results = {}

        electrode_stats = {}
        for electrode_idx in self.electrodes:
            trials = self._electrode_trial_results[electrode_idx]
            if not trials:
                continue

            response_rate = sum(1 for t in trials if t.responded) / len(trials)
            all_latencies = []
            all_amplitudes = []
            for t in trials:
                all_latencies.extend(t.latencies_ms)
                all_amplitudes.extend(t.amplitudes_uv)

            mean_latency = float(np.mean(all_latencies)) if all_latencies else 0.0
            std_latency = float(np.std(all_latencies)) if all_latencies else 0.0
            mean_amplitude = float(np.mean(all_amplitudes)) if all_amplitudes else 0.0

            waveform_cluster_info = self._cluster_waveforms(electrode_idx, all_amplitudes)

            electrode_stats[electrode_idx] = {
                "response_rate": response_rate,
                "total_spikes": len(all_latencies),
                "mean_latency_ms": mean_latency,
                "std_latency_ms": std_latency,
                "mean_amplitude_uv": mean_amplitude,
                "waveform_clusters": waveform_cluster_info,
            }

        self._analysis_results["electrode_stats"] = electrode_stats

        mw_results = self._mann_whitney_comparisons(electrode_stats)
        self._analysis_results["mann_whitney_tests"] = mw_results

        raster_data = self._build_raster_data()
        self._analysis_results["raster_data"] = raster_data

        logger.info("Analysis complete for %d electrodes", len(electrode_stats))

    def _cluster_waveforms(self, electrode_idx: int, amplitudes: List[float]) -> Dict[str, Any]:
        if len(amplitudes) < self.n_waveform_clusters:
            return {"n_clusters": 0, "cluster_centers": [], "cluster_labels": [], "note": "insufficient data"}

        n_clusters = min(self.n_waveform_clusters, len(amplitudes))
        data = np.array(amplitudes).reshape(-1, 1)

        centers = np.linspace(np.min(data), np.max(data), n_clusters)
        labels = []
        for _ in range(10):
            new_labels = []
            for val in data.flatten():
                dists = [abs(val - c) for c in centers]
                new_labels.append(int(np.argmin(dists)))
            labels = new_labels
            new_centers = []
            for k in range(n_clusters):
                cluster_vals = [data.flatten()[i] for i, l in enumerate(labels) if l == k]
                if cluster_vals:
                    new_centers.append(float(np.mean(cluster_vals)))
                else:
                    new_centers.append(float(centers[k]))
            centers = new_centers

        cluster_counts = [labels.count(k) for k in range(n_clusters)]
        dominant = int(np.argmax(cluster_counts))
        dominant_fraction = cluster_counts[dominant] / len(labels) if labels else 0.0

        return {
            "n_clusters": n_clusters,
            "cluster_centers_uv": [float(c) for c in centers],
            "cluster_counts": cluster_counts,
            "dominant_cluster": dominant,
            "dominant_cluster_fraction": float(dominant_fraction),
            "cluster_labels": labels,
        }

    def _mann_whitney_comparisons(self, electrode_stats: Dict[int, Any]) -> List[Dict[str, Any]]:
        results = []
        electrode_list = [e for e in self.electrodes if e in electrode_stats]

        for i in range(len(electrode_list)):
            for j in range(i + 1, len(electrode_list)):
                e1 = electrode_list[i]
                e2 = electrode_list[j]

                trials1 = self._electrode_trial_results[e1]
                trials2 = self._electrode_trial_results[e2]

                lat1 = []
                for t in trials1:
                    lat1.extend(t.latencies_ms)
                lat2 = []
                for t in trials2:
                    lat2.extend(t.latencies_ms)

                if len(lat1) < 2 or len(lat2) < 2:
                    results.append({
                        "electrode_a": e1,
                        "electrode_b": e2,
                        "u_statistic": None,
                        "p_value": None,
                        "note": "insufficient data",
                    })
                    continue

                u_stat, p_val = self._mann_whitney_u(lat1, lat2)
                results.append({
                    "electrode_a": e1,
                    "electrode_b": e2,
                    "u_statistic": float(u_stat),
                    "p_value": float(p_val),
                    "significant_p05": bool(p_val < 0.05),
                    "n1": len(lat1),
                    "n2": len(lat2),
                })

        return results

    def _mann_whitney_u(self, x: List[float], y: List[float]) -> Tuple[float, float]:
        n1 = len(x)
        n2 = len(y)
        u1 = 0.0
        for xi in x:
            for yi in y:
                if xi > yi:
                    u1 += 1.0
                elif xi == yi:
                    u1 += 0.5
        u2 = n1 * n2 - u1
        u_stat = min(u1, u2)

        mu_u = n1 * n2 / 2.0
        sigma_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
        if sigma_u == 0:
            return u_stat, 1.0

        z = (u_stat - mu_u) / sigma_u
        p_val = 2.0 * self._norm_cdf(-abs(z))
        return u_stat, p_val

    def _norm_cdf(self, z: float) -> float:
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def _build_raster_data(self) -> Dict[str, Any]:
        raster: Dict[int, List[Dict[str, Any]]] = {}
        for electrode_idx in self.electrodes:
            trials = self._electrode_trial_results[electrode_idx]
            raster[electrode_idx] = []
            for t in trials:
                raster[electrode_idx].append({
                    "trial_index": t.trial_index,
                    "latencies_ms": t.latencies_ms,
                    "responded": t.responded,
                })
        return raster

    def _stimulate_and_record(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity = StimPolarity.PositiveFirst,
        trigger_key: int = 0,
        post_stim_wait_s: float = 0.3,
        recording_window_s: float = 0.5,
        trial_index: int = 0,
    ) -> pd.DataFrame:
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
            timestamp_utc=datetime_now().isoformat(),
            trigger_key=trigger_key,
            trial_index=trial_index,
        ))

        self._wait(post_stim_wait_s)

        query_stop = datetime_now()
        query_start = query_stop - timedelta(seconds=post_stim_wait_s + recording_window_s)
        spike_df = self.database.get_spike_event_electrode(
            query_start, query_stop, electrode_idx
        )
        return spike_df

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
            "electrodes_scanned": self.electrodes,
            "num_repetitions": self.num_repetitions,
            "stim_amplitude_ua": self.stim_amplitude_ua,
            "stim_duration_us": self.stim_duration_us,
        }
        saver.save_summary(summary)

        waveform_records = self._fetch_spike_waveforms(
            fs_name, spike_df, trigger_df, recording_start, recording_stop
        )
        saver.save_spike_waveforms(waveform_records)

        saver.save_analysis(self._analysis_results)

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

        electrode_summary = {}
        for electrode_idx in self.electrodes:
            trials = self._electrode_trial_results[electrode_idx]
            if not trials:
                electrode_summary[electrode_idx] = {
                    "response_rate": 0.0,
                    "total_spikes": 0,
                    "mean_latency_ms": None,
                }
                continue
            response_rate = sum(1 for t in trials if t.responded) / len(trials)
            all_latencies = []
            for t in trials:
                all_latencies.extend(t.latencies_ms)
            mean_latency = float(np.mean(all_latencies)) if all_latencies else None
            electrode_summary[electrode_idx] = {
                "response_rate": float(response_rate),
                "total_spikes": len(all_latencies),
                "mean_latency_ms": mean_latency,
            }

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "electrodes_scanned": self.electrodes,
            "num_repetitions": self.num_repetitions,
            "stim_amplitude_ua": self.stim_amplitude_ua,
            "stim_duration_us": self.stim_duration_us,
            "electrode_summary": electrode_summary,
            "total_stimulations": len(self._stimulation_log),
            "analysis_keys": list(self._analysis_results.keys()),
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
