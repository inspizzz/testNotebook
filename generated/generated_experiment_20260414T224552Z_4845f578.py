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
    condition: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialResult:
    trial_idx: int
    phase: str
    condition: str
    stim_electrode: int
    resp_electrode: int
    amplitude_ua: float
    duration_us: float
    polarity: str
    stim_time_utc: str
    spike_count_in_window: int
    responded: bool
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
    """
    Closed-loop vs open-loop synchrony experiment on FinalSpark NeuroPlatform.

    Hypothesis: Closed-loop stimulation delivered during low-synchrony network
    states produces greater reduction in SPIKE-distance (increased synchrony)
    than charge-equated open-loop stimulation.

    Session structure:
      - Baseline epoch: 20 minutes of spontaneous recording
      - Stimulation epoch: 30 minutes (CL or OL condition)
      - Washout epoch: 10 minutes of spontaneous recording

    Primary electrode pairs (from deep scan, highest response rates):
      - Stim 14 -> Resp 12 (response_rate=0.94, amplitude=1.0uA, dur=400us, NegativeFirst)
      - Stim 18 -> Resp 17 (response_rate=0.89, amplitude=1.0uA, dur=400us, PositiveFirst)
      - Stim 22 -> Resp 21 (response_rate=0.93, amplitude=3.0uA, dur=400us, PositiveFirst)
      - Stim 9  -> Resp 10 (response_rate=0.94, amplitude=3.0uA, dur=400us, NegativeFirst)
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        baseline_duration_min: float = 20.0,
        stimulation_duration_min: float = 30.0,
        washout_duration_min: float = 10.0,
        cl_window_ms: float = 200.0,
        cl_threshold_percentile: float = 25.0,
        cl_refractory_s: float = 2.0,
        cl_min_active_electrodes: int = 3,
        max_triggers_per_session: int = 300,
        inter_stim_interval_s: float = 1.0,
        ol_inter_stim_interval_s: float = 4.0,
        response_window_ms: float = 50.0,
        artifact_blank_ms: float = 5.0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.baseline_duration_s = baseline_duration_min * 60.0
        self.stimulation_duration_s = stimulation_duration_min * 60.0
        self.washout_duration_s = washout_duration_min * 60.0
        self.cl_window_ms = cl_window_ms
        self.cl_threshold_percentile = cl_threshold_percentile
        self.cl_refractory_s = cl_refractory_s
        self.cl_min_active_electrodes = cl_min_active_electrodes
        self.max_triggers_per_session = max_triggers_per_session
        self.inter_stim_interval_s = inter_stim_interval_s
        self.ol_inter_stim_interval_s = ol_inter_stim_interval_s
        self.response_window_ms = response_window_ms
        self.artifact_blank_ms = artifact_blank_ms

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._trial_results: List[TrialResult] = []

        self._stim_pairs = [
            {
                "stim_electrode": 14,
                "resp_electrode": 12,
                "amplitude": 1.0,
                "duration": 400.0,
                "polarity": StimPolarity.NegativeFirst,
                "polarity_str": "NegativeFirst",
                "trigger_key": 0,
                "median_latency_ms": 22.72,
                "response_rate": 0.94,
            },
            {
                "stim_electrode": 18,
                "resp_electrode": 17,
                "amplitude": 1.0,
                "duration": 400.0,
                "polarity": StimPolarity.PositiveFirst,
                "polarity_str": "PositiveFirst",
                "trigger_key": 1,
                "median_latency_ms": 25.075,
                "response_rate": 0.89,
            },
            {
                "stim_electrode": 22,
                "resp_electrode": 21,
                "amplitude": 3.0,
                "duration": 400.0,
                "polarity": StimPolarity.PositiveFirst,
                "polarity_str": "PositiveFirst",
                "trigger_key": 2,
                "median_latency_ms": 14.03,
                "response_rate": 0.93,
            },
            {
                "stim_electrode": 9,
                "resp_electrode": 10,
                "amplitude": 3.0,
                "duration": 400.0,
                "polarity": StimPolarity.NegativeFirst,
                "polarity_str": "NegativeFirst",
                "trigger_key": 3,
                "median_latency_ms": 11.035,
                "response_rate": 0.94,
            },
        ]

        self._all_electrodes = list(set(
            [p["stim_electrode"] for p in self._stim_pairs] +
            [p["resp_electrode"] for p in self._stim_pairs]
        ))

        self._baseline_spike_counts: List[int] = []
        self._cl_threshold: float = 0.0
        self._last_trigger_time: float = 0.0
        self._cl_trigger_count: int = 0
        self._ol_trigger_count: int = 0

        self._phase_boundaries: Dict[str, Any] = {}

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
            self._phase_boundaries["recording_start"] = recording_start.isoformat()

            self._configure_stimulation_params()

            self._phase_baseline(recording_start)

            self._phase_closed_loop_stimulation()

            self._phase_washout()

            recording_stop = datetime_now()
            self._phase_boundaries["recording_stop"] = recording_stop.isoformat()

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _configure_stimulation_params(self) -> None:
        logger.info("Configuring stimulation parameters for all electrode pairs")
        stim_params = []
        for pair in self._stim_pairs:
            sp = self._build_stim_param(
                electrode_idx=pair["stim_electrode"],
                amplitude_ua=pair["amplitude"],
                duration_us=pair["duration"],
                polarity=pair["polarity"],
                trigger_key=pair["trigger_key"],
            )
            stim_params.append(sp)
        self.intan.send_stimparam(stim_params)
        logger.info("Stimulation parameters configured for %d pairs", len(stim_params))

    def _build_stim_param(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity,
        trigger_key: int,
    ) -> StimParam:
        amplitude_ua = min(abs(amplitude_ua), 4.0)
        duration_us = min(abs(duration_us), 400.0)

        sp = StimParam()
        sp.index = electrode_idx
        sp.enable = True
        sp.trigger_key = trigger_key
        sp.trigger_delay = 0
        sp.nb_pulse = 0
        sp.pulse_train_period = 10000
        sp.post_stim_ref_period = 1000.0
        sp.stim_shape = StimShape.Biphasic
        sp.polarity = polarity
        sp.phase_amplitude1 = amplitude_ua
        sp.phase_duration1 = duration_us
        sp.phase_amplitude2 = amplitude_ua
        sp.phase_duration2 = duration_us
        sp.enable_amp_settle = True
        sp.pre_stim_amp_settle = 0.0
        sp.post_stim_amp_settle = 1000.0
        sp.enable_charge_recovery = True
        sp.post_charge_recovery_on = 0.0
        sp.post_charge_recovery_off = 100.0
        sp.interphase_delay = 0.0
        return sp

    def _phase_baseline(self, recording_start: datetime) -> None:
        logger.info("Phase: Baseline recording (%.1f min)", self.baseline_duration_s / 60.0)
        baseline_start = datetime_now()
        self._phase_boundaries["baseline_start"] = baseline_start.isoformat()

        bin_duration_s = 10.0
        num_bins = int(self.baseline_duration_s / bin_duration_s)
        spike_counts_per_bin: List[int] = []

        for bin_idx in range(num_bins):
            bin_start = datetime_now()
            self._wait(bin_duration_s)
            bin_stop = datetime_now()

            try:
                spike_df = self.database.get_spike_event(
                    bin_start, bin_stop, self.experiment.exp_name
                )
                count = len(spike_df) if not spike_df.empty else 0
            except Exception as exc:
                logger.warning("Baseline bin %d spike query failed: %s", bin_idx, exc)
                count = 0

            spike_counts_per_bin.append(count)
            logger.info("Baseline bin %d/%d: %d spikes", bin_idx + 1, num_bins, count)

        self._baseline_spike_counts = spike_counts_per_bin

        if spike_counts_per_bin:
            sorted_counts = sorted(spike_counts_per_bin)
            idx = int(math.floor(self.cl_threshold_percentile / 100.0 * len(sorted_counts)))
            idx = max(0, min(idx, len(sorted_counts) - 1))
            self._cl_threshold = float(sorted_counts[idx])
        else:
            self._cl_threshold = 0.0

        logger.info(
            "Baseline complete. CL threshold (%.0fth percentile): %.1f spikes/bin",
            self.cl_threshold_percentile,
            self._cl_threshold,
        )
        self._phase_boundaries["baseline_stop"] = datetime_now().isoformat()

    def _get_recent_spike_count(self, window_s: float) -> Tuple[int, int]:
        query_stop = datetime_now()
        query_start = query_stop - timedelta(seconds=window_s)
        try:
            spike_df = self.database.get_spike_event(
                query_start, query_stop, self.experiment.exp_name
            )
            if spike_df.empty:
                return 0, 0
            total_spikes = len(spike_df)
            if "channel" in spike_df.columns:
                active_electrodes = spike_df["channel"].nunique()
            else:
                active_electrodes = 0
            return total_spikes, active_electrodes
        except Exception as exc:
            logger.warning("Spike count query failed: %s", exc)
            return 0, 0

    def _fire_trigger(self, pair: Dict[str, Any], phase: str, condition: str, trial_idx: int) -> None:
        trigger_key = pair["trigger_key"]
        pattern = np.zeros(16, dtype=np.uint8)
        pattern[trigger_key] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.02)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

        stim_time = datetime_now()

        self._stimulation_log.append(StimulationRecord(
            electrode_idx=pair["stim_electrode"],
            amplitude_ua=pair["amplitude"],
            duration_us=pair["duration"],
            polarity=pair["polarity_str"],
            timestamp_utc=stim_time.isoformat(),
            trigger_key=trigger_key,
            phase=phase,
            condition=condition,
        ))

        self._wait(self.artifact_blank_ms / 1000.0)

        response_window_s = self.response_window_ms / 1000.0
        self._wait(response_window_s)

        query_stop = datetime_now()
        query_start = query_stop - timedelta(seconds=response_window_s + 0.01)
        try:
            spike_df = self.database.get_spike_event(
                query_start, query_stop, self.experiment.exp_name
            )
            if not spike_df.empty and "channel" in spike_df.columns:
                resp_spikes = spike_df[spike_df["channel"] == pair["resp_electrode"]]
                spike_count = len(resp_spikes)
                responded = spike_count > 0
                if responded and "Time" in resp_spikes.columns:
                    try:
                        first_spike_time = pd.to_datetime(resp_spikes["Time"].iloc[0])
                        stim_dt = pd.to_datetime(stim_time)
                        latency_ms = (first_spike_time - stim_dt).total_seconds() * 1000.0
                    except Exception:
                        latency_ms = None
                else:
                    latency_ms = None
            else:
                spike_count = 0
                responded = False
                latency_ms = None
        except Exception as exc:
            logger.warning("Response query failed: %s", exc)
            spike_count = 0
            responded = False
            latency_ms = None

        self._trial_results.append(TrialResult(
            trial_idx=trial_idx,
            phase=phase,
            condition=condition,
            stim_electrode=pair["stim_electrode"],
            resp_electrode=pair["resp_electrode"],
            amplitude_ua=pair["amplitude"],
            duration_us=pair["duration"],
            polarity=pair["polarity_str"],
            stim_time_utc=stim_time.isoformat(),
            spike_count_in_window=spike_count,
            responded=responded,
            latency_ms=latency_ms,
        ))

    def _phase_closed_loop_stimulation(self) -> None:
        logger.info("Phase: Closed-loop stimulation (%.1f min)", self.stimulation_duration_s / 60.0)
        phase_start = datetime_now()
        self._phase_boundaries["cl_stim_start"] = phase_start.isoformat()

        cl_window_s = self.cl_window_ms / 1000.0
        self._last_trigger_time = 0.0
        self._cl_trigger_count = 0
        trial_idx = 0
        pair_cycle_idx = 0

        phase_end_time = phase_start + timedelta(seconds=self.stimulation_duration_s)

        while datetime_now() < phase_end_time:
            if self._cl_trigger_count >= self.max_triggers_per_session:
                logger.info("Max CL trigger count reached (%d)", self.max_triggers_per_session)
                break

            now_ts = datetime_now().timestamp()
            time_since_last = now_ts - self._last_trigger_time

            if time_since_last < self.cl_refractory_s:
                self._wait(0.1)
                continue

            spike_count, active_electrodes = self._get_recent_spike_count(cl_window_s)

            low_synchrony = spike_count <= self._cl_threshold
            network_active = active_electrodes >= self.cl_min_active_electrodes

            if low_synchrony and network_active:
                pair = self._stim_pairs[pair_cycle_idx % len(self._stim_pairs)]
                self._fire_trigger(pair, phase="cl_stimulation", condition="closed_loop", trial_idx=trial_idx)
                self._last_trigger_time = datetime_now().timestamp()
                self._cl_trigger_count += 1
                trial_idx += 1
                pair_cycle_idx += 1
                logger.info(
                    "CL trigger %d: electrode %d, spikes=%d, active_elec=%d",
                    self._cl_trigger_count,
                    pair["stim_electrode"],
                    spike_count,
                    active_electrodes,
                )
                self._wait(self.inter_stim_interval_s)
            else:
                self._wait(0.05)

        self._phase_boundaries["cl_stim_stop"] = datetime_now().isoformat()
        logger.info("CL stimulation complete. Total triggers: %d", self._cl_trigger_count)

    def _phase_open_loop_stimulation(self) -> None:
        logger.info("Phase: Open-loop stimulation (%.1f min)", self.stimulation_duration_s / 60.0)
        phase_start = datetime_now()
        self._phase_boundaries["ol_stim_start"] = phase_start.isoformat()

        self._ol_trigger_count = 0
        trial_idx = 0
        pair_cycle_idx = 0

        phase_end_time = phase_start + timedelta(seconds=self.stimulation_duration_s)

        while datetime_now() < phase_end_time:
            if self._ol_trigger_count >= self.max_triggers_per_session:
                logger.info("Max OL trigger count reached (%d)", self.max_triggers_per_session)
                break

            pair = self._stim_pairs[pair_cycle_idx % len(self._stim_pairs)]
            self._fire_trigger(pair, phase="ol_stimulation", condition="open_loop", trial_idx=trial_idx)
            self._ol_trigger_count += 1
            trial_idx += 1
            pair_cycle_idx += 1
            logger.info(
                "OL trigger %d: electrode %d",
                self._ol_trigger_count,
                pair["stim_electrode"],
            )
            self._wait(self.ol_inter_stim_interval_s)

        self._phase_boundaries["ol_stim_stop"] = datetime_now().isoformat()
        logger.info("OL stimulation complete. Total triggers: %d", self._ol_trigger_count)

    def _phase_washout(self) -> None:
        logger.info("Phase: Washout recording (%.1f min)", self.washout_duration_s / 60.0)
        washout_start = datetime_now()
        self._phase_boundaries["washout_start"] = washout_start.isoformat()

        bin_duration_s = 10.0
        num_bins = int(self.washout_duration_s / bin_duration_s)
        washout_spike_counts: List[int] = []

        for bin_idx in range(num_bins):
            bin_start = datetime_now()
            self._wait(bin_duration_s)
            bin_stop = datetime_now()

            try:
                spike_df = self.database.get_spike_event(
                    bin_start, bin_stop, self.experiment.exp_name
                )
                count = len(spike_df) if not spike_df.empty else 0
            except Exception as exc:
                logger.warning("Washout bin %d spike query failed: %s", bin_idx, exc)
                count = 0

            washout_spike_counts.append(count)
            logger.info("Washout bin %d/%d: %d spikes", bin_idx + 1, num_bins, count)

        self._washout_spike_counts = washout_spike_counts
        self._phase_boundaries["washout_stop"] = datetime_now().isoformat()
        logger.info("Washout complete.")

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        logger.info("Compiling results")

        total_trials = len(self._trial_results)
        responded_trials = sum(1 for t in self._trial_results if t.responded)
        response_rate = responded_trials / total_trials if total_trials > 0 else 0.0

        latencies = [t.latency_ms for t in self._trial_results if t.latency_ms is not None]
        mean_latency = float(np.mean(latencies)) if latencies else None
        median_latency = float(np.median(latencies)) if latencies else None

        cl_trials = [t for t in self._trial_results if t.condition == "closed_loop"]
        ol_trials = [t for t in self._trial_results if t.condition == "open_loop"]

        cl_response_rate = (
            sum(1 for t in cl_trials if t.responded) / len(cl_trials)
            if cl_trials else 0.0
        )
        ol_response_rate = (
            sum(1 for t in ol_trials if t.responded) / len(ol_trials)
            if ol_trials else 0.0
        )

        baseline_mean = float(np.mean(self._baseline_spike_counts)) if self._baseline_spike_counts else 0.0
        washout_counts = getattr(self, "_washout_spike_counts", [])
        washout_mean = float(np.mean(washout_counts)) if washout_counts else 0.0

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "phase_boundaries": self._phase_boundaries,
            "total_stimulations": len(self._stimulation_log),
            "total_trials": total_trials,
            "responded_trials": responded_trials,
            "overall_response_rate": response_rate,
            "mean_latency_ms": mean_latency,
            "median_latency_ms": median_latency,
            "cl_trigger_count": self._cl_trigger_count,
            "ol_trigger_count": self._ol_trigger_count,
            "cl_response_rate": cl_response_rate,
            "ol_response_rate": ol_response_rate,
            "cl_threshold_spikes_per_bin": self._cl_threshold,
            "baseline_mean_spikes_per_bin": baseline_mean,
            "washout_mean_spikes_per_bin": washout_mean,
            "baseline_spike_counts": self._baseline_spike_counts,
            "washout_spike_counts": washout_counts,
            "stim_pairs": [
                {
                    "stim_electrode": p["stim_electrode"],
                    "resp_electrode": p["resp_electrode"],
                    "amplitude": p["amplitude"],
                    "duration": p["duration"],
                    "polarity": p["polarity_str"],
                    "response_rate_scan": p["response_rate"],
                    "median_latency_ms_scan": p["median_latency_ms"],
                }
                for p in self._stim_pairs
            ],
        }

        return summary

    def _save_all(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> None:
        fs_name = getattr(self.experiment, "exp_name", "unknown")
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

        trial_dicts = [asdict(t) for t in self._trial_results]
        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "cl_trigger_count": self._cl_trigger_count,
            "ol_trigger_count": self._ol_trigger_count,
            "cl_threshold": self._cl_threshold,
            "baseline_spike_counts": self._baseline_spike_counts,
            "washout_spike_counts": getattr(self, "_washout_spike_counts", []),
            "phase_boundaries": self._phase_boundaries,
            "trial_results": trial_dicts,
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

        electrodes_to_fetch = self._all_electrodes

        if not spike_df.empty:
            electrode_col = None
            for col in ["channel", "index", "electrode"]:
                if col in spike_df.columns:
                    electrode_col = col
                    break
            if electrode_col is not None:
                electrodes_to_fetch = list(spike_df[electrode_col].unique())

        for electrode_idx in electrodes_to_fetch:
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
