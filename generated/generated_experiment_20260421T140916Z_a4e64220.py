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
    condition: str = "open_loop"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpikeCountWindow:
    window_start_utc: str
    window_stop_utc: str
    electrode: int
    spike_count: int
    condition: str


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

    Hypothesis: Biphasic pulses delivered contingent on detected low-synchrony
    network states (closed-loop, CL) will produce a larger reduction in network
    asynchrony than charge-equated pulses delivered at fixed intervals (open-loop, OL).

    Session structure:
      1. Baseline recording (baseline_duration_s seconds)
      2. Stimulation epoch (stim_epoch_duration_s seconds):
         - CL condition: stimulate when spike count in sliding window is below T_low
         - OL condition: stimulate at fixed intervals derived from CL session
      3. Washout recording (washout_duration_s seconds)
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        stim_electrode: int = 21,
        resp_electrode: int = 22,
        stim_amplitude_ua: float = 3.0,
        stim_duration_us: float = 400.0,
        stim_polarity: str = "NegativeFirst",
        baseline_duration_s: float = 120.0,
        stim_epoch_duration_s: float = 180.0,
        washout_duration_s: float = 60.0,
        cl_window_s: float = 0.2,
        cl_threshold_percentile: float = 25.0,
        cl_refractory_s: float = 2.0,
        cl_min_active_electrodes: int = 3,
        max_cl_triggers: int = 300,
        inter_stim_interval_s: float = 2.0,
        artifact_blank_ms: float = 5.0,
        trigger_key: int = 0,
        monitor_electrodes: Tuple = (6, 7, 17, 18, 19, 21, 22),
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.stim_electrode = stim_electrode
        self.resp_electrode = resp_electrode
        self.stim_amplitude_ua = min(abs(stim_amplitude_ua), 4.0)
        self.stim_duration_us = min(abs(stim_duration_us), 400.0)
        self.stim_polarity_str = stim_polarity

        self.baseline_duration_s = baseline_duration_s
        self.stim_epoch_duration_s = stim_epoch_duration_s
        self.washout_duration_s = washout_duration_s

        self.cl_window_s = cl_window_s
        self.cl_threshold_percentile = cl_threshold_percentile
        self.cl_refractory_s = cl_refractory_s
        self.cl_min_active_electrodes = cl_min_active_electrodes
        self.max_cl_triggers = max_cl_triggers
        self.inter_stim_interval_s = inter_stim_interval_s
        self.artifact_blank_ms = artifact_blank_ms
        self.trigger_key = trigger_key
        self.monitor_electrodes = list(monitor_electrodes)

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._spike_count_windows: List[SpikeCountWindow] = []
        self._baseline_spike_counts: List[int] = []
        self._cl_trigger_times: List[datetime] = []
        self._ol_trigger_times: List[datetime] = []
        self._phase_boundaries: Dict[str, str] = {}

        self._cl_threshold: float = 0.0
        self._total_cl_triggers: int = 0
        self._total_ol_triggers: int = 0

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
            logger.info("Electrodes available: %s", self.experiment.electrodes)

            if not self.experiment.start():
                logger.error("Failed to start experiment")
                return {"status": "failed", "error": "Could not start experiment"}

            recording_start = datetime_now()
            self._phase_boundaries["recording_start"] = recording_start.isoformat()

            self._configure_stimulation()

            logger.info("=== Phase 1: Baseline Recording (%.0f s) ===", self.baseline_duration_s)
            baseline_start = datetime_now()
            self._phase_boundaries["baseline_start"] = baseline_start.isoformat()
            self._phase_baseline()
            baseline_stop = datetime_now()
            self._phase_boundaries["baseline_stop"] = baseline_stop.isoformat()

            self._compute_cl_threshold(baseline_start, baseline_stop)

            logger.info("=== Phase 2: Closed-Loop Stimulation Epoch (%.0f s) ===", self.stim_epoch_duration_s)
            cl_epoch_start = datetime_now()
            self._phase_boundaries["cl_epoch_start"] = cl_epoch_start.isoformat()
            self._phase_closed_loop(cl_epoch_start)
            cl_epoch_stop = datetime_now()
            self._phase_boundaries["cl_epoch_stop"] = cl_epoch_stop.isoformat()

            logger.info("=== Phase 3: Washout (%.0f s) ===", self.washout_duration_s)
            washout_start = datetime_now()
            self._phase_boundaries["washout_start"] = washout_start.isoformat()
            self._wait(self.washout_duration_s)
            washout_stop = datetime_now()
            self._phase_boundaries["washout_stop"] = washout_stop.isoformat()

            logger.info("=== Phase 4: Open-Loop Stimulation Epoch (%.0f s) ===", self.stim_epoch_duration_s)
            ol_epoch_start = datetime_now()
            self._phase_boundaries["ol_epoch_start"] = ol_epoch_start.isoformat()
            self._phase_open_loop(ol_epoch_start)
            ol_epoch_stop = datetime_now()
            self._phase_boundaries["ol_epoch_stop"] = ol_epoch_stop.isoformat()

            logger.info("=== Phase 5: Final Washout (%.0f s) ===", self.washout_duration_s)
            final_washout_start = datetime_now()
            self._phase_boundaries["final_washout_start"] = final_washout_start.isoformat()
            self._wait(self.washout_duration_s)
            final_washout_stop = datetime_now()
            self._phase_boundaries["final_washout_stop"] = final_washout_stop.isoformat()

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

    def _configure_stimulation(self) -> None:
        polarity = StimPolarity.NegativeFirst if self.stim_polarity_str == "NegativeFirst" else StimPolarity.PositiveFirst

        stim = StimParam()
        stim.index = self.stim_electrode
        stim.enable = True
        stim.trigger_key = self.trigger_key
        stim.trigger_delay = 0
        stim.nb_pulse = 0
        stim.pulse_train_period = 10000
        stim.post_stim_ref_period = 1000.0
        stim.stim_shape = StimShape.Biphasic
        stim.polarity = polarity
        stim.phase_amplitude1 = self.stim_amplitude_ua
        stim.phase_duration1 = self.stim_duration_us
        stim.phase_amplitude2 = self.stim_amplitude_ua
        stim.phase_duration2 = self.stim_duration_us
        stim.enable_amp_settle = True
        stim.pre_stim_amp_settle = 0.0
        stim.post_stim_amp_settle = 1000.0
        stim.enable_charge_recovery = True
        stim.post_charge_recovery_on = 0.0
        stim.post_charge_recovery_off = 100.0
        stim.interphase_delay = 0.0

        self.intan.send_stimparam([stim])
        logger.info(
            "Stimulation configured: electrode=%d, amplitude=%.1f uA, duration=%.0f us, polarity=%s",
            self.stim_electrode, self.stim_amplitude_ua, self.stim_duration_us, self.stim_polarity_str
        )

    def _phase_baseline(self) -> None:
        elapsed = 0.0
        poll_interval = min(self.cl_window_s, 1.0)
        while elapsed < self.baseline_duration_s:
            self._wait(poll_interval)
            elapsed += poll_interval
        logger.info("Baseline phase complete (%.1f s elapsed)", elapsed)

    def _compute_cl_threshold(self, baseline_start: datetime, baseline_stop: datetime) -> None:
        logger.info("Computing CL threshold from baseline spike counts")
        try:
            spike_df = self.database.get_spike_event(baseline_start, baseline_stop, self.experiment.exp_name)
            if spike_df.empty:
                logger.warning("No spikes in baseline; using threshold=0")
                self._cl_threshold = 0.0
                return

            window_s = self.cl_window_s
            baseline_duration = (baseline_stop - baseline_start).total_seconds()
            n_windows = max(1, int(baseline_duration / window_s))

            spike_counts_per_window = []
            for i in range(n_windows):
                w_start = baseline_start + timedelta(seconds=i * window_s)
                w_stop = w_start + timedelta(seconds=window_s)
                if "Time" in spike_df.columns:
                    mask = (spike_df["Time"] >= w_start) & (spike_df["Time"] < w_stop)
                    count = int(mask.sum())
                else:
                    count = 0
                spike_counts_per_window.append(count)

            self._baseline_spike_counts = spike_counts_per_window
            arr = np.array(spike_counts_per_window, dtype=float)
            self._cl_threshold = float(np.percentile(arr, self.cl_threshold_percentile))
            logger.info(
                "CL threshold set to %.2f spikes/window (%.0fth percentile of %d windows)",
                self._cl_threshold, self.cl_threshold_percentile, n_windows
            )
        except Exception as exc:
            logger.warning("Could not compute CL threshold from baseline: %s", exc)
            self._cl_threshold = 0.0

    def _count_spikes_in_window(self, window_start: datetime, window_stop: datetime) -> Tuple[int, int]:
        try:
            spike_df = self.database.get_spike_event(window_start, window_stop, self.experiment.exp_name)
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

    def _fire_trigger(self, condition: str) -> None:
        pattern = np.zeros(16, dtype=np.uint8)
        pattern[self.trigger_key] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.05)
        pattern[self.trigger_key] = 0
        self.trigger_controller.send(pattern)

        ts = datetime_now()
        self._stimulation_log.append(StimulationRecord(
            electrode_idx=self.stim_electrode,
            amplitude_ua=self.stim_amplitude_ua,
            duration_us=self.stim_duration_us,
            polarity=self.stim_polarity_str,
            timestamp_utc=ts.isoformat(),
            trigger_key=self.trigger_key,
            condition=condition,
        ))
        logger.debug("Trigger fired: condition=%s at %s", condition, ts.isoformat())

    def _phase_closed_loop(self, epoch_start: datetime) -> None:
        last_trigger_time: Optional[datetime] = None
        poll_interval = self.cl_window_s
        n_triggers = 0

        while True:
            now = datetime_now()
            elapsed = (now - epoch_start).total_seconds()
            if elapsed >= self.stim_epoch_duration_s:
                break
            if n_triggers >= self.max_cl_triggers:
                logger.info("CL: max trigger count (%d) reached", self.max_cl_triggers)
                remaining = self.stim_epoch_duration_s - elapsed
                if remaining > 0:
                    self._wait(remaining)
                break

            refractory_ok = True
            if last_trigger_time is not None:
                since_last = (now - last_trigger_time).total_seconds()
                if since_last < self.cl_refractory_s:
                    refractory_ok = False

            if refractory_ok:
                window_start = now - timedelta(seconds=self.cl_window_s)
                spike_count, active_electrodes = self._count_spikes_in_window(window_start, now)

                low_sync = spike_count <= self._cl_threshold
                network_active = active_electrodes >= self.cl_min_active_electrodes

                if low_sync and network_active:
                    self._fire_trigger("closed_loop")
                    last_trigger_time = datetime_now()
                    self._cl_trigger_times.append(last_trigger_time)
                    n_triggers += 1
                    self._wait(self.artifact_blank_ms / 1000.0)

            self._wait(poll_interval)

        self._total_cl_triggers = n_triggers
        logger.info("CL epoch complete: %d triggers fired", n_triggers)

    def _phase_open_loop(self, epoch_start: datetime) -> None:
        if self._cl_trigger_times:
            intervals = []
            sorted_times = sorted(self._cl_trigger_times)
            for i in range(1, len(sorted_times)):
                dt = (sorted_times[i] - sorted_times[i - 1]).total_seconds()
                intervals.append(dt)
            if intervals:
                mean_interval = float(np.mean(intervals))
            else:
                mean_interval = self.inter_stim_interval_s
        else:
            mean_interval = self.inter_stim_interval_s

        mean_interval = max(mean_interval, self.cl_refractory_s)
        n_triggers = 0
        max_ol_triggers = self._total_cl_triggers if self._total_cl_triggers > 0 else int(self.stim_epoch_duration_s / mean_interval)

        logger.info("OL epoch: mean_interval=%.2f s, max_triggers=%d", mean_interval, max_ol_triggers)

        while True:
            now = datetime_now()
            elapsed = (now - epoch_start).total_seconds()
            if elapsed >= self.stim_epoch_duration_s:
                break
            if n_triggers >= max_ol_triggers:
                remaining = self.stim_epoch_duration_s - elapsed
                if remaining > 0:
                    self._wait(remaining)
                break

            self._fire_trigger("open_loop")
            self._ol_trigger_times.append(datetime_now())
            n_triggers += 1
            self._wait(mean_interval)

        self._total_ol_triggers = n_triggers
        logger.info("OL epoch complete: %d triggers fired", n_triggers)

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
            "total_cl_triggers": self._total_cl_triggers,
            "total_ol_triggers": self._total_ol_triggers,
            "cl_threshold_spikes_per_window": self._cl_threshold,
            "cl_window_s": self.cl_window_s,
            "cl_refractory_s": self.cl_refractory_s,
            "stim_electrode": self.stim_electrode,
            "resp_electrode": self.resp_electrode,
            "stim_amplitude_ua": self.stim_amplitude_ua,
            "stim_duration_us": self.stim_duration_us,
            "stim_polarity": self.stim_polarity_str,
            "baseline_duration_s": self.baseline_duration_s,
            "stim_epoch_duration_s": self.stim_epoch_duration_s,
            "washout_duration_s": self.washout_duration_s,
            "phase_boundaries": self._phase_boundaries,
            "total_spike_events": len(spike_df),
            "total_triggers_db": len(trigger_df),
        }
        saver.save_summary(summary)

        waveform_records = self._fetch_spike_waveforms(spike_df, recording_start, recording_stop)
        saver.save_spike_waveforms(waveform_records)

    def _fetch_spike_waveforms(
        self,
        spike_df: pd.DataFrame,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> list:
        waveform_records = []
        if spike_df.empty:
            return waveform_records

        electrode_col = None
        for col in ("channel", "index", "electrode"):
            if col in spike_df.columns:
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

        cl_rate = self._total_cl_triggers / max(self.stim_epoch_duration_s, 1.0)
        ol_rate = self._total_ol_triggers / max(self.stim_epoch_duration_s, 1.0)

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": duration_s,
            "stim_electrode": self.stim_electrode,
            "resp_electrode": self.resp_electrode,
            "stim_amplitude_ua": self.stim_amplitude_ua,
            "stim_duration_us": self.stim_duration_us,
            "stim_polarity": self.stim_polarity_str,
            "charge_balance_check": abs(self.stim_amplitude_ua * self.stim_duration_us - self.stim_amplitude_ua * self.stim_duration_us) < 1e-9,
            "cl_threshold_spikes_per_window": self._cl_threshold,
            "total_cl_triggers": self._total_cl_triggers,
            "total_ol_triggers": self._total_ol_triggers,
            "cl_trigger_rate_per_s": cl_rate,
            "ol_trigger_rate_per_s": ol_rate,
            "baseline_spike_count_windows": len(self._baseline_spike_counts),
            "phase_boundaries": self._phase_boundaries,
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
