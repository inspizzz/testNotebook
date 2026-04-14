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
    electrode_idx: int
    amplitude_ua: float
    duration_us: float
    timestamp_utc: str
    trigger_key: int = 0
    phase: str = ""
    trial_index: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialResult:
    trial_index: int
    phase: str
    stim_electrode: int
    record_electrode: int
    stim_time_utc: str
    responded: bool
    latency_ms: float
    spike_count: int


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
    STDP experiment: Baseline -> Induction -> Post-test.

    Pre-synaptic electrode: 0 (electrode_from)
    Post-synaptic electrode: 1 (electrode_to)
    Amplitude: 3.0 uA, Duration: 300 us (charge balanced: 3*300 = 900 = 3*300)

    Phase 1 - Baseline (10 min): stimulate pre at 0.5 Hz, record post response.
    Phase 2 - Induction (30 min): paired pre+post stimulations at 0.2 Hz, 10ms delay.
    Phase 3 - Post-test (10 min): repeat baseline protocol.
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        pre_electrode: int = 0,
        post_electrode: int = 1,
        amplitude_ua: float = 3.0,
        duration_us: float = 300.0,
        baseline_duration_s: float = 600.0,
        induction_duration_s: float = 1800.0,
        posttest_duration_s: float = 600.0,
        baseline_freq_hz: float = 0.5,
        induction_freq_hz: float = 0.2,
        stdp_delay_ms: float = 10.0,
        response_window_ms: float = 50.0,
        trigger_key_pre: int = 0,
        trigger_key_post: int = 1,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.pre_electrode = pre_electrode
        self.post_electrode = post_electrode
        self.amplitude_ua = min(abs(amplitude_ua), 4.0)
        self.duration_us = min(abs(duration_us), 400.0)

        # Charge balance: A1*D1 == A2*D2 => same amplitude and duration on both phases
        assert abs(self.amplitude_ua * self.duration_us - self.amplitude_ua * self.duration_us) < 1e-9

        self.baseline_duration_s = baseline_duration_s
        self.induction_duration_s = induction_duration_s
        self.posttest_duration_s = posttest_duration_s
        self.baseline_freq_hz = baseline_freq_hz
        self.induction_freq_hz = induction_freq_hz
        self.stdp_delay_ms = stdp_delay_ms
        self.response_window_ms = response_window_ms
        self.trigger_key_pre = trigger_key_pre
        self.trigger_key_post = trigger_key_post

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._baseline_trials: List[TrialResult] = []
        self._posttest_trials: List[TrialResult] = []
        self._induction_count: int = 0

        self._recording_start: Optional[datetime] = None
        self._recording_stop: Optional[datetime] = None

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

            self._recording_start = datetime_now()

            self._configure_stimulation_params()

            logger.info("=== Phase 1: Baseline (%.0f s at %.2f Hz) ===", self.baseline_duration_s, self.baseline_freq_hz)
            self._phase_baseline()

            logger.info("=== Phase 2: Induction (%.0f s at %.2f Hz, STDP +%.1f ms) ===",
                        self.induction_duration_s, self.induction_freq_hz, self.stdp_delay_ms)
            self._phase_induction()

            logger.info("=== Phase 3: Post-test (%.0f s at %.2f Hz) ===", self.posttest_duration_s, self.baseline_freq_hz)
            self._phase_posttest()

            self._recording_stop = datetime_now()

            results = self._compile_results(self._recording_start, self._recording_stop)

            self._save_all(self._recording_start, self._recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _configure_stimulation_params(self) -> None:
        """Pre-configure stimulation parameters for pre and post electrodes."""
        # Configure pre-synaptic electrode on trigger_key_pre
        stim_pre = self._build_stim_param(self.pre_electrode, self.trigger_key_pre)
        # Configure post-synaptic electrode on trigger_key_post
        stim_post = self._build_stim_param(self.post_electrode, self.trigger_key_post)
        self.intan.send_stimparam([stim_pre, stim_post])
        logger.info("Stimulation parameters configured for electrodes %d (key %d) and %d (key %d)",
                    self.pre_electrode, self.trigger_key_pre,
                    self.post_electrode, self.trigger_key_post)

    def _build_stim_param(self, electrode_idx: int, trigger_key: int) -> StimParam:
        stim = StimParam()
        stim.index = electrode_idx
        stim.enable = True
        stim.trigger_key = trigger_key
        stim.trigger_delay = 0
        stim.nb_pulse = 0
        stim.pulse_train_period = 10000
        stim.post_stim_ref_period = 1000.0
        stim.stim_shape = StimShape.Biphasic
        stim.polarity = StimPolarity.NegativeFirst
        stim.phase_amplitude1 = self.amplitude_ua
        stim.phase_duration1 = self.duration_us
        stim.phase_amplitude2 = self.amplitude_ua
        stim.phase_duration2 = self.duration_us
        stim.enable_amp_settle = True
        stim.pre_stim_amp_settle = 0.0
        stim.post_stim_amp_settle = 1000.0
        stim.enable_charge_recovery = True
        stim.post_charge_recovery_on = 0.0
        stim.post_charge_recovery_off = 100.0
        stim.interphase_delay = 0.0
        return stim

    def _fire_trigger(self, trigger_key: int) -> None:
        pattern = np.zeros(16, dtype=np.uint8)
        pattern[trigger_key] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.02)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

    def _stimulate_pre(self, trial_index: int, phase: str) -> datetime:
        stim_time = datetime_now()
        self._fire_trigger(self.trigger_key_pre)
        self._stimulation_log.append(StimulationRecord(
            electrode_idx=self.pre_electrode,
            amplitude_ua=self.amplitude_ua,
            duration_us=self.duration_us,
            timestamp_utc=stim_time.isoformat(),
            trigger_key=self.trigger_key_pre,
            phase=phase,
            trial_index=trial_index,
        ))
        return stim_time

    def _stimulate_post(self, trial_index: int, phase: str) -> datetime:
        stim_time = datetime_now()
        self._fire_trigger(self.trigger_key_post)
        self._stimulation_log.append(StimulationRecord(
            electrode_idx=self.post_electrode,
            amplitude_ua=self.amplitude_ua,
            duration_us=self.duration_us,
            timestamp_utc=stim_time.isoformat(),
            trigger_key=self.trigger_key_post,
            phase=phase,
            trial_index=trial_index,
        ))
        return stim_time

    def _query_post_response(self, stim_time: datetime) -> Tuple[bool, float, int]:
        """
        Query spike events on the post-synaptic electrode within the response window.
        Returns (responded, latency_ms, spike_count).
        """
        window_s = self.response_window_ms / 1000.0
        query_start = stim_time
        query_stop = stim_time + timedelta(seconds=window_s + 0.1)
        try:
            spike_df = self.database.get_spike_event(
                query_start, query_stop, self.experiment.exp_name
            )
            if spike_df.empty:
                return False, 0.0, 0

            # Filter to post electrode
            channel_col = None
            for col in ["channel", "index", "electrode"]:
                if col in spike_df.columns:
                    channel_col = col
                    break

            if channel_col is not None:
                post_spikes = spike_df[spike_df[channel_col] == self.post_electrode]
            else:
                post_spikes = spike_df

            if post_spikes.empty:
                return False, 0.0, 0

            # Compute latency from stim_time
            time_col = "Time" if "Time" in post_spikes.columns else post_spikes.columns[0]
            spike_times = pd.to_datetime(post_spikes[time_col], utc=True)
            stim_dt = pd.Timestamp(stim_time)
            if stim_dt.tzinfo is None:
                stim_dt = stim_dt.tz_localize("UTC")

            latencies_ms = [(t - stim_dt).total_seconds() * 1000.0 for t in spike_times]
            valid_latencies = [l for l in latencies_ms if 0 < l <= self.response_window_ms]

            if not valid_latencies:
                return False, 0.0, 0

            return True, float(np.min(valid_latencies)), len(valid_latencies)

        except Exception as exc:
            logger.warning("Error querying post response: %s", exc)
            return False, 0.0, 0

    def _phase_baseline(self) -> None:
        inter_stim_s = 1.0 / self.baseline_freq_hz
        phase_start = datetime_now()
        trial_index = 0

        while True:
            elapsed = (datetime_now() - phase_start).total_seconds()
            if elapsed >= self.baseline_duration_s:
                break

            stim_time = self._stimulate_pre(trial_index, "baseline")
            # Wait for response window
            self._wait(self.response_window_ms / 1000.0 + 0.05)
            responded, latency_ms, spike_count = self._query_post_response(stim_time)

            self._baseline_trials.append(TrialResult(
                trial_index=trial_index,
                phase="baseline",
                stim_electrode=self.pre_electrode,
                record_electrode=self.post_electrode,
                stim_time_utc=stim_time.isoformat(),
                responded=responded,
                latency_ms=latency_ms,
                spike_count=spike_count,
            ))

            logger.info("Baseline trial %d: responded=%s, latency=%.2f ms",
                        trial_index, responded, latency_ms)
            trial_index += 1

            # Wait remainder of inter-stim interval
            elapsed_trial = (datetime_now() - stim_time).total_seconds()
            remaining = inter_stim_s - elapsed_trial
            if remaining > 0:
                self._wait(remaining)

        logger.info("Baseline complete: %d trials", len(self._baseline_trials))

    def _phase_induction(self) -> None:
        inter_stim_s = 1.0 / self.induction_freq_hz
        stdp_delay_s = self.stdp_delay_ms / 1000.0
        phase_start = datetime_now()
        trial_index = 0

        while True:
            elapsed = (datetime_now() - phase_start).total_seconds()
            if elapsed >= self.induction_duration_s:
                break

            # Fire pre-synaptic stimulation
            pre_stim_time = self._stimulate_pre(trial_index, "induction")

            # Wait STDP delay then fire post-synaptic stimulation
            self._wait(stdp_delay_s)
            self._stimulate_post(trial_index, "induction")

            self._induction_count += 1
            logger.info("Induction trial %d (total paired: %d)", trial_index, self._induction_count)
            trial_index += 1

            # Wait remainder of inter-stim interval
            elapsed_trial = (datetime_now() - pre_stim_time).total_seconds()
            remaining = inter_stim_s - elapsed_trial
            if remaining > 0:
                self._wait(remaining)

        logger.info("Induction complete: %d paired stimulations", self._induction_count)

    def _phase_posttest(self) -> None:
        inter_stim_s = 1.0 / self.baseline_freq_hz
        phase_start = datetime_now()
        trial_index = 0

        while True:
            elapsed = (datetime_now() - phase_start).total_seconds()
            if elapsed >= self.posttest_duration_s:
                break

            stim_time = self._stimulate_pre(trial_index, "posttest")
            self._wait(self.response_window_ms / 1000.0 + 0.05)
            responded, latency_ms, spike_count = self._query_post_response(stim_time)

            self._posttest_trials.append(TrialResult(
                trial_index=trial_index,
                phase="posttest",
                stim_electrode=self.pre_electrode,
                record_electrode=self.post_electrode,
                stim_time_utc=stim_time.isoformat(),
                responded=responded,
                latency_ms=latency_ms,
                spike_count=spike_count,
            ))

            logger.info("Post-test trial %d: responded=%s, latency=%.2f ms",
                        trial_index, responded, latency_ms)
            trial_index += 1

            elapsed_trial = (datetime_now() - stim_time).total_seconds()
            remaining = inter_stim_s - elapsed_trial
            if remaining > 0:
                self._wait(remaining)

        logger.info("Post-test complete: %d trials", len(self._posttest_trials))

    def _compute_response_probability(self, trials: List[TrialResult]) -> float:
        if not trials:
            return 0.0
        return sum(1 for t in trials if t.responded) / len(trials)

    def _compute_mean_latency(self, trials: List[TrialResult]) -> float:
        valid = [t.latency_ms for t in trials if t.responded and t.latency_ms > 0]
        if not valid:
            return 0.0
        return float(np.mean(valid))

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        logger.info("Compiling results")

        baseline_prob = self._compute_response_probability(self._baseline_trials)
        posttest_prob = self._compute_response_probability(self._posttest_trials)
        baseline_latency = self._compute_mean_latency(self._baseline_trials)
        posttest_latency = self._compute_mean_latency(self._posttest_trials)

        delta_prob = posttest_prob - baseline_prob

        logger.info("Baseline response probability: %.3f", baseline_prob)
        logger.info("Post-test response probability: %.3f", posttest_prob)
        logger.info("Delta response probability: %.3f", delta_prob)
        logger.info("Baseline mean latency: %.2f ms", baseline_latency)
        logger.info("Post-test mean latency: %.2f ms", posttest_latency)

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "pre_electrode": self.pre_electrode,
            "post_electrode": self.post_electrode,
            "amplitude_ua": self.amplitude_ua,
            "duration_us": self.duration_us,
            "stdp_delay_ms": self.stdp_delay_ms,
            "baseline_trials": len(self._baseline_trials),
            "induction_paired_stimulations": self._induction_count,
            "posttest_trials": len(self._posttest_trials),
            "baseline_response_probability": baseline_prob,
            "posttest_response_probability": posttest_prob,
            "delta_response_probability": delta_prob,
            "baseline_mean_latency_ms": baseline_latency,
            "posttest_mean_latency_ms": posttest_latency,
            "ltp_indicated": delta_prob > 0.1,
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

        spike_df = self.database.get_spike_event(
            recording_start, recording_stop, fs_name
        )
        saver.save_spike_events(spike_df)

        trigger_df = self.database.get_all_triggers(
            recording_start, recording_stop
        )
        saver.save_triggers(trigger_df)

        baseline_prob = self._compute_response_probability(self._baseline_trials)
        posttest_prob = self._compute_response_probability(self._posttest_trials)
        delta_prob = posttest_prob - baseline_prob

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "pre_electrode": self.pre_electrode,
            "post_electrode": self.post_electrode,
            "amplitude_ua": self.amplitude_ua,
            "duration_us": self.duration_us,
            "stdp_delay_ms": self.stdp_delay_ms,
            "baseline_freq_hz": self.baseline_freq_hz,
            "induction_freq_hz": self.induction_freq_hz,
            "baseline_duration_s": self.baseline_duration_s,
            "induction_duration_s": self.induction_duration_s,
            "posttest_duration_s": self.posttest_duration_s,
            "total_stimulations": len(self._stimulation_log),
            "baseline_trials": len(self._baseline_trials),
            "induction_paired_stimulations": self._induction_count,
            "posttest_trials": len(self._posttest_trials),
            "baseline_response_probability": baseline_prob,
            "posttest_response_probability": posttest_prob,
            "delta_response_probability": delta_prob,
            "baseline_mean_latency_ms": self._compute_mean_latency(self._baseline_trials),
            "posttest_mean_latency_ms": self._compute_mean_latency(self._posttest_trials),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "ltp_indicated": delta_prob > 0.1,
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
