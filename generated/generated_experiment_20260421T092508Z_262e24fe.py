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
    trigger_key: int
    phase: str
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialRecord:
    trial_index: int
    phase: str
    pre_electrode: int
    post_electrode: int
    stim_time_utc: str
    responded: bool
    latency_ms: float


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
    STDP experiment with three phases:
      Phase 1 - Baseline (10 min): pre-synaptic stimulation at 0.5 Hz, record post-synaptic response.
      Phase 2 - Induction (30 min): paired pre+post stimulation with 10 ms delay at 0.2 Hz.
      Phase 3 - Post-test (10 min): repeat baseline protocol to measure plasticity changes.

    Uses electrode pair 0 (pre) -> 1 (post) based on scan results showing reliable connection
    with 3 uA, 300 us, NegativeFirst polarity (hits_k=5, median_latency=12.37 ms).
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
        polarity: str = "NegativeFirst",
        baseline_duration_s: float = 600.0,
        induction_duration_s: float = 1800.0,
        posttest_duration_s: float = 600.0,
        baseline_freq_hz: float = 0.5,
        induction_freq_hz: float = 0.2,
        stdp_delay_ms: float = 10.0,
        response_window_ms: float = 50.0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.pre_electrode = pre_electrode
        self.post_electrode = post_electrode
        self.amplitude_ua = min(abs(amplitude_ua), 4.0)
        self.duration_us = min(abs(duration_us), 400.0)
        self.polarity = StimPolarity.NegativeFirst if polarity == "NegativeFirst" else StimPolarity.PositiveFirst

        self.baseline_duration_s = baseline_duration_s
        self.induction_duration_s = induction_duration_s
        self.posttest_duration_s = posttest_duration_s
        self.baseline_freq_hz = baseline_freq_hz
        self.induction_freq_hz = induction_freq_hz
        self.stdp_delay_ms = stdp_delay_ms
        self.response_window_ms = response_window_ms

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._trial_records: List[TrialRecord] = []

        self._baseline_trials: List[TrialRecord] = []
        self._posttest_trials: List[TrialRecord] = []

        self._pre_trigger_key = 0
        self._post_trigger_key = 1

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
            logger.info("Pre electrode: %d, Post electrode: %d", self.pre_electrode, self.post_electrode)
            logger.info("Amplitude: %.1f uA, Duration: %.1f us", self.amplitude_ua, self.duration_us)

            if not self.experiment.start():
                logger.error("Failed to start experiment")
                return {"status": "failed", "error": "Could not start experiment"}

            recording_start = datetime_now()

            self._configure_stimulation_params()

            logger.info("=== Phase 1: Baseline (%d s at %.2f Hz) ===", int(self.baseline_duration_s), self.baseline_freq_hz)
            phase1_start = datetime_now()
            self._phase_baseline_or_posttest("baseline")
            phase1_stop = datetime_now()
            logger.info("Phase 1 complete. Trials: %d", len(self._baseline_trials))

            logger.info("=== Phase 2: Induction (%d s at %.2f Hz) ===", int(self.induction_duration_s), self.induction_freq_hz)
            phase2_start = datetime_now()
            self._phase_induction()
            phase2_stop = datetime_now()
            logger.info("Phase 2 complete.")

            logger.info("=== Phase 3: Post-test (%d s at %.2f Hz) ===", int(self.posttest_duration_s), self.baseline_freq_hz)
            phase3_start = datetime_now()
            self._phase_baseline_or_posttest("posttest")
            phase3_stop = datetime_now()
            logger.info("Phase 3 complete. Trials: %d", len(self._posttest_trials))

            recording_stop = datetime_now()

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _configure_stimulation_params(self) -> None:
        """Pre-configure stimulation parameters for both pre and post electrodes."""
        for electrode_idx, trigger_key in [
            (self.pre_electrode, self._pre_trigger_key),
            (self.post_electrode, self._post_trigger_key),
        ]:
            stim = self._build_stim_param(electrode_idx, trigger_key)
            self.intan.send_stimparam([stim])
            logger.info(
                "Configured stim params for electrode %d (trigger key %d)",
                electrode_idx, trigger_key
            )

    def _build_stim_param(self, electrode_idx: int, trigger_key: int) -> StimParam:
        """Build a charge-balanced StimParam. A1*D1 == A2*D2."""
        stim = StimParam()
        stim.index = electrode_idx
        stim.enable = True
        stim.trigger_key = trigger_key
        stim.trigger_delay = 0
        stim.nb_pulse = 0
        stim.pulse_train_period = 10000
        stim.post_stim_ref_period = 1000.0
        stim.stim_shape = StimShape.Biphasic
        stim.polarity = self.polarity
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
        """Fire a single trigger pulse."""
        pattern = np.zeros(16, dtype=np.uint8)
        pattern[trigger_key] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.005)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

    def _stimulate_electrode(self, electrode_idx: int, trigger_key: int, phase: str) -> datetime:
        """Fire stimulation on one electrode and log it."""
        stim_time = datetime_now()
        self._fire_trigger(trigger_key)
        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=self.amplitude_ua,
            duration_us=self.duration_us,
            timestamp_utc=stim_time.isoformat(),
            trigger_key=trigger_key,
            phase=phase,
        ))
        return stim_time

    def _check_response(self, stim_time: datetime) -> Tuple[bool, float]:
        """
        Check if the post-synaptic electrode responded within the response window.
        Returns (responded: bool, latency_ms: float).
        """
        window_s = self.response_window_ms / 1000.0
        self._wait(window_s + 0.01)

        query_start = stim_time
        query_stop = stim_time + timedelta(seconds=window_s + 0.05)

        try:
            spike_df = self.database.get_spike_event_electrode(
                query_start, query_stop, self.post_electrode
            )
        except Exception as exc:
            logger.warning("Failed to query spike events: %s", exc)
            return False, 0.0

        if spike_df is None or spike_df.empty:
            return False, 0.0

        time_col = None
        for col in spike_df.columns:
            if col.lower() in ("time", "_time", "timestamp"):
                time_col = col
                break

        if time_col is None:
            return len(spike_df) > 0, 0.0

        spike_times = pd.to_datetime(spike_df[time_col], utc=True)
        stim_time_aware = stim_time if stim_time.tzinfo is not None else stim_time.replace(tzinfo=timezone.utc)

        valid_spikes = spike_times[spike_times > stim_time_aware]
        if valid_spikes.empty:
            return False, 0.0

        first_spike = valid_spikes.min()
        latency_ms = (first_spike - stim_time_aware).total_seconds() * 1000.0

        if 0 < latency_ms <= self.response_window_ms:
            return True, latency_ms

        return False, 0.0

    def _phase_baseline_or_posttest(self, phase: str) -> None:
        """
        Run baseline or post-test phase.
        Stimulate pre-synaptic electrode at baseline_freq_hz for the phase duration.
        Record response on post-synaptic electrode.
        """
        iti_s = 1.0 / self.baseline_freq_hz
        phase_duration_s = self.baseline_duration_s if phase == "baseline" else self.posttest_duration_s

        num_trials = int(phase_duration_s * self.baseline_freq_hz)
        logger.info("Phase %s: %d trials, ITI=%.1f s", phase, num_trials, iti_s)

        for trial_idx in range(num_trials):
            stim_time = self._stimulate_electrode(
                self.pre_electrode, self._pre_trigger_key, phase
            )

            responded, latency_ms = self._check_response(stim_time)

            record = TrialRecord(
                trial_index=trial_idx,
                phase=phase,
                pre_electrode=self.pre_electrode,
                post_electrode=self.post_electrode,
                stim_time_utc=stim_time.isoformat(),
                responded=responded,
                latency_ms=latency_ms,
            )
            self._trial_records.append(record)

            if phase == "baseline":
                self._baseline_trials.append(record)
            else:
                self._posttest_trials.append(record)

            if (trial_idx + 1) % 10 == 0:
                logger.info(
                    "Phase %s: trial %d/%d, responded=%s, latency=%.2f ms",
                    phase, trial_idx + 1, num_trials, responded, latency_ms
                )

            elapsed_wait = (self.response_window_ms / 1000.0) + 0.01
            remaining_iti = iti_s - elapsed_wait
            if remaining_iti > 0:
                self._wait(remaining_iti)

    def _phase_induction(self) -> None:
        """
        Run induction phase.
        Deliver paired pre-then-post stimulations with stdp_delay_ms delay at induction_freq_hz.
        """
        iti_s = 1.0 / self.induction_freq_hz
        stdp_delay_s = self.stdp_delay_ms / 1000.0
        num_trials = int(self.induction_duration_s * self.induction_freq_hz)

        logger.info(
            "Induction: %d paired trials, ITI=%.1f s, STDP delay=%.1f ms",
            num_trials, iti_s, self.stdp_delay_ms
        )

        for trial_idx in range(num_trials):
            pre_stim_time = self._stimulate_electrode(
                self.pre_electrode, self._pre_trigger_key, "induction"
            )

            self._wait(stdp_delay_s)

            post_stim_time = self._stimulate_electrode(
                self.post_electrode, self._post_trigger_key, "induction"
            )

            if (trial_idx + 1) % 20 == 0:
                logger.info(
                    "Induction: trial %d/%d complete",
                    trial_idx + 1, num_trials
                )

            elapsed = stdp_delay_s + 0.01
            remaining_iti = iti_s - elapsed
            if remaining_iti > 0:
                self._wait(remaining_iti)

    def _compute_response_probability(self, trials: List[TrialRecord]) -> float:
        if not trials:
            return 0.0
        return sum(1 for t in trials if t.responded) / len(trials)

    def _compute_mean_latency(self, trials: List[TrialRecord]) -> float:
        responding = [t.latency_ms for t in trials if t.responded and t.latency_ms > 0]
        if not responding:
            return 0.0
        return sum(responding) / len(responding)

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
        ltp_detected = delta_prob > 0.1

        logger.info("Baseline response probability: %.3f", baseline_prob)
        logger.info("Post-test response probability: %.3f", posttest_prob)
        logger.info("Delta probability: %.3f", delta_prob)
        logger.info("Baseline mean latency: %.2f ms", baseline_latency)
        logger.info("Post-test mean latency: %.2f ms", posttest_latency)
        logger.info("LTP detected: %s", ltp_detected)

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
            "posttest_trials": len(self._posttest_trials),
            "baseline_response_probability": baseline_prob,
            "posttest_response_probability": posttest_prob,
            "delta_response_probability": delta_prob,
            "baseline_mean_latency_ms": baseline_latency,
            "posttest_mean_latency_ms": posttest_latency,
            "ltp_detected": ltp_detected,
            "total_stimulations": len(self._stimulation_log),
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

        baseline_prob = self._compute_response_probability(self._baseline_trials)
        posttest_prob = self._compute_response_probability(self._posttest_trials)

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
            "total_baseline_trials": len(self._baseline_trials),
            "total_posttest_trials": len(self._posttest_trials),
            "baseline_response_probability": baseline_prob,
            "posttest_response_probability": posttest_prob,
            "delta_response_probability": posttest_prob - baseline_prob,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
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
        if spike_df is None or spike_df.empty:
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
