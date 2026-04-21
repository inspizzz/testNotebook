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
    phase: str = ""
    trial_index: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialResult:
    trial_index: int
    phase: str
    stim_electrode: int
    resp_electrode: int
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

    Phase 1 - Baseline (10 min): stimulate pre-synaptic electrode at 0.5 Hz,
    record response on post-synaptic electrode.
    Phase 2 - Induction (30 min): paired pre-then-post stimulations with 10 ms
    delay at 0.2 Hz to induce LTP.
    Phase 3 - Post-test (10 min): repeat baseline protocol to measure changes.
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        pre_electrode: int = 17,
        post_electrode: int = 18,
        amplitude_ua: float = 3.0,
        duration_us: float = 300.0,
        baseline_duration_s: float = 600.0,
        induction_duration_s: float = 1800.0,
        posttest_duration_s: float = 600.0,
        baseline_freq_hz: float = 0.5,
        induction_freq_hz: float = 0.2,
        stdp_delay_ms: float = 10.0,
        response_window_ms: float = 50.0,
        pre_trigger_key: int = 0,
        post_trigger_key: int = 1,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.pre_electrode = pre_electrode
        self.post_electrode = post_electrode
        self.amplitude_ua = min(abs(amplitude_ua), 4.0)
        self.duration_us = min(abs(duration_us), 400.0)

        self.baseline_duration_s = baseline_duration_s
        self.induction_duration_s = induction_duration_s
        self.posttest_duration_s = posttest_duration_s
        self.baseline_freq_hz = baseline_freq_hz
        self.induction_freq_hz = induction_freq_hz
        self.stdp_delay_ms = stdp_delay_ms
        self.response_window_ms = response_window_ms
        self.pre_trigger_key = pre_trigger_key
        self.post_trigger_key = post_trigger_key

        # Verify charge balance: A1*D1 == A2*D2 (equal amplitudes and durations)
        # Both phases use same amplitude and duration so balance is guaranteed.

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._baseline_trials: List[TrialResult] = []
        self._induction_trials: List[TrialResult] = []
        self._posttest_trials: List[TrialResult] = []

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

            self._configure_stim_params()

            logger.info("=== Phase 1: Baseline ===")
            self._phase_baseline()

            logger.info("=== Phase 2: Induction ===")
            self._phase_induction()

            logger.info("=== Phase 3: Post-test ===")
            self._phase_posttest()

            recording_stop = datetime_now()

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _configure_stim_params(self) -> None:
        """Pre-configure stimulation parameters for both electrodes."""
        for electrode_idx, trigger_key in [
            (self.pre_electrode, self.pre_trigger_key),
            (self.post_electrode, self.post_trigger_key),
        ]:
            stim = self._build_stim_param(electrode_idx, trigger_key)
            self.intan.send_stimparam([stim])
            logger.info(
                "Configured stim params for electrode %d (trigger key %d)",
                electrode_idx, trigger_key
            )

    def _build_stim_param(self, electrode_idx: int, trigger_key: int) -> StimParam:
        """Build a charge-balanced StimParam. A1*D1 == A2*D2 (equal phases)."""
        stim = StimParam()
        stim.index = electrode_idx
        stim.enable = True
        stim.trigger_key = trigger_key
        stim.trigger_delay = 0
        stim.nb_pulse = 0
        stim.pulse_train_period = 10000
        stim.post_stim_ref_period = 1000.0
        stim.stim_shape = StimShape.Biphasic
        stim.polarity = StimPolarity.PositiveFirst
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
        """Send a single trigger pulse on the specified key."""
        pattern = np.zeros(16, dtype=np.uint8)
        pattern[trigger_key] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.05)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

    def _stimulate_electrode(
        self,
        electrode_idx: int,
        trigger_key: int,
        phase: str,
        trial_index: int,
    ) -> datetime:
        """Fire a stimulation on one electrode and log it."""
        stim_time = datetime_now()
        self._fire_trigger(trigger_key)
        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=self.amplitude_ua,
            duration_us=self.duration_us,
            timestamp_utc=stim_time.isoformat(),
            trigger_key=trigger_key,
            phase=phase,
            trial_index=trial_index,
        ))
        return stim_time

    def _query_response(
        self,
        stim_time: datetime,
        resp_electrode: int,
    ) -> Tuple[bool, float, int]:
        """
        Query spike events on the response electrode within the response window.

        Returns:
            (responded, latency_ms, spike_count)
        """
        window_s = self.response_window_ms / 1000.0
        query_start = stim_time
        query_stop = stim_time + timedelta(seconds=window_s + 0.1)
        self._wait(window_s + 0.05)

        spike_df = self.database.get_spike_event_electrode(
            query_start, query_stop, resp_electrode
        )

        if spike_df.empty:
            return False, 0.0, 0

        spike_count = len(spike_df)
        time_col = "Time" if "Time" in spike_df.columns else spike_df.columns[0]
        spike_times = pd.to_datetime(spike_df[time_col], utc=True)
        stim_time_aware = stim_time if stim_time.tzinfo else stim_time.replace(tzinfo=timezone.utc)
        latencies_ms = [
            (t - stim_time_aware).total_seconds() * 1000.0
            for t in spike_times
            if (t - stim_time_aware).total_seconds() * 1000.0 > 0
        ]

        if not latencies_ms:
            return False, 0.0, spike_count

        first_latency_ms = min(latencies_ms)
        return True, first_latency_ms, spike_count

    def _phase_baseline(self) -> None:
        """
        Phase 1: Stimulate pre-synaptic electrode at 0.5 Hz for baseline_duration_s.
        Record response on post-synaptic electrode.
        """
        iti_s = 1.0 / self.baseline_freq_hz
        n_trials = int(self.baseline_duration_s * self.baseline_freq_hz)
        logger.info("Baseline: %d trials at %.2f Hz (ITI=%.1f s)", n_trials, self.baseline_freq_hz, iti_s)

        for trial_idx in range(n_trials):
            stim_time = self._stimulate_electrode(
                self.pre_electrode, self.pre_trigger_key, "baseline", trial_idx
            )
            responded, latency_ms, spike_count = self._query_response(
                stim_time, self.post_electrode
            )
            self._baseline_trials.append(TrialResult(
                trial_index=trial_idx,
                phase="baseline",
                stim_electrode=self.pre_electrode,
                resp_electrode=self.post_electrode,
                stim_time_utc=stim_time.isoformat(),
                responded=responded,
                latency_ms=latency_ms,
                spike_count=spike_count,
            ))
            if (trial_idx + 1) % 10 == 0:
                n_resp = sum(1 for t in self._baseline_trials if t.responded)
                logger.info(
                    "Baseline trial %d/%d | Response rate so far: %.2f",
                    trial_idx + 1, n_trials, n_resp / (trial_idx + 1)
                )
            elapsed_s = (datetime_now() - stim_time).total_seconds()
            remaining_wait = iti_s - elapsed_s
            if remaining_wait > 0:
                self._wait(remaining_wait)

    def _phase_induction(self) -> None:
        """
        Phase 2: Paired pre-then-post stimulations with stdp_delay_ms at induction_freq_hz.
        Induces LTP via STDP.
        """
        iti_s = 1.0 / self.induction_freq_hz
        n_trials = int(self.induction_duration_s * self.induction_freq_hz)
        delay_s = self.stdp_delay_ms / 1000.0
        logger.info(
            "Induction: %d paired trials at %.2f Hz (ITI=%.1f s, delay=%.1f ms)",
            n_trials, self.induction_freq_hz, iti_s, self.stdp_delay_ms
        )

        for trial_idx in range(n_trials):
            trial_start = datetime_now()

            # Pre-synaptic stimulation
            pre_stim_time = self._stimulate_electrode(
                self.pre_electrode, self.pre_trigger_key, "induction_pre", trial_idx
            )

            # Wait for STDP delay
            self._wait(delay_s)

            # Post-synaptic stimulation
            post_stim_time = self._stimulate_electrode(
                self.post_electrode, self.post_trigger_key, "induction_post", trial_idx
            )

            self._induction_trials.append(TrialResult(
                trial_index=trial_idx,
                phase="induction",
                stim_electrode=self.pre_electrode,
                resp_electrode=self.post_electrode,
                stim_time_utc=pre_stim_time.isoformat(),
                responded=True,
                latency_ms=self.stdp_delay_ms,
                spike_count=2,
            ))

            if (trial_idx + 1) % 20 == 0:
                logger.info("Induction trial %d/%d completed", trial_idx + 1, n_trials)

            elapsed_s = (datetime_now() - trial_start).total_seconds()
            remaining_wait = iti_s - elapsed_s
            if remaining_wait > 0:
                self._wait(remaining_wait)

    def _phase_posttest(self) -> None:
        """
        Phase 3: Repeat baseline protocol to measure plasticity changes.
        """
        iti_s = 1.0 / self.baseline_freq_hz
        n_trials = int(self.posttest_duration_s * self.baseline_freq_hz)
        logger.info("Post-test: %d trials at %.2f Hz (ITI=%.1f s)", n_trials, self.baseline_freq_hz, iti_s)

        for trial_idx in range(n_trials):
            stim_time = self._stimulate_electrode(
                self.pre_electrode, self.pre_trigger_key, "posttest", trial_idx
            )
            responded, latency_ms, spike_count = self._query_response(
                stim_time, self.post_electrode
            )
            self._posttest_trials.append(TrialResult(
                trial_index=trial_idx,
                phase="posttest",
                stim_electrode=self.pre_electrode,
                resp_electrode=self.post_electrode,
                stim_time_utc=stim_time.isoformat(),
                responded=responded,
                latency_ms=latency_ms,
                spike_count=spike_count,
            ))
            if (trial_idx + 1) % 10 == 0:
                n_resp = sum(1 for t in self._posttest_trials if t.responded)
                logger.info(
                    "Post-test trial %d/%d | Response rate so far: %.2f",
                    trial_idx + 1, n_trials, n_resp / (trial_idx + 1)
                )
            elapsed_s = (datetime_now() - stim_time).total_seconds()
            remaining_wait = iti_s - elapsed_s
            if remaining_wait > 0:
                self._wait(remaining_wait)

    def _compute_response_probability(self, trials: List[TrialResult]) -> float:
        if not trials:
            return 0.0
        return sum(1 for t in trials if t.responded) / len(trials)

    def _compute_mean_latency(self, trials: List[TrialResult]) -> float:
        valid = [t.latency_ms for t in trials if t.responded and t.latency_ms > 0]
        if not valid:
            return 0.0
        return sum(valid) / len(valid)

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
            "baseline_n_trials": len(self._baseline_trials),
            "baseline_response_probability": baseline_prob,
            "baseline_mean_latency_ms": baseline_latency,
            "induction_n_trials": len(self._induction_trials),
            "posttest_n_trials": len(self._posttest_trials),
            "posttest_response_probability": posttest_prob,
            "posttest_mean_latency_ms": posttest_latency,
            "delta_response_probability": delta_prob,
            "ltp_detected": ltp_detected,
            "total_stimulations": len(self._stimulation_log),
        }

        logger.info("Baseline response probability: %.3f", baseline_prob)
        logger.info("Post-test response probability: %.3f", posttest_prob)
        logger.info("Delta response probability: %.3f", delta_prob)
        logger.info("LTP detected: %s", ltp_detected)

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
            "baseline_n_trials": len(self._baseline_trials),
            "baseline_response_probability": baseline_prob,
            "baseline_mean_latency_ms": self._compute_mean_latency(self._baseline_trials),
            "induction_n_trials": len(self._induction_trials),
            "posttest_n_trials": len(self._posttest_trials),
            "posttest_response_probability": posttest_prob,
            "posttest_mean_latency_ms": self._compute_mean_latency(self._posttest_trials),
            "delta_response_probability": delta_prob,
            "ltp_detected": delta_prob > 0.1,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "baseline_trials": [asdict(t) for t in self._baseline_trials],
            "posttest_trials": [asdict(t) for t in self._posttest_trials],
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
