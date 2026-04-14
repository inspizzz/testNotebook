"""
FinalSpark Experiment: State-Dependent Gating of Evoked Responses
=================================================================
Tests whether electrically evoked spike counts differ between spontaneous
network burst states (B) and quiescent states (Q) in human cortical organoids.

Hypothesis: Evoked spike counts will be significantly lower when identical
biphasic stimuli are delivered during burst states vs. quiescent states.

Phases:
  0 - Baseline recording (5 min) to estimate spontaneous firing thresholds
  1 - Calibration block (fixed timing, 0.05 Hz) to confirm evoked responses
  2 - State-dependent stimulation (interleaved Q and B trials)
  3 - Recovery recording (5 min, no stimulation)
"""

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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StimulationRecord:
    """A single stimulation event for the persistence log."""
    trial_index: int
    electrode_from: int
    electrode_to: int
    amplitude_ua: float
    duration_us: float
    polarity: str
    network_state: str
    pre_burst_spike_count: int
    timestamp_utc: str
    trigger_key: int = 0
    phase: str = "stimulation"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialResult:
    """Result of a single stimulation trial."""
    trial_index: int
    network_state: str
    pre_burst_spike_count: int
    evoked_spike_count: int
    response_probability: float
    first_spike_latency_ms: Optional[float]
    stim_electrode: int
    resp_electrode: int
    timestamp_utc: str


# ---------------------------------------------------------------------------
# Data persistence
# ---------------------------------------------------------------------------

class DataSaver:
    """Handles persistence of stimulation records, spike events, and triggers."""

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


# ---------------------------------------------------------------------------
# Main experiment class
# ---------------------------------------------------------------------------

class Experiment:
    """
    State-dependent gating experiment for FinalSpark NeuroPlatform.

    Tests whether evoked spike counts differ between burst (B) and
    quiescent (Q) network states using charge-balanced biphasic stimulation.

    Selected electrode pairs from deep scan (highest response rates):
      - Primary:   stim=14, resp=12  (response_rate=0.94, amp=1.0uA, dur=400us, NegativeFirst)
      - Secondary: stim=18, resp=17  (response_rate=0.89, amp=1.0uA, dur=400us, PositiveFirst)
      - Tertiary:  stim=22, resp=21  (response_rate=0.93, amp=3.0uA, dur=400us, PositiveFirst)

    State classification uses a 500ms pre-stimulus window on reference
    electrodes (0, 7, 24, 31) to estimate instantaneous spike rate.
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        # Stimulation parameters (from deep scan best performers)
        stim_electrode: int = 14,
        resp_electrode: int = 12,
        stim_amplitude_ua: float = 1.0,
        stim_duration_us: float = 400.0,
        stim_polarity: str = "NegativeFirst",
        # State classification
        state_window_s: float = 0.5,
        burst_multiplier: float = 2.0,
        # Trial parameters
        num_trials_per_condition: int = 50,
        min_isi_s: float = 10.0,
        max_wait_for_state_s: float = 120.0,
        # Phase durations
        baseline_duration_s: float = 120.0,
        calibration_trials: int = 10,
        recovery_duration_s: float = 120.0,
        # Analysis window
        evoked_window_start_ms: float = 2.0,
        evoked_window_stop_ms: float = 100.0,
        # Reference electrodes for state classification
        reference_electrodes: Optional[List[int]] = None,
        trigger_key: int = 0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        # Stimulation config
        self.stim_electrode = stim_electrode
        self.resp_electrode = resp_electrode
        self.stim_amplitude_ua = min(abs(stim_amplitude_ua), 4.0)
        self.stim_duration_us = min(abs(stim_duration_us), 400.0)
        self.stim_polarity_str = stim_polarity
        self.stim_polarity = (
            StimPolarity.NegativeFirst
            if stim_polarity == "NegativeFirst"
            else StimPolarity.PositiveFirst
        )

        # State classification
        self.state_window_s = state_window_s
        self.burst_multiplier = burst_multiplier
        self.quiescence_threshold: float = 0.0
        self.burst_threshold: float = 0.0

        # Trial parameters
        self.num_trials_per_condition = num_trials_per_condition
        self.min_isi_s = min_isi_s
        self.max_wait_for_state_s = max_wait_for_state_s

        # Phase durations
        self.baseline_duration_s = baseline_duration_s
        self.calibration_trials = calibration_trials
        self.recovery_duration_s = recovery_duration_s

        # Analysis window
        self.evoked_window_start_ms = evoked_window_start_ms
        self.evoked_window_stop_ms = evoked_window_stop_ms

        # Reference electrodes (corners of MEA, far from stim electrode)
        self.reference_electrodes = reference_electrodes if reference_electrodes is not None else [0, 7, 24, 31]
        self.trigger_key = trigger_key

        # Hardware handles
        self.np_experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        # Results storage
        self._stimulation_log: List[StimulationRecord] = []
        self._trial_results: List[TrialResult] = []
        self._baseline_spike_counts: List[float] = []
        self._calibration_results: List[Dict[str, Any]] = []
        self._trial_index: int = 0

        # Phase timestamps
        self._phase_timestamps: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Delay helper
    # ------------------------------------------------------------------
    def _wait(self, seconds: float) -> None:
        wait(seconds)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """Execute the full experiment and return a results dict."""
        try:
            logger.info("Initialising hardware connections")

            from neuroplatform import Experiment as NeuroPlatformExperiment

            self.np_experiment = NeuroPlatformExperiment(self.token)
            self.trigger_controller = TriggerController(email=self.booking_email)
            self.intan = IntanSofware()
            self.database = Database()

            logger.info("Experiment: %s", self.np_experiment.exp_name)
            logger.info("Electrodes available: %s", self.np_experiment.electrodes)

            if not self.np_experiment.start():
                logger.error("Failed to start experiment")
                return {"status": "failed", "error": "Could not start experiment"}

            recording_start = datetime_now()
            self._phase_timestamps["recording_start"] = recording_start.isoformat()

            # --- Phase 0: Baseline recording ---
            self._phase_baseline()

            # --- Phase 1: Calibration block ---
            self._phase_calibration()

            # --- Phase 2: State-dependent stimulation ---
            self._phase_state_dependent_stimulation()

            # --- Phase 3: Recovery recording ---
            self._phase_recovery()

            recording_stop = datetime_now()
            self._phase_timestamps["recording_stop"] = recording_stop.isoformat()

            results = self._compile_results(recording_start, recording_stop)

            # Persist all raw data
            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    # ------------------------------------------------------------------
    # Phase 0: Baseline recording
    # ------------------------------------------------------------------
    def _phase_baseline(self) -> None:
        """Record spontaneous activity to estimate firing rate thresholds."""
        logger.info("Phase 0: Baseline recording (%.0f s)", self.baseline_duration_s)
        self._phase_timestamps["baseline_start"] = datetime_now().isoformat()

        baseline_start = datetime_now()
        self._wait(self.baseline_duration_s)
        baseline_stop = datetime_now()

        self._phase_timestamps["baseline_stop"] = baseline_stop.isoformat()

        # Query spike events from reference electrodes during baseline
        try:
            spike_df = self.database.get_spike_event(
                baseline_start, baseline_stop, self.np_experiment.exp_name
            )

            if not spike_df.empty:
                # Estimate spike rate per 500ms window across reference electrodes
                ref_spikes = spike_df
                if "channel" in spike_df.columns:
                    ref_spikes = spike_df[spike_df["channel"].isin(self.reference_electrodes)]

                duration_s = (baseline_stop - baseline_start).total_seconds()
                num_windows = max(1, int(duration_s / self.state_window_s))
                total_ref_spikes = len(ref_spikes)
                mean_spikes_per_window = total_ref_spikes / num_windows

                self.quiescence_threshold = mean_spikes_per_window
                self.burst_threshold = self.burst_multiplier * mean_spikes_per_window

                logger.info(
                    "Baseline: mean=%.2f spikes/window, Q_thresh=%.2f, B_thresh=%.2f",
                    mean_spikes_per_window,
                    self.quiescence_threshold,
                    self.burst_threshold,
                )
                self._baseline_spike_counts.append(mean_spikes_per_window)
            else:
                logger.warning("No spikes detected during baseline; using default thresholds")
                self.quiescence_threshold = 2.0
                self.burst_threshold = 4.0

        except Exception as exc:
            logger.warning("Baseline query failed: %s; using default thresholds", exc)
            self.quiescence_threshold = 2.0
            self.burst_threshold = 4.0

        logger.info("Phase 0 complete")

    # ------------------------------------------------------------------
    # Phase 1: Calibration block
    # ------------------------------------------------------------------
    def _phase_calibration(self) -> None:
        """Fixed-timing calibration at 0.05 Hz to confirm evoked responses."""
        logger.info("Phase 1: Calibration block (%d trials at 0.05 Hz)", self.calibration_trials)
        self._phase_timestamps["calibration_start"] = datetime_now().isoformat()

        isi_s = 20.0  # 0.05 Hz

        for i in range(self.calibration_trials):
            logger.info("Calibration trial %d/%d", i + 1, self.calibration_trials)

            stim_time = datetime_now()
            self._send_stimulation(
                electrode_idx=self.stim_electrode,
                amplitude_ua=self.stim_amplitude_ua,
                duration_us=self.stim_duration_us,
                polarity=self.stim_polarity,
                trigger_key=self.trigger_key,
                network_state="calibration",
                pre_burst_spike_count=0,
                phase="calibration",
            )

            # Query evoked response
            self._wait(0.15)
            query_start = stim_time
            query_stop = datetime_now()

            try:
                spike_df = self.database.get_spike_event(
                    query_start, query_stop, self.np_experiment.exp_name
                )
                evoked_count = self._count_evoked_spikes(spike_df, stim_time)
                self._calibration_results.append({
                    "trial": i,
                    "evoked_spike_count": evoked_count,
                    "timestamp": stim_time.isoformat(),
                })
                logger.info("Calibration trial %d: %d evoked spikes", i + 1, evoked_count)
            except Exception as exc:
                logger.warning("Calibration query failed: %s", exc)

            if i < self.calibration_trials - 1:
                self._wait(isi_s)

        self._phase_timestamps["calibration_stop"] = datetime_now().isoformat()
        logger.info("Phase 1 complete")

    # ------------------------------------------------------------------
    # Phase 2: State-dependent stimulation
    # ------------------------------------------------------------------
    def _phase_state_dependent_stimulation(self) -> None:
        """Interleaved Q and B trials based on real-time state classification."""
        logger.info(
            "Phase 2: State-dependent stimulation (%d trials per condition)",
            self.num_trials_per_condition,
        )
        self._phase_timestamps["stim_phase_start"] = datetime_now().isoformat()

        q_count = 0
        b_count = 0
        total_attempts = 0
        max_attempts = (self.num_trials_per_condition * 2) * 10

        last_stim_time = datetime_now()

        while (q_count < self.num_trials_per_condition or b_count < self.num_trials_per_condition):
            if total_attempts >= max_attempts:
                logger.warning(
                    "Max attempts reached. Q=%d, B=%d", q_count, b_count
                )
                break

            total_attempts += 1

            # Enforce minimum ISI
            elapsed = (datetime_now() - last_stim_time).total_seconds()
            if elapsed < self.min_isi_s:
                self._wait(self.min_isi_s - elapsed)

            # Classify current network state
            state, spike_count = self._classify_network_state()

            # Determine if we need this state
            need_q = q_count < self.num_trials_per_condition
            need_b = b_count < self.num_trials_per_condition

            if state == "Q" and not need_q:
                self._wait(1.0)
                continue
            if state == "B" and not need_b:
                self._wait(1.0)
                continue
            if state == "intermediate":
                self._wait(1.0)
                continue

            # Deliver stimulation
            logger.info(
                "Trial attempt %d: state=%s (spikes=%.1f), Q=%d/%d, B=%d/%d",
                total_attempts, state, spike_count,
                q_count, self.num_trials_per_condition,
                b_count, self.num_trials_per_condition,
            )

            stim_time = datetime_now()
            self._send_stimulation(
                electrode_idx=self.stim_electrode,
                amplitude_ua=self.stim_amplitude_ua,
                duration_us=self.stim_duration_us,
                polarity=self.stim_polarity,
                trigger_key=self.trigger_key,
                network_state=state,
                pre_burst_spike_count=int(spike_count),
                phase="state_dependent",
            )
            last_stim_time = datetime_now()

            # Wait for evoked response window
            self._wait(self.evoked_window_stop_ms / 1000.0 + 0.05)

            # Query evoked response
            query_start = stim_time
            query_stop = datetime_now()

            try:
                spike_df = self.database.get_spike_event(
                    query_start, query_stop, self.np_experiment.exp_name
                )
                evoked_count = self._count_evoked_spikes(spike_df, stim_time)
                first_latency = self._compute_first_latency(spike_df, stim_time)

                trial_result = TrialResult(
                    trial_index=self._trial_index,
                    network_state=state,
                    pre_burst_spike_count=int(spike_count),
                    evoked_spike_count=evoked_count,
                    response_probability=1.0 if evoked_count > 0 else 0.0,
                    first_spike_latency_ms=first_latency,
                    stim_electrode=self.stim_electrode,
                    resp_electrode=self.resp_electrode,
                    timestamp_utc=stim_time.isoformat(),
                )
                self._trial_results.append(trial_result)
                self._trial_index += 1

                if state == "Q":
                    q_count += 1
                elif state == "B":
                    b_count += 1

                logger.info(
                    "Trial %d [%s]: evoked=%d spikes, latency=%s ms",
                    self._trial_index, state, evoked_count,
                    f"{first_latency:.2f}" if first_latency is not None else "N/A",
                )

            except Exception as exc:
                logger.warning("Trial query failed: %s", exc)
                self._trial_index += 1
                if state == "Q":
                    q_count += 1
                elif state == "B":
                    b_count += 1

        self._phase_timestamps["stim_phase_stop"] = datetime_now().isoformat()
        logger.info("Phase 2 complete: Q=%d, B=%d trials", q_count, b_count)

    # ------------------------------------------------------------------
    # Phase 3: Recovery recording
    # ------------------------------------------------------------------
    def _phase_recovery(self) -> None:
        """Record spontaneous activity post-stimulation to check for plasticity."""
        logger.info("Phase 3: Recovery recording (%.0f s)", self.recovery_duration_s)
        self._phase_timestamps["recovery_start"] = datetime_now().isoformat()
        self._wait(self.recovery_duration_s)
        self._phase_timestamps["recovery_stop"] = datetime_now().isoformat()
        logger.info("Phase 3 complete")

    # ------------------------------------------------------------------
    # State classification
    # ------------------------------------------------------------------
    def _classify_network_state(self) -> Tuple[str, float]:
        """
        Classify current network state as Q (quiescent), B (burst), or intermediate.

        Queries spike events from reference electrodes over the last state_window_s
        and compares to thresholds derived from baseline.

        Returns:
            Tuple of (state_label, spike_count_in_window)
        """
        window_stop = datetime_now()
        window_start = window_stop - timedelta(seconds=self.state_window_s)

        try:
            spike_df = self.database.get_spike_event(
                window_start, window_stop, self.np_experiment.exp_name
            )

            if spike_df.empty:
                spike_count = 0.0
            else:
                if "channel" in spike_df.columns:
                    ref_spikes = spike_df[spike_df["channel"].isin(self.reference_electrodes)]
                    spike_count = float(len(ref_spikes))
                else:
                    spike_count = float(len(spike_df))

        except Exception as exc:
            logger.warning("State classification query failed: %s", exc)
            spike_count = 0.0

        if spike_count < self.quiescence_threshold:
            state = "Q"
        elif spike_count > self.burst_threshold:
            state = "B"
        else:
            state = "intermediate"

        return state, spike_count

    # ------------------------------------------------------------------
    # Stimulation helper
    # ------------------------------------------------------------------
    def _send_stimulation(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity,
        trigger_key: int,
        network_state: str,
        pre_burst_spike_count: int,
        phase: str = "stimulation",
    ) -> None:
        """Configure and fire a single charge-balanced biphasic pulse."""
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

        # Charge balance: A1*D1 == A2*D2 (equal amplitudes, equal durations)
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
        stim.interphase_delay = 0.0

        self.intan.send_stimparam([stim])

        # Fire trigger
        pattern = np.zeros(16, dtype=np.uint8)
        pattern[trigger_key] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.05)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

        # Log stimulation
        self._stimulation_log.append(StimulationRecord(
            trial_index=self._trial_index,
            electrode_from=electrode_idx,
            electrode_to=self.resp_electrode,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            polarity=polarity.name,
            network_state=network_state,
            pre_burst_spike_count=pre_burst_spike_count,
            timestamp_utc=datetime_now().isoformat(),
            trigger_key=trigger_key,
            phase=phase,
        ))

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------
    def _count_evoked_spikes(
        self, spike_df: pd.DataFrame, stim_time: datetime
    ) -> int:
        """Count spikes on the response electrode within the evoked window."""
        if spike_df.empty:
            return 0

        window_start_s = self.evoked_window_start_ms / 1000.0
        window_stop_s = self.evoked_window_stop_ms / 1000.0

        try:
            if "Time" in spike_df.columns:
                time_col = "Time"
            else:
                time_col = spike_df.columns[0]

            times = pd.to_datetime(spike_df[time_col], utc=True)
            latencies_s = (times - stim_time).dt.total_seconds()

            mask = (latencies_s >= window_start_s) & (latencies_s <= window_stop_s)

            if "channel" in spike_df.columns:
                mask = mask & (spike_df["channel"] == self.resp_electrode)

            return int(mask.sum())

        except Exception as exc:
            logger.warning("Evoked spike count failed: %s", exc)
            return 0

    def _compute_first_latency(
        self, spike_df: pd.DataFrame, stim_time: datetime
    ) -> Optional[float]:
        """Compute first-spike latency in ms for the response electrode."""
        if spike_df.empty:
            return None

        window_start_s = self.evoked_window_start_ms / 1000.0
        window_stop_s = self.evoked_window_stop_ms / 1000.0

        try:
            if "Time" in spike_df.columns:
                time_col = "Time"
            else:
                time_col = spike_df.columns[0]

            times = pd.to_datetime(spike_df[time_col], utc=True)
            latencies_s = (times - stim_time).dt.total_seconds()

            mask = (latencies_s >= window_start_s) & (latencies_s <= window_stop_s)

            if "channel" in spike_df.columns:
                mask = mask & (spike_df["channel"] == self.resp_electrode)

            valid_latencies = latencies_s[mask]
            if valid_latencies.empty:
                return None

            return float(valid_latencies.min() * 1000.0)

        except Exception as exc:
            logger.warning("First latency computation failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Data persistence
    # ------------------------------------------------------------------
    def _save_all(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> None:
        """Persist all raw experiment data for downstream analysis."""
        fs_name = getattr(self.np_experiment, "exp_name", "unknown")
        saver = DataSaver(self._output_dir, fs_name)

        # Save stimulation log
        saver.save_stimulation_log(self._stimulation_log)

        # Fetch and save ALL spike events
        try:
            spike_df = self.database.get_spike_event(
                recording_start, recording_stop, fs_name
            )
        except Exception as exc:
            logger.warning("Failed to fetch spike events: %s", exc)
            spike_df = pd.DataFrame()
        saver.save_spike_events(spike_df)

        # Fetch and save ALL triggers
        try:
            trigger_df = self.database.get_all_triggers(
                recording_start, recording_stop
            )
        except Exception as exc:
            logger.warning("Failed to fetch triggers: %s", exc)
            trigger_df = pd.DataFrame()
        saver.save_triggers(trigger_df)

        # Compute summary statistics
        q_trials = [t for t in self._trial_results if t.network_state == "Q"]
        b_trials = [t for t in self._trial_results if t.network_state == "B"]

        q_evoked = [t.evoked_spike_count for t in q_trials]
        b_evoked = [t.evoked_spike_count for t in b_trials]

        q_mean = float(np.mean(q_evoked)) if q_evoked else 0.0
        b_mean = float(np.mean(b_evoked)) if b_evoked else 0.0
        q_resp_prob = float(np.mean([t.response_probability for t in q_trials])) if q_trials else 0.0
        b_resp_prob = float(np.mean([t.response_probability for t in b_trials])) if b_trials else 0.0

        q_latencies = [t.first_spike_latency_ms for t in q_trials if t.first_spike_latency_ms is not None]
        b_latencies = [t.first_spike_latency_ms for t in b_trials if t.first_spike_latency_ms is not None]

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "stim_electrode": self.stim_electrode,
            "resp_electrode": self.resp_electrode,
            "stim_amplitude_ua": self.stim_amplitude_ua,
            "stim_duration_us": self.stim_duration_us,
            "stim_polarity": self.stim_polarity_str,
            "quiescence_threshold": self.quiescence_threshold,
            "burst_threshold": self.burst_threshold,
            "num_q_trials": len(q_trials),
            "num_b_trials": len(b_trials),
            "q_mean_evoked_spikes": q_mean,
            "b_mean_evoked_spikes": b_mean,
            "q_response_probability": q_resp_prob,
            "b_response_probability": b_resp_prob,
            "q_mean_latency_ms": float(np.mean(q_latencies)) if q_latencies else None,
            "b_mean_latency_ms": float(np.mean(b_latencies)) if b_latencies else None,
            "calibration_trials": len(self._calibration_results),
            "phase_timestamps": self._phase_timestamps,
            "trial_results": [asdict(t) for t in self._trial_results],
        }
        saver.save_summary(summary)

        # Fetch and save spike waveforms
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
        """Fetch raw spike waveform data for each electrode that had spikes."""
        waveform_records = []
        if spike_df.empty:
            return waveform_records

        electrode_col = None
        for col in ["channel", "index", "electrode"]:
            if col in spike_df.columns:
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
                    electrode_idx, exc,
                )

        return waveform_records

    # ------------------------------------------------------------------
    # Results compilation
    # ------------------------------------------------------------------
    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        """Assemble a summary dict to be returned from run()."""
        logger.info("Compiling results")

        q_trials = [t for t in self._trial_results if t.network_state == "Q"]
        b_trials = [t for t in self._trial_results if t.network_state == "B"]

        q_evoked = [t.evoked_spike_count for t in q_trials]
        b_evoked = [t.evoked_spike_count for t in b_trials]

        q_mean = float(np.mean(q_evoked)) if q_evoked else 0.0
        b_mean = float(np.mean(b_evoked)) if b_evoked else 0.0

        q_resp_prob = float(np.mean([t.response_probability for t in q_trials])) if q_trials else 0.0
        b_resp_prob = float(np.mean([t.response_probability for t in b_trials])) if b_trials else 0.0

        q_latencies = [t.first_spike_latency_ms for t in q_trials if t.first_spike_latency_ms is not None]
        b_latencies = [t.first_spike_latency_ms for t in b_trials if t.first_spike_latency_ms is not None]

        # Fano factor
        q_fano = (float(np.var(q_evoked)) / q_mean) if (q_mean > 0 and len(q_evoked) > 1) else None
        b_fano = (float(np.var(b_evoked)) / b_mean) if (b_mean > 0 and len(b_evoked) > 1) else None

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.np_experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "stim_electrode": self.stim_electrode,
            "resp_electrode": self.resp_electrode,
            "stim_amplitude_ua": self.stim_amplitude_ua,
            "stim_duration_us": self.stim_duration_us,
            "stim_polarity": self.stim_polarity_str,
            "quiescence_threshold": self.quiescence_threshold,
            "burst_threshold": self.burst_threshold,
            "num_q_trials": len(q_trials),
            "num_b_trials": len(b_trials),
            "q_mean_evoked_spikes": q_mean,
            "b_mean_evoked_spikes": b_mean,
            "q_response_probability": q_resp_prob,
            "b_response_probability": b_resp_prob,
            "q_mean_latency_ms": float(np.mean(q_latencies)) if q_latencies else None,
            "b_mean_latency_ms": float(np.mean(b_latencies)) if b_latencies else None,
            "q_fano_factor": q_fano,
            "b_fano_factor": b_fano,
            "total_stimulations": len(self._stimulation_log),
            "calibration_trials": len(self._calibration_results),
            "phase_timestamps": self._phase_timestamps,
        }

        logger.info(
            "Results: Q_mean=%.2f, B_mean=%.2f, Q_resp=%.2f, B_resp=%.2f",
            q_mean, b_mean, q_resp_prob, b_resp_prob,
        )

        return summary

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def _cleanup(self) -> None:
        """Release all hardware resources. Called from the finally block."""
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
