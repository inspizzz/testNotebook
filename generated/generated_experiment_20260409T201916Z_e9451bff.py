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
    Experiment as NeuroPlatformExperiment,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StimulationTrial:
    """Record of a single stimulation trial."""
    electrode_idx: int
    frequency_hz: float
    amplitude_ua: float
    duration_us: float
    trial_num: int
    timestamp: datetime
    spike_count: int
    spike_latencies_ms: List[float] = field(default_factory=list)
    spike_amplitudes_ua: List[float] = field(default_factory=list)


@dataclass
class FrequencyBlockResults:
    """Results from stimulation at a single frequency."""
    frequency_hz: float
    electrode_idx: int
    trials: List[StimulationTrial] = field(default_factory=list)
    adaptation_index: float = 0.0
    mean_spike_count: float = 0.0
    std_spike_count: float = 0.0
    mean_latency_ms: float = 0.0
    latency_jitter_ms: float = 0.0


class Experiment:
    """Frequency-dependent adaptation experiment targeting active electrodes.
    
    Tests the hypothesis that low-frequency stimulation (<0.2 Hz) enables
    complete ion channel recovery, yielding non-adapting responses (adaptation
    index >0.8), while higher frequencies (>1 Hz) induce adaptation
    (adaptation index <0.5).
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        frequencies_hz: Optional[List[float]] = None,
        trials_per_frequency: int = 100,
        stimulus_amplitude_ua: float = 2.0,
        stimulus_duration_us: float = 100.0,
        inter_block_rest_s: float = 60.0,
        post_stim_wait_s: float = 0.5,
        recording_window_s: float = 1.0,
        target_electrodes: Optional[List[int]] = None,
        max_electrodes: int = 8,
    ):
        """Initialize the frequency-dependent adaptation experiment.
        
        Args:
            token: FinalSpark experiment token.
            booking_email: Booking email for trigger controller.
            testing: Flag for testing mode (stored but not used for branching).
            frequencies_hz: List of frequencies to test. Defaults to literature-derived.
            trials_per_frequency: Number of trials per frequency block.
            stimulus_amplitude_ua: Amplitude in uA (max 5.0).
            stimulus_duration_us: Duration in us (max 500).
            inter_block_rest_s: Rest time between frequency blocks (seconds).
            post_stim_wait_s: Wait time after stimulus before recording.
            recording_window_s: Duration of spike recording window.
            target_electrodes: Specific electrodes to target. If None, auto-select.
            max_electrodes: Maximum number of electrodes to use.
        """
        self.token = token
        self.booking_email = booking_email
        self.testing = testing

        # Stimulation parameters
        self.frequencies_hz = frequencies_hz or [0.1, 0.2, 0.5, 1.0, 5.0, 10.0]
        self.trials_per_frequency = trials_per_frequency
        self.stimulus_amplitude_ua = min(abs(stimulus_amplitude_ua), 5.0)
        self.stimulus_duration_us = min(abs(stimulus_duration_us), 500.0)
        self.inter_block_rest_s = inter_block_rest_s
        self.post_stim_wait_s = post_stim_wait_s
        self.recording_window_s = recording_window_s
        self.target_electrodes = target_electrodes
        self.max_electrodes = max_electrodes

        # Hardware handles
        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        # Results storage
        self.baseline_spike_rate = {}
        self.frequency_results: Dict[float, List[FrequencyBlockResults]] = defaultdict(list)
        self.all_trials: List[StimulationTrial] = []
        self.recording_start = None
        self.recording_stop = None

    def run(self) -> Dict[str, Any]:
        """Execute the full frequency-dependent adaptation experiment."""
        try:
            logger.info("Initializing hardware connections")
            self.experiment = NeuroPlatformExperiment(self.token)
            self.trigger_controller = TriggerController(email=self.booking_email)
            self.intan = IntanSofware()
            self.database = Database()

            logger.info("Experiment: %s", self.experiment.exp_name)
            logger.info("Available electrodes: %s", self.experiment.electrodes)

            if not self.experiment.start():
                logger.error("Failed to start experiment")
                return {"status": "failed", "error": "Could not start experiment"}

            self.recording_start = datetime.now(timezone.utc)

            # Select target electrodes
            selected_electrodes = self._select_target_electrodes()
            logger.info("Selected electrodes for stimulation: %s", selected_electrodes)

            # Phase 1: Baseline recording
            logger.info("Phase 1: Baseline spontaneous activity recording")
            self._phase_baseline_recording(selected_electrodes)

            # Phase 2: Frequency sweep with repeated trials
            logger.info("Phase 2: Frequency-dependent stimulation sweep")
            self._phase_frequency_sweep(selected_electrodes)

            # Phase 3: Day 2 validation (repeat 0.2 Hz)
            logger.info("Phase 3: Day 2 validation at 0.2 Hz")
            self._phase_validation_block(selected_electrodes)

            self.recording_stop = datetime.now(timezone.utc)

            results = self._compile_results()
            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _select_target_electrodes(self) -> List[int]:
        """Select target electrodes from available electrodes.
        
        Prioritizes central electrodes and limits to max_electrodes.
        """
        available = self.experiment.electrodes
        if self.target_electrodes:
            selected = [e for e in self.target_electrodes if e in available]
        else:
            # Prefer central region (indices 40-87 are typically central in 128-electrode array)
            central_range = [e for e in available if 40 <= e <= 87]
            selected = central_range[:self.max_electrodes]
            if len(selected) < self.max_electrodes:
                selected.extend(available[:self.max_electrodes - len(selected)])

        return selected[:self.max_electrodes]

    def _phase_baseline_recording(self, electrodes: List[int]) -> None:
        """Record 60 seconds of baseline spontaneous activity."""
        logger.info("Recording baseline activity for 60 seconds")
        baseline_start = datetime.now(timezone.utc)
        time.sleep(60.0)
        baseline_stop = datetime.now(timezone.utc)

        for electrode_idx in electrodes:
            spike_df = self.database.get_spike_event_electrode(
                baseline_start, baseline_stop, electrode_idx
            )
            spike_count = len(spike_df) if not spike_df.empty else 0
            spike_rate = spike_count / 60.0
            self.baseline_spike_rate[electrode_idx] = spike_rate
            logger.info(
                "Electrode %d baseline spike rate: %.2f spikes/sec",
                electrode_idx,
                spike_rate,
            )

    def _phase_frequency_sweep(self, electrodes: List[int]) -> None:
        """Execute frequency sweep with randomized block order."""
        # Randomize frequency order to avoid order effects
        freq_order = np.random.permutation(self.frequencies_hz).tolist()
        logger.info("Frequency block order: %s", freq_order)

        for freq_idx, frequency in enumerate(freq_order):
            logger.info(
                "Frequency block %d/%d: %.2f Hz",
                freq_idx + 1,
                len(freq_order),
                frequency,
            )

            inter_stimulus_interval = 1.0 / frequency if frequency > 0 else 10.0
            logger.info("Inter-stimulus interval: %.3f seconds", inter_stimulus_interval)

            for electrode_idx in electrodes:
                block_results = self._stimulate_frequency_block(
                    electrode_idx, frequency, inter_stimulus_interval
                )
                self.frequency_results[frequency].append(block_results)

            # Rest period between blocks
            if freq_idx < len(freq_order) - 1:
                logger.info("Inter-block rest: %.1f seconds", self.inter_block_rest_s)
                time.sleep(self.inter_block_rest_s)

    def _stimulate_frequency_block(
        self, electrode_idx: int, frequency: float, inter_stimulus_interval: float
    ) -> FrequencyBlockResults:
        """Execute one frequency block with repeated trials."""
        block_results = FrequencyBlockResults(
            frequency_hz=frequency, electrode_idx=electrode_idx
        )

        spike_counts = []
        latencies_all = []

        for trial_num in range(self.trials_per_frequency):
            trial_timestamp = datetime.now(timezone.utc)

            # Send stimulus
            spike_df = self._stimulate_and_record(
                electrode_idx,
                self.stimulus_amplitude_ua,
                self.stimulus_duration_us,
                trigger_key=0,
                post_stim_wait_s=self.post_stim_wait_s,
                recording_window_s=self.recording_window_s,
            )

            spike_count = len(spike_df) if not spike_df.empty else 0
            spike_counts.append(spike_count)

            # Extract latencies (time from stimulus to spike)
            latencies_ms = []
            if not spike_df.empty and "Time" in spike_df.columns:
                for _, row in spike_df.iterrows():
                    latency_ms = (row["Time"] - trial_timestamp).total_seconds() * 1000.0
                    if 0 < latency_ms < self.recording_window_s * 1000.0:
                        latencies_ms.append(latency_ms)
                        latencies_all.append(latency_ms)

            trial = StimulationTrial(
                electrode_idx=electrode_idx,
                frequency_hz=frequency,
                amplitude_ua=self.stimulus_amplitude_ua,
                duration_us=self.stimulus_duration_us,
                trial_num=trial_num,
                timestamp=trial_timestamp,
                spike_count=spike_count,
                spike_latencies_ms=latencies_ms,
                spike_amplitudes_ua=(
                    spike_df["Amplitude"].tolist() if not spike_df.empty else []
                ),
            )
            block_results.trials.append(trial)
            self.all_trials.append(trial)

            # Inter-stimulus interval
            time.sleep(inter_stimulus_interval)

        # Compute block statistics
        if spike_counts:
            block_results.mean_spike_count = np.mean(spike_counts)
            block_results.std_spike_count = np.std(spike_counts)
            block_results.adaptation_index = self._compute_adaptation_index(spike_counts)

        if latencies_all:
            block_results.mean_latency_ms = np.mean(latencies_all)
            block_results.latency_jitter_ms = np.std(latencies_all)

        logger.info(
            "Electrode %d @ %.2f Hz: mean spikes=%.2f, adaptation_idx=%.3f, "
            "latency_jitter=%.2f ms",
            electrode_idx,
            frequency,
            block_results.mean_spike_count,
            block_results.adaptation_index,
            block_results.latency_jitter_ms,
        )

        return block_results

    def _phase_validation_block(self, electrodes: List[int]) -> None:
        """Repeat 0.2 Hz stimulation for Day 2 validation."""
        logger.info("Validation phase: repeating 0.2 Hz stimulation")
        frequency = 0.2
        inter_stimulus_interval = 1.0 / frequency

        for electrode_idx in electrodes:
            block_results = self._stimulate_frequency_block(
                electrode_idx, frequency, inter_stimulus_interval
            )
            self.frequency_results[frequency].append(block_results)

    def _stimulate_and_record(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity = StimPolarity.NegativeFirst,
        trigger_key: int = 0,
        post_stim_wait_s: float = 0.3,
        recording_window_s: float = 0.5,
    ) -> pd.DataFrame:
        """Send one charge-balanced biphasic pulse and return spike events."""
        amplitude_ua = min(abs(amplitude_ua), 5.0)
        duration_us = min(abs(duration_us), 500.0)

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

        # Charge-balanced: A1*D1 == A2*D2
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

        # Fire trigger
        pattern = np.zeros(16, dtype=np.uint8)
        pattern[trigger_key] = 1
        self.trigger_controller.send(pattern)
        time.sleep(0.05)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

        time.sleep(post_stim_wait_s)

        # Query spike events
        query_start = datetime.now(timezone.utc) - timedelta(
            seconds=post_stim_wait_s + recording_window_s
        )
        query_stop = datetime.now(timezone.utc)
        spike_df = self.database.get_spike_event_electrode(
            query_start, query_stop, electrode_idx
        )
        return spike_df

    def _compute_adaptation_index(self, spike_counts: List[int]) -> float:
        """Compute adaptation index from spike count trajectory.
        
        Adaptation index = 1 - (final_response / initial_response).
        Values >0.8 indicate minimal adaptation (complete recovery).
        Values <0.5 indicate strong adaptation (incomplete recovery).
        """
        if len(spike_counts) < 2:
            return 0.0

        # Fit exponential decay: y = A + B*exp(-t/tau)
        # Compute first and last quartile means
        n = len(spike_counts)
        first_quartile = np.mean(spike_counts[: max(1, n // 4)])
        last_quartile = np.mean(spike_counts[max(1, 3 * n // 4) :])

        if first_quartile == 0:
            return 0.0

        adaptation_index = 1.0 - (last_quartile / first_quartile)
        return max(0.0, min(1.0, adaptation_index))

    def _compile_results(self) -> Dict[str, Any]:
        """Assemble comprehensive results summary."""
        logger.info("Compiling results")

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": self.recording_start.isoformat(),
            "recording_stop": self.recording_stop.isoformat(),
            "duration_seconds": (self.recording_stop - self.recording_start).total_seconds(),
            "total_trials": len(self.all_trials),
            "frequencies_tested": sorted(self.frequency_results.keys()),
        }

        # Baseline summary
        summary["baseline_spike_rates"] = {
            str(k): v for k, v in self.baseline_spike_rate.items()
        }

        # Frequency-dependent results
        frequency_summary = {}
        for frequency in sorted(self.frequency_results.keys()):
            blocks = self.frequency_results[frequency]
            if blocks:
                mean_adaptation = np.mean([b.adaptation_index for b in blocks])
                mean_spike_count = np.mean([b.mean_spike_count for b in blocks])
                mean_latency_jitter = np.mean([b.latency_jitter_ms for b in blocks])

                frequency_summary[str(frequency)] = {
                    "num_blocks": len(blocks),
                    "mean_adaptation_index": float(mean_adaptation),
                    "mean_spike_count": float(mean_spike_count),
                    "mean_latency_jitter_ms": float(mean_latency_jitter),
                }

        summary["frequency_summary"] = frequency_summary

        # Test predictions
        predictions_met = self._evaluate_predictions()
        summary["predictions_met"] = predictions_met

        # Save detailed results to CSV
        self._save_trials_csv()

        return summary

    def _evaluate_predictions(self) -> Dict[str, bool]:
        """Evaluate literature-derived testable predictions."""
        predictions = {}

        # Prediction 1: Adaptation index at 0.2 Hz > 0.8
        if 0.2 in self.frequency_results:
            blocks_02 = self.frequency_results[0.2]
            if blocks_02:
                mean_adapt_02 = np.mean([b.adaptation_index for b in blocks_02])
                predictions["adaptation_index_0.2Hz_gt_0.8"] = mean_adapt_02 > 0.8

        # Prediction 2: Adaptation index at 10 Hz < 0.5
        if 10.0 in self.frequency_results:
            blocks_10 = self.frequency_results[10.0]
            if blocks_10:
                mean_adapt_10 = np.mean([b.adaptation_index for b in blocks_10])
                predictions["adaptation_index_10Hz_lt_0.5"] = mean_adapt_10 < 0.5

        # Prediction 3: Latency jitter at 0.2 Hz < 5 ms
        if 0.2 in self.frequency_results:
            blocks_02 = self.frequency_results[0.2]
            if blocks_02:
                mean_jitter_02 = np.mean([b.latency_jitter_ms for b in blocks_02])
                predictions["latency_jitter_0.2Hz_lt_5ms"] = mean_jitter_02 < 5.0

        # Prediction 4: Latency jitter at 10 Hz > 10 ms
        if 10.0 in self.frequency_results:
            blocks_10 = self.frequency_results[10.0]
            if blocks_10:
                mean_jitter_10 = np.mean([b.latency_jitter_ms for b in blocks_10])
                predictions["latency_jitter_10Hz_gt_10ms"] = mean_jitter_10 > 10.0

        # Prediction 7: Normalized spike rate at 0.2 Hz is 2-5x baseline
        if 0.2 in self.frequency_results:
            blocks_02 = self.frequency_results[0.2]
            if blocks_02 and self.baseline_spike_rate:
                mean_spike_02 = np.mean([b.mean_spike_count for b in blocks_02])
                mean_baseline = np.mean(list(self.baseline_spike_rate.values()))
                if mean_baseline > 0:
                    ratio = mean_spike_02 / mean_baseline
                    predictions["spike_rate_0.2Hz_2to5x_baseline"] = 2.0 <= ratio <= 5.0

        return predictions

    def _save_trials_csv(self) -> None:
        """Save all trial data to CSV file."""
        if not self.all_trials:
            return

        csv_path = Path("frequency_adaptation_trials.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "electrode_idx",
                "frequency_hz",
                "amplitude_ua",
                "duration_us",
                "trial_num",
                "timestamp",
                "spike_count",
                "mean_latency_ms",
                "latency_jitter_ms",
            ])
            for trial in self.all_trials:
                mean_lat = (
                    np.mean(trial.spike_latencies_ms)
                    if trial.spike_latencies_ms
                    else 0.0
                )
                jitter = (
                    np.std(trial.spike_latencies_ms)
                    if trial.spike_latencies_ms
                    else 0.0
                )
                writer.writerow([
                    trial.electrode_idx,
                    trial.frequency_hz,
                    trial.amplitude_ua,
                    trial.duration_us,
                    trial.trial_num,
                    trial.timestamp.isoformat(),
                    trial.spike_count,
                    mean_lat,
                    jitter,
                ])

        logger.info("Saved trial data to %s", csv_path)

    def _cleanup(self) -> None:
        """Release all hardware resources."""
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
