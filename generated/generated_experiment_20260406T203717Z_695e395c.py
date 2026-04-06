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
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StimulationTrial:
    """Record of a single stimulation trial."""
    electrode_idx: int
    amplitude_ua: float
    duration_us: float
    frequency_hz: float
    polarity: str
    trigger_time: datetime
    spike_count: int
    spike_latencies: List[float] = field(default_factory=list)
    spike_amplitudes: List[float] = field(default_factory=list)


@dataclass
class BaselineMetrics:
    """Baseline network metrics before stimulation."""
    recording_duration_s: float
    total_spike_count: int
    spike_rate_hz: float
    active_electrodes: int
    timestamp: datetime


@dataclass
class ResponseMetrics:
    """Response metrics during and after stimulation."""
    trial_id: int
    electrode_idx: int
    amplitude_ua: float
    duration_us: float
    frequency_hz: float
    spike_count_during: int
    spike_count_post: int
    latency_to_first_spike_ms: Optional[float]
    mean_spike_amplitude_uv: float
    synchrony_index: float


class Experiment:
    """
    Amplitude and duration variation experiment for neural organoid stimulation.
    
    Explores the effect of nearby amplitude and duration variations on neuronal
    response using parameter sweeps and deep scans to identify optimal stimulation
    parameters and responsive electrode pairs.
    """

    def __init__(
        self,
        token: str,
        booking_email: str,
        testing: bool = False,
        amplitude_range: Tuple[float, float] = (0.5, 5.0),
        duration_range: Tuple[float, float] = (50.0, 200.0),
        frequency_range: Tuple[float, float] = (10.0, 200.0),
        num_amplitude_levels: int = 5,
        num_duration_levels: int = 3,
        num_frequency_levels: int = 8,
        baseline_duration_s: float = 60.0,
        stimulation_duration_s: float = 10.0,
        inter_trial_interval_s: float = 180.0,
        post_stim_recovery_s: float = 120.0,
    ):
        """Initialize the experiment with parameter ranges and configuration."""
        self.token = token
        self.booking_email = booking_email
        self.testing = testing

        self.amplitude_range = amplitude_range
        self.duration_range = duration_range
        self.frequency_range = frequency_range
        self.num_amplitude_levels = num_amplitude_levels
        self.num_duration_levels = num_duration_levels
        self.num_frequency_levels = num_frequency_levels

        self.baseline_duration_s = baseline_duration_s
        self.stimulation_duration_s = stimulation_duration_s
        self.inter_trial_interval_s = inter_trial_interval_s
        self.post_stim_recovery_s = post_stim_recovery_s

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self.trials: List[StimulationTrial] = []
        self.baseline_metrics: Optional[BaselineMetrics] = None
        self.response_metrics: List[ResponseMetrics] = []
        self.electrode_responsiveness: Dict[int, float] = {}

    def run(self) -> Dict[str, Any]:
        """Execute the full experiment and return results."""
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

            recording_start = datetime.now(timezone.utc)

            self._phase_baseline_recording()
            self._phase_deep_scan()
            self._phase_amplitude_sweep()
            self._phase_duration_sweep()
            self._phase_frequency_sweep()
            self._phase_recovery_assessment()

            recording_stop = datetime.now(timezone.utc)

            results = self._compile_results(recording_start, recording_stop)
            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _phase_baseline_recording(self) -> None:
        """Record baseline network activity without stimulation."""
        logger.info("Phase: baseline recording (%s seconds)", self.baseline_duration_s)

        baseline_start = datetime.now(timezone.utc)
        time.sleep(self.baseline_duration_s)
        baseline_stop = datetime.now(timezone.utc)

        spike_df = self.database.get_spike_event(baseline_start, baseline_stop, self.experiment.exp_name)

        total_spikes = len(spike_df) if not spike_df.empty else 0
        duration = (baseline_stop - baseline_start).total_seconds()
        spike_rate = total_spikes / duration if duration > 0 else 0.0
        active_electrodes = len(spike_df['channel'].unique()) if not spike_df.empty else 0

        self.baseline_metrics = BaselineMetrics(
            recording_duration_s=duration,
            total_spike_count=total_spikes,
            spike_rate_hz=spike_rate,
            active_electrodes=active_electrodes,
            timestamp=baseline_start,
        )

        logger.info(
            "Baseline: %d spikes, %.2f Hz, %d active electrodes",
            total_spikes,
            spike_rate,
            active_electrodes,
        )

    def _phase_deep_scan(self) -> None:
        """Deep scan to identify most responsive electrode pairs."""
        logger.info("Phase: deep scan of electrode responsiveness")

        electrodes_to_test = self.experiment.electrodes[:16]
        test_amplitude = 2.0
        test_duration = 100.0
        test_frequency = 100.0

        for electrode_idx in electrodes_to_test:
            logger.info("Deep scan: testing electrode %d", electrode_idx)

            spike_df = self._stimulate_and_record(
                electrode_idx=electrode_idx,
                amplitude_ua=test_amplitude,
                duration_us=test_duration,
                frequency_hz=test_frequency,
                trigger_key=0,
                post_stim_wait_s=0.5,
                recording_window_s=2.0,
            )

            spike_count = len(spike_df) if not spike_df.empty else 0
            responsiveness = spike_count / (self.stimulation_duration_s + 1.0)
            self.electrode_responsiveness[electrode_idx] = responsiveness

            logger.info("Electrode %d: %d spikes, responsiveness %.2f", electrode_idx, spike_count, responsiveness)

            time.sleep(self.inter_trial_interval_s)

    def _phase_amplitude_sweep(self) -> None:
        """Sweep amplitude values to identify optimal amplitude."""
        logger.info("Phase: amplitude sweep")

        amplitudes = np.linspace(
            self.amplitude_range[0],
            self.amplitude_range[1],
            self.num_amplitude_levels,
        )

        best_electrodes = sorted(
            self.electrode_responsiveness.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:4]

        for electrode_idx, _ in best_electrodes:
            logger.info("Amplitude sweep: electrode %d", electrode_idx)

            for amplitude in amplitudes:
                duration = 100.0
                frequency = 100.0

                spike_df = self._stimulate_and_record(
                    electrode_idx=electrode_idx,
                    amplitude_ua=float(amplitude),
                    duration_us=duration,
                    frequency_hz=frequency,
                    trigger_key=0,
                    post_stim_wait_s=0.5,
                    recording_window_s=1.5,
                )

                spike_count = len(spike_df) if not spike_df.empty else 0
                mean_amplitude = spike_df['Amplitude'].mean() if not spike_df.empty else 0.0
                latency = self._compute_latency(spike_df) if not spike_df.empty else None

                trial = StimulationTrial(
                    electrode_idx=electrode_idx,
                    amplitude_ua=float(amplitude),
                    duration_us=duration,
                    frequency_hz=frequency,
                    polarity="NegativeFirst",
                    trigger_time=datetime.now(timezone.utc),
                    spike_count=spike_count,
                    spike_amplitudes=[float(a) for a in spike_df['Amplitude'].tolist()] if not spike_df.empty else [],
                )
                self.trials.append(trial)

                response = ResponseMetrics(
                    trial_id=len(self.response_metrics),
                    electrode_idx=electrode_idx,
                    amplitude_ua=float(amplitude),
                    duration_us=duration,
                    frequency_hz=frequency,
                    spike_count_during=spike_count,
                    spike_count_post=0,
                    latency_to_first_spike_ms=latency,
                    mean_spike_amplitude_uv=mean_amplitude,
                    synchrony_index=0.0,
                )
                self.response_metrics.append(response)

                logger.info(
                    "Amplitude %.2f uA: %d spikes, mean amplitude %.2f uV",
                    amplitude,
                    spike_count,
                    mean_amplitude,
                )

                time.sleep(self.inter_trial_interval_s)

    def _phase_duration_sweep(self) -> None:
        """Sweep duration values to identify optimal duration."""
        logger.info("Phase: duration sweep")

        durations = np.linspace(
            self.duration_range[0],
            self.duration_range[1],
            self.num_duration_levels,
        )

        best_electrodes = sorted(
            self.electrode_responsiveness.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:4]

        for electrode_idx, _ in best_electrodes:
            logger.info("Duration sweep: electrode %d", electrode_idx)

            for duration in durations:
                amplitude = 2.0
                frequency = 100.0

                spike_df = self._stimulate_and_record(
                    electrode_idx=electrode_idx,
                    amplitude_ua=amplitude,
                    duration_us=float(duration),
                    frequency_hz=frequency,
                    trigger_key=0,
                    post_stim_wait_s=0.5,
                    recording_window_s=1.5,
                )

                spike_count = len(spike_df) if not spike_df.empty else 0
                mean_amplitude = spike_df['Amplitude'].mean() if not spike_df.empty else 0.0
                latency = self._compute_latency(spike_df) if not spike_df.empty else None

                trial = StimulationTrial(
                    electrode_idx=electrode_idx,
                    amplitude_ua=amplitude,
                    duration_us=float(duration),
                    frequency_hz=frequency,
                    polarity="NegativeFirst",
                    trigger_time=datetime.now(timezone.utc),
                    spike_count=spike_count,
                    spike_amplitudes=[float(a) for a in spike_df['Amplitude'].tolist()] if not spike_df.empty else [],
                )
                self.trials.append(trial)

                response = ResponseMetrics(
                    trial_id=len(self.response_metrics),
                    electrode_idx=electrode_idx,
                    amplitude_ua=amplitude,
                    duration_us=float(duration),
                    frequency_hz=frequency,
                    spike_count_during=spike_count,
                    spike_count_post=0,
                    latency_to_first_spike_ms=latency,
                    mean_spike_amplitude_uv=mean_amplitude,
                    synchrony_index=0.0,
                )
                self.response_metrics.append(response)

                logger.info(
                    "Duration %.1f us: %d spikes, mean amplitude %.2f uV",
                    duration,
                    spike_count,
                    mean_amplitude,
                )

                time.sleep(self.inter_trial_interval_s)

    def _phase_frequency_sweep(self) -> None:
        """Sweep frequency values to identify optimal frequency."""
        logger.info("Phase: frequency sweep")

        frequencies = np.logspace(
            np.log10(self.frequency_range[0]),
            np.log10(self.frequency_range[1]),
            self.num_frequency_levels,
        )

        best_electrodes = sorted(
            self.electrode_responsiveness.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:4]

        for electrode_idx, _ in best_electrodes:
            logger.info("Frequency sweep: electrode %d", electrode_idx)

            for frequency in frequencies:
                amplitude = 2.0
                duration = 100.0

                spike_df = self._stimulate_and_record(
                    electrode_idx=electrode_idx,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    frequency_hz=float(frequency),
                    trigger_key=0,
                    post_stim_wait_s=0.5,
                    recording_window_s=1.5,
                )

                spike_count = len(spike_df) if not spike_df.empty else 0
                mean_amplitude = spike_df['Amplitude'].mean() if not spike_df.empty else 0.0
                latency = self._compute_latency(spike_df) if not spike_df.empty else None

                trial = StimulationTrial(
                    electrode_idx=electrode_idx,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    frequency_hz=float(frequency),
                    polarity="NegativeFirst",
                    trigger_time=datetime.now(timezone.utc),
                    spike_count=spike_count,
                    spike_amplitudes=[float(a) for a in spike_df['Amplitude'].tolist()] if not spike_df.empty else [],
                )
                self.trials.append(trial)

                response = ResponseMetrics(
                    trial_id=len(self.response_metrics),
                    electrode_idx=electrode_idx,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    frequency_hz=float(frequency),
                    spike_count_during=spike_count,
                    spike_count_post=0,
                    latency_to_first_spike_ms=latency,
                    mean_spike_amplitude_uv=mean_amplitude,
                    synchrony_index=0.0,
                )
                self.response_metrics.append(response)

                logger.info(
                    "Frequency %.1f Hz: %d spikes, mean amplitude %.2f uV",
                    frequency,
                    spike_count,
                    mean_amplitude,
                )

                time.sleep(self.inter_trial_interval_s)

    def _phase_recovery_assessment(self) -> None:
        """Assess network recovery post-stimulation."""
        logger.info("Phase: recovery assessment")

        recovery_start = datetime.now(timezone.utc)
        time.sleep(self.post_stim_recovery_s)
        recovery_stop = datetime.now(timezone.utc)

        spike_df = self.database.get_spike_event(recovery_start, recovery_stop, self.experiment.exp_name)

        total_spikes = len(spike_df) if not spike_df.empty else 0
        duration = (recovery_stop - recovery_start).total_seconds()
        spike_rate = total_spikes / duration if duration > 0 else 0.0

        logger.info(
            "Recovery: %d spikes, %.2f Hz",
            total_spikes,
            spike_rate,
        )

    def _stimulate_and_record(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        frequency_hz: float,
        trigger_key: int = 0,
        post_stim_wait_s: float = 0.5,
        recording_window_s: float = 1.5,
    ) -> pd.DataFrame:
        """Send charge-balanced biphasic pulses and return spike events."""
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
        stim.polarity = StimPolarity.NegativeFirst

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
        time.sleep(0.05)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

        time.sleep(post_stim_wait_s)

        query_start = datetime.now(timezone.utc) - timedelta(
            seconds=post_stim_wait_s + recording_window_s
        )
        query_stop = datetime.now(timezone.utc)
        spike_df = self.database.get_spike_event_electrode(
            query_start, query_stop, electrode_idx
        )
        return spike_df

    def _compute_latency(self, spike_df: pd.DataFrame) -> Optional[float]:
        """Compute latency to first spike in milliseconds."""
        if spike_df.empty:
            return None
        spike_times = pd.to_datetime(spike_df['Time'])
        if len(spike_times) == 0:
            return None
        first_spike = spike_times.min()
        now = datetime.now(timezone.utc)
        latency_ms = (first_spike - now).total_seconds() * 1000.0
        return max(0.0, latency_ms)

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        """Assemble results summary."""
        logger.info("Compiling results")

        amplitude_responses = defaultdict(list)
        duration_responses = defaultdict(list)
        frequency_responses = defaultdict(list)

        for response in self.response_metrics:
            amplitude_responses[response.amplitude_ua].append(response.spike_count_during)
            duration_responses[response.duration_us].append(response.spike_count_during)
            frequency_responses[response.frequency_hz].append(response.spike_count_during)

        optimal_amplitude = max(
            amplitude_responses.items(),
            key=lambda x: np.mean(x[1]),
            default=(0.0, [0]),
        )[0]

        optimal_duration = max(
            duration_responses.items(),
            key=lambda x: np.mean(x[1]),
            default=(0.0, [0]),
        )[0]

        optimal_frequency = max(
            frequency_responses.items(),
            key=lambda x: np.mean(x[1]),
            default=(0.0, [0]),
        )[0]

        most_responsive_electrode = max(
            self.electrode_responsiveness.items(),
            key=lambda x: x[1],
            default=(0, 0.0),
        )[0]

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "baseline_metrics": asdict(self.baseline_metrics) if self.baseline_metrics else {},
            "total_trials": len(self.trials),
            "total_responses": len(self.response_metrics),
            "optimal_amplitude_ua": float(optimal_amplitude),
            "optimal_duration_us": float(optimal_duration),
            "optimal_frequency_hz": float(optimal_frequency),
            "most_responsive_electrode": int(most_responsive_electrode),
            "electrode_responsiveness": {int(k): float(v) for k, v in self.electrode_responsiveness.items()},
        }

        return summary

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
