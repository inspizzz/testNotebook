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
    timestamp: datetime
    spike_count: int
    spike_latencies_ms: List[float] = field(default_factory=list)
    response_quality: str = "unknown"


@dataclass
class ElectrodeResponseProfile:
    """Summary of an electrode's response characteristics."""
    electrode_idx: int
    responsive: bool
    optimal_amplitude_ua: Optional[float] = None
    optimal_duration_us: Optional[float] = None
    optimal_frequency_hz: Optional[float] = None
    max_spike_count: int = 0
    trials_count: int = 0


@dataclass
class ExperimentResults:
    """Complete experiment results summary."""
    status: str
    experiment_name: str
    recording_start: str
    recording_stop: str
    duration_seconds: float
    responsive_electrodes: List[int] = field(default_factory=list)
    electrode_profiles: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    optimal_parameters: Dict[str, float] = field(default_factory=dict)
    total_trials: int = 0
    total_spikes: int = 0


class Experiment:
    """
    Frequency and amplitude sweep experiment to identify optimal stimulation
    parameters for neural organoid networks.
    
    Based on literature hypothesis: optimal desynchronization occurs at ~100 Hz
    with amplitude-dependent scaling. This experiment explores nearby parameter
    variations to identify responsive electrode pairs and optimal stimulation
    parameters.
    """

    def __init__(
        self,
        token: str = "",
        booking_email: str = "",
        testing: bool = False,
        frequencies_hz: Optional[List[float]] = None,
        amplitudes_ua: Optional[List[float]] = None,
        durations_us: Optional[List[float]] = None,
        num_trials_per_condition: int = 2,
        inter_trial_interval_s: float = 5.0,
        recording_window_s: float = 1.0,
        target_electrodes: Optional[List[int]] = None,
    ):
        """
        Initialize the frequency/amplitude sweep experiment.
        
        Args:
            token: FinalSpark experiment token.
            booking_email: Booking email for trigger access.
            testing: Flag for testing mode (stored but not used for branching).
            frequencies_hz: List of frequencies to test (default: [50, 75, 100, 125, 150]).
            amplitudes_ua: List of amplitudes to test (default: [2.0, 3.5, 5.0]).
            durations_us: List of pulse durations to test (default: [100, 200, 300]).
            num_trials_per_condition: Repetitions per condition.
            inter_trial_interval_s: Rest time between trials.
            recording_window_s: Duration of spike recording window post-stimulus.
            target_electrodes: Specific electrodes to test (default: first 8).
        """
        self.token = token
        self.booking_email = booking_email
        self.testing = testing

        # Default parameter ranges based on literature
        self.frequencies_hz = frequencies_hz or [50, 75, 100, 125, 150]
        self.amplitudes_ua = amplitudes_ua or [2.0, 3.5, 5.0]
        self.durations_us = durations_us or [100, 200, 300]
        self.num_trials_per_condition = num_trials_per_condition
        self.inter_trial_interval_s = inter_trial_interval_s
        self.recording_window_s = recording_window_s
        self.target_electrodes = target_electrodes

        # Hardware handles
        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        # Results storage
        self.trials: List[StimulationTrial] = []
        self.electrode_profiles: Dict[int, ElectrodeResponseProfile] = {}
        self.baseline_activity: Dict[int, int] = {}

    def run(self) -> Dict[str, Any]:
        """Execute the full parameter sweep experiment."""
        try:
            logger.info("Initializing hardware connections")

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

            # Determine target electrodes
            if self.target_electrodes is None:
                self.target_electrodes = self.experiment.electrodes[:8]
            else:
                self.target_electrodes = [
                    e for e in self.target_electrodes if e in self.experiment.electrodes
                ]

            logger.info("Target electrodes: %s", self.target_electrodes)

            # Initialize electrode profiles
            for elec in self.target_electrodes:
                self.electrode_profiles[elec] = ElectrodeResponseProfile(
                    electrode_idx=elec, responsive=False
                )

            # Experiment phases
            self._phase_baseline_recording()
            self._phase_amplitude_sweep()
            self._phase_frequency_sweep()
            self._phase_duration_sweep()
            self._phase_optimal_parameter_validation()

            recording_stop = datetime.now(timezone.utc)

            results = self._compile_results(recording_start, recording_stop)
            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _phase_baseline_recording(self) -> None:
        """Record baseline activity without stimulation."""
        logger.info("Phase: baseline recording (no stimulation)")
        
        for elec in self.target_electrodes:
            time.sleep(0.5)
            query_start = datetime.now(timezone.utc) - timedelta(seconds=2.0)
            query_stop = datetime.now(timezone.utc)
            
            spike_df = self.database.get_spike_event_electrode(
                query_start, query_stop, elec
            )
            baseline_count = len(spike_df) if not spike_df.empty else 0
            self.baseline_activity[elec] = baseline_count
            logger.info(
                "Electrode %d baseline spikes: %d", elec, baseline_count
            )

    def _phase_amplitude_sweep(self) -> None:
        """Sweep amplitudes at fixed frequency (100 Hz) and duration (100 us)."""
        logger.info("Phase: amplitude sweep (fixed frequency=100 Hz, duration=100 us)")
        
        fixed_frequency = 100.0
        fixed_duration = 100.0
        
        for elec in self.target_electrodes:
            logger.info("Testing electrode %d", elec)
            
            for amplitude in self.amplitudes_ua:
                for trial_num in range(self.num_trials_per_condition):
                    spike_df = self._stimulate_and_record(
                        elec, amplitude, fixed_duration, fixed_frequency
                    )
                    
                    spike_count = len(spike_df) if not spike_df.empty else 0
                    spike_latencies = self._extract_latencies(spike_df)
                    
                    trial = StimulationTrial(
                        electrode_idx=elec,
                        amplitude_ua=amplitude,
                        duration_us=fixed_duration,
                        frequency_hz=fixed_frequency,
                        polarity="NegativeFirst",
                        timestamp=datetime.now(timezone.utc),
                        spike_count=spike_count,
                        spike_latencies_ms=spike_latencies,
                        response_quality=self._classify_response(spike_count),
                    )
                    self.trials.append(trial)
                    
                    logger.info(
                        "Electrode %d, Amplitude %.1f uA, Trial %d: %d spikes",
                        elec, amplitude, trial_num + 1, spike_count
                    )
                    
                    time.sleep(self.inter_trial_interval_s)
                    
                    # Update electrode profile
                    if spike_count > self.electrode_profiles[elec].max_spike_count:
                        self.electrode_profiles[elec].max_spike_count = spike_count
                        self.electrode_profiles[elec].optimal_amplitude_ua = amplitude
                        self.electrode_profiles[elec].responsive = True

    def _phase_frequency_sweep(self) -> None:
        """Sweep frequencies at optimal amplitude and fixed duration."""
        logger.info("Phase: frequency sweep")
        
        # Use median amplitude as fixed value
        fixed_amplitude = self.amplitudes_ua[len(self.amplitudes_ua) // 2]
        fixed_duration = 100.0
        
        for elec in self.target_electrodes:
            logger.info("Testing electrode %d", elec)
            
            for frequency in self.frequencies_hz:
                for trial_num in range(self.num_trials_per_condition):
                    spike_df = self._stimulate_and_record(
                        elec, fixed_amplitude, fixed_duration, frequency
                    )
                    
                    spike_count = len(spike_df) if not spike_df.empty else 0
                    spike_latencies = self._extract_latencies(spike_df)
                    
                    trial = StimulationTrial(
                        electrode_idx=elec,
                        amplitude_ua=fixed_amplitude,
                        duration_us=fixed_duration,
                        frequency_hz=frequency,
                        polarity="NegativeFirst",
                        timestamp=datetime.now(timezone.utc),
                        spike_count=spike_count,
                        spike_latencies_ms=spike_latencies,
                        response_quality=self._classify_response(spike_count),
                    )
                    self.trials.append(trial)
                    
                    logger.info(
                        "Electrode %d, Frequency %.1f Hz, Trial %d: %d spikes",
                        elec, frequency, trial_num + 1, spike_count
                    )
                    
                    time.sleep(self.inter_trial_interval_s)
                    
                    # Update electrode profile
                    if spike_count > self.electrode_profiles[elec].max_spike_count:
                        self.electrode_profiles[elec].max_spike_count = spike_count
                        self.electrode_profiles[elec].optimal_frequency_hz = frequency
                        self.electrode_profiles[elec].responsive = True

    def _phase_duration_sweep(self) -> None:
        """Sweep durations at optimal amplitude and frequency."""
        logger.info("Phase: duration sweep")
        
        fixed_amplitude = self.amplitudes_ua[len(self.amplitudes_ua) // 2]
        fixed_frequency = 100.0
        
        for elec in self.target_electrodes:
            logger.info("Testing electrode %d", elec)
            
            for duration in self.durations_us:
                for trial_num in range(self.num_trials_per_condition):
                    spike_df = self._stimulate_and_record(
                        elec, fixed_amplitude, duration, fixed_frequency
                    )
                    
                    spike_count = len(spike_df) if not spike_df.empty else 0
                    spike_latencies = self._extract_latencies(spike_df)
                    
                    trial = StimulationTrial(
                        electrode_idx=elec,
                        amplitude_ua=fixed_amplitude,
                        duration_us=duration,
                        frequency_hz=fixed_frequency,
                        polarity="NegativeFirst",
                        timestamp=datetime.now(timezone.utc),
                        spike_count=spike_count,
                        spike_latencies_ms=spike_latencies,
                        response_quality=self._classify_response(spike_count),
                    )
                    self.trials.append(trial)
                    
                    logger.info(
                        "Electrode %d, Duration %.0f us, Trial %d: %d spikes",
                        elec, duration, trial_num + 1, spike_count
                    )
                    
                    time.sleep(self.inter_trial_interval_s)
                    
                    # Update electrode profile
                    if spike_count > self.electrode_profiles[elec].max_spike_count:
                        self.electrode_profiles[elec].max_spike_count = spike_count
                        self.electrode_profiles[elec].optimal_duration_us = duration
                        self.electrode_profiles[elec].responsive = True

    def _phase_optimal_parameter_validation(self) -> None:
        """Validate optimal parameters with additional trials."""
        logger.info("Phase: optimal parameter validation")
        
        responsive_electrodes = [
            elec for elec, profile in self.electrode_profiles.items()
            if profile.responsive
        ]
        
        if not responsive_electrodes:
            logger.warning("No responsive electrodes found")
            return
        
        logger.info("Responsive electrodes: %s", responsive_electrodes)
        
        for elec in responsive_electrodes:
            profile = self.electrode_profiles[elec]
            
            # Use identified optimal parameters
            amplitude = profile.optimal_amplitude_ua or self.amplitudes_ua[0]
            duration = profile.optimal_duration_us or 100.0
            frequency = profile.optimal_frequency_hz or 100.0
            
            logger.info(
                "Validating electrode %d with A=%.1f uA, D=%.0f us, F=%.1f Hz",
                elec, amplitude, duration, frequency
            )
            
            for trial_num in range(3):
                spike_df = self._stimulate_and_record(
                    elec, amplitude, duration, frequency
                )
                
                spike_count = len(spike_df) if not spike_df.empty else 0
                spike_latencies = self._extract_latencies(spike_df)
                
                trial = StimulationTrial(
                    electrode_idx=elec,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    frequency_hz=frequency,
                    polarity="NegativeFirst",
                    timestamp=datetime.now(timezone.utc),
                    spike_count=spike_count,
                    spike_latencies_ms=spike_latencies,
                    response_quality=self._classify_response(spike_count),
                )
                self.trials.append(trial)
                
                logger.info(
                    "Electrode %d validation trial %d: %d spikes",
                    elec, trial_num + 1, spike_count
                )
                
                time.sleep(self.inter_trial_interval_s)

    def _stimulate_and_record(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        frequency_hz: float,
        trigger_key: int = 0,
    ) -> pd.DataFrame:
        """
        Send charge-balanced biphasic pulse(s) and record spike response.
        
        For pulse trains, calculate pulse interval from frequency.
        Charge balance: amplitude1 * duration1 == amplitude2 * duration2.
        """
        # Safety clamps
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

        # Charge-balanced biphasic: A1*D1 == A2*D2
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
        stim.interphase_delay = 10.0

        self.intan.send_stimparam([stim])

        # Fire trigger
        pattern = np.zeros(16, dtype=np.uint8)
        pattern[trigger_key] = 1
        self.trigger_controller.send(pattern)
        time.sleep(0.05)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

        time.sleep(0.1)

        # Query spike events
        query_start = datetime.now(timezone.utc) - timedelta(
            seconds=self.recording_window_s + 0.2
        )
        query_stop = datetime.now(timezone.utc)
        spike_df = self.database.get_spike_event_electrode(
            query_start, query_stop, electrode_idx
        )
        return spike_df

    def _extract_latencies(self, spike_df: pd.DataFrame) -> List[float]:
        """Extract spike latencies in milliseconds from spike dataframe."""
        if spike_df.empty:
            return []
        
        latencies = []
        try:
            if "Time" in spike_df.columns:
                times = pd.to_datetime(spike_df["Time"])
                first_time = times.min()
                latencies = [(t - first_time).total_seconds() * 1000 for t in times]
        except Exception as e:
            logger.warning("Error extracting latencies: %s", e)
        
        return latencies

    def _classify_response(self, spike_count: int) -> str:
        """Classify response quality based on spike count."""
        if spike_count == 0:
            return "no_response"
        elif spike_count < 3:
            return "weak"
        elif spike_count < 10:
            return "moderate"
        else:
            return "strong"

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        """Assemble comprehensive results summary."""
        logger.info("Compiling results")

        responsive_electrodes = [
            elec for elec, profile in self.electrode_profiles.items()
            if profile.responsive
        ]

        # Calculate optimal parameters across all responsive electrodes
        optimal_params = self._calculate_optimal_parameters()

        # Build electrode profiles dict
        electrode_profiles_dict = {}
        for elec, profile in self.electrode_profiles.items():
            electrode_profiles_dict[elec] = {
                "responsive": profile.responsive,
                "optimal_amplitude_ua": profile.optimal_amplitude_ua,
                "optimal_duration_us": profile.optimal_duration_us,
                "optimal_frequency_hz": profile.optimal_frequency_hz,
                "max_spike_count": profile.max_spike_count,
                "trials_count": profile.trials_count,
            }

        total_spikes = sum(trial.spike_count for trial in self.trials)

        results = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "responsive_electrodes": responsive_electrodes,
            "electrode_profiles": electrode_profiles_dict,
            "optimal_parameters": optimal_params,
            "total_trials": len(self.trials),
            "total_spikes": total_spikes,
            "baseline_activity": self.baseline_activity,
        }

        return results

    def _calculate_optimal_parameters(self) -> Dict[str, float]:
        """Calculate globally optimal parameters from all trials."""
        if not self.trials:
            return {}

        # Find trial with maximum spike count
        best_trial = max(self.trials, key=lambda t: t.spike_count)

        return {
            "optimal_amplitude_ua": best_trial.amplitude_ua,
            "optimal_duration_us": best_trial.duration_us,
            "optimal_frequency_hz": best_trial.frequency_hz,
            "max_spikes_achieved": best_trial.spike_count,
        }

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
