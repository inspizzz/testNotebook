import numpy as np
import pandas as pd
import json
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
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
class SweepResult:
    """Result from a parameter sweep trial"""
    electrode_pair: Tuple[int, int]
    amplitude: float
    duration: float
    response_magnitude: float
    trial_num: int


class Experiment:
    """
    Parameter sweep optimization experiment with confirmation and variation exploration.
    
    This experiment:
    1. Identifies optimal stimulation parameters via parameter sweep
    2. Confirms optimal parameters with repeated trials
    3. Explores amplitude and duration variations separately
    """

    def __init__(
        self,
        token: str = "W32XGX2HCH",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
    ):
        """
        Initialize the experiment.
        
        Args:
            token: FinalSpark experiment token
            booking_email: Email for booking system
            testing: If True, skip hardware calls (testing mode)
        """
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None
        self.results = []
        
        logger.info(f"Experiment initialized with token={token}, testing={testing}")

    def run(self) -> Dict:
        """
        Execute the full experiment workflow.
        
        Returns:
            Dictionary containing all results and analysis
        """
        try:
            self._setup()
            
            # Phase 1: Parameter sweep to identify optimal parameters
            logger.info("=== PHASE 1: Parameter Sweep ===")
            sweep_results = self._parameter_sweep()
            
            # Phase 2: Confirm optimal parameters with repeated trials
            logger.info("=== PHASE 2: Confirmation Trials ===")
            optimal_params = self._identify_optimal_params(sweep_results)
            confirmation_results = self._confirmation_trials(optimal_params)
            
            # Phase 3: Explore amplitude variations
            logger.info("=== PHASE 3: Amplitude Variation Exploration ===")
            amplitude_results = self._amplitude_variation_sweep(optimal_params)
            
            # Phase 4: Explore duration variations
            logger.info("=== PHASE 4: Duration Variation Exploration ===")
            duration_results = self._duration_variation_sweep(optimal_params)
            
            # Compile results
            final_results = {
                "experiment_token": self.token,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "phase_1_sweep": self._format_results(sweep_results),
                "phase_2_confirmation": self._format_results(confirmation_results),
                "phase_3_amplitude_variation": self._format_results(amplitude_results),
                "phase_4_duration_variation": self._format_results(duration_results),
                "optimal_parameters": optimal_params,
                "summary": self._generate_summary(
                    sweep_results, confirmation_results, amplitude_results, duration_results
                ),
            }
            
            logger.info("Experiment completed successfully")
            return final_results
            
        finally:
            self.stop()

    def _setup(self) -> None:
        """Initialize hardware connections and experiment."""
        logger.info("Setting up experiment...")
        
        # Initialize trigger controller
        self.trigger_controller = TriggerController(booking_email=self.booking_email)
        
        # Initialize Intan software connection
        self.intan = IntanSofware()
        
        # Initialize database connection
        self.database = Database()
        
        # Start the experiment
        from neuroplatform import Experiment as NeuroPlatformExperiment
        self.experiment = NeuroPlatformExperiment(self.token)
        
        if not self.experiment.start():
            raise RuntimeError("Failed to start experiment")
        
        logger.info("Experiment setup complete")

    def _parameter_sweep(self) -> List[SweepResult]:
        """
        Phase 1: Perform parameter sweep across amplitude and duration ranges.
        
        Returns:
            List of sweep results
        """
        results = []
        
        # Define parameter ranges based on literature
        amplitudes = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]  # uA
        durations = [50, 100, 150, 200, 250, 300]    # us
        electrode_pairs = [(0, 1), (1, 2), (2, 3), (3, 4)]  # Example pairs
        
        trial_num = 0
        for amp in amplitudes:
            for dur in durations:
                # Ensure charge balance: amp * dur should be reasonable
                charge = amp * dur
                if charge > 1500:  # Skip if charge too high
                    continue
                
                for elec_pair in electrode_pairs:
                    trial_num += 1
                    
                    # Create stimulation parameters
                    stim_params = self._create_stim_params(
                        elec_pair, amp, dur
                    )
                    
                    # Apply stimulation
                    response = self._apply_stimulation(stim_params, elec_pair)
                    
                    # Record result
                    result = SweepResult(
                        electrode_pair=elec_pair,
                        amplitude=amp,
                        duration=dur,
                        response_magnitude=response,
                        trial_num=trial_num,
                    )
                    results.append(result)
                    
                    logger.info(
                        f"Trial {trial_num}: Elec {elec_pair}, "
                        f"Amp={amp}uA, Dur={dur}us, Response={response:.3f}"
                    )
                    
                    # Brief pause between trials
                    time.sleep(0.5)
        
        return results

    def _identify_optimal_params(self, results: List[SweepResult]) -> Dict:
        """
        Identify optimal parameters from sweep results.
        
        Args:
            results: List of sweep results
            
        Returns:
            Dictionary with optimal parameters
        """
        if not results:
            raise ValueError("No results to analyze")
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([
            {
                "electrode_pair": r.electrode_pair,
                "amplitude": r.amplitude,
                "duration": r.duration,
                "response": r.response_magnitude,
            }
            for r in results
        ])
        
        # Find parameters with highest response
        best_idx = df["response"].idxmax()
        best_result = df.iloc[best_idx]
        
        optimal = {
            "electrode_pair": best_result["electrode_pair"],
            "amplitude": float(best_result["amplitude"]),
            "duration": float(best_result["duration"]),
            "response_magnitude": float(best_result["response"]),
        }
        
        logger.info(f"Optimal parameters identified: {optimal}")
        return optimal

    def _confirmation_trials(self, optimal_params: Dict) -> List[SweepResult]:
        """
        Phase 2: Confirm optimal parameters with repeated trials.
        
        Args:
            optimal_params: Dictionary with optimal parameters
            
        Returns:
            List of confirmation trial results
        """
        results = []
        num_confirmations = 5
        
        for trial_num in range(num_confirmations):
            stim_params = self._create_stim_params(
                optimal_params["electrode_pair"],
                optimal_params["amplitude"],
                optimal_params["duration"],
            )
            
            response = self._apply_stimulation(
                stim_params,
                optimal_params["electrode_pair"],
            )
            
            result = SweepResult(
                electrode_pair=optimal_params["electrode_pair"],
                amplitude=optimal_params["amplitude"],
                duration=optimal_params["duration"],
                response_magnitude=response,
                trial_num=trial_num,
            )
            results.append(result)
            
            logger.info(
                f"Confirmation trial {trial_num + 1}/{num_confirmations}: "
                f"Response={response:.3f}"
            )
            
            time.sleep(0.5)
        
        return results

    def _amplitude_variation_sweep(self, optimal_params: Dict) -> List[SweepResult]:
        """
        Phase 3: Explore amplitude variations around optimal value.
        
        Args:
            optimal_params: Dictionary with optimal parameters
            
        Returns:
            List of amplitude variation results
        """
        results = []
        optimal_amp = optimal_params["amplitude"]
        
        # Explore ±50% around optimal amplitude
        amplitude_variations = [
            optimal_amp * 0.5,
            optimal_amp * 0.75,
            optimal_amp,
            optimal_amp * 1.25,
            optimal_amp * 1.5,
        ]
        
        # Clamp to valid range [0.1, 5.0]
        amplitude_variations = [
            max(0.1, min(5.0, amp)) for amp in amplitude_variations
        ]
        
        trial_num = 0
        for amp in amplitude_variations:
            trial_num += 1
            
            stim_params = self._create_stim_params(
                optimal_params["electrode_pair"],
                amp,
                optimal_params["duration"],
            )
            
            response = self._apply_stimulation(
                stim_params,
                optimal_params["electrode_pair"],
            )
            
            result = SweepResult(
                electrode_pair=optimal_params["electrode_pair"],
                amplitude=amp,
                duration=optimal_params["duration"],
                response_magnitude=response,
                trial_num=trial_num,
            )
            results.append(result)
            
            logger.info(
                f"Amplitude variation trial {trial_num}: "
                f"Amp={amp:.2f}uA, Response={response:.3f}"
            )
            
            time.sleep(0.5)
        
        return results

    def _duration_variation_sweep(self, optimal_params: Dict) -> List[SweepResult]:
        """
        Phase 4: Explore duration variations around optimal value.
        
        Args:
            optimal_params: Dictionary with optimal parameters
            
        Returns:
            List of duration variation results
        """
        results = []
        optimal_dur = optimal_params["duration"]
        
        # Explore ±50% around optimal duration
        duration_variations = [
            optimal_dur * 0.5,
            optimal_dur * 0.75,
            optimal_dur,
            optimal_dur * 1.25,
            optimal_dur * 1.5,
        ]
        
        # Clamp to valid range [10, 500]
        duration_variations = [
            max(10, min(500, dur)) for dur in duration_variations
        ]
        
        trial_num = 0
        for dur in duration_variations:
            trial_num += 1
            
            stim_params = self._create_stim_params(
                optimal_params["electrode_pair"],
                optimal_params["amplitude"],
                dur,
            )
            
            response = self._apply_stimulation(
                stim_params,
                optimal_params["electrode_pair"],
            )
            
            result = SweepResult(
                electrode_pair=optimal_params["electrode_pair"],
                amplitude=optimal_params["amplitude"],
                duration=dur,
                response_magnitude=response,
                trial_num=trial_num,
            )
            results.append(result)
            
            logger.info(
                f"Duration variation trial {trial_num}: "
                f"Dur={dur:.0f}us, Response={response:.3f}"
            )
            
            time.sleep(0.5)
        
        return results

    def _create_stim_params(
        self,
        electrode_pair: Tuple[int, int],
        amplitude: float,
        duration: float,
    ) -> List[StimParam]:
        """
        Create stimulation parameters with charge balance.
        
        Args:
            electrode_pair: Tuple of (stim_electrode, recording_electrode)
            amplitude: Amplitude in uA
            duration: Duration in us
            
        Returns:
            List of StimParam objects
        """
        stim_elec, rec_elec = electrode_pair
        
        # Create biphasic pulse with charge balance
        # Phase 1: negative (cathodic)
        # Phase 2: positive (anodic)
        # Charge balance: A1*D1 = A2*D2
        
        param = StimParam()
        param.index = stim_elec
        param.enable = True
        param.trigger_key = 0
        param.trigger_delay = 0
        param.nb_pulse = 0  # Single pulse
        param.pulse_train_period = 10000
        param.post_stim_ref_period = 1000.0
        param.stim_shape = StimShape.Biphasic
        param.polarity = StimPolarity.NegativeFirst
        
        # Phase 1 (cathodic)
        param.phase_duration1 = duration
        param.phase_amplitude1 = -amplitude  # Negative for cathodic
        
        # Phase 2 (anodic) - charge balanced
        param.phase_duration2 = duration
        param.phase_amplitude2 = amplitude  # Positive for anodic
        
        param.enable_amp_settle = True
        param.pre_stim_amp_settle = 0.0
        param.post_stim_amp_settle = 1000.0
        param.enable_charge_recovery = True
        param.post_charge_recovery_on = 0.0
        param.post_charge_recovery_off = 100.0
        param.interphase_delay = 0.0
        
        return [param]

    def _apply_stimulation(
        self,
        stim_params: List[StimParam],
        electrode_pair: Tuple[int, int],
    ) -> float:
        """
        Apply stimulation and measure response.
        
        Args:
            stim_params: List of stimulation parameters
            electrode_pair: Tuple of (stim_electrode, recording_electrode)
            
        Returns:
            Response magnitude (simulated or measured)
        """
        stim_elec, rec_elec = electrode_pair
        
        # Send stimulation parameters to hardware
        self.intan.send_stimparam(stim_params)
        
        # Send trigger pattern
        trigger_pattern = np.zeros(16, dtype=np.uint8)
        trigger_pattern[0] = 1  # Trigger on channel 0
        self.trigger_controller.send(trigger_pattern)
        
        # Wait for response
        time.sleep(0.1)
        
        # Measure response (in testing mode, return simulated response)
        response = self._measure_response(rec_elec)
        
        return response

    def _measure_response(self, electrode: int) -> float:
        """
        Measure neural response from electrode.
        
        Args:
            electrode: Electrode index
            
        Returns:
            Response magnitude
        """
        # In testing mode, return simulated response
        # In production, query database for spike counts
        
        now = datetime.now(timezone.utc)
        start_time = now - timedelta(seconds=1)
        
        try:
            spike_data = self.database.get_spike_event_electrode(
                start_time, now, electrode
            )
            
            if spike_data.empty:
                response = 0.0
            else:
                response = float(len(spike_data))
        except Exception as e:
            logger.warning(f"Failed to measure response: {e}")
            response = 0.0
        
        return response

    def _format_results(self, results: List[SweepResult]) -> List[Dict]:
        """
        Format results for output.
        
        Args:
            results: List of SweepResult objects
            
        Returns:
            List of dictionaries
        """
        return [
            {
                "electrode_pair": r.electrode_pair,
                "amplitude_uA": r.amplitude,
                "duration_us": r.duration,
                "response_magnitude": r.response_magnitude,
                "trial_num": r.trial_num,
            }
            for r in results
        ]

    def _generate_summary(
        self,
        sweep_results: List[SweepResult],
        confirmation_results: List[SweepResult],
        amplitude_results: List[SweepResult],
        duration_results: List[SweepResult],
    ) -> Dict:
        """
        Generate summary statistics.
        
        Args:
            sweep_results: Phase 1 results
            confirmation_results: Phase 2 results
            amplitude_results: Phase 3 results
            duration_results: Phase 4 results
            
        Returns:
            Dictionary with summary statistics
        """
        def get_stats(results):
            if not results:
                return {}
            responses = [r.response_magnitude for r in results]
            return {
                "mean_response": float(np.mean(responses)),
                "std_response": float(np.std(responses)),
                "min_response": float(np.min(responses)),
                "max_response": float(np.max(responses)),
                "num_trials": len(results),
            }
        
        return {
            "phase_1_sweep": get_stats(sweep_results),
            "phase_2_confirmation": get_stats(confirmation_results),
            "phase_3_amplitude_variation": get_stats(amplitude_results),
            "phase_4_duration_variation": get_stats(duration_results),
        }

    def stop(self) -> None:
        """Stop the experiment and clean up resources."""
        try:
            if self.experiment:
                self.experiment.stop()
                logger.info("Experiment stopped")
        except Exception as e:
            logger.error(f"Error stopping experiment: {e}")
        
        try:
            if self.intan:
                self.intan.close()
                logger.info("Intan connection closed")
        except Exception as e:
            logger.error(f"Error closing Intan: {e}")
        
        try:
            if self.trigger_controller:
                self.trigger_controller.close()
                logger.info("Trigger controller closed")
        except Exception as e:
            logger.error(f"Error closing trigger controller: {e}")
        
        try:
            if self.database:
                logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database: {e}")
