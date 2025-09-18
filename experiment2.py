import numpy as np
import time
import itertools
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

# Import FinalSpark NeuroPlatform modules
from neuroplatform import (
    StimParam, 
    IntanSofware, 
    Trigger, 
    StimPolarity, 
    Experiment,
    Database
)

@dataclass
class StimulationConfig:
    """Configuration for a single stimulation parameter set"""
    amplitude: float  # microAmps
    duration: float   # microseconds
    polarity: StimPolarity
    electrode_idx: int
    
@dataclass
class ResponseMetrics:
    """Metrics for evaluating stimulation response"""
    spike_count_pre: int
    spike_count_post: int
    response_latency: float
    peak_amplitude: float
    response_strength: float
    
class ClosedLoopStimulationExperiment:
    """
    A closed-loop stimulation experiment that tests combinations of amplitudes,
    durations, and polarities while monitoring responses and updating stimulation center.
    """
    
    def __init__(self, 
                 token: str,
                 testing_mode: bool = False,
                 amplitude_range: List[float] = [1.0, 2.0, 3.0],
                 duration_range: List[float] = [100.0, 200.0, 300.0],
                 polarity_options: List[StimPolarity] = None,
                 response_window_ms: float = 200.0,
                 baseline_window_ms: float = 200.0,
                 inter_stim_delay: float = 5.0):
        """
        Initialize the closed-loop stimulation experiment.
        
        Args:
            token: Experiment token for FinalSpark platform
            testing_mode: If True, doesn't send actual stimulations (for testing)
            amplitude_range: List of amplitudes to test (microAmps)
            duration_range: List of durations to test (microseconds)  
            polarity_options: List of polarities to test
            response_window_ms: Time window to analyze response after stimulation
            baseline_window_ms: Time window to analyze baseline before stimulation
            inter_stim_delay: Delay between stimulations (seconds)
        """
        self.token = token
        self.testing_mode = testing_mode
        self.amplitude_range = amplitude_range
        self.duration_range = duration_range
        self.polarity_options = polarity_options or [StimPolarity.NegativeFirst, StimPolarity.PositiveFirst]
        self.response_window_ms = response_window_ms
        self.baseline_window_ms = baseline_window_ms
        self.inter_stim_delay = inter_stim_delay
        
        # Initialize components
        self.exp = None
        self.intan = None
        self.trigger_gen = None
        self.db = None
        
        # Experiment state
        self.available_electrodes = []
        self.current_center_electrode = None
        self.stimulation_history = []
        self.response_history = []
        self.best_configs = {}
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for experiment tracking"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def initialize_connections(self):
        """Initialize all necessary connections to the platform"""
        try:
            self.exp = Experiment(self.token)
            self.available_electrodes = self.exp.electrodes
            self.current_center_electrode = self.available_electrodes[0] if self.available_electrodes else 0
            
            if not self.testing_mode:
                self.intan = IntanSofware()
                self.trigger_gen = Trigger()
            
            self.db = Database()
            
            self.logger.info(f"Initialized connections. Available electrodes: {self.available_electrodes}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize connections: {e}")
            raise
            
    def generate_stimulation_combinations(self) -> List[StimulationConfig]:
        """Generate all combinations of stimulation parameters"""
        combinations = []
        
        for amplitude, duration, polarity in itertools.product(
            self.amplitude_range, self.duration_range, self.polarity_options
        ):
            config = StimulationConfig(
                amplitude=amplitude,
                duration=duration,
                polarity=polarity,
                electrode_idx=self.current_center_electrode
            )
            combinations.append(config)
            
        self.logger.info(f"Generated {len(combinations)} stimulation combinations")
        return combinations
        
    def create_stim_param(self, config: StimulationConfig, trigger_key: int = 0) -> StimParam:
        """Create a StimParam object from configuration"""
        stim_param = StimParam()
        stim_param.enable = True
        stim_param.index = config.electrode_idx
        stim_param.trigger_key = trigger_key
        stim_param.polarity = config.polarity
        
        # Set phase durations and amplitudes for biphasic stimulation
        stim_param.phase_duration1 = config.duration
        stim_param.phase_duration2 = config.duration
        stim_param.phase_amplitude1 = config.amplitude
        stim_param.phase_amplitude2 = config.amplitude  # Charge balanced
        
        # Configure other parameters with safe defaults
        stim_param.interphase_delay = 0.0
        stim_param.post_trigger_delay = 0
        stim_param.nb_pulse = 1  # Single pulse
        stim_param.post_stim_refractory_period = 1000.0
        
        return stim_param
        
    def send_stimulation(self, config: StimulationConfig) -> datetime:
        """Send a stimulation and return the timestamp"""
        if self.testing_mode:
            self.logger.info(f"TEST MODE: Would stimulate electrode {config.electrode_idx} "
                           f"with {config.amplitude}µA, {config.duration}µs, {config.polarity}")
            return datetime.utcnow()
            
        try:
            # Create and send stimulation parameter
            stim_param = self.create_stim_param(config, trigger_key=0)
            self.intan.send_stimparam([stim_param])
            
            # Send trigger
            trigger_array = np.zeros(16, dtype=np.uint8)
            trigger_array[0] = 1
            
            stim_time = datetime.utcnow()
            self.trigger_gen.send(trigger_array)
            
            self.logger.info(f"Stimulated electrode {config.electrode_idx} at {stim_time}")
            return stim_time
            
        except Exception as e:
            self.logger.error(f"Failed to send stimulation: {e}")
            raise
            
    def analyze_response(self, stim_time: datetime, electrode_idx: int) -> ResponseMetrics:
        """Analyze the response to a stimulation"""
        try:
            # Define time windows
            baseline_start = stim_time - timedelta(milliseconds=self.baseline_window_ms)
            baseline_end = stim_time
            response_start = stim_time + timedelta(milliseconds=10)  # Avoid artifacts
            response_end = stim_time + timedelta(milliseconds=self.response_window_ms)
            
            if self.testing_mode:
                # Generate mock data for testing
                spike_count_pre = np.random.poisson(5)
                spike_count_post = np.random.poisson(15)
                response_latency = np.random.uniform(15, 50)
                peak_amplitude = np.random.uniform(50, 200)
            else:
                # Get actual spike data from database
                fs_name = self.exp.exp_name
                
                # Get baseline activity
                baseline_spikes = self.db.get_spike_event(baseline_start, baseline_end, fs_name)
                baseline_electrode_spikes = baseline_spikes[baseline_spikes['channel'] == electrode_idx]
                spike_count_pre = len(baseline_electrode_spikes)
                
                # Get response activity
                response_spikes = self.db.get_spike_event(response_start, response_end, fs_name)
                response_electrode_spikes = response_spikes[response_spikes['channel'] == electrode_idx]
                spike_count_post = len(response_electrode_spikes)
                
                # Calculate response latency and peak amplitude
                if len(response_electrode_spikes) > 0:
                    first_spike_time = response_electrode_spikes.iloc[0]['Time']
                    response_latency = (first_spike_time - stim_time).total_seconds() * 1000
                    peak_amplitude = response_electrode_spikes['amplitude'].max()
                else:
                    response_latency = float('inf')
                    peak_amplitude = 0.0
            
            # Calculate response strength (corrected for baseline)
            baseline_rate = spike_count_pre / (self.baseline_window_ms / 1000.0)
            response_rate = spike_count_post / ((self.response_window_ms - 10) / 1000.0)  # -10ms for artifact window
            response_strength = max(0, response_rate - baseline_rate)
            
            metrics = ResponseMetrics(
                spike_count_pre=spike_count_pre,
                spike_count_post=spike_count_post,
                response_latency=response_latency,
                peak_amplitude=peak_amplitude,
                response_strength=response_strength
            )
            
            self.logger.info(f"Response analysis: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to analyze response: {e}")
            # Return default metrics on error
            return ResponseMetrics(0, 0, float('inf'), 0.0, 0.0)
            
    def update_stimulation_center(self):
        """Update the center electrode based on response history"""
        if len(self.response_history) < 3:  # Need some history
            return
            
        # Analyze recent responses to find best performing electrode
        recent_responses = self.response_history[-10:]  # Last 10 stimulations
        electrode_performance = {}
        
        for stim_config, response in recent_responses:
            electrode = stim_config.electrode_idx
            if electrode not in electrode_performance:
                electrode_performance[electrode] = []
            electrode_performance[electrode].append(response.response_strength)
            
        # Find electrode with highest average response strength
        best_electrode = None
        best_avg_response = -1
        
        for electrode, responses in electrode_performance.items():
            avg_response = np.mean(responses)
            if avg_response > best_avg_response:
                best_avg_response = avg_response
                best_electrode = electrode
                
        # Update center if we found a better electrode
        if best_electrode and best_electrode != self.current_center_electrode:
            old_center = self.current_center_electrode
            self.current_center_electrode = best_electrode
            self.logger.info(f"Updated stimulation center from electrode {old_center} to {best_electrode}")
            
    def get_neighboring_electrodes(self, center_electrode: int, radius: int = 1) -> List[int]:
        """Get electrodes neighboring the center electrode (simplified for example)"""
        # This is a simplified version - in practice you'd use the actual MEA layout
        available_set = set(self.available_electrodes)
        neighbors = []
        
        for offset in range(-radius, radius + 1):
            neighbor = center_electrode + offset
            if neighbor in available_set and neighbor != center_electrode:
                neighbors.append(neighbor)
                
        return neighbors[:3]  # Limit to 3 neighbors for this example
        
    def run_parameter_sweep(self, electrode_idx: int) -> Dict:
        """Run parameter sweep on a specific electrode"""
        combinations = []
        
        # Generate combinations for this electrode
        for amplitude, duration, polarity in itertools.product(
            self.amplitude_range, self.duration_range, self.polarity_options
        ):
            config = StimulationConfig(
                amplitude=amplitude,
                duration=duration,
                polarity=polarity,
                electrode_idx=electrode_idx
            )
            combinations.append(config)
            
        electrode_results = []
        
        for config in combinations:
            # Send stimulation
            stim_time = self.send_stimulation(config)
            
            # Wait for response
            time.sleep(self.response_window_ms / 1000.0 + 0.1)  # Wait for response window + buffer
            
            # Analyze response
            response = self.analyze_response(stim_time, electrode_idx)
            
            # Store results
            electrode_results.append((config, response))
            self.stimulation_history.append(config)
            self.response_history.append((config, response))
            
            # Inter-stimulation delay
            time.sleep(self.inter_stim_delay)
            
        # Find best configuration for this electrode
        best_config, best_response = max(electrode_results, 
                                       key=lambda x: x[1].response_strength)
        
        return {
            'electrode': electrode_idx,
            'best_config': best_config,
            'best_response': best_response,
            'all_results': electrode_results
        }
        
    def run_closed_loop_experiment(self, max_iterations: int = 5):
        """Run the main closed-loop experiment"""
        self.logger.info(f"Starting closed-loop experiment with {max_iterations} iterations")
        
        for iteration in range(max_iterations):
            self.logger.info(f"=== Iteration {iteration + 1}/{max_iterations} ===")
            
            # Test current center electrode
            center_results = self.run_parameter_sweep(self.current_center_electrode)
            self.best_configs[self.current_center_electrode] = center_results
            
            # Test neighboring electrodes
            neighbors = self.get_neighboring_electrodes(self.current_center_electrode)
            
            for neighbor in neighbors:
                neighbor_results = self.run_parameter_sweep(neighbor)
                self.best_configs[neighbor] = neighbor_results
                
            # Update stimulation center based on results
            self.update_stimulation_center()
            
            # Log iteration summary
            center_performance = center_results['best_response'].response_strength
            self.logger.info(f"Iteration {iteration + 1} complete. "
                           f"Center electrode: {self.current_center_electrode}, "
                           f"Best response strength: {center_performance:.3f}")
                           
    def cleanup(self):
        """Clean up connections and disable stimulations"""
        try:
            if not self.testing_mode and self.intan:
                # Disable all stimulation parameters
                for electrode in self.available_electrodes:
                    stim_param = StimParam()
                    stim_param.index = electrode
                    stim_param.enable = False
                    stim_param.trigger_key = 0
                    
                self.intan.send_stimparam([stim_param])
                self.intan.close()
                
            if not self.testing_mode and self.trigger_gen:
                self.trigger_gen.close()
                
            if self.exp:
                self.exp.stop()
                
            self.logger.info("Experiment cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            
    def generate_report(self) -> Dict:
        """Generate a comprehensive experiment report"""
        report = {
            'experiment_summary': {
                'token': self.token,
                'testing_mode': self.testing_mode,
                'total_stimulations': len(self.stimulation_history),
                'electrodes_tested': list(self.best_configs.keys()),
                'final_center_electrode': self.current_center_electrode
            },
            'parameter_ranges': {
                'amplitudes': self.amplitude_range,
                'durations': self.duration_range,
                'polarities': [str(p) for p in self.polarity_options]
            },
            'best_configurations': {},
            'electrode_performance': {}
        }
        
        # Compile best configurations
        for electrode, results in self.best_configs.items():
            best_config = results['best_config']
            best_response = results['best_response']
            
            report['best_configurations'][electrode] = {
                'amplitude': best_config.amplitude,
                'duration': best_config.duration,
                'polarity': str(best_config.polarity),
                'response_strength': best_response.response_strength,
                'response_latency': best_response.response_latency,
                'peak_amplitude': best_response.peak_amplitude
            }
            
            report['electrode_performance'][electrode] = best_response.response_strength
            
        return report
        
    def run(self):
        """
        Main function to run the closed-loop stimulation experiment
        """
        
        try:
            self.logger.info("=== Starting FinalSpark Closed-Loop Stimulation Experiment ===")
            
            # Initialize all connections
            self.initialize_connections()
            
            # Start the experiment
            if not self.testing_mode:
                if not self.exp.start():
                    raise Exception("Failed to start experiment - another experiment may be running")
            
            # Run the closed-loop experiment
            self.run_closed_loop_experiment()
            
            # Generate and log final report
            report = self.generate_report()
            self.logger.info(f"Experiment completed successfully. Final report: {report}")
            
            print("\n=== EXPERIMENT COMPLETE ===")
            print("Final Report:")
            for key, value in report['experiment_summary'].items():
                print(f"{key}: {value}")
                
            print("\nBest performing electrode configurations:")
            for electrode, config in report['best_configurations'].items():
                print(f"Electrode {electrode}: {config['response_strength']:.3f} strength, "
                      f"{config['amplitude']}µA, {config['duration']}µs, {config['polarity']}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            print(f"Experiment failed: {e}")
            raise
            
        finally:
            self.cleanup()


def run(token: str = "9T5KLS6T7X", testing_mode: bool = True):
    """
    Convenience function to create and run the experiment
    
    Args:
        token: Experiment token for FinalSpark platform
        testing_mode: If True, runs in testing mode without actual stimulations
    """
    
    # Create and run experiment
    experiment = ClosedLoopStimulationExperiment(
        token=token,
        testing_mode=testing_mode
    )
    
    return experiment.run()


if __name__ == "__main__":
    # Run the experiment in testing mode by default
    run(testing_mode=True)
