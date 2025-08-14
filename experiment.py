import numpy as np
import time
from datetime import datetime
from neuroplatform import StimParam, IntanSofware, Trigger, StimPolarity, Experiment


class NeuroStimulationExperiment:
    """
    A simple neuron stimulation experiment class that runs a 5-10 minute experiment
    stimulating neurons on specified electrodes with configurable parameters.
    """
    
    def __init__(self):
        self.experiment = None
        self.intan = None
        self.trigger_gen = None
        self.stim_params = []
    
    def _create_stimulation_parameters(self, electrode_indices, trigger_keys, amplitudes, durations):
        """Create stimulation parameters for the specified electrodes."""
        self.stim_params = []
        
        for i, (electrode_idx, trigger_key, amplitude, duration) in enumerate(
            zip(electrode_indices, trigger_keys, amplitudes, durations)
        ):
            stim_param = StimParam()
            stim_param.enable = True
            stim_param.index = electrode_idx
            stim_param.trigger_key = trigger_key
            stim_param.polarity = StimPolarity.NegativeFirst
            stim_param.phase_duration1 = duration
            stim_param.phase_duration2 = duration
            stim_param.phase_amplitude1 = amplitude
            stim_param.phase_amplitude2 = amplitude
            
            self.stim_params.append(stim_param)
            print(f"Created StimParam {i+1}: Electrode {electrode_idx}, Trigger {trigger_key}, "
                  f"Amplitude {amplitude}µA, Duration {duration}µs")
    
    def _setup_connections(self):
        """Setup connections to Intan software and trigger generator."""
        self.intan = IntanSofware()
        self.trigger_gen = Trigger()
        print("Connected to Intan software and trigger generator")
    
    def _send_parameters(self):
        """Send stimulation parameters to Intan software."""
        print("Sending stimulation parameters (this takes 10 seconds)...")
        self.intan.send_stimparam(self.stim_params)
        print("Parameters sent successfully")
    
    def _run_stimulation_sequence(self, experiment_duration_minutes=7, 
                                stimulation_interval_seconds=2):
        """Run the main stimulation sequence."""
        start_time = datetime.utcnow()
        print(f"Starting experiment at {start_time}")
        
        # Calculate total iterations based on duration and interval
        total_iterations = int((experiment_duration_minutes * 60) / stimulation_interval_seconds)
        
        for iteration in range(total_iterations):
            # Create trigger array for all configured triggers
            trigger_array = np.zeros(16, dtype=np.uint8)
            
            # Enable triggers for all configured stimulation parameters
            for stim_param in self.stim_params:
                trigger_array[stim_param.trigger_key] = 1
            
            # Send stimulation trigger
            self.trigger_gen.send(trigger_array)
            
            print(f"Stimulation {iteration + 1}/{total_iterations} sent at "
                  f"{datetime.utcnow().strftime('%H:%M:%S')}")
            
            # Wait for next stimulation
            time.sleep(stimulation_interval_seconds)
        
        end_time = datetime.utcnow()
        print(f"Experiment completed at {end_time}")
        print(f"Total duration: {end_time - start_time}")
    
    def _cleanup(self):
        """Disable stimulation parameters and close connections."""
        print("Cleaning up experiment...")
        
        # Disable all stimulation parameters
        for stim_param in self.stim_params:
            stim_param.enable = False
        
        # Send disabled parameters to Intan
        if self.intan and self.stim_params:
            self.intan.send_stimparam(self.stim_params)
            print("Disabled all stimulation parameters")
        
        # Close connections
        if self.trigger_gen:
            self.trigger_gen.close()
            print("Closed trigger generator connection")
        
        if self.intan:
            self.intan.close()
            print("Closed Intan software connection")
        
        if self.experiment:
            self.experiment.stop()
            print("Stopped experiment")
    
    def run(self, token, electrode_indices=None, amplitudes=None, durations=None,
            experiment_duration_minutes=7, stimulation_interval_seconds=2):
        """
        Run the neuron stimulation experiment.
        
        Args:
            token (str): Experiment token provided by FinalSpark
            electrode_indices (list): List of electrode indices to stimulate (default: [0, 1])
            amplitudes (list): List of stimulation amplitudes in µA (default: [10, 15])
            durations (list): List of phase durations in µs (default: [100, 100])
            experiment_duration_minutes (int): Total experiment duration in minutes (default: 7)
            stimulation_interval_seconds (int): Interval between stimulations in seconds (default: 2)
        """
        # Set default parameters if not provided
        if electrode_indices is None:
            electrode_indices = [0, 1]
        if amplitudes is None:
            amplitudes = [10, 15]
        if durations is None:
            durations = [100, 100]
        
        # Validate input parameters
        if len(electrode_indices) != len(amplitudes) or len(amplitudes) != len(durations):
            raise ValueError("electrode_indices, amplitudes, and durations must have the same length")
        
        # Generate trigger keys (one for each electrode)
        trigger_keys = list(range(len(electrode_indices)))
        
        try:
            # Initialize experiment
            self.experiment = Experiment(token)
            print(f"Available electrodes: {self.experiment.electrodes}")
            
            # Start experiment
            if not self.experiment.start():
                raise RuntimeError("Failed to start experiment")
            
            print("Experiment started successfully")
            
            # Create stimulation parameters
            self._create_stimulation_parameters(electrode_indices, trigger_keys, 
                                             amplitudes, durations)
            
            # Setup hardware connections
            self._setup_connections()
            
            # Send parameters to Intan
            self._send_parameters()
            
            # Run the stimulation sequence
            self._run_stimulation_sequence(experiment_duration_minutes, 
                                         stimulation_interval_seconds)
            
        except Exception as e:
            print(f"Error during experiment: {e}")
            raise
        
        finally:
            # Always cleanup, even if an error occurred
            self._cleanup()
