import numpy as np
import pandas as pd
import json
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from neuroplatform import StimParam, StimPolarity, StimShape, IntanSofware, TriggerController, Database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StimulationConfig:
    electrode_from: int
    electrode_to: int
    amplitude: float
    duration: float
    polarity: str
    repeats: int = 5


class TrialRecorder:
    def __init__(self):
        self.trials = []
    
    def record_trial(self, stim_config: StimulationConfig, timestamp: datetime, response_detected: bool = False):
        self.trials.append({
            'electrode_from': stim_config.electrode_from,
            'electrode_to': stim_config.electrode_to,
            'amplitude': stim_config.amplitude,
            'duration': stim_config.duration,
            'polarity': stim_config.polarity,
            'timestamp': timestamp,
            'response_detected': response_detected
        })
    
    def get_summary(self) -> Dict:
        return {
            'total_trials': len(self.trials),
            'trials': self.trials
        }


class StimulationController:
    def __init__(self, testing: bool = False):
        self.testing = testing
        self.trigger_controller = None
        self.intan_software = None
        self.trial_recorder = TrialRecorder()
    
    def initialize_hardware(self, booking_email: str):
        if self.testing:
            logger.info("Testing mode: skipping hardware initialization")
            return
        
        try:
            self.trigger_controller = TriggerController(booking_email=booking_email)
            self.intan_software = IntanSofware()
            logger.info("Hardware initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize hardware: {e}")
            raise
    
    def create_stim_param(self, config: StimulationConfig) -> StimParam:
        param = StimParam()
        param.index = config.electrode_from
        param.enable = True
        param.trigger_key = 0
        param.trigger_delay = 0
        param.nb_pulse = 0
        param.pulse_train_period = 10000
        param.post_stim_ref_period = 1000.0
        param.stim_shape = StimShape.Biphasic
        
        if config.polarity == "StimPolarity.NegativeFirst":
            param.polarity = StimPolarity.NegativeFirst
        else:
            param.polarity = StimPolarity.PositiveFirst
        
        param.phase_duration1 = config.duration
        param.phase_duration2 = config.duration
        param.phase_amplitude1 = config.amplitude
        param.phase_amplitude2 = config.amplitude
        param.enable_amp_settle = True
        param.pre_stim_amp_settle = 0.0
        param.post_stim_amp_settle = 1000.0
        param.enable_charge_recovery = True
        param.post_charge_recovery_on = 0.0
        param.post_charge_recovery_off = 100.0
        param.interphase_delay = 0.0
        
        return param
    
    def send_stimulation(self, config: StimulationConfig):
        timestamp = datetime.now(timezone.utc)
        
        if self.testing:
            logger.info(f"[TEST] Would send stimulation: {config.electrode_from} -> {config.electrode_to}, "
                       f"A={config.amplitude}uA, D={config.duration}us")
            self.trial_recorder.record_trial(config, timestamp, response_detected=False)
            return
        
        try:
            param = self.create_stim_param(config)
            if self.intan_software:
                self.intan_software.send_stimparam([param])
            
            pattern = np.zeros(16, dtype=np.uint8)
            pattern[0] = 1
            
            if self.trigger_controller:
                self.trigger_controller.send(pattern)
            
            self.trial_recorder.record_trial(config, timestamp, response_detected=False)
            logger.info(f"Stimulation sent: {config.electrode_from} -> {config.electrode_to}")
        except Exception as e:
            logger.error(f"Failed to send stimulation: {e}")
            raise
    
    def cleanup(self):
        if self.testing:
            return
        
        try:
            if self.intan_software:
                self.intan_software.close()
            if self.trigger_controller:
                self.trigger_controller.close()
            logger.info("Hardware cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


class Experiment:
    def __init__(self, token: str = "", testing: bool = False, booking_email: str = ""):
        self.token = token
        self.testing = testing
        self.booking_email = booking_email
        self.stim_controller = StimulationController(testing=testing)
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def _validate_charge_balance(self, amplitude: float, duration: float) -> bool:
        a1 = amplitude
        d1 = duration
        a2 = amplitude
        d2 = duration
        return abs(a1 * d1 - a2 * d2) < 0.01
    
    def _validate_parameters(self, config: StimulationConfig) -> bool:
        if config.amplitude > 5.0:
            logger.error(f"Amplitude {config.amplitude} exceeds maximum 5.0 uA")
            return False
        if config.duration > 500.0:
            logger.error(f"Duration {config.duration} exceeds maximum 500 us")
            return False
        if not self._validate_charge_balance(config.amplitude, config.duration):
            logger.error(f"Charge imbalance detected for A={config.amplitude}, D={config.duration}")
            return False
        return True
    
    def _get_stimulation_configs(self) -> List[StimulationConfig]:
        configs = [
            StimulationConfig(
                electrode_from=4,
                electrode_to=0,
                amplitude=1.0,
                duration=200.0,
                polarity="StimPolarity.NegativeFirst",
                repeats=5
            ),
            StimulationConfig(
                electrode_from=4,
                electrode_to=0,
                amplitude=2.0,
                duration=100.0,
                polarity="StimPolarity.NegativeFirst",
                repeats=5
            ),
            StimulationConfig(
                electrode_from=4,
                electrode_to=0,
                amplitude=2.0,
                duration=200.0,
                polarity="StimPolarity.NegativeFirst",
                repeats=5
            ),
            StimulationConfig(
                electrode_from=5,
                electrode_to=0,
                amplitude=3.0,
                duration=200.0,
                polarity="StimPolarity.PositiveFirst",
                repeats=5
            ),
            StimulationConfig(
                electrode_from=5,
                electrode_to=1,
                amplitude=1.0,
                duration=200.0,
                polarity="StimPolarity.NegativeFirst",
                repeats=5
            ),
            StimulationConfig(
                electrode_from=5,
                electrode_to=1,
                amplitude=2.0,
                duration=200.0,
                polarity="StimPolarity.NegativeFirst",
                repeats=5
            ),
            StimulationConfig(
                electrode_from=5,
                electrode_to=1,
                amplitude=3.0,
                duration=100.0,
                polarity="StimPolarity.NegativeFirst",
                repeats=5
            ),
        ]
        return configs
    
    def run(self, token: str = "", testing: bool = False, booking_email: str = ""):
        self.token = token if token else self.token
        self.testing = testing if testing is not None else self.testing
        self.booking_email = booking_email if booking_email else self.booking_email
        
        self.start_time = datetime.now(timezone.utc)
        logger.info(f"Starting experiment at {self.start_time}")
        logger.info(f"Testing mode: {self.testing}")
        
        try:
            self.stim_controller.initialize_hardware(self.booking_email)
            
            configs = self._get_stimulation_configs()
            total_stimulations = 0
            
            for config in configs:
                if not self._validate_parameters(config):
                    logger.warning(f"Skipping invalid configuration: {config}")
                    continue
                
                for repeat in range(config.repeats):
                    if total_stimulations >= 50:
                        logger.info("Reached maximum stimulations limit (50)")
                        break
                    
                    try:
                        self.stim_controller.send_stimulation(config)
                        total_stimulations += 1
                        time.sleep(0.1)
                    except Exception as e:
                        logger.error(f"Error during stimulation repeat {repeat}: {e}")
                        continue
                
                if total_stimulations >= 50:
                    break
            
            self.end_time = datetime.now(timezone.utc)
            
            self.results = {
                'status': 'completed',
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat(),
                'total_stimulations': total_stimulations,
                'testing_mode': self.testing,
                'trial_summary': self.stim_controller.trial_recorder.get_summary()
            }
            
            logger.info(f"Experiment completed. Total stimulations: {total_stimulations}")
            return self.results
        
        except Exception as e:
            logger.error(f"Experiment failed with error: {e}")
            self.end_time = datetime.now(timezone.utc)
            self.results = {
                'status': 'failed',
                'error': str(e),
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': self.end_time.isoformat() if self.end_time else None
            }
            return self.results
        
        finally:
            self.stop()
    
    def stop(self):
        try:
            self.stim_controller.cleanup()
            logger.info("Experiment stopped and cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
