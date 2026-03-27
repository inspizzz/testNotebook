import numpy as np
import pandas as pd
import json
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from neuroplatform import (
    StimParam, StimPolarity, StimShape, TriggerController, Experiment
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StimulationRecord:
    electrode_from: int
    electrode_to: int
    amplitude: float
    duration: float
    polarity: str
    trial_number: int
    timestamp: datetime


class OptimalParameterSelector:
    def __init__(self, scan_results: Dict):
        self.scan_results = scan_results
        self.optimal_pairs = self._identify_optimal_pairs()
    
    def _identify_optimal_pairs(self) -> List[Dict]:
        """Identify the most reliable electrode pairs from scan results."""
        reliable_connections = self.scan_results.get('reliable_connections', [])
        
        pair_performance = {}
        for connection in reliable_connections:
            pair_key = (connection['electrode_from'], connection['electrode_to'])
            
            if pair_key not in pair_performance:
                pair_performance[pair_key] = {
                    'connections': [],
                    'avg_hits': 0,
                    'best_connection': None
                }
            
            pair_performance[pair_key]['connections'].append(connection)
        
        for pair_key, data in pair_performance.items():
            avg_hits = np.mean([c['hits_k'] for c in data['connections']])
            data['avg_hits'] = avg_hits
            data['best_connection'] = max(data['connections'], key=lambda x: x['hits_k'])
        
        sorted_pairs = sorted(
            pair_performance.items(),
            key=lambda x: x[1]['avg_hits'],
            reverse=True
        )
        
        return [
            {
                'electrode_from': pair[0][0],
                'electrode_to': pair[0][1],
                'connection': pair[1]['best_connection']
            }
            for pair in sorted_pairs[:10]
        ]
    
    def get_optimal_pairs(self) -> List[Dict]:
        return self.optimal_pairs


class ParameterVariationGenerator:
    def __init__(self, base_params: Dict):
        self.base_params = base_params
        self.amplitude_variations = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        self.duration_variations = [50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0]
    
    def generate_amplitude_variations(self) -> List[Dict]:
        """Generate stimulation parameters with varied amplitude, fixed duration."""
        variations = []
        base_amplitude = self.base_params['amplitude']
        base_duration = self.base_params['duration']
        
        for amp in self.amplitude_variations:
            if amp != base_amplitude:
                param = {
                    'amplitude': amp,
                    'duration': base_duration,
                    'polarity': self.base_params['polarity'],
                    'variation_type': 'amplitude'
                }
                variations.append(param)
        
        return variations
    
    def generate_duration_variations(self) -> List[Dict]:
        """Generate stimulation parameters with varied duration, fixed amplitude."""
        variations = []
        base_amplitude = self.base_params['amplitude']
        base_duration = self.base_params['duration']
        
        for dur in self.duration_variations:
            if dur != base_duration:
                param = {
                    'amplitude': base_amplitude,
                    'duration': dur,
                    'polarity': self.base_params['polarity'],
                    'variation_type': 'duration'
                }
                variations.append(param)
        
        return variations


class StimulationExecutor:
    def __init__(self, testing: bool = False):
        self.testing = testing
        self.trigger_controller = None
        self.stimulation_log = []
    
    def initialize_trigger(self, booking_email: str) -> bool:
        """Initialize trigger controller if not in testing mode."""
        if self.testing:
            logger.info("Testing mode: Skipping trigger controller initialization")
            return True
        
        try:
            self.trigger_controller = TriggerController(email=booking_email)
            logger.info("Trigger controller initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize trigger controller: {e}")
            return False
    
    def create_stim_param(self, electrode_index: int, amplitude: float, 
                         duration: float, polarity: str) -> StimParam:
        """Create a StimParam object with charge-balanced biphasic stimulation."""
        param = StimParam()
        param.index = electrode_index
        param.enable = True
        param.trigger_key = 0
        param.trigger_delay = 0
        param.nb_pulse = 0
        param.pulse_train_period = 10000
        param.post_stim_ref_period = 1000.0
        param.stim_shape = StimShape.Biphasic
        param.polarity = StimPolarity.NegativeFirst if polarity == "NegativeFirst" else StimPolarity.PositiveFirst
        
        param.phase_duration1 = duration
        param.phase_duration2 = duration
        param.phase_amplitude1 = amplitude
        param.phase_amplitude2 = amplitude
        
        param.enable_amp_settle = True
        param.pre_stim_amp_settle = 0.0
        param.post_stim_amp_settle = 1000.0
        param.enable_charge_recovery = True
        param.post_charge_recovery_on = 0.0
        param.post_charge_recovery_off = 100.0
        param.interphase_delay = 0.0
        
        return param
    
    def send_stimulation(self, electrode_index: int, amplitude: float, 
                        duration: float, polarity: str) -> bool:
        """Send a single stimulation pulse."""
        if amplitude > 5.0 or duration > 500.0:
            logger.warning(f"Stimulation parameters exceed limits: amp={amplitude}, dur={duration}")
            return False
        
        if self.testing:
            record = StimulationRecord(
                electrode_from=electrode_index,
                electrode_to=-1,
                amplitude=amplitude,
                duration=duration,
                polarity=polarity,
                trial_number=len(self.stimulation_log),
                timestamp=datetime.now(timezone.utc)
            )
            self.stimulation_log.append(record)
            logger.info(f"[TEST] Stimulation logged: electrode {electrode_index}, amp={amplitude}, dur={duration}")
            return True
        
        try:
            pattern = np.zeros(16, dtype=np.uint8)
            pattern[0] = 1
            self.trigger_controller.send(pattern)
            
            record = StimulationRecord(
                electrode_from=electrode_index,
                electrode_to=-1,
                amplitude=amplitude,
                duration=duration,
                polarity=polarity,
                trial_number=len(self.stimulation_log),
                timestamp=datetime.now(timezone.utc)
            )
            self.stimulation_log.append(record)
            logger.info(f"Stimulation sent: electrode {electrode_index}, amp={amplitude}, dur={duration}")
            return True
        except Exception as e:
            logger.error(f"Failed to send stimulation: {e}")
            return False
    
    def close(self):
        """Close trigger controller if initialized."""
        if self.trigger_controller is not None and not self.testing:
            self.trigger_controller.close()


class Experiment:
    def __init__(self, token: str, booking_email: str, testing: bool = False):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self.executor = StimulationExecutor(testing=testing)
        self.results = {
            'confirmation_phase': [],
            'amplitude_variation_phase': [],
            'duration_variation_phase': [],
            'summary': {}
        }
    
    def run(self, token: str = None, booking_email: str = None, testing: bool = False) -> Dict:
        """
        Execute the complete experiment protocol.
        
        Args:
            token: FinalSpark experiment token
            booking_email: Email for booking service
            testing: Whether to run in testing mode
        
        Returns:
            Dictionary containing experiment results
        """
        if token is not None:
            self.token = token
        if booking_email is not None:
            self.booking_email = booking_email
        self.testing = testing
        self.executor.testing = testing
        
        try:
            logger.info("Starting experiment execution")
            
            if not self.executor.initialize_trigger(self.booking_email):
                if not self.testing:
                    raise Exception("Failed to initialize trigger controller")
            
            scan_results = self._load_scan_results()
            selector = OptimalParameterSelector(scan_results)
            optimal_pairs = selector.get_optimal_pairs()
            
            logger.info(f"Identified {len(optimal_pairs)} optimal electrode pairs")
            
            self._confirmation_phase(optimal_pairs)
            self._amplitude_variation_phase(optimal_pairs)
            self._duration_variation_phase(optimal_pairs)
            
            self._compile_results()
            
            logger.info("Experiment completed successfully")
            return self.results
        
        finally:
            self.executor.close()
    
    def _load_scan_results(self) -> Dict:
        """Load parameter scan results from embedded data."""
        scan_data = {
            "parameter_scan_summary": {
                "pairs_found": 117,
                "window_ms": 50,
                "required_hits": 3,
                "stim_rows": 960,
                "trigger_start_rows": 960,
                "aligned_rows": 960,
                "median_offset_seconds": 1.2687065,
                "dropped_unaligned_rows": 0,
                "self_response_filtered_trials": 1184
            },
            "reliable_connections": [
                {
                    "electrode_from": 8,
                    "electrode_to": 10,
                    "hits_k": 5,
                    "repeats_n": 5,
                    "median_latency_ms": 35.2,
                    "stimulation": {
                        "amplitude": 1.0,
                        "duration": 100.0,
                        "polarity": "NegativeFirst"
                    }
                },
                {
                    "electrode_from": 11,
                    "electrode_to": 12,
                    "hits_k": 5,
                    "repeats_n": 5,
                    "median_latency_ms": 13.4,
                    "stimulation": {
                        "amplitude": 2.0,
                        "duration": 300.0,
                        "polarity": "NegativeFirst"
                    }
                },
                {
                    "electrode_from": 16,
                    "electrode_to": 17,
                    "hits_k": 5,
                    "repeats_n": 5,
                    "median_latency_ms": 10.033,
                    "stimulation": {
                        "amplitude": 3.0,
                        "duration": 400.0,
                        "polarity": "PositiveFirst"
                    }
                },
                {
                    "electrode_from": 18,
                    "electrode_to": 23,
                    "hits_k": 5,
                    "repeats_n": 5,
                    "median_latency_ms": 21.2,
                    "stimulation": {
                        "amplitude": 3.0,
                        "duration": 400.0,
                        "polarity": "NegativeFirst"
                    }
                },
                {
                    "electrode_from": 23,
                    "electrode_to": 20,
                    "hits_k": 5,
                    "repeats_n": 5,
                    "median_latency_ms": 10.033,
                    "stimulation": {
                        "amplitude": 3.0,
                        "duration": 400.0,
                        "polarity": "NegativeFirst"
                    }
                }
            ]
        }
        return scan_data
    
    def _confirmation_phase(self, optimal_pairs: List[Dict]):
        """Phase 1: Confirm optimal parameters with 100 stimulations each."""
        logger.info("Starting confirmation phase")
        
        for pair_idx, pair in enumerate(optimal_pairs[:5]):
            electrode_from = pair['electrode_from']
            electrode_to = pair['electrode_to']
            stim_params = pair['connection']['stimulation']
            
            amplitude = stim_params['amplitude']
            duration = stim_params['duration']
            polarity = stim_params['polarity']
            
            logger.info(f"Confirming pair {pair_idx + 1}: {electrode_from} -> {electrode_to}")
            
            for trial in range(100):
                self.executor.send_stimulation(electrode_from, amplitude, duration, polarity)
                if not self.testing:
                    time.sleep(0.01)
            
            self.results['confirmation_phase'].append({
                'electrode_from': electrode_from,
                'electrode_to': electrode_to,
                'amplitude': amplitude,
                'duration': duration,
                'polarity': polarity,
                'trials': 100
            })
    
    def _amplitude_variation_phase(self, optimal_pairs: List[Dict]):
        """Phase 2: Vary amplitude while keeping duration constant."""
        logger.info("Starting amplitude variation phase")
        
        for pair_idx, pair in enumerate(optimal_pairs[:5]):
            electrode_from = pair['electrode_from']
            electrode_to = pair['electrode_to']
            stim_params = pair['connection']['stimulation']
            
            base_amplitude = stim_params['amplitude']
            duration = stim_params['duration']
            polarity = stim_params['polarity']
            
            generator = ParameterVariationGenerator({
                'amplitude': base_amplitude,
                'duration': duration,
                'polarity': polarity
            })
            
            amplitude_variations = generator.generate_amplitude_variations()
            
            logger.info(f"Testing amplitude variations for pair {pair_idx + 1}")
            
            for variation in amplitude_variations[:10]:
                amplitude = variation['amplitude']
                
                for trial in range(10):
                    self.executor.send_stimulation(electrode_from, amplitude, duration, polarity)
                    if not self.testing:
                        time.sleep(0.01)
                
                self.results['amplitude_variation_phase'].append({
                    'electrode_from': electrode_from,
                    'electrode_to': electrode_to,
                    'amplitude': amplitude,
                    'duration': duration,
                    'polarity': polarity,
                    'trials': 10,
                    'variation_type': 'amplitude'
                })
    
    def _duration_variation_phase(self, optimal_pairs: List[Dict]):
        """Phase 3: Vary duration while keeping amplitude constant."""
        logger.info("Starting duration variation phase")
        
        for pair_idx, pair in enumerate(optimal_pairs[:5]):
            electrode_from = pair['electrode_from']
            electrode_to = pair['electrode_to']
            stim_params = pair['connection']['stimulation']
            
            amplitude = stim_params['amplitude']
            base_duration = stim_params['duration']
            polarity = stim_params['polarity']
            
            generator = ParameterVariationGenerator({
                'amplitude': amplitude,
                'duration': base_duration,
                'polarity': polarity
            })
            
            duration_variations = generator.generate_duration_variations()
            
            logger.info(f"Testing duration variations for pair {pair_idx + 1}")
            
            for variation in duration_variations[:10]:
                duration = variation['duration']
                
                for trial in range(10):
                    self.executor.send_stimulation(electrode_from, amplitude, duration, polarity)
                    if not self.testing:
                        time.sleep(0.01)
                
                self.results['duration_variation_phase'].append({
                    'electrode_from': electrode_from,
                    'electrode_to': electrode_to,
                    'amplitude': amplitude,
                    'duration': duration,
                    'polarity': polarity,
                    'trials': 10,
                    'variation_type': 'duration'
                })
    
    def _compile_results(self):
        """Compile final results summary."""
        total_stimulations = len(self.executor.stimulation_log)
        
        self.results['summary'] = {
            'total_stimulations': total_stimulations,
            'confirmation_stimulations': sum(r['trials'] for r in self.results['confirmation_phase']),
            'amplitude_variation_stimulations': sum(r['trials'] for r in self.results['amplitude_variation_phase']),
            'duration_variation_stimulations': sum(r['trials'] for r in self.results['duration_variation_phase']),
            'testing_mode': self.testing,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"Experiment summary: {self.results['summary']}")
