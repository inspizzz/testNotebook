import logging
import time
from datetime import datetime
from typing import List, Dict
import numpy as np

from neuroplatform import Experiment as NPExperiment, StimParam, IntanSofware, TriggerController, StimPolarity

class Experiment:
    def __init__(self, token: str, booking_email: str, testing: bool = False):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        # Define reliable connections from prior parameter scan
        self.reliable_connections = [
            {'electrode': 1, 'repeats': 5, 'amplitude': 3.0, 'duration': 200.0, 'polarity': StimPolarity.PositiveFirst},
            {'electrode': 1, 'repeats': 5, 'amplitude': 1.0, 'duration': 200.0, 'polarity': StimPolarity.NegativeFirst},
            {'electrode': 1, 'repeats': 5, 'amplitude': 2.0, 'duration': 200.0, 'polarity': StimPolarity.NegativeFirst},
            {'electrode': 1, 'repeats': 5, 'amplitude': 3.0, 'duration': 100.0, 'polarity': StimPolarity.NegativeFirst},
        ]
        # Safety limits
        self.max_amplitude = 5.0  # uA
        self.max_duration = 500.0  # us
        self.max_repeats = 50
        self.inter_stim_delay = 0.1  # seconds

    def run(self) -> List[Dict]:
        results: List[Dict] = []
        np_experiment = None
        intan = None
        trigger_ctrl = None
        try:
            if not self.testing:
                logging.info("Connecting to NeuroPlatform Experiment...")
                np_experiment = NPExperiment(self.token)
                logging.info("Connecting to IntanSoftware...")
                intan = IntanSofware()
                logging.info("Connecting to TriggerController...")
                trigger_ctrl = TriggerController(self.booking_email)
                if not np_experiment.start():
                    raise RuntimeError("Could not start the experiment: session conflict or maintenance.")
            else:
                logging.info("TEST MODE: Skipping hardware connections; simulating stimulations.")
            for idx, conn in enumerate(self.reliable_connections):
                electrode = conn['electrode']
                repeats = min(conn['repeats'], self.max_repeats)
                amp = conn['amplitude']
                dur = conn['duration']
                pol = conn['polarity']
                # Safety checks
                if amp > self.max_amplitude or dur > self.max_duration:
                    raise ValueError(f"Stimulation exceeds safety limits: amp={amp}, dur={dur}")
                # Build StimParam
                sp = StimParam()
                sp.index = electrode
                sp.enable = True
                sp.polarity = pol
                sp.phase_duration1 = dur
                sp.phase_duration2 = dur
                sp.phase_amplitude1 = amp
                sp.phase_amplitude2 = amp
                trigger_key = idx if idx < 16 else 15
                sp.trigger_key = trigger_key
                # Send stimulation parameters
                if not self.testing:
                    intan.send_stimparam([sp])
                # Prepare trigger pattern
                pattern = np.zeros(16, dtype=np.uint8)
                pattern[trigger_key] = 1
                # Repeated stimulations
                for rep in range(1, repeats + 1):
                    if not self.testing:
                        trigger_ctrl.send(pattern)
                    timestamp = datetime.utcnow().isoformat() + "Z"
                    results.append({
                        'electrode': electrode,
                        'amplitude': amp,
                        'duration': dur,
                        'polarity': pol.name,
                        'trigger_key': trigger_key,
                        'rep': rep,
                        'timestamp': timestamp,
                        'testing': self.testing
                    })
                    time.sleep(self.inter_stim_delay)
                # Disable this stimulation
                sp.enable = False
                if not self.testing:
                    intan.send_stimparam([sp])
            return results
        finally:
            # Ensure experiment stop and hardware close
            if np_experiment:
                try:
                    np_experiment.stop()
                except Exception:
                    pass
            if trigger_ctrl and not self.testing:
                try:
                    trigger_ctrl.close()
                except Exception:
                    pass
            if intan and not self.testing:
                try:
                    intan.close()
                except Exception:
                    pass
