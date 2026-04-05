import numpy as np
import pandas as pd
import json
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import math

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
    electrode_index: int
    amplitude_ua: float
    duration_us: float
    response_count: int
    timestamp: datetime


@dataclass
class TrialResult:
    trial_id: int
    electrode_index: int
    amplitude_ua: float
    duration_us: float
    response_count: int
    timestamp: datetime


@dataclass
class VariationResult:
    variation_type: str
    base_amplitude_ua: float
    base_duration_us: float
    varied_amplitude_ua: float
    varied_duration_us: float
    response_count: int
    timestamp: datetime


class Experiment:
    def __init__(
        self,
        token: str = "W32XGX2HCH",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None
        self.sweep_results: List[SweepResult] = []
        self.trial_results: List[TrialResult] = []
        self.variation_results: List[VariationResult] = []
        self.optimal_params: Dict[int, Dict] = {}

    def run(self) -> Dict:
        try:
            logger.info("Starting FinalSpark parameter optimization experiment")

            from neuroplatform import Experiment as NeuroPlatformExperiment

            self.experiment = NeuroPlatformExperiment(self.token)
            self.trigger_controller = TriggerController(
                email=self.booking_email
            )
            self.intan = IntanSofware()
            self.database = Database()

            logger.info(f"Experiment: {self.experiment.exp_name}")
            logger.info(f"Electrodes: {self.experiment.electrodes}")

            if not self.experiment.start():
                logger.error("Failed to start experiment")
                return {"status": "failed", "error": "Could not start experiment"}

            recording_start = datetime.now(timezone.utc)

            self._run_parameter_sweep()

            self._identify_optimal_parameters()

            self._confirm_optimal_parameters()

            self._explore_amplitude_variations()

            self._explore_duration_variations()

            recording_stop = datetime.now(timezone.utc)

            results = self._compile_results(recording_start, recording_stop)

            return results

        except Exception as e:
            logger.error(f"Experiment error: {str(e)}", exc_info=True)
            return {"status": "failed", "error": str(e)}
        finally:
            self._cleanup()

    def _run_parameter_sweep(self) -> None:
        logger.info("Phase 1: Running parameter sweep")

        test_amplitudes = [10.0, 20.0, 30.0]
        test_durations = [50.0, 100.0, 150.0]
        electrode_subset = self.experiment.electrodes[:4]

        for electrode_idx in electrode_subset:
            for amplitude in test_amplitudes:
                for duration in test_durations:
                    if amplitude > 5.0 or duration > 500.0:
                        continue

                    response_count = self._stimulate_and_count(
                        electrode_idx, amplitude, duration, num_pulses=3
                    )

                    result = SweepResult(
                        electrode_index=electrode_idx,
                        amplitude_ua=amplitude,
                        duration_us=duration,
                        response_count=response_count,
                        timestamp=datetime.now(timezone.utc),
                    )
                    self.sweep_results.append(result)
                    logger.info(
                        f"Sweep: Electrode {electrode_idx}, "
                        f"Amplitude {amplitude} uA, Duration {duration} us, "
                        f"Responses: {response_count}"
                    )

                    time.sleep(0.5)

    def _identify_optimal_parameters(self) -> None:
        logger.info("Phase 2: Identifying optimal parameters")

        if not self.sweep_results:
            logger.warning("No sweep results available")
            return

        df = pd.DataFrame([asdict(r) for r in self.sweep_results])

        for electrode_idx in df["electrode_index"].unique():
            electrode_data = df[df["electrode_index"] == electrode_idx]

            if len(electrode_data) == 0:
                continue

            best_row = electrode_data.loc[electrode_data["response_count"].idxmax()]

            self.optimal_params[electrode_idx] = {
                "amplitude_ua": float(best_row["amplitude_ua"]),
                "duration_us": float(best_row["duration_us"]),
                "response_count": int(best_row["response_count"]),
            }

            logger.info(
                f"Optimal for electrode {electrode_idx}: "
                f"Amplitude {best_row['amplitude_ua']} uA, "
                f"Duration {best_row['duration_us']} us"
            )

    def _confirm_optimal_parameters(self) -> None:
        logger.info("Phase 3: Confirming optimal parameters with repeated trials")

        num_trials = 5

        for electrode_idx, params in self.optimal_params.items():
            amplitude = params["amplitude_ua"]
            duration = params["duration_us"]

            for trial_id in range(num_trials):
                response_count = self._stimulate_and_count(
                    electrode_idx, amplitude, duration, num_pulses=1
                )

                result = TrialResult(
                    trial_id=trial_id,
                    electrode_index=electrode_idx,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    response_count=response_count,
                    timestamp=datetime.now(timezone.utc),
                )
                self.trial_results.append(result)

                logger.info(
                    f"Trial {trial_id} for electrode {electrode_idx}: "
                    f"{response_count} responses"
                )

                time.sleep(0.3)

    def _explore_amplitude_variations(self) -> None:
        logger.info("Phase 4: Exploring amplitude variations")

        for electrode_idx, params in self.optimal_params.items():
            base_amplitude = params["amplitude_ua"]
            base_duration = params["duration_us"]

            amplitude_offsets = [-5.0, -2.5, 2.5, 5.0]

            for offset in amplitude_offsets:
                varied_amplitude = base_amplitude + offset

                if varied_amplitude <= 0 or varied_amplitude > 5.0:
                    continue

                response_count = self._stimulate_and_count(
                    electrode_idx, varied_amplitude, base_duration, num_pulses=2
                )

                result = VariationResult(
                    variation_type="amplitude",
                    base_amplitude_ua=base_amplitude,
                    base_duration_us=base_duration,
                    varied_amplitude_ua=varied_amplitude,
                    varied_duration_us=base_duration,
                    response_count=response_count,
                    timestamp=datetime.now(timezone.utc),
                )
                self.variation_results.append(result)

                logger.info(
                    f"Amplitude variation for electrode {electrode_idx}: "
                    f"{varied_amplitude} uA -> {response_count} responses"
                )

                time.sleep(0.3)

    def _explore_duration_variations(self) -> None:
        logger.info("Phase 5: Exploring duration variations")

        for electrode_idx, params in self.optimal_params.items():
            base_amplitude = params["amplitude_ua"]
            base_duration = params["duration_us"]

            duration_offsets = [-25.0, -10.0, 10.0, 25.0]

            for offset in duration_offsets:
                varied_duration = base_duration + offset

                if varied_duration <= 0 or varied_duration > 500.0:
                    continue

                response_count = self._stimulate_and_count(
                    electrode_idx, base_amplitude, varied_duration, num_pulses=2
                )

                result = VariationResult(
                    variation_type="duration",
                    base_amplitude_ua=base_amplitude,
                    base_duration_us=base_duration,
                    varied_amplitude_ua=base_amplitude,
                    varied_duration_us=varied_duration,
                    response_count=response_count,
                    timestamp=datetime.now(timezone.utc),
                )
                self.variation_results.append(result)

                logger.info(
                    f"Duration variation for electrode {electrode_idx}: "
                    f"{varied_duration} us -> {response_count} responses"
                )

                time.sleep(0.3)

    def _stimulate_and_count(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        num_pulses: int = 1,
    ) -> int:
        if self.testing:
            return np.random.randint(5, 20)

        try:
            stim_param = StimParam()
            stim_param.index = electrode_idx
            stim_param.enable = True
            stim_param.trigger_key = 0
            stim_param.trigger_delay = 0
            stim_param.nb_pulse = num_pulses - 1
            stim_param.pulse_train_period = 10000
            stim_param.post_stim_ref_period = 1000.0
            stim_param.stim_shape = StimShape.Biphasic
            stim_param.polarity = StimPolarity.NegativeFirst
            stim_param.phase_duration1 = duration_us
            stim_param.phase_duration2 = duration_us
            stim_param.phase_amplitude1 = amplitude_ua
            stim_param.phase_amplitude2 = -amplitude_ua
            stim_param.enable_amp_settle = True
            stim_param.post_stim_amp_settle = 1000.0
            stim_param.enable_charge_recovery = True

            self.intan.send_stimparam([stim_param])

            trigger_pattern = np.zeros(16, dtype=np.uint8)
            trigger_pattern[0] = 1
            self.trigger_controller.send(trigger_pattern)

            time.sleep(0.1)

            trigger_pattern[0] = 0
            self.trigger_controller.send(trigger_pattern)

            time.sleep(0.2)

            stim_time = datetime.now(timezone.utc) - timedelta(seconds=0.5)
            spike_data = self.database.get_spike_event_electrode(
                stim_time, datetime.now(timezone.utc), electrode_idx
            )

            response_count = len(spike_data) if spike_data is not None else 0
            return response_count

        except Exception as e:
            logger.error(f"Error during stimulation: {str(e)}")
            return 0

    def _compile_results(
        self, recording_start: datetime, recording_stop: datetime
    ) -> Dict:
        logger.info("Compiling results")

        summary = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "token": self.token,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "optimal_parameters": self.optimal_params,
            "sweep_results_count": len(self.sweep_results),
            "trial_results_count": len(self.trial_results),
            "variation_results_count": len(self.variation_results),
        }

        if self.sweep_results:
            sweep_df = pd.DataFrame([asdict(r) for r in self.sweep_results])
            summary["sweep_statistics"] = {
                "mean_response_count": float(sweep_df["response_count"].mean()),
                "max_response_count": int(sweep_df["response_count"].max()),
                "min_response_count": int(sweep_df["response_count"].min()),
            }

        if self.trial_results:
            trial_df = pd.DataFrame([asdict(r) for r in self.trial_results])
            summary["trial_statistics"] = {
                "mean_response_count": float(trial_df["response_count"].mean()),
                "std_response_count": float(trial_df["response_count"].std()),
                "cv_response_count": float(
                    trial_df["response_count"].std()
                    / trial_df["response_count"].mean()
                    if trial_df["response_count"].mean() > 0
                    else 0
                ),
            }

        if self.variation_results:
            var_df = pd.DataFrame([asdict(r) for r in self.variation_results])
            amplitude_vars = var_df[var_df["variation_type"] == "amplitude"]
            duration_vars = var_df[var_df["variation_type"] == "duration"]

            if len(amplitude_vars) > 0:
                summary["amplitude_variation_statistics"] = {
                    "mean_response_count": float(amplitude_vars["response_count"].mean()),
                    "max_response_count": int(amplitude_vars["response_count"].max()),
                }

            if len(duration_vars) > 0:
                summary["duration_variation_statistics"] = {
                    "mean_response_count": float(duration_vars["response_count"].mean()),
                    "max_response_count": int(duration_vars["response_count"].max()),
                }

        return summary

    def _cleanup(self) -> None:
        logger.info("Cleaning up resources")

        try:
            if self.experiment is not None:
                self.experiment.stop()
        except Exception as e:
            logger.error(f"Error stopping experiment: {str(e)}")

        try:
            if self.intan is not None:
                self.intan.close()
        except Exception as e:
            logger.error(f"Error closing intan: {str(e)}")

        try:
            if self.trigger_controller is not None:
                self.trigger_controller.close()
        except Exception as e:
            logger.error(f"Error closing trigger controller: {str(e)}")
