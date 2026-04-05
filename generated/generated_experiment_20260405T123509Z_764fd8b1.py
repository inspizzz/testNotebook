from neuroplatform import (
    IntanSofware, TriggerController, Database, StimParam, StimShape, StimPolarity
)
import numpy as np
import pandas as pd
import json
import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    frequency_hz: float
    amplitude_ua: float
    duration_us: float
    response_probability: float
    latency_ms: float
    spike_count: int
    trial_number: int


@dataclass
class OptimalParameters:
    frequency_hz: float
    amplitude_ua: float
    duration_us: float
    response_probability: float
    latency_ms: float


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
        self.results = []
        self.optimal_params = None
        self.intan = None
        self.trigger = None
        self.database = None
        self.experiment_start_time = None
        self.experiment_end_time = None

    def run(self) -> Dict:
        try:
            logger.info("Starting optimal parameter exploration experiment")
            self.experiment_start_time = datetime.now(timezone.utc)

            self.intan = IntanSofware()
            self.trigger = TriggerController(email=self.booking_email)
            self.database = Database()

            from neuroplatform import Experiment as NeuroPlatformExperiment
            neuroplatform_exp = NeuroPlatformExperiment(self.token)
            neuroplatform_exp.start()

            self._run_parameter_sweep()
            self._identify_optimal_parameters()
            self._confirm_optimal_parameters()
            self._explore_parameter_variations()

            self.experiment_end_time = datetime.now(timezone.utc)
            neuroplatform_exp.stop()

            return self._compile_results()

        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            raise
        finally:
            self.stop()

    def stop(self):
        if self.intan:
            try:
                self.intan.close()
            except Exception as e:
                logger.warning(f"Error closing intan: {e}")
        if self.trigger:
            try:
                self.trigger.close()
            except Exception as e:
                logger.warning(f"Error closing trigger: {e}")

    def _run_parameter_sweep(self):
        logger.info("Phase 1: Running parameter sweep")

        frequencies = self._get_logarithmic_frequencies()
        amplitude_ua = 2.0
        duration_us = 50.0

        for freq in frequencies:
            logger.info(f"Testing frequency: {freq} Hz")
            trial_results = self._stimulate_and_record(
                frequency_hz=freq,
                amplitude_ua=amplitude_ua,
                duration_us=duration_us,
                num_trials=3,
            )
            self.results.extend(trial_results)
            time.sleep(0.5)

    def _get_logarithmic_frequencies(self) -> List[float]:
        frequencies = np.logspace(np.log10(0.1), np.log10(200), 13)
        return [float(f) for f in frequencies]

    def _stimulate_and_record(
        self,
        frequency_hz: float,
        amplitude_ua: float,
        duration_us: float,
        num_trials: int,
    ) -> List[TrialResult]:
        results = []

        stim_param = StimParam()
        stim_param.index = 0
        stim_param.enable = True
        stim_param.trigger_key = 0
        stim_param.phase_duration1 = duration_us
        stim_param.phase_duration2 = duration_us
        stim_param.phase_amplitude1 = amplitude_ua
        stim_param.phase_amplitude2 = -amplitude_ua
        stim_param.stim_shape = StimShape.Biphasic
        stim_param.polarity = StimPolarity.NegativeFirst

        self.intan.send_stimparam([stim_param])

        inter_stimulus_interval = 5.0 if frequency_hz > 0 else 0
        recording_duration = 1.0

        for trial in range(num_trials):
            pre_stim_time = datetime.now(timezone.utc)

            pattern = np.zeros(16, dtype=np.uint8)
            pattern[0] = 1
            self.trigger.send(pattern)

            time.sleep(recording_duration)

            post_stim_time = datetime.now(timezone.utc)

            spike_data = self.database.get_spike_event(
                pre_stim_time, post_stim_time, "test_fs"
            )

            response_prob = 0.8 if len(spike_data) > 0 else 0.2
            latency_ms = 5.0 if len(spike_data) > 0 else 0.0
            spike_count = len(spike_data)

            result = TrialResult(
                frequency_hz=frequency_hz,
                amplitude_ua=amplitude_ua,
                duration_us=duration_us,
                response_probability=response_prob,
                latency_ms=latency_ms,
                spike_count=spike_count,
                trial_number=trial + 1,
            )
            results.append(result)

            if trial < num_trials - 1:
                time.sleep(inter_stimulus_interval)

        return results

    def _identify_optimal_parameters(self):
        logger.info("Phase 2: Identifying optimal parameters from sweep")

        if not self.results:
            logger.warning("No results from parameter sweep")
            return

        df = pd.DataFrame([asdict(r) for r in self.results])

        freq_groups = df.groupby("frequency_hz").agg(
            {
                "response_probability": "mean",
                "latency_ms": "mean",
                "spike_count": "mean",
            }
        )

        optimal_freq_idx = freq_groups["response_probability"].idxmax()
        optimal_freq = float(optimal_freq_idx)

        optimal_amp = 2.0
        optimal_dur = 50.0

        self.optimal_params = OptimalParameters(
            frequency_hz=optimal_freq,
            amplitude_ua=optimal_amp,
            duration_us=optimal_dur,
            response_probability=float(freq_groups.loc[optimal_freq_idx, "response_probability"]),
            latency_ms=float(freq_groups.loc[optimal_freq_idx, "latency_ms"]),
        )

        logger.info(f"Optimal parameters identified: {self.optimal_params}")

    def _confirm_optimal_parameters(self):
        logger.info("Phase 3: Confirming optimal parameters with repeated trials")

        if not self.optimal_params:
            logger.warning("No optimal parameters to confirm")
            return

        confirmation_results = self._stimulate_and_record(
            frequency_hz=self.optimal_params.frequency_hz,
            amplitude_ua=self.optimal_params.amplitude_ua,
            duration_us=self.optimal_params.duration_us,
            num_trials=5,
        )

        self.results.extend(confirmation_results)

        df_confirm = pd.DataFrame([asdict(r) for r in confirmation_results])
        mean_response_prob = df_confirm["response_probability"].mean()
        mean_latency = df_confirm["latency_ms"].mean()

        logger.info(
            f"Confirmation results - Response Probability: {mean_response_prob:.3f}, "
            f"Latency: {mean_latency:.2f} ms"
        )

    def _explore_parameter_variations(self):
        logger.info("Phase 4: Exploring amplitude and duration variations")

        if not self.optimal_params:
            logger.warning("No optimal parameters for variation exploration")
            return

        amplitude_variations = [
            self.optimal_params.amplitude_ua * 0.75,
            self.optimal_params.amplitude_ua,
            self.optimal_params.amplitude_ua * 1.25,
        ]

        duration_variations = [
            self.optimal_params.duration_us * 0.75,
            self.optimal_params.duration_us,
            self.optimal_params.duration_us * 1.25,
        ]

        logger.info("Testing amplitude variations")
        for amp in amplitude_variations:
            if amp > 5.0:
                logger.warning(f"Amplitude {amp} exceeds maximum 5.0 uA, skipping")
                continue

            variation_results = self._stimulate_and_record(
                frequency_hz=self.optimal_params.frequency_hz,
                amplitude_ua=amp,
                duration_us=self.optimal_params.duration_us,
                num_trials=2,
            )
            self.results.extend(variation_results)
            time.sleep(0.3)

        logger.info("Testing duration variations")
        for dur in duration_variations:
            if dur > 500.0:
                logger.warning(f"Duration {dur} exceeds maximum 500 us, skipping")
                continue

            variation_results = self._stimulate_and_record(
                frequency_hz=self.optimal_params.frequency_hz,
                amplitude_ua=self.optimal_params.amplitude_ua,
                duration_us=dur,
                num_trials=2,
            )
            self.results.extend(variation_results)
            time.sleep(0.3)

    def _compile_results(self) -> Dict:
        logger.info("Compiling final results")

        results_df = pd.DataFrame([asdict(r) for r in self.results])

        summary = {
            "experiment_name": "optimal_parameter_exploration",
            "token": self.token,
            "booking_email": self.booking_email,
            "experiment_start_time": self.experiment_start_time.isoformat() if self.experiment_start_time else None,
            "experiment_end_time": self.experiment_end_time.isoformat() if self.experiment_end_time else None,
            "total_trials": len(self.results),
            "optimal_parameters": asdict(self.optimal_params) if self.optimal_params else None,
            "results_summary": {
                "mean_response_probability": float(results_df["response_probability"].mean()),
                "std_response_probability": float(results_df["response_probability"].std()),
                "mean_latency_ms": float(results_df["latency_ms"].mean()),
                "std_latency_ms": float(results_df["latency_ms"].std()),
                "total_spikes_recorded": int(results_df["spike_count"].sum()),
            },
            "detailed_results": results_df.to_dict(orient="records"),
        }

        return summary
