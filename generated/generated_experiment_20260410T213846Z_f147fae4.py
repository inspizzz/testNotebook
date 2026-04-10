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
class StimulationRecord:
    """A single stimulation event for the persistence log."""
    electrode_idx: int
    amplitude_ua: float
    duration_us: float
    frequency_hz: float
    timestamp_utc: str
    trigger_key: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrequencyTestResult:
    """Results from testing a specific frequency on an electrode."""
    electrode_idx: int
    frequency_hz: float
    baseline_spike_count: int
    post_stim_spike_count: int
    response_ratio: float
    timestamp_utc: str


class DataSaver:
    """Handles persistence of stimulation records, spike events, and triggers."""

    def __init__(self, output_dir: Path, fs_name: str) -> None:
        self._dir = output_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._prefix = self._dir / f"{fs_name}_{timestamp}"

    def save_stimulation_log(self, stimulations: List[StimulationRecord]) -> Path:
        path = Path(f"{self._prefix}_stimulations.json")
        records = [asdict(s) for s in stimulations]
        path.write_text(json.dumps(records, indent=2, default=str))
        logger.info("Saved stimulation log -> %s  (%d records)", path, len(records))
        return path

    def save_spike_events(self, df: pd.DataFrame) -> Path:
        path = Path(f"{self._prefix}_spike_events.csv")
        df.to_csv(path, index=False)
        logger.info("Saved spike events -> %s  (%d rows)", path, len(df))
        return path

    def save_triggers(self, df: pd.DataFrame) -> Path:
        path = Path(f"{self._prefix}_triggers.csv")
        df.to_csv(path, index=False)
        logger.info("Saved triggers -> %s  (%d rows)", path, len(df))
        return path

    def save_summary(self, summary: Dict[str, Any]) -> Path:
        path = Path(f"{self._prefix}_summary.json")
        path.write_text(json.dumps(summary, indent=2, default=str))
        logger.info("Saved summary -> %s", path)
        return path

    def save_spike_waveforms(self, waveform_records: list) -> Path:
        path = Path(f"{self._prefix}_spike_waveforms.json")
        path.write_text(json.dumps(waveform_records, indent=2, default=str))
        logger.info("Saved spike waveforms -> %s  (%d spike(s))", path, len(waveform_records))
        return path


class Experiment:
    """Frequency-dependent neural modulation experiment targeting active electrodes."""

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        stimulation_amplitude: float = 2.0,
        pulse_duration: float = 200.0,
        test_frequencies: List[float] = None,
        baseline_duration: float = 10.0,
        post_stim_duration: float = 10.0,
        inter_frequency_delay: float = 30.0,
        pulses_per_frequency: int = 10,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        # Experiment parameters
        self.stimulation_amplitude = min(stimulation_amplitude, 5.0)
        self.pulse_duration = min(pulse_duration, 500.0)
        self.test_frequencies = test_frequencies or [0.5, 1.0, 10.0, 25.0, 50.0, 100.0, 150.0]
        self.baseline_duration = baseline_duration
        self.post_stim_duration = post_stim_duration
        self.inter_frequency_delay = inter_frequency_delay
        self.pulses_per_frequency = pulses_per_frequency

        # Hardware handles
        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        # Data storage
        self._stimulation_log: List[StimulationRecord] = []
        self._frequency_results: List[FrequencyTestResult] = []
        self._active_electrodes: List[int] = []

    def run(self) -> Dict[str, Any]:
        """Execute the frequency modulation experiment."""
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

            # Identify active electrodes
            self._identify_active_electrodes()

            # Test frequency effects on active electrodes
            self._test_frequency_modulation()

            recording_stop = datetime.now(timezone.utc)

            results = self._compile_results(recording_start, recording_stop)

            # Persist all raw data
            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _identify_active_electrodes(self) -> None:
        """Identify electrodes with spontaneous activity."""
        logger.info("Identifying active electrodes")
        
        # Record baseline activity for 30 seconds
        baseline_start = datetime.now(timezone.utc)
        time.sleep(30.0)
        baseline_stop = datetime.now(timezone.utc)

        # Get spike events for all electrodes
        spike_df = self.database.get_spike_event(
            baseline_start, baseline_stop, self.experiment.exp_name
        )

        if not spike_df.empty:
            electrode_col = "channel" if "channel" in spike_df.columns else "index"
            if electrode_col in spike_df.columns:
                spike_counts = spike_df[electrode_col].value_counts()
                # Consider electrodes with >5 spikes as active
                active_electrodes = spike_counts[spike_counts > 5].index.tolist()
                self._active_electrodes = [int(e) for e in active_electrodes if int(e) in self.experiment.electrodes]
            
        if not self._active_electrodes:
            # Fallback to first few available electrodes
            self._active_electrodes = self.experiment.electrodes[:4]
            
        logger.info("Active electrodes identified: %s", self._active_electrodes)

    def _test_frequency_modulation(self) -> None:
        """Test different stimulation frequencies on active electrodes."""
        logger.info("Testing frequency modulation on %d electrodes", len(self._active_electrodes))

        for electrode_idx in self._active_electrodes:
            logger.info("Testing electrode %d", electrode_idx)
            
            for frequency in self.test_frequencies:
                logger.info("Testing frequency %.1f Hz on electrode %d", frequency, electrode_idx)
                
                # Measure baseline activity
                baseline_count = self._measure_baseline_activity(electrode_idx)
                
                # Apply frequency-specific stimulation
                self._apply_frequency_stimulation(electrode_idx, frequency)
                
                # Measure post-stimulation activity
                post_stim_count = self._measure_post_stimulation_activity(electrode_idx)
                
                # Calculate response ratio
                response_ratio = post_stim_count / max(baseline_count, 1)
                
                # Store result
                result = FrequencyTestResult(
                    electrode_idx=electrode_idx,
                    frequency_hz=frequency,
                    baseline_spike_count=baseline_count,
                    post_stim_spike_count=post_stim_count,
                    response_ratio=response_ratio,
                    timestamp_utc=datetime.now(timezone.utc).isoformat()
                )
                self._frequency_results.append(result)
                
                logger.info("Frequency %.1f Hz: baseline=%d, post=%d, ratio=%.2f", 
                           frequency, baseline_count, post_stim_count, response_ratio)
                
                # Wait between frequencies
                time.sleep(self.inter_frequency_delay)

    def _measure_baseline_activity(self, electrode_idx: int) -> int:
        """Measure baseline spike activity for an electrode."""
        baseline_start = datetime.now(timezone.utc)
        time.sleep(self.baseline_duration)
        baseline_stop = datetime.now(timezone.utc)
        
        spike_df = self.database.get_spike_event_electrode(
            baseline_start, baseline_stop, electrode_idx
        )
        return len(spike_df)

    def _apply_frequency_stimulation(self, electrode_idx: int, frequency_hz: float) -> None:
        """Apply stimulation at specified frequency."""
        if frequency_hz <= 0:
            return
            
        inter_pulse_interval = 1.0 / frequency_hz
        
        for pulse_idx in range(self.pulses_per_frequency):
            self._deliver_single_pulse(electrode_idx, frequency_hz)
            
            if pulse_idx < self.pulses_per_frequency - 1:
                time.sleep(inter_pulse_interval)

    def _deliver_single_pulse(self, electrode_idx: int, frequency_hz: float) -> None:
        """Deliver a single charge-balanced biphasic pulse."""
        stim = StimParam()
        stim.index = electrode_idx
        stim.enable = True
        stim.trigger_key = 0
        stim.trigger_delay = 0
        stim.nb_pulse = 0
        stim.pulse_train_period = 10000
        stim.post_stim_ref_period = 1000.0
        stim.stim_shape = StimShape.Biphasic
        stim.polarity = StimPolarity.NegativeFirst

        # Charge-balanced biphasic pulse
        stim.phase_amplitude1 = self.stimulation_amplitude
        stim.phase_duration1 = self.pulse_duration
        stim.phase_amplitude2 = self.stimulation_amplitude
        stim.phase_duration2 = self.pulse_duration

        stim.enable_amp_settle = True
        stim.pre_stim_amp_settle = 0.0
        stim.post_stim_amp_settle = 1000.0
        stim.enable_charge_recovery = True
        stim.post_charge_recovery_on = 0.0
        stim.post_charge_recovery_off = 100.0

        self.intan.send_stimparam([stim])

        # Fire trigger
        pattern = np.zeros(16, dtype=np.uint8)
        pattern[0] = 1
        self.trigger_controller.send(pattern)
        time.sleep(0.05)
        pattern[0] = 0
        self.trigger_controller.send(pattern)

        # Log stimulation
        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=self.stimulation_amplitude,
            duration_us=self.pulse_duration,
            frequency_hz=frequency_hz,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            trigger_key=0,
        ))

    def _measure_post_stimulation_activity(self, electrode_idx: int) -> int:
        """Measure post-stimulation spike activity."""
        post_start = datetime.now(timezone.utc)
        time.sleep(self.post_stim_duration)
        post_stop = datetime.now(timezone.utc)
        
        spike_df = self.database.get_spike_event_electrode(
            post_start, post_stop, electrode_idx
        )
        return len(spike_df)

    def _save_all(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> None:
        """Persist all raw experiment data."""
        fs_name = getattr(self.experiment, "exp_name", "unknown")
        saver = DataSaver(self._output_dir, fs_name)

        # Save stimulation log
        saver.save_stimulation_log(self._stimulation_log)

        # Fetch and save ALL spike events
        spike_df = self.database.get_spike_event(
            recording_start, recording_stop, fs_name
        )
        saver.save_spike_events(spike_df)

        # Fetch and save ALL triggers
        trigger_df = self.database.get_all_triggers(
            recording_start, recording_stop
        )
        saver.save_triggers(trigger_df)

        # Save experiment summary
        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "active_electrodes": self._active_electrodes,
            "test_frequencies": self.test_frequencies,
            "frequency_results": [asdict(r) for r in self._frequency_results],
        }
        saver.save_summary(summary)

        # Fetch and save spike waveforms
        waveform_records = self._fetch_spike_waveforms(
            fs_name, spike_df, trigger_df, recording_start, recording_stop
        )
        saver.save_spike_waveforms(waveform_records)

    def _fetch_spike_waveforms(
        self,
        fs_name: str,
        spike_df: pd.DataFrame,
        trigger_df: pd.DataFrame,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> list:
        """Fetch raw spike waveform data."""
        waveform_records = []
        if spike_df.empty:
            return waveform_records

        electrode_col = "channel" if "channel" in spike_df.columns else "index"
        if electrode_col not in spike_df.columns:
            logger.warning("Cannot determine electrode column for waveform fetch")
            return waveform_records

        unique_electrodes = spike_df[electrode_col].unique()
        for electrode_idx in unique_electrodes:
            try:
                raw_df = self.database.get_raw_spike(
                    recording_start, recording_stop, int(electrode_idx)
                )
                if not raw_df.empty:
                    waveform_records.append({
                        "electrode_idx": int(electrode_idx),
                        "num_waveforms": len(raw_df),
                        "waveform_samples": raw_df.values.tolist(),
                    })
            except Exception as exc:
                logger.warning(
                    "Failed to fetch waveforms for electrode %s: %s",
                    electrode_idx, exc
                )

        return waveform_records

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        """Compile experiment results."""
        logger.info("Compiling results")

        # Analyze frequency effects
        frequency_analysis = self._analyze_frequency_effects()

        summary = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "active_electrodes": self._active_electrodes,
            "test_frequencies": self.test_frequencies,
            "total_stimulations": len(self._stimulation_log),
            "total_frequency_tests": len(self._frequency_results),
            "frequency_analysis": frequency_analysis,
        }

        return summary

    def _analyze_frequency_effects(self) -> Dict[str, Any]:
        """Analyze the effects of different frequencies."""
        if not self._frequency_results:
            return {"error": "No frequency test results available"}

        # Group results by frequency
        freq_groups = defaultdict(list)
        for result in self._frequency_results:
            freq_groups[result.frequency_hz].append(result.response_ratio)

        # Calculate statistics for each frequency
        freq_stats = {}
        for freq, ratios in freq_groups.items():
            freq_stats[freq] = {
                "mean_response_ratio": np.mean(ratios),
                "std_response_ratio": np.std(ratios),
                "num_tests": len(ratios),
                "max_ratio": max(ratios),
                "min_ratio": min(ratios),
            }

        # Find optimal frequencies
        mean_ratios = {freq: stats["mean_response_ratio"] for freq, stats in freq_stats.items()}
        
        # Frequencies that increase activity (ratio > 1.5)
        enhancing_freqs = [freq for freq, ratio in mean_ratios.items() if ratio > 1.5]
        
        # Frequencies that decrease activity (ratio < 0.5)
        suppressing_freqs = [freq for freq, ratio in mean_ratios.items() if ratio < 0.5]
        
        # Best frequency for enhancement
        best_enhancing = max(mean_ratios.items(), key=lambda x: x[1]) if mean_ratios else (None, 0)
        
        # Best frequency for suppression
        best_suppressing = min(mean_ratios.items(), key=lambda x: x[1]) if mean_ratios else (None, 0)

        return {
            "frequency_statistics": freq_stats,
            "enhancing_frequencies": enhancing_freqs,
            "suppressing_frequencies": suppressing_freqs,
            "best_enhancing_frequency": best_enhancing[0],
            "best_enhancing_ratio": best_enhancing[1],
            "best_suppressing_frequency": best_suppressing[0],
            "best_suppressing_ratio": best_suppressing[1],
            "hypothesis_validation": {
                "intermediate_freq_enhancement": any(10 <= f <= 50 and r > 1.5 for f, r in mean_ratios.items()),
                "high_freq_suppression": any(f >= 100 and r < 0.5 for f, r in mean_ratios.items()),
                "low_freq_baseline": any(f < 1 and 0.8 <= r <= 1.2 for f, r in mean_ratios.items()),
            }
        }

    def _cleanup(self) -> None:
        """Release hardware resources."""
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
