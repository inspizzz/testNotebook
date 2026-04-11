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
    electrode_idx: int
    amplitude_ua: float
    duration_us: float
    timestamp_utc: str
    trigger_key: int = 0
    phase: str = ""
    protocol: str = ""
    burst_index: int = 0
    pulse_index: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ElectrodeActivity:
    electrode_idx: int
    baseline_spike_count: int = 0
    post_tetanic_spike_count: int = 0
    post_low_freq_spike_count: int = 0
    baseline_rate_hz: float = 0.0
    post_tetanic_rate_hz: float = 0.0
    post_low_freq_rate_hz: float = 0.0
    normalized_efficacy_tetanic: float = 0.0
    normalized_efficacy_low_freq: float = 0.0
    is_responsive: bool = False


class DataSaver:
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
    """
    Electrode-Targeted Activity Modulation Experiment
    ==================================================
    Goal: Identify active electrodes on the organoid, then attempt to
    increase or decrease their response activity using two contrasting
    stimulation protocols:

    1. TETANIC PROTOCOL (potentiation attempt):
       High-frequency burst stimulation (multiple pulses per burst,
       multiple bursts with inter-burst intervals) designed to induce
       LTP-like potentiation at target electrodes.

    2. LOW-FREQUENCY PROTOCOL (depression attempt):
       Low-frequency single-pulse stimulation designed to induce
       LTD-like depression at target electrodes.

    The experiment proceeds in phases:
      A. Discovery: probe all available electrodes to find responsive ones.
      B. Baseline: record spontaneous activity on responsive electrodes.
      C. Tetanic stimulation on a subset of responsive electrodes.
      D. Post-tetanic monitoring to measure potentiation.
      E. Low-frequency stimulation on another subset.
      F. Post-low-freq monitoring to measure depression.
      G. Data persistence and cleanup.

    Literature grounding:
      - Tetanic frequency ~20 Hz from 2602.12050v1 (optogenetic LTP).
      - Spaced burst protocol from 1805.10116v1 (bistable consolidation).
      - Amplitude ramping from 2402.05886v1 to avoid onset transients.
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        discovery_amplitude_ua: float = 2.0,
        discovery_duration_us: float = 200.0,
        discovery_trials: int = 3,
        discovery_wait_s: float = 1.0,
        tetanic_amplitude_ua: float = 3.0,
        tetanic_duration_us: float = 200.0,
        tetanic_num_bursts: int = 3,
        tetanic_pulses_per_burst: int = 5,
        tetanic_inter_pulse_s: float = 0.05,
        tetanic_inter_burst_s: float = 10.0,
        low_freq_amplitude_ua: float = 1.5,
        low_freq_duration_us: float = 200.0,
        low_freq_num_pulses: int = 10,
        low_freq_inter_pulse_s: float = 5.0,
        baseline_duration_s: float = 30.0,
        post_stim_monitor_s: float = 30.0,
        test_probe_amplitude_ua: float = 2.0,
        test_probe_duration_us: float = 200.0,
        test_probe_trials: int = 5,
        test_probe_interval_s: float = 3.0,
        max_electrodes_to_test: int = 8,
        ne_potentiation_threshold: float = 1.20,
        ne_depression_threshold: float = 0.80,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.discovery_amplitude_ua = min(abs(discovery_amplitude_ua), 4.0)
        self.discovery_duration_us = min(abs(discovery_duration_us), 400.0)
        self.discovery_trials = discovery_trials
        self.discovery_wait_s = discovery_wait_s

        self.tetanic_amplitude_ua = min(abs(tetanic_amplitude_ua), 4.0)
        self.tetanic_duration_us = min(abs(tetanic_duration_us), 400.0)
        self.tetanic_num_bursts = tetanic_num_bursts
        self.tetanic_pulses_per_burst = tetanic_pulses_per_burst
        self.tetanic_inter_pulse_s = tetanic_inter_pulse_s
        self.tetanic_inter_burst_s = tetanic_inter_burst_s

        self.low_freq_amplitude_ua = min(abs(low_freq_amplitude_ua), 4.0)
        self.low_freq_duration_us = min(abs(low_freq_duration_us), 400.0)
        self.low_freq_num_pulses = low_freq_num_pulses
        self.low_freq_inter_pulse_s = low_freq_inter_pulse_s

        self.baseline_duration_s = baseline_duration_s
        self.post_stim_monitor_s = post_stim_monitor_s

        self.test_probe_amplitude_ua = min(abs(test_probe_amplitude_ua), 4.0)
        self.test_probe_duration_us = min(abs(test_probe_duration_us), 400.0)
        self.test_probe_trials = test_probe_trials
        self.test_probe_interval_s = test_probe_interval_s

        self.max_electrodes_to_test = max_electrodes_to_test
        self.ne_potentiation_threshold = ne_potentiation_threshold
        self.ne_depression_threshold = ne_depression_threshold

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._electrode_activities: Dict[int, ElectrodeActivity] = {}
        self._responsive_electrodes: List[int] = []
        self._tetanic_group: List[int] = []
        self._low_freq_group: List[int] = []
        self._discovery_results: Dict[int, int] = {}
        self._baseline_counts: Dict[int, int] = {}
        self._post_tetanic_counts: Dict[int, int] = {}
        self._post_low_freq_counts: Dict[int, int] = {}

    def run(self) -> Dict[str, Any]:
        try:
            logger.info("Initialising hardware connections")

            from neuroplatform import Experiment as NeuroPlatformExperiment

            self.experiment = NeuroPlatformExperiment(self.token)
            self.trigger_controller = TriggerController(email=self.booking_email)
            self.intan = IntanSofware()
            self.database = Database()

            fs_name = self.experiment.exp_name
            electrodes = self.experiment.electrodes
            logger.info("Experiment: %s", fs_name)
            logger.info("Available electrodes: %s", electrodes)

            if not self.experiment.start():
                logger.error("Failed to start experiment")
                return {"status": "failed", "error": "Could not start experiment"}

            recording_start = datetime.now(timezone.utc)

            self._phase_discovery(electrodes)

            if not self._responsive_electrodes:
                logger.warning("No responsive electrodes found. Using first available electrodes for protocol.")
                self._responsive_electrodes = electrodes[:min(self.max_electrodes_to_test, len(electrodes))]

            self._assign_electrode_groups()

            self._phase_baseline()

            self._phase_tetanic_stimulation()

            self._phase_post_tetanic_monitoring()

            self._phase_low_frequency_stimulation()

            self._phase_post_low_freq_monitoring()

            self._phase_final_probe()

            recording_stop = datetime.now(timezone.utc)

            self._compute_normalized_efficacy()

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _phase_discovery(self, electrodes: List[int]) -> None:
        logger.info("Phase: Discovery - probing %d electrodes", len(electrodes))
        candidates = electrodes[:min(self.max_electrodes_to_test, len(electrodes))]

        for electrode_idx in candidates:
            total_spikes = 0
            for trial in range(self.discovery_trials):
                spike_df = self._stimulate_and_record(
                    electrode_idx=electrode_idx,
                    amplitude_ua=self.discovery_amplitude_ua,
                    duration_us=self.discovery_duration_us,
                    trigger_key=0,
                    post_stim_wait_s=0.5,
                    recording_window_s=1.0,
                    phase="discovery",
                    protocol="probe",
                )
                total_spikes += len(spike_df)
                time.sleep(self.discovery_wait_s)

            self._discovery_results[electrode_idx] = total_spikes
            logger.info("Electrode %d: %d spikes in %d discovery trials",
                        electrode_idx, total_spikes, self.discovery_trials)

        sorted_electrodes = sorted(
            self._discovery_results.items(), key=lambda x: x[1], reverse=True
        )
        self._responsive_electrodes = [
            e for e, count in sorted_electrodes if count > 0
        ]

        if len(self._responsive_electrodes) == 0:
            self._responsive_electrodes = [e for e, _ in sorted_electrodes[:4]]

        logger.info("Responsive electrodes: %s", self._responsive_electrodes)

    def _assign_electrode_groups(self) -> None:
        n = len(self._responsive_electrodes)
        if n == 0:
            return
        mid = max(1, n // 2)
        self._tetanic_group = self._responsive_electrodes[:mid]
        self._low_freq_group = self._responsive_electrodes[mid:]
        if not self._low_freq_group:
            self._low_freq_group = self._responsive_electrodes[:1]

        logger.info("Tetanic group: %s", self._tetanic_group)
        logger.info("Low-freq group: %s", self._low_freq_group)

        for e in self._responsive_electrodes:
            self._electrode_activities[e] = ElectrodeActivity(electrode_idx=e)

    def _phase_baseline(self) -> None:
        logger.info("Phase: Baseline recording for %.1f seconds", self.baseline_duration_s)
        baseline_start = datetime.now(timezone.utc)
        time.sleep(self.baseline_duration_s)
        baseline_stop = datetime.now(timezone.utc)

        fs_name = self.experiment.exp_name
        for electrode_idx in self._responsive_electrodes:
            spike_df = self.database.get_spike_event_electrode(
                baseline_start, baseline_stop, electrode_idx
            )
            count = len(spike_df)
            self._baseline_counts[electrode_idx] = count
            duration_s = (baseline_stop - baseline_start).total_seconds()
            rate = count / max(duration_s, 0.001)
            if electrode_idx in self._electrode_activities:
                self._electrode_activities[electrode_idx].baseline_spike_count = count
                self._electrode_activities[electrode_idx].baseline_rate_hz = rate
            logger.info("Baseline electrode %d: %d spikes (%.2f Hz)",
                        electrode_idx, count, rate)

    def _phase_tetanic_stimulation(self) -> None:
        logger.info("Phase: Tetanic stimulation on %d electrodes", len(self._tetanic_group))

        for electrode_idx in self._tetanic_group:
            logger.info("Tetanic protocol on electrode %d: %d bursts x %d pulses",
                        electrode_idx, self.tetanic_num_bursts, self.tetanic_pulses_per_burst)

            for burst_idx in range(self.tetanic_num_bursts):
                ramp_factor = min(1.0, (burst_idx + 1) / max(self.tetanic_num_bursts, 1))
                ramped_amplitude = self.tetanic_amplitude_ua * (0.5 + 0.5 * ramp_factor)
                ramped_amplitude = min(ramped_amplitude, 4.0)

                for pulse_idx in range(self.tetanic_pulses_per_burst):
                    self._stimulate_and_record(
                        electrode_idx=electrode_idx,
                        amplitude_ua=ramped_amplitude,
                        duration_us=self.tetanic_duration_us,
                        trigger_key=1,
                        post_stim_wait_s=0.02,
                        recording_window_s=0.1,
                        phase="tetanic",
                        protocol="burst",
                        burst_index=burst_idx,
                        pulse_index=pulse_idx,
                    )
                    time.sleep(self.tetanic_inter_pulse_s)

                if burst_idx < self.tetanic_num_bursts - 1:
                    logger.info("Inter-burst interval: %.1f s", self.tetanic_inter_burst_s)
                    time.sleep(self.tetanic_inter_burst_s)

            logger.info("Tetanic protocol complete for electrode %d", electrode_idx)

    def _phase_post_tetanic_monitoring(self) -> None:
        logger.info("Phase: Post-tetanic monitoring for %.1f seconds", self.post_stim_monitor_s)
        monitor_start = datetime.now(timezone.utc)
        time.sleep(self.post_stim_monitor_s)
        monitor_stop = datetime.now(timezone.utc)

        for electrode_idx in self._tetanic_group:
            spike_df = self.database.get_spike_event_electrode(
                monitor_start, monitor_stop, electrode_idx
            )
            count = len(spike_df)
            self._post_tetanic_counts[electrode_idx] = count
            duration_s = (monitor_stop - monitor_start).total_seconds()
            rate = count / max(duration_s, 0.001)
            if electrode_idx in self._electrode_activities:
                self._electrode_activities[electrode_idx].post_tetanic_spike_count = count
                self._electrode_activities[electrode_idx].post_tetanic_rate_hz = rate
            logger.info("Post-tetanic electrode %d: %d spikes (%.2f Hz)",
                        electrode_idx, count, rate)

    def _phase_low_frequency_stimulation(self) -> None:
        logger.info("Phase: Low-frequency stimulation on %d electrodes", len(self._low_freq_group))

        for electrode_idx in self._low_freq_group:
            logger.info("Low-freq protocol on electrode %d: %d pulses at %.2f Hz",
                        electrode_idx, self.low_freq_num_pulses,
                        1.0 / max(self.low_freq_inter_pulse_s, 0.001))

            for pulse_idx in range(self.low_freq_num_pulses):
                self._stimulate_and_record(
                    electrode_idx=electrode_idx,
                    amplitude_ua=self.low_freq_amplitude_ua,
                    duration_us=self.low_freq_duration_us,
                    trigger_key=2,
                    post_stim_wait_s=0.3,
                    recording_window_s=0.5,
                    phase="low_frequency",
                    protocol="single_pulse",
                    pulse_index=pulse_idx,
                )
                if pulse_idx < self.low_freq_num_pulses - 1:
                    time.sleep(self.low_freq_inter_pulse_s)

            logger.info("Low-freq protocol complete for electrode %d", electrode_idx)

    def _phase_post_low_freq_monitoring(self) -> None:
        logger.info("Phase: Post-low-freq monitoring for %.1f seconds", self.post_stim_monitor_s)
        monitor_start = datetime.now(timezone.utc)
        time.sleep(self.post_stim_monitor_s)
        monitor_stop = datetime.now(timezone.utc)

        for electrode_idx in self._low_freq_group:
            spike_df = self.database.get_spike_event_electrode(
                monitor_start, monitor_stop, electrode_idx
            )
            count = len(spike_df)
            self._post_low_freq_counts[electrode_idx] = count
            duration_s = (monitor_stop - monitor_start).total_seconds()
            rate = count / max(duration_s, 0.001)
            if electrode_idx in self._electrode_activities:
                self._electrode_activities[electrode_idx].post_low_freq_spike_count = count
                self._electrode_activities[electrode_idx].post_low_freq_rate_hz = rate
            logger.info("Post-low-freq electrode %d: %d spikes (%.2f Hz)",
                        electrode_idx, count, rate)

    def _phase_final_probe(self) -> None:
        logger.info("Phase: Final probe - testing all responsive electrodes")

        for electrode_idx in self._responsive_electrodes:
            total_spikes = 0
            for trial in range(self.test_probe_trials):
                spike_df = self._stimulate_and_record(
                    electrode_idx=electrode_idx,
                    amplitude_ua=self.test_probe_amplitude_ua,
                    duration_us=self.test_probe_duration_us,
                    trigger_key=3,
                    post_stim_wait_s=0.5,
                    recording_window_s=1.0,
                    phase="final_probe",
                    protocol="test_pulse",
                    pulse_index=trial,
                )
                total_spikes += len(spike_df)
                time.sleep(self.test_probe_interval_s)

            logger.info("Final probe electrode %d: %d spikes in %d trials",
                        electrode_idx, total_spikes, self.test_probe_trials)

    def _compute_normalized_efficacy(self) -> None:
        logger.info("Computing normalized efficacy for all electrodes")

        for electrode_idx in self._tetanic_group:
            act = self._electrode_activities.get(electrode_idx)
            if act and act.baseline_rate_hz > 0:
                act.normalized_efficacy_tetanic = act.post_tetanic_rate_hz / act.baseline_rate_hz
                act.is_responsive = act.normalized_efficacy_tetanic >= self.ne_potentiation_threshold
                logger.info("Electrode %d: NE_tetanic=%.3f (responsive=%s)",
                            electrode_idx, act.normalized_efficacy_tetanic, act.is_responsive)
            elif act:
                act.normalized_efficacy_tetanic = 0.0
                logger.info("Electrode %d: baseline rate zero, NE undefined", electrode_idx)

        for electrode_idx in self._low_freq_group:
            act = self._electrode_activities.get(electrode_idx)
            if act and act.baseline_rate_hz > 0:
                act.normalized_efficacy_low_freq = act.post_low_freq_rate_hz / act.baseline_rate_hz
                logger.info("Electrode %d: NE_low_freq=%.3f",
                            electrode_idx, act.normalized_efficacy_low_freq)
            elif act:
                act.normalized_efficacy_low_freq = 0.0

    def _stimulate_and_record(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity = StimPolarity.NegativeFirst,
        trigger_key: int = 0,
        post_stim_wait_s: float = 0.3,
        recording_window_s: float = 0.5,
        phase: str = "",
        protocol: str = "",
        burst_index: int = 0,
        pulse_index: int = 0,
    ) -> pd.DataFrame:
        amplitude_ua = min(abs(amplitude_ua), 4.0)
        duration_us = min(abs(duration_us), 400.0)

        stim = StimParam()
        stim.index = electrode_idx
        stim.enable = True
        stim.trigger_key = trigger_key
        stim.trigger_delay = 0
        stim.nb_pulse = 0
        stim.pulse_train_period = 10000
        stim.post_stim_ref_period = 1000.0
        stim.stim_shape = StimShape.Biphasic
        stim.polarity = polarity

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

        self.intan.send_stimparam([stim])

        pattern = np.zeros(16, dtype=np.uint8)
        pattern[trigger_key] = 1
        self.trigger_controller.send(pattern)
        time.sleep(0.05)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            trigger_key=trigger_key,
            phase=phase,
            protocol=protocol,
            burst_index=burst_index,
            pulse_index=pulse_index,
        ))

        time.sleep(post_stim_wait_s)

        query_start = datetime.now(timezone.utc) - timedelta(
            seconds=post_stim_wait_s + recording_window_s
        )
        query_stop = datetime.now(timezone.utc)
        spike_df = self.database.get_spike_event_electrode(
            query_start, query_stop, electrode_idx
        )
        return spike_df

    def _save_all(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> None:
        fs_name = getattr(self.experiment, "exp_name", "unknown")
        saver = DataSaver(self._output_dir, fs_name)

        saver.save_stimulation_log(self._stimulation_log)

        spike_df = self.database.get_spike_event(
            recording_start, recording_stop, fs_name
        )
        saver.save_spike_events(spike_df)

        trigger_df = self.database.get_all_triggers(
            recording_start, recording_stop
        )
        saver.save_triggers(trigger_df)

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "responsive_electrodes": self._responsive_electrodes,
            "tetanic_group": self._tetanic_group,
            "low_freq_group": self._low_freq_group,
            "discovery_results": self._discovery_results,
            "baseline_counts": self._baseline_counts,
            "post_tetanic_counts": self._post_tetanic_counts,
            "post_low_freq_counts": self._post_low_freq_counts,
            "electrode_activities": {
                str(k): asdict(v) for k, v in self._electrode_activities.items()
            },
            "parameters": {
                "discovery_amplitude_ua": self.discovery_amplitude_ua,
                "discovery_duration_us": self.discovery_duration_us,
                "discovery_trials": self.discovery_trials,
                "tetanic_amplitude_ua": self.tetanic_amplitude_ua,
                "tetanic_duration_us": self.tetanic_duration_us,
                "tetanic_num_bursts": self.tetanic_num_bursts,
                "tetanic_pulses_per_burst": self.tetanic_pulses_per_burst,
                "tetanic_inter_pulse_s": self.tetanic_inter_pulse_s,
                "tetanic_inter_burst_s": self.tetanic_inter_burst_s,
                "low_freq_amplitude_ua": self.low_freq_amplitude_ua,
                "low_freq_duration_us": self.low_freq_duration_us,
                "low_freq_num_pulses": self.low_freq_num_pulses,
                "low_freq_inter_pulse_s": self.low_freq_inter_pulse_s,
                "baseline_duration_s": self.baseline_duration_s,
                "post_stim_monitor_s": self.post_stim_monitor_s,
                "ne_potentiation_threshold": self.ne_potentiation_threshold,
                "ne_depression_threshold": self.ne_depression_threshold,
            },
        }
        saver.save_summary(summary)

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
        waveform_records = []
        if spike_df.empty:
            return waveform_records

        electrode_col = None
        for candidate in ["channel", "index", "electrode_idx", "electrode"]:
            if candidate in spike_df.columns:
                electrode_col = candidate
                break

        if electrode_col is None:
            for col in spike_df.columns:
                if "electrode" in col.lower() or "idx" in col.lower() or "channel" in col.lower():
                    electrode_col = col
                    break

        if electrode_col is None:
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
        logger.info("Compiling results")

        potentiated = []
        depressed = []
        for e_idx, act in self._electrode_activities.items():
            if act.normalized_efficacy_tetanic >= self.ne_potentiation_threshold:
                potentiated.append(e_idx)
            if act.normalized_efficacy_low_freq > 0 and act.normalized_efficacy_low_freq <= self.ne_depression_threshold:
                depressed.append(e_idx)

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "total_stimulations": len(self._stimulation_log),
            "responsive_electrodes": self._responsive_electrodes,
            "tetanic_group": self._tetanic_group,
            "low_freq_group": self._low_freq_group,
            "potentiated_electrodes": potentiated,
            "depressed_electrodes": depressed,
            "electrode_activities": {
                str(k): asdict(v) for k, v in self._electrode_activities.items()
            },
            "discovery_results": self._discovery_results,
            "hypothesis": (
                "Tetanic burst stimulation (20 Hz-like) increases neural response "
                "activity (NE >= 1.20), while low-frequency stimulation decreases "
                "it (NE <= 0.80). Grounded in LTP/LTD literature: 2602.12050v1, "
                "1805.10116v1, 2402.05886v1."
            ),
        }

        return summary

    def _cleanup(self) -> None:
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
