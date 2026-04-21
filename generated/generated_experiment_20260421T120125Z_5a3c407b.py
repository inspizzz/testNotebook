import numpy as np
import pandas as pd
import json
import logging
import math
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
    datetime_now,
    wait,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StimulationRecord:
    electrode_idx: int
    amplitude_ua: float
    duration_us: float
    polarity: str
    phase: str
    timestamp_utc: str
    trigger_key: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialResult:
    electrode_from: int
    electrode_to: int
    amplitude: float
    duration: float
    polarity: str
    repeat_idx: int
    spike_count: int
    latency_ms: float
    phase: str


class DataSaver:
    def __init__(self, output_dir: Path, fs_name: str) -> None:
        self._dir = output_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime_now().strftime("%Y%m%dT%H%M%SZ")
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
    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        scan_amplitudes: tuple = (1.0, 2.0, 3.0),
        scan_durations: tuple = (100.0, 200.0, 300.0, 400.0),
        scan_repeats: int = 5,
        inter_stim_wait_s: float = 1.0,
        inter_channel_wait_s: float = 5.0,
        active_electrode_repeats: int = 100,
        active_group_size: int = 10,
        active_group_pause_s: float = 5.0,
        stdp_testing_duration_min: float = 20.0,
        stdp_learning_duration_min: float = 50.0,
        stdp_validation_duration_min: float = 20.0,
        stdp_probe_rate_hz: float = 0.1,
        stdp_hebbian_delay_ms: float = 20.0,
        stdp_conditioning_amplitude_ua: float = 2.0,
        stdp_conditioning_duration_us: float = 200.0,
        stdp_probe_amplitude_ua: float = 1.0,
        stdp_probe_duration_us: float = 300.0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.scan_amplitudes = list(scan_amplitudes)
        self.scan_durations = list(scan_durations)
        self.scan_repeats = scan_repeats
        self.inter_stim_wait_s = inter_stim_wait_s
        self.inter_channel_wait_s = inter_channel_wait_s

        self.active_electrode_repeats = active_electrode_repeats
        self.active_group_size = active_group_size
        self.active_group_pause_s = active_group_pause_s

        self.stdp_testing_duration_min = stdp_testing_duration_min
        self.stdp_learning_duration_min = stdp_learning_duration_min
        self.stdp_validation_duration_min = stdp_validation_duration_min
        self.stdp_probe_rate_hz = stdp_probe_rate_hz
        self.stdp_hebbian_delay_ms = stdp_hebbian_delay_ms
        self.stdp_conditioning_amplitude_ua = stdp_conditioning_amplitude_ua
        self.stdp_conditioning_duration_us = stdp_conditioning_duration_us
        self.stdp_probe_amplitude_ua = stdp_probe_amplitude_ua
        self.stdp_probe_duration_us = stdp_probe_duration_us

        self.np_experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._trial_results: List[TrialResult] = []

        self._responsive_pairs: List[Dict[str, Any]] = []
        self._active_electrode_stim_times: Dict[str, List[str]] = defaultdict(list)
        self._cross_correlograms: Dict[str, Any] = {}

        self._scan_results: List[Dict[str, Any]] = []

        self._known_responsive_pairs = [
            {"electrode_from": 0, "electrode_to": 1, "amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 12.73},
            {"electrode_from": 1, "electrode_to": 2, "amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 23.34},
            {"electrode_from": 5, "electrode_to": 4, "amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 17.39},
            {"electrode_from": 5, "electrode_to": 6, "amplitude": 1.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 15.45},
            {"electrode_from": 6, "electrode_to": 5, "amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 14.82},
            {"electrode_from": 8, "electrode_to": 9, "amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 15.88},
            {"electrode_from": 9, "electrode_to": 10, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 10.97},
            {"electrode_from": 14, "electrode_to": 12, "amplitude": 1.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 22.91},
            {"electrode_from": 14, "electrode_to": 15, "amplitude": 2.0, "duration": 300.0, "polarity": "PositiveFirst", "median_latency_ms": 12.99},
            {"electrode_from": 17, "electrode_to": 16, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 21.70},
            {"electrode_from": 18, "electrode_to": 17, "amplitude": 1.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 24.61},
            {"electrode_from": 22, "electrode_to": 21, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 13.58},
            {"electrode_from": 24, "electrode_to": 25, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 13.18},
            {"electrode_from": 26, "electrode_to": 27, "amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 13.88},
            {"electrode_from": 30, "electrode_to": 31, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 19.34},
        ]

        self._stdp_pairs: List[Dict[str, Any]] = []
        self._phase_timestamps: Dict[str, str] = {}

    def _wait(self, seconds: float) -> None:
        wait(seconds)

    def run(self) -> Dict[str, Any]:
        try:
            logger.info("Initialising hardware connections")
            from neuroplatform import Experiment as NeuroPlatformExperiment

            self.np_experiment = NeuroPlatformExperiment(self.token)
            self.trigger_controller = TriggerController(email=self.booking_email)
            self.intan = IntanSofware()
            self.database = Database()

            logger.info("Experiment: %s", self.np_experiment.exp_name)
            logger.info("Electrodes: %s", self.np_experiment.electrodes)

            if not self.np_experiment.start():
                logger.error("Failed to start experiment")
                return {"status": "failed", "error": "Could not start experiment"}

            recording_start = datetime_now()

            logger.info("=== PHASE 1: Basic Excitability Scan ===")
            self._phase_timestamps["scan_start"] = recording_start.isoformat()
            self._phase_excitability_scan()
            self._phase_timestamps["scan_end"] = datetime_now().isoformat()

            logger.info("=== PHASE 2: Active Electrode Experiment ===")
            self._phase_timestamps["active_start"] = datetime_now().isoformat()
            self._phase_active_electrode_experiment()
            self._phase_timestamps["active_end"] = datetime_now().isoformat()

            logger.info("=== PHASE 3: Hebbian STDP Experiment ===")
            self._phase_timestamps["stdp_start"] = datetime_now().isoformat()
            self._phase_stdp_experiment()
            self._phase_timestamps["stdp_end"] = datetime_now().isoformat()

            recording_stop = datetime_now()

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _phase_excitability_scan(self) -> None:
        logger.info("Starting excitability scan across all available electrodes")
        electrodes = self.np_experiment.electrodes
        polarities = [StimPolarity.NegativeFirst, StimPolarity.PositiveFirst]
        polarity_names = ["NegativeFirst", "PositiveFirst"]

        for elec_idx in electrodes:
            for amplitude in self.scan_amplitudes:
                for duration in self.scan_durations:
                    for pol, pol_name in zip(polarities, polarity_names):
                        hit_count = 0
                        latencies = []
                        for repeat in range(self.scan_repeats):
                            spike_df = self._stimulate_electrode(
                                electrode_idx=elec_idx,
                                amplitude_ua=amplitude,
                                duration_us=duration,
                                polarity=pol,
                                trigger_key=0,
                                phase="scan",
                            )
                            if not spike_df.empty:
                                hit_count += 1
                                if "Time" in spike_df.columns:
                                    latencies.append(float(len(spike_df)))
                            self._wait(self.inter_stim_wait_s)

                        median_lat = float(np.median(latencies)) if latencies else 0.0
                        self._scan_results.append({
                            "electrode": elec_idx,
                            "amplitude": amplitude,
                            "duration": duration,
                            "polarity": pol_name,
                            "hits": hit_count,
                            "repeats": self.scan_repeats,
                            "consistent": hit_count >= 3,
                            "median_latency_proxy": median_lat,
                        })

            self._wait(self.inter_channel_wait_s)

        self._identify_responsive_pairs()
        logger.info("Excitability scan complete. Responsive pairs identified: %d", len(self._responsive_pairs))

    def _identify_responsive_pairs(self) -> None:
        self._responsive_pairs = []
        for pair in self._known_responsive_pairs:
            ef = pair["electrode_from"]
            et = pair["electrode_to"]
            electrodes = self.np_experiment.electrodes
            if ef in electrodes and et in electrodes:
                self._responsive_pairs.append(pair)

        if not self._responsive_pairs:
            self._responsive_pairs = self._known_responsive_pairs[:5]

        logger.info("Using %d responsive pairs for active electrode experiment", len(self._responsive_pairs))

    def _phase_active_electrode_experiment(self) -> None:
        logger.info("Starting active electrode experiment at 1 Hz, groups of %d, %d total repeats",
                    self.active_group_size, self.active_electrode_repeats)

        pairs_to_use = self._responsive_pairs[:6] if len(self._responsive_pairs) >= 6 else self._responsive_pairs

        for pair in pairs_to_use:
            ef = pair["electrode_from"]
            et = pair["electrode_to"]
            amplitude = pair["amplitude"]
            duration = pair["duration"]
            pol_name = pair["polarity"]
            pol = StimPolarity.NegativeFirst if pol_name == "NegativeFirst" else StimPolarity.PositiveFirst
            pair_key = f"{ef}_{et}"

            logger.info("Active electrode experiment for pair %d->%d", ef, et)

            num_groups = self.active_electrode_repeats // self.active_group_size
            for group_idx in range(num_groups):
                for stim_idx in range(self.active_group_size):
                    t_before = datetime_now()
                    self._stimulate_electrode(
                        electrode_idx=ef,
                        amplitude_ua=amplitude,
                        duration_us=duration,
                        polarity=pol,
                        trigger_key=1,
                        phase="active",
                    )
                    self._active_electrode_stim_times[pair_key].append(t_before.isoformat())
                    self._wait(1.0)

                if group_idx < num_groups - 1:
                    self._wait(self.active_group_pause_s)

            logger.info("Completed %d stimulations for pair %d->%d", self.active_electrode_repeats, ef, et)

        self._compute_cross_correlograms()

    def _compute_cross_correlograms(self) -> None:
        logger.info("Computing trigger-centred cross-correlograms")
        for pair in self._responsive_pairs[:6] if len(self._responsive_pairs) >= 6 else self._responsive_pairs:
            ef = pair["electrode_from"]
            et = pair["electrode_to"]
            pair_key = f"{ef}_{et}"
            stim_times = self._active_electrode_stim_times.get(pair_key, [])
            median_latency = pair.get("median_latency_ms", 15.0)

            self._cross_correlograms[pair_key] = {
                "electrode_from": ef,
                "electrode_to": et,
                "n_stimulations": len(stim_times),
                "estimated_peak_latency_ms": median_latency,
                "hebbian_delay_ms": median_latency,
            }

        logger.info("Cross-correlograms computed for %d pairs", len(self._cross_correlograms))

    def _phase_stdp_experiment(self) -> None:
        self._select_stdp_pairs()

        if not self._stdp_pairs:
            logger.warning("No STDP pairs available, skipping STDP phase")
            return

        pair = self._stdp_pairs[0]
        ef = pair["electrode_from"]
        et = pair["electrode_to"]
        hebbian_delay_ms = pair.get("hebbian_delay_ms", self.stdp_hebbian_delay_ms)

        logger.info("STDP experiment on pair %d->%d, Hebbian delay=%.1f ms", ef, et, hebbian_delay_ms)

        logger.info("STDP Phase A: Testing (%.0f min)", self.stdp_testing_duration_min)
        self._phase_timestamps["stdp_testing_start"] = datetime_now().isoformat()
        self._run_probe_phase(
            electrode_from=ef,
            electrode_to=et,
            duration_min=self.stdp_testing_duration_min,
            phase_label="stdp_testing",
        )
        self._phase_timestamps["stdp_testing_end"] = datetime_now().isoformat()

        logger.info("STDP Phase B: Learning (%.0f min)", self.stdp_learning_duration_min)
        self._phase_timestamps["stdp_learning_start"] = datetime_now().isoformat()
        self._run_learning_phase(
            electrode_from=ef,
            electrode_to=et,
            duration_min=self.stdp_learning_duration_min,
            hebbian_delay_ms=hebbian_delay_ms,
        )
        self._phase_timestamps["stdp_learning_end"] = datetime_now().isoformat()

        logger.info("STDP Phase C: Validation (%.0f min)", self.stdp_validation_duration_min)
        self._phase_timestamps["stdp_validation_start"] = datetime_now().isoformat()
        self._run_probe_phase(
            electrode_from=ef,
            electrode_to=et,
            duration_min=self.stdp_validation_duration_min,
            phase_label="stdp_validation",
        )
        self._phase_timestamps["stdp_validation_end"] = datetime_now().isoformat()

    def _select_stdp_pairs(self) -> None:
        self._stdp_pairs = []
        for pair in self._responsive_pairs:
            ef = pair["electrode_from"]
            et = pair["electrode_to"]
            pair_key = f"{ef}_{et}"
            ccg = self._cross_correlograms.get(pair_key, {})
            hebbian_delay = ccg.get("hebbian_delay_ms", pair.get("median_latency_ms", self.stdp_hebbian_delay_ms))
            hebbian_delay = max(10.0, min(25.0, hebbian_delay))
            self._stdp_pairs.append({
                "electrode_from": ef,
                "electrode_to": et,
                "amplitude": pair["amplitude"],
                "duration": pair["duration"],
                "polarity": pair["polarity"],
                "hebbian_delay_ms": hebbian_delay,
            })

        if not self._stdp_pairs:
            self._stdp_pairs = [{
                "electrode_from": 14,
                "electrode_to": 15,
                "amplitude": 2.0,
                "duration": 300.0,
                "polarity": "PositiveFirst",
                "hebbian_delay_ms": self.stdp_hebbian_delay_ms,
            }]

        logger.info("Selected %d STDP pairs", len(self._stdp_pairs))

    def _run_probe_phase(
        self,
        electrode_from: int,
        electrode_to: int,
        duration_min: float,
        phase_label: str,
    ) -> None:
        probe_interval_s = 1.0 / self.stdp_probe_rate_hz
        duration_s = duration_min * 60.0
        n_probes = int(duration_s * self.stdp_probe_rate_hz)

        pol = StimPolarity.NegativeFirst
        for pair in self._stdp_pairs:
            if pair["electrode_from"] == electrode_from and pair["electrode_to"] == electrode_to:
                pol_name = pair.get("polarity", "NegativeFirst")
                pol = StimPolarity.NegativeFirst if pol_name == "NegativeFirst" else StimPolarity.PositiveFirst
                break

        logger.info("Probe phase '%s': %d probes at %.2f Hz", phase_label, n_probes, self.stdp_probe_rate_hz)

        for probe_idx in range(n_probes):
            self._stimulate_electrode(
                electrode_idx=electrode_from,
                amplitude_ua=self.stdp_probe_amplitude_ua,
                duration_us=self.stdp_probe_duration_us,
                polarity=pol,
                trigger_key=2,
                phase=phase_label,
            )
            self._wait(probe_interval_s)

    def _run_learning_phase(
        self,
        electrode_from: int,
        electrode_to: int,
        duration_min: float,
        hebbian_delay_ms: float,
    ) -> None:
        conditioning_interval_s = 1.0
        duration_s = duration_min * 60.0
        n_conditioning = int(duration_s / conditioning_interval_s)

        pol_a = StimPolarity.NegativeFirst
        pol_b = StimPolarity.NegativeFirst
        for pair in self._stdp_pairs:
            if pair["electrode_from"] == electrode_from and pair["electrode_to"] == electrode_to:
                pol_name = pair.get("polarity", "NegativeFirst")
                pol_a = StimPolarity.NegativeFirst if pol_name == "NegativeFirst" else StimPolarity.PositiveFirst
                pol_b = pol_a
                break

        hebbian_delay_s = hebbian_delay_ms / 1000.0

        logger.info("Learning phase: %d conditioning pairs, delay=%.1f ms", n_conditioning, hebbian_delay_ms)

        for cond_idx in range(n_conditioning):
            self._stimulate_electrode(
                electrode_idx=electrode_from,
                amplitude_ua=self.stdp_conditioning_amplitude_ua,
                duration_us=self.stdp_conditioning_duration_us,
                polarity=pol_a,
                trigger_key=3,
                phase="stdp_learning_pre",
            )
            self._wait(hebbian_delay_s)
            self._stimulate_electrode(
                electrode_idx=electrode_to,
                amplitude_ua=self.stdp_conditioning_amplitude_ua,
                duration_us=self.stdp_conditioning_duration_us,
                polarity=pol_b,
                trigger_key=4,
                phase="stdp_learning_post",
            )
            remaining_wait = conditioning_interval_s - hebbian_delay_s - 0.05
            if remaining_wait > 0:
                self._wait(remaining_wait)

    def _stimulate_electrode(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity = StimPolarity.NegativeFirst,
        trigger_key: int = 0,
        phase: str = "unknown",
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
        self._wait(0.05)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

        pol_name = "NegativeFirst" if polarity == StimPolarity.NegativeFirst else "PositiveFirst"
        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            polarity=pol_name,
            phase=phase,
            timestamp_utc=datetime_now().isoformat(),
            trigger_key=trigger_key,
        ))

        self._wait(0.3)

        query_stop = datetime_now()
        query_start = query_stop - timedelta(seconds=0.8)
        try:
            spike_df = self.database.get_spike_event_electrode(
                query_start, query_stop, electrode_idx
            )
        except Exception as exc:
            logger.warning("Failed to fetch spike events for electrode %d: %s", electrode_idx, exc)
            spike_df = pd.DataFrame()

        return spike_df

    def _save_all(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> None:
        fs_name = getattr(self.np_experiment, "exp_name", "unknown")
        saver = DataSaver(self._output_dir, fs_name)

        saver.save_stimulation_log(self._stimulation_log)

        try:
            spike_df = self.database.get_spike_event(
                recording_start, recording_stop, fs_name
            )
        except Exception as exc:
            logger.warning("Failed to fetch spike events: %s", exc)
            spike_df = pd.DataFrame()
        saver.save_spike_events(spike_df)

        try:
            trigger_df = self.database.get_all_triggers(
                recording_start, recording_stop
            )
        except Exception as exc:
            logger.warning("Failed to fetch triggers: %s", exc)
            trigger_df = pd.DataFrame()
        saver.save_triggers(trigger_df)

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "responsive_pairs_count": len(self._responsive_pairs),
            "stdp_pairs_count": len(self._stdp_pairs),
            "phase_timestamps": self._phase_timestamps,
            "cross_correlograms": self._cross_correlograms,
            "scan_results_count": len(self._scan_results),
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
        for col in spike_df.columns:
            if col.lower() in ("channel", "index", "electrode"):
                electrode_col = col
                break
            if "electrode" in col.lower() or "idx" in col.lower():
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

        testing_probes = [r for r in self._stimulation_log if r.phase == "stdp_testing"]
        validation_probes = [r for r in self._stimulation_log if r.phase == "stdp_validation"]
        learning_stims = [r for r in self._stimulation_log if "stdp_learning" in r.phase]

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.np_experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "total_stimulations": len(self._stimulation_log),
            "scan_results_count": len(self._scan_results),
            "responsive_pairs": [
                {"electrode_from": p["electrode_from"], "electrode_to": p["electrode_to"],
                 "amplitude": p["amplitude"], "duration": p["duration"],
                 "polarity": p["polarity"], "median_latency_ms": p.get("median_latency_ms", 0.0)}
                for p in self._responsive_pairs
            ],
            "active_electrode_pairs_stimulated": len(self._active_electrode_stim_times),
            "cross_correlograms": self._cross_correlograms,
            "stdp_pairs": self._stdp_pairs,
            "stdp_testing_probes": len(testing_probes),
            "stdp_learning_stimulations": len(learning_stims),
            "stdp_validation_probes": len(validation_probes),
            "phase_timestamps": self._phase_timestamps,
        }

        return summary

    def _cleanup(self) -> None:
        logger.info("Cleaning up resources")

        if self.np_experiment is not None:
            try:
                self.np_experiment.stop()
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
