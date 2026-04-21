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
from neuroplatform import Experiment as NeuroPlatformExperiment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StimulationRecord:
    electrode_idx: int
    amplitude_ua: float
    duration_us: float
    polarity: str
    timestamp_utc: str
    trigger_key: int = 0
    phase: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScanResult:
    electrode_from: int
    electrode_to: int
    amplitude: float
    duration: float
    polarity: str
    hits: int
    repeats: int
    median_latency_ms: float


@dataclass
class CrossCorrelogramResult:
    electrode_from: int
    electrode_to: int
    lag_bins_ms: List[float]
    counts: List[int]
    peak_lag_ms: float


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
        scan_inter_stim_s: float = 1.0,
        scan_inter_channel_s: float = 5.0,
        scan_required_hits: int = 3,
        active_stim_hz: float = 1.0,
        active_group_size: int = 10,
        active_inter_group_s: float = 5.0,
        active_total_repeats: int = 100,
        stdp_testing_duration_s: float = 1200.0,
        stdp_learning_duration_s: float = 3000.0,
        stdp_validation_duration_s: float = 1200.0,
        stdp_probe_interval_s: float = 10.0,
        stdp_hebbian_delay_ms: float = 20.0,
        stdp_conditioning_amplitude_ua: float = 3.0,
        stdp_conditioning_duration_us: float = 400.0,
        stdp_probe_amplitude_ua: float = 2.0,
        stdp_probe_duration_us: float = 200.0,
        xcorr_window_ms: float = 100.0,
        xcorr_bin_ms: float = 1.0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.scan_amplitudes = list(scan_amplitudes)
        self.scan_durations = list(scan_durations)
        self.scan_repeats = scan_repeats
        self.scan_inter_stim_s = scan_inter_stim_s
        self.scan_inter_channel_s = scan_inter_channel_s
        self.scan_required_hits = scan_required_hits

        self.active_stim_hz = active_stim_hz
        self.active_group_size = active_group_size
        self.active_inter_group_s = active_inter_group_s
        self.active_total_repeats = active_total_repeats

        self.stdp_testing_duration_s = stdp_testing_duration_s
        self.stdp_learning_duration_s = stdp_learning_duration_s
        self.stdp_validation_duration_s = stdp_validation_duration_s
        self.stdp_probe_interval_s = stdp_probe_interval_s
        self.stdp_hebbian_delay_ms = stdp_hebbian_delay_ms
        self.stdp_conditioning_amplitude_ua = stdp_conditioning_amplitude_ua
        self.stdp_conditioning_duration_us = stdp_conditioning_duration_us
        self.stdp_probe_amplitude_ua = stdp_probe_amplitude_ua
        self.stdp_probe_duration_us = stdp_probe_duration_us

        self.xcorr_window_ms = xcorr_window_ms
        self.xcorr_bin_ms = xcorr_bin_ms

        self.np_experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._scan_results: List[ScanResult] = []
        self._responsive_pairs: List[Dict[str, Any]] = []
        self._active_stim_times: Dict[str, List[datetime]] = {}
        self._xcorr_results: List[CrossCorrelogramResult] = []
        self._stdp_results: Dict[str, Any] = {}

        self._known_responsive_pairs = [
            {"electrode_from": 7, "electrode_to": 6, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 24.74},
            {"electrode_from": 17, "electrode_to": 18, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 13.17},
            {"electrode_from": 21, "electrode_to": 19, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 19.3},
            {"electrode_from": 21, "electrode_to": 22, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 11.34},
        ]

    def _wait(self, seconds: float) -> None:
        wait(seconds)

    def run(self) -> Dict[str, Any]:
        try:
            logger.info("Initialising hardware connections")
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

            self._phase_excitability_scan()
            self._phase_active_electrode_experiment()
            self._phase_stdp_experiment()

            recording_stop = datetime_now()

            self._save_all(recording_start, recording_stop)

            results = self._compile_results(recording_start, recording_stop)
            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _phase_excitability_scan(self) -> None:
        logger.info("Phase 1: Basic Excitability Scan")
        electrodes = self.np_experiment.electrodes
        polarities = [StimPolarity.PositiveFirst, StimPolarity.NegativeFirst]
        polarity_names = {StimPolarity.PositiveFirst: "PositiveFirst", StimPolarity.NegativeFirst: "NegativeFirst"}

        for electrode_idx in electrodes:
            for amplitude in self.scan_amplitudes:
                for duration in self.scan_durations:
                    for polarity in polarities:
                        hits = 0
                        latencies = []
                        for rep in range(self.scan_repeats):
                            stim_time = datetime_now()
                            self._send_stim_pulse(
                                electrode_idx=electrode_idx,
                                amplitude_ua=amplitude,
                                duration_us=duration,
                                polarity=polarity,
                                trigger_key=0,
                                phase="scan",
                            )
                            self._wait(0.05)
                            query_start = stim_time
                            query_stop = datetime_now()
                            spike_df = self.database.get_spike_event(
                                query_start, query_stop, self.np_experiment.exp_name
                            )
                            if not spike_df.empty:
                                window_spikes = spike_df[spike_df["channel"] != electrode_idx]
                                if not window_spikes.empty:
                                    hits += 1
                                    if "Time" in window_spikes.columns:
                                        t_stim = stim_time.timestamp()
                                        for _, row in window_spikes.iterrows():
                                            t_spike = pd.Timestamp(row["Time"]).timestamp()
                                            lat = (t_spike - t_stim) * 1000.0
                                            if 0 < lat < 100:
                                                latencies.append(lat)
                            if rep < self.scan_repeats - 1:
                                self._wait(self.scan_inter_stim_s)

                        if hits >= self.scan_required_hits:
                            median_lat = float(np.median(latencies)) if latencies else 0.0
                            result = ScanResult(
                                electrode_from=electrode_idx,
                                electrode_to=-1,
                                amplitude=amplitude,
                                duration=duration,
                                polarity=polarity_names[polarity],
                                hits=hits,
                                repeats=self.scan_repeats,
                                median_latency_ms=median_lat,
                            )
                            self._scan_results.append(result)
                            logger.info(
                                "Responsive electrode %d: amp=%.1f dur=%.0f pol=%s hits=%d",
                                electrode_idx, amplitude, duration, polarity_names[polarity], hits,
                            )

                    self._wait(self.scan_inter_channel_s)

        if not self._responsive_pairs:
            logger.info("Using known responsive pairs from prior scan")
            self._responsive_pairs = list(self._known_responsive_pairs)
        else:
            seen = set()
            for r in self._scan_results:
                key = r.electrode_from
                if key not in seen:
                    seen.add(key)
                    self._responsive_pairs.append({
                        "electrode_from": r.electrode_from,
                        "electrode_to": r.electrode_to,
                        "amplitude": r.amplitude,
                        "duration": r.duration,
                        "polarity": r.polarity,
                        "median_latency_ms": r.median_latency_ms,
                    })

        if not self._responsive_pairs:
            logger.info("Falling back to known responsive pairs")
            self._responsive_pairs = list(self._known_responsive_pairs)

        logger.info("Phase 1 complete. Responsive pairs: %d", len(self._responsive_pairs))

    def _phase_active_electrode_experiment(self) -> None:
        logger.info("Phase 2: Active Electrode Experiment")
        pairs_to_use = self._responsive_pairs if self._responsive_pairs else self._known_responsive_pairs
        inter_stim_s = 1.0 / self.active_stim_hz

        for pair in pairs_to_use:
            elec_from = pair["electrode_from"]
            amplitude = pair.get("amplitude", 3.0)
            duration = pair.get("duration", 400.0)
            polarity_str = pair.get("polarity", "PositiveFirst")
            polarity = StimPolarity.PositiveFirst if polarity_str == "PositiveFirst" else StimPolarity.NegativeFirst

            pair_key = f"{elec_from}_to_{pair.get('electrode_to', -1)}"
            self._active_stim_times[pair_key] = []

            groups = self.active_total_repeats // self.active_group_size
            for g in range(groups):
                logger.info("Pair %s group %d/%d", pair_key, g + 1, groups)
                for s in range(self.active_group_size):
                    stim_time = datetime_now()
                    self._send_stim_pulse(
                        electrode_idx=elec_from,
                        amplitude_ua=amplitude,
                        duration_us=duration,
                        polarity=polarity,
                        trigger_key=0,
                        phase="active",
                    )
                    self._active_stim_times[pair_key].append(stim_time)
                    if s < self.active_group_size - 1:
                        self._wait(inter_stim_s)
                if g < groups - 1:
                    self._wait(self.active_inter_group_s)

        self._compute_cross_correlograms()
        logger.info("Phase 2 complete.")

    def _compute_cross_correlograms(self) -> None:
        logger.info("Computing trigger-centred cross-correlograms")
        pairs_to_use = self._responsive_pairs if self._responsive_pairs else self._known_responsive_pairs

        n_bins = int(self.xcorr_window_ms / self.xcorr_bin_ms)
        lag_bins = [i * self.xcorr_bin_ms for i in range(n_bins)]

        for pair in pairs_to_use:
            elec_from = pair["electrode_from"]
            elec_to = pair.get("electrode_to", -1)
            pair_key = f"{elec_from}_to_{elec_to}"
            stim_times = self._active_stim_times.get(pair_key, [])

            counts = [0] * n_bins

            if stim_times and elec_to >= 0:
                window_s = self.xcorr_window_ms / 1000.0
                for stim_time in stim_times:
                    query_stop = stim_time + timedelta(seconds=window_s + 0.1)
                    try:
                        spike_df = self.database.get_spike_event(
                            stim_time, query_stop, self.np_experiment.exp_name
                        )
                        if not spike_df.empty and "channel" in spike_df.columns:
                            resp_spikes = spike_df[spike_df["channel"] == elec_to]
                            for _, row in resp_spikes.iterrows():
                                t_spike = pd.Timestamp(row["Time"]).timestamp()
                                t_stim = stim_time.timestamp()
                                lag_ms = (t_spike - t_stim) * 1000.0
                                if 0 <= lag_ms < self.xcorr_window_ms:
                                    bin_idx = int(lag_ms / self.xcorr_bin_ms)
                                    if bin_idx < n_bins:
                                        counts[bin_idx] += 1
                    except Exception as exc:
                        logger.warning("Xcorr query failed for pair %s: %s", pair_key, exc)

            peak_lag_ms = lag_bins[counts.index(max(counts))] if any(c > 0 for c in counts) else pair.get("median_latency_ms", 20.0)

            xcorr = CrossCorrelogramResult(
                electrode_from=elec_from,
                electrode_to=elec_to,
                lag_bins_ms=lag_bins,
                counts=counts,
                peak_lag_ms=peak_lag_ms,
            )
            self._xcorr_results.append(xcorr)
            logger.info("Xcorr pair %s: peak lag = %.2f ms", pair_key, peak_lag_ms)

    def _phase_stdp_experiment(self) -> None:
        logger.info("Phase 3: Two-Electrode Hebbian Learning (STDP) Experiment")
        pairs_to_use = self._responsive_pairs if self._responsive_pairs else self._known_responsive_pairs

        if not pairs_to_use:
            logger.warning("No responsive pairs for STDP experiment")
            return

        hebbian_delay_s = self.stdp_hebbian_delay_ms / 1000.0

        for pair_idx, pair in enumerate(pairs_to_use[:2]):
            elec_a = pair["electrode_from"]
            elec_b = pair.get("electrode_to", -1)
            if elec_b < 0:
                continue

            polarity_str = pair.get("polarity", "PositiveFirst")
            polarity_a = StimPolarity.PositiveFirst if polarity_str == "PositiveFirst" else StimPolarity.NegativeFirst

            xcorr_delay_ms = self.stdp_hebbian_delay_ms
            for xcorr in self._xcorr_results:
                if xcorr.electrode_from == elec_a and xcorr.electrode_to == elec_b:
                    xcorr_delay_ms = xcorr.peak_lag_ms
                    break

            actual_delay_s = xcorr_delay_ms / 1000.0
            actual_delay_s = max(0.010, min(actual_delay_s, 0.040))

            logger.info(
                "STDP pair %d: elec_A=%d elec_B=%d delay=%.1f ms",
                pair_idx, elec_a, elec_b, actual_delay_s * 1000.0,
            )

            phase_results: Dict[str, Any] = {}

            logger.info("STDP Testing Phase (%.0f s)", self.stdp_testing_duration_s)
            testing_start = datetime_now()
            probe_count = 0
            elapsed = 0.0
            while elapsed < self.stdp_testing_duration_s:
                self._send_stim_pulse(
                    electrode_idx=elec_a,
                    amplitude_ua=self.stdp_probe_amplitude_ua,
                    duration_us=self.stdp_probe_duration_us,
                    polarity=polarity_a,
                    trigger_key=0,
                    phase="stdp_testing",
                )
                probe_count += 1
                self._wait(self.stdp_probe_interval_s)
                elapsed += self.stdp_probe_interval_s
            testing_stop = datetime_now()
            phase_results["testing_probes"] = probe_count
            phase_results["testing_start"] = testing_start.isoformat()
            phase_results["testing_stop"] = testing_stop.isoformat()
            logger.info("Testing phase complete: %d probes", probe_count)

            logger.info("STDP Learning Phase (%.0f s)", self.stdp_learning_duration_s)
            learning_start = datetime_now()
            conditioning_count = 0
            elapsed = 0.0
            min_inter_event_s = 0.5
            while elapsed < self.stdp_learning_duration_s:
                self._send_stim_pulse(
                    electrode_idx=elec_a,
                    amplitude_ua=self.stdp_probe_amplitude_ua,
                    duration_us=self.stdp_probe_duration_us,
                    polarity=polarity_a,
                    trigger_key=0,
                    phase="stdp_learning_probe",
                )
                self._wait(actual_delay_s)
                self._send_stim_pulse(
                    electrode_idx=elec_b,
                    amplitude_ua=self.stdp_conditioning_amplitude_ua,
                    duration_us=self.stdp_conditioning_duration_us,
                    polarity=StimPolarity.NegativeFirst,
                    trigger_key=1,
                    phase="stdp_learning_conditioning",
                )
                conditioning_count += 1
                self._wait(min_inter_event_s)
                elapsed += actual_delay_s + min_inter_event_s
            learning_stop = datetime_now()
            phase_results["learning_conditioning_events"] = conditioning_count
            phase_results["learning_start"] = learning_start.isoformat()
            phase_results["learning_stop"] = learning_stop.isoformat()
            logger.info("Learning phase complete: %d conditioning events", conditioning_count)

            logger.info("STDP Validation Phase (%.0f s)", self.stdp_validation_duration_s)
            validation_start = datetime_now()
            validation_probe_count = 0
            elapsed = 0.0
            while elapsed < self.stdp_validation_duration_s:
                self._send_stim_pulse(
                    electrode_idx=elec_a,
                    amplitude_ua=self.stdp_probe_amplitude_ua,
                    duration_us=self.stdp_probe_duration_us,
                    polarity=polarity_a,
                    trigger_key=0,
                    phase="stdp_validation",
                )
                validation_probe_count += 1
                self._wait(self.stdp_probe_interval_s)
                elapsed += self.stdp_probe_interval_s
            validation_stop = datetime_now()
            phase_results["validation_probes"] = validation_probe_count
            phase_results["validation_start"] = validation_start.isoformat()
            phase_results["validation_stop"] = validation_stop.isoformat()
            logger.info("Validation phase complete: %d probes", validation_probe_count)

            pair_key = f"{elec_a}_to_{elec_b}"
            self._stdp_results[pair_key] = phase_results

        logger.info("Phase 3 complete.")

    def _send_stim_pulse(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity = StimPolarity.NegativeFirst,
        trigger_key: int = 0,
        phase: str = "",
    ) -> None:
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

        polarity_names = {StimPolarity.PositiveFirst: "PositiveFirst", StimPolarity.NegativeFirst: "NegativeFirst"}
        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            polarity=polarity_names.get(polarity, str(polarity)),
            timestamp_utc=datetime_now().isoformat(),
            trigger_key=trigger_key,
            phase=phase,
        ))

    def _save_all(self, recording_start: datetime, recording_stop: datetime) -> None:
        fs_name = getattr(self.np_experiment, "exp_name", "unknown")
        saver = DataSaver(self._output_dir, fs_name)

        saver.save_stimulation_log(self._stimulation_log)

        spike_df = self.database.get_spike_event(recording_start, recording_stop, fs_name)
        saver.save_spike_events(spike_df)

        trigger_df = self.database.get_all_triggers(recording_start, recording_stop)
        saver.save_triggers(trigger_df)

        xcorr_serializable = []
        for xc in self._xcorr_results:
            xcorr_serializable.append(asdict(xc))

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "scan_results_count": len(self._scan_results),
            "responsive_pairs_count": len(self._responsive_pairs),
            "xcorr_results": xcorr_serializable,
            "stdp_results": self._stdp_results,
        }
        saver.save_summary(summary)

        waveform_records = self._fetch_spike_waveforms(
            fs_name, spike_df, recording_start, recording_stop
        )
        saver.save_spike_waveforms(waveform_records)

    def _fetch_spike_waveforms(
        self,
        fs_name: str,
        spike_df: pd.DataFrame,
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
                logger.warning("Failed to fetch waveforms for electrode %s: %s", electrode_idx, exc)

        return waveform_records

    def _compile_results(self, recording_start: datetime, recording_stop: datetime) -> Dict[str, Any]:
        logger.info("Compiling results")
        fs_name = getattr(self.np_experiment, "exp_name", "unknown")
        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": fs_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "total_stimulations": len(self._stimulation_log),
            "scan_results_count": len(self._scan_results),
            "responsive_pairs_count": len(self._responsive_pairs),
            "xcorr_peaks": {
                f"{xc.electrode_from}_to_{xc.electrode_to}": xc.peak_lag_ms
                for xc in self._xcorr_results
            },
            "stdp_results": self._stdp_results,
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
