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
    """
    Full neuronal plasticity experiment pipeline:
      Stage 1 - Basic Excitability Scan
      Stage 2 - Active Electrode Experiment (1 Hz, cross-correlograms)
      Stage 3 - Two-Electrode Hebbian (STDP) Learning Experiment
    """

    KNOWN_PAIRS = [
        {"electrode_from": 17, "electrode_to": 18, "amplitude": 3.0, "duration": 400.0,
         "polarity": "PositiveFirst", "median_latency_ms": 13.477},
        {"electrode_from": 21, "electrode_to": 19, "amplitude": 3.0, "duration": 400.0,
         "polarity": "PositiveFirst", "median_latency_ms": 18.979},
        {"electrode_from": 21, "electrode_to": 22, "amplitude": 3.0, "duration": 400.0,
         "polarity": "NegativeFirst", "median_latency_ms": 10.859},
        {"electrode_from": 7, "electrode_to": 6, "amplitude": 3.0, "duration": 400.0,
         "polarity": "PositiveFirst", "median_latency_ms": 24.622},
        {"electrode_from": 6, "electrode_to": 7, "amplitude": 3.0, "duration": 400.0,
         "polarity": "PositiveFirst", "median_latency_ms": 19.294},
    ]

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        scan_amplitudes: tuple = (1.0, 2.0, 3.0),
        scan_durations: tuple = (100.0, 200.0, 300.0, 400.0),
        scan_repeats: int = 5,
        scan_required_hits: int = 3,
        inter_stim_s: float = 1.0,
        inter_channel_s: float = 5.0,
        active_total_repeats: int = 100,
        active_group_size: int = 10,
        active_group_pause_s: float = 5.0,
        stdp_testing_min: float = 20.0,
        stdp_learning_min: float = 50.0,
        stdp_validation_min: float = 20.0,
        stdp_delta_t_ms: float = 15.0,
        stdp_stim_amplitude: float = 3.0,
        stdp_stim_duration: float = 200.0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.scan_amplitudes = list(scan_amplitudes)
        self.scan_durations = list(scan_durations)
        self.scan_repeats = scan_repeats
        self.scan_required_hits = scan_required_hits
        self.inter_stim_s = inter_stim_s
        self.inter_channel_s = inter_channel_s

        self.active_total_repeats = active_total_repeats
        self.active_group_size = active_group_size
        self.active_group_pause_s = active_group_pause_s

        self.stdp_testing_min = stdp_testing_min
        self.stdp_learning_min = stdp_learning_min
        self.stdp_validation_min = stdp_validation_min
        self.stdp_delta_t_ms = stdp_delta_t_ms
        self.stdp_stim_amplitude = min(stdp_stim_amplitude, 4.0)
        self.stdp_stim_duration = min(stdp_stim_duration, 400.0)

        self.np_experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._scan_results: List[ScanResult] = []
        self._responsive_pairs: List[Dict[str, Any]] = []
        self._active_stim_times: Dict[str, List[str]] = defaultdict(list)
        self._correlograms: Dict[str, Any] = {}
        self._stdp_results: Dict[str, Any] = {}

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

            logger.info("=== Stage 1: Basic Excitability Scan ===")
            self._phase_excitability_scan()

            logger.info("=== Stage 2: Active Electrode Experiment ===")
            self._phase_active_electrode()

            logger.info("=== Stage 3: Hebbian STDP Experiment ===")
            self._phase_stdp()

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
        logger.info("Phase: excitability scan")
        available_electrodes = list(self.np_experiment.electrodes)
        polarities = [StimPolarity.PositiveFirst, StimPolarity.NegativeFirst]
        polarity_names = {StimPolarity.PositiveFirst: "PositiveFirst",
                          StimPolarity.NegativeFirst: "NegativeFirst"}

        for electrode_idx in available_electrodes:
            for amplitude in self.scan_amplitudes:
                for duration in self.scan_durations:
                    for polarity in polarities:
                        hits = 0
                        latencies = []
                        for rep in range(self.scan_repeats):
                            t_before = datetime_now()
                            self._send_stim_pulse(
                                electrode_idx=electrode_idx,
                                amplitude_ua=amplitude,
                                duration_us=duration,
                                polarity=polarity,
                                trigger_key=0,
                                phase="scan",
                            )
                            self._wait(0.05)
                            t_after = datetime_now()
                            window_start = t_before
                            window_stop = t_after + timedelta(seconds=0.1)
                            try:
                                spike_df = self.database.get_spike_event(
                                    window_start, window_stop,
                                    self.np_experiment.exp_name
                                )
                                if not spike_df.empty:
                                    stim_ts = t_before.timestamp()
                                    for _, row in spike_df.iterrows():
                                        try:
                                            spike_ts = row["Time"].timestamp()
                                        except Exception:
                                            spike_ts = stim_ts
                                        lat_ms = (spike_ts - stim_ts) * 1000.0
                                        if 1.0 < lat_ms < 50.0:
                                            hits += 1
                                            latencies.append(lat_ms)
                                            break
                            except Exception as exc:
                                logger.warning("Scan spike query error: %s", exc)
                            if rep < self.scan_repeats - 1:
                                self._wait(self.inter_stim_s)

                        median_lat = float(np.median(latencies)) if latencies else 0.0
                        if hits >= self.scan_required_hits:
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
                                electrode_idx, amplitude, duration,
                                polarity_names[polarity], hits
                            )

            self._wait(self.inter_channel_s)

        self._build_responsive_pairs()

    def _build_responsive_pairs(self) -> None:
        for pair in self.KNOWN_PAIRS:
            self._responsive_pairs.append(dict(pair))
        logger.info("Using %d known responsive pairs for stages 2 & 3",
                    len(self._responsive_pairs))

    def _phase_active_electrode(self) -> None:
        logger.info("Phase: active electrode experiment (1 Hz, %d repeats)",
                    self.active_total_repeats)

        pairs_to_use = self._responsive_pairs[:4] if len(self._responsive_pairs) >= 4 \
            else self._responsive_pairs

        for pair in pairs_to_use:
            stim_elec = pair["electrode_from"]
            resp_elec = pair["electrode_to"]
            amplitude = min(pair["amplitude"], 4.0)
            duration = min(pair["duration"], 400.0)
            polarity = (StimPolarity.PositiveFirst
                        if pair["polarity"] == "PositiveFirst"
                        else StimPolarity.NegativeFirst)
            pair_key = f"{stim_elec}->{resp_elec}"
            stim_times_this_pair: List[float] = []
            spike_times_this_pair: List[float] = []

            logger.info("Active electrode: pair %s, %d repeats", pair_key,
                        self.active_total_repeats)

            groups = self.active_total_repeats // self.active_group_size
            for g in range(groups):
                for s in range(self.active_group_size):
                    t_stim = datetime_now()
                    self._send_stim_pulse(
                        electrode_idx=stim_elec,
                        amplitude_ua=amplitude,
                        duration_us=duration,
                        polarity=polarity,
                        trigger_key=1,
                        phase="active",
                    )
                    stim_times_this_pair.append(t_stim.timestamp())
                    self._active_stim_times[pair_key].append(t_stim.isoformat())
                    self._wait(0.05)
                    t_after = datetime_now()
                    try:
                        spike_df = self.database.get_spike_event(
                            t_stim,
                            t_after + timedelta(seconds=0.1),
                            self.np_experiment.exp_name
                        )
                        if not spike_df.empty:
                            for _, row in spike_df.iterrows():
                                try:
                                    spike_times_this_pair.append(row["Time"].timestamp())
                                except Exception:
                                    pass
                    except Exception as exc:
                        logger.warning("Active spike query error: %s", exc)
                    if s < self.active_group_size - 1:
                        self._wait(1.0 - 0.05)
                    else:
                        self._wait(0.95)

                if g < groups - 1:
                    self._wait(self.active_group_pause_s)

            ccg = self._compute_ccg(stim_times_this_pair, spike_times_this_pair,
                                    window_ms=50.0, bin_ms=1.0)
            self._correlograms[pair_key] = ccg
            logger.info("CCG computed for pair %s: %d bins", pair_key, len(ccg["bins"]))

    def _compute_ccg(
        self,
        stim_times: List[float],
        spike_times: List[float],
        window_ms: float = 50.0,
        bin_ms: float = 1.0,
    ) -> Dict[str, Any]:
        window_s = window_ms / 1000.0
        bin_s = bin_ms / 1000.0
        n_bins = int(window_ms / bin_ms)
        counts = [0] * n_bins
        bin_edges = [i * bin_ms for i in range(n_bins + 1)]

        for st in stim_times:
            for sp in spike_times:
                delta = sp - st
                if 0.0 <= delta < window_s:
                    bin_idx = int(delta / bin_s)
                    if bin_idx < n_bins:
                        counts[bin_idx] += 1

        peak_bin = int(np.argmax(counts)) if counts else 0
        peak_latency_ms = (peak_bin + 0.5) * bin_ms

        return {
            "bins": bin_edges,
            "counts": counts,
            "peak_latency_ms": peak_latency_ms,
            "n_stim": len(stim_times),
            "n_spikes": len(spike_times),
        }

    def _phase_stdp(self) -> None:
        logger.info("Phase: STDP Hebbian learning experiment")

        if not self._responsive_pairs:
            logger.warning("No responsive pairs for STDP phase")
            return

        pair = self._responsive_pairs[0]
        stim_elec = pair["electrode_from"]
        resp_elec = pair["electrode_to"]
        amplitude = min(self.stdp_stim_amplitude, 4.0)
        duration = min(self.stdp_stim_duration, 400.0)
        polarity = (StimPolarity.PositiveFirst
                    if pair["polarity"] == "PositiveFirst"
                    else StimPolarity.NegativeFirst)
        pair_key = f"{stim_elec}->{resp_elec}"
        delta_t_s = self.stdp_delta_t_ms / 1000.0

        logger.info("STDP pair: %s, delta_t=%.1f ms", pair_key, self.stdp_delta_t_ms)

        testing_spikes = self._stdp_probe_phase(
            stim_elec, resp_elec, amplitude, duration, polarity,
            duration_min=self.stdp_testing_min, phase_name="testing"
        )

        self._stdp_learning_phase(
            stim_elec, resp_elec, amplitude, duration, polarity,
            delta_t_s=delta_t_s,
            duration_min=self.stdp_learning_min,
        )

        validation_spikes = self._stdp_probe_phase(
            stim_elec, resp_elec, amplitude, duration, polarity,
            duration_min=self.stdp_validation_min, phase_name="validation"
        )

        pre_rate = testing_spikes / max(self.stdp_testing_min, 1.0)
        post_rate = validation_spikes / max(self.stdp_validation_min, 1.0)
        delta_stp = post_rate - pre_rate

        self._stdp_results = {
            "pair": pair_key,
            "stim_electrode": stim_elec,
            "resp_electrode": resp_elec,
            "delta_t_ms": self.stdp_delta_t_ms,
            "testing_spike_count": testing_spikes,
            "validation_spike_count": validation_spikes,
            "pre_rate_spikes_per_min": pre_rate,
            "post_rate_spikes_per_min": post_rate,
            "delta_stp": delta_stp,
        }
        logger.info("STDP result: delta_STP=%.4f (pre=%.2f, post=%.2f spk/min)",
                    delta_stp, pre_rate, post_rate)

    def _stdp_probe_phase(
        self,
        stim_elec: int,
        resp_elec: int,
        amplitude: float,
        duration: float,
        polarity: StimPolarity,
        duration_min: float,
        phase_name: str,
    ) -> int:
        logger.info("STDP %s phase: %.0f min", phase_name, duration_min)
        phase_end = datetime_now() + timedelta(minutes=duration_min)
        spike_count = 0
        stim_interval_s = 2.0

        while datetime_now() < phase_end:
            t_stim = datetime_now()
            self._send_stim_pulse(
                electrode_idx=stim_elec,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=polarity,
                trigger_key=2,
                phase=phase_name,
            )
            self._wait(0.05)
            t_after = datetime_now()
            try:
                spike_df = self.database.get_spike_event(
                    t_stim,
                    t_after + timedelta(seconds=0.1),
                    self.np_experiment.exp_name
                )
                if not spike_df.empty:
                    spike_count += len(spike_df)
            except Exception as exc:
                logger.warning("STDP probe spike query error: %s", exc)
            self._wait(stim_interval_s - 0.05)

        logger.info("STDP %s phase complete: %d spikes detected", phase_name, spike_count)
        return spike_count

    def _stdp_learning_phase(
        self,
        stim_elec: int,
        resp_elec: int,
        amplitude: float,
        duration: float,
        polarity: StimPolarity,
        delta_t_s: float,
        duration_min: float,
    ) -> None:
        logger.info("STDP learning phase: %.0f min, delta_t=%.1f ms",
                    duration_min, delta_t_s * 1000.0)
        phase_end = datetime_now() + timedelta(minutes=duration_min)
        pair_interval_s = 2.0

        while datetime_now() < phase_end:
            self._send_stim_pulse(
                electrode_idx=stim_elec,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=polarity,
                trigger_key=3,
                phase="learning_pre",
            )
            self._wait(delta_t_s)
            self._send_stim_pulse(
                electrode_idx=resp_elec,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=polarity,
                trigger_key=4,
                phase="learning_post",
            )
            self._wait(pair_interval_s - delta_t_s)

        logger.info("STDP learning phase complete")

    def _send_stim_pulse(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity,
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
        stim.interphase_delay = 0.0

        self.intan.send_stimparam([stim])

        pattern = np.zeros(16, dtype=np.uint8)
        pattern[trigger_key] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.01)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

        polarity_name = ("PositiveFirst" if polarity == StimPolarity.PositiveFirst
                         else "NegativeFirst")
        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            polarity=polarity_name,
            timestamp_utc=datetime_now().isoformat(),
            trigger_key=trigger_key,
            phase=phase,
        ))

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
            "scan_results_count": len(self._scan_results),
            "responsive_pairs_count": len(self._responsive_pairs),
            "correlograms": {k: {"peak_latency_ms": v["peak_latency_ms"],
                                  "n_stim": v["n_stim"],
                                  "n_spikes": v["n_spikes"]}
                             for k, v in self._correlograms.items()},
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
        duration_s = (recording_stop - recording_start).total_seconds()

        scan_summary = []
        for r in self._scan_results:
            scan_summary.append({
                "electrode_from": r.electrode_from,
                "amplitude": r.amplitude,
                "duration": r.duration,
                "polarity": r.polarity,
                "hits": r.hits,
                "repeats": r.repeats,
                "median_latency_ms": r.median_latency_ms,
            })

        ccg_summary = {}
        for k, v in self._correlograms.items():
            ccg_summary[k] = {
                "peak_latency_ms": v["peak_latency_ms"],
                "n_stim": v["n_stim"],
                "n_spikes": v["n_spikes"],
            }

        return {
            "status": "completed",
            "experiment_name": self.np_experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": duration_s,
            "total_stimulations": len(self._stimulation_log),
            "scan_responsive_electrodes": len(self._scan_results),
            "responsive_pairs": len(self._responsive_pairs),
            "correlograms": ccg_summary,
            "stdp_results": self._stdp_results,
        }

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
