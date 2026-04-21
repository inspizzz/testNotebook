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
    phase: str
    timestamp_utc: str
    trigger_key: int = 0
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
class PairConfig:
    electrode_from: int
    electrode_to: int
    amplitude: float
    duration: float
    polarity: StimPolarity
    median_latency_ms: float
    hebbian_delay_ms: float = 15.0


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

    def save_summary(self, summary: dict) -> Path:
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
    Three-stage neuronal plasticity experiment:
    1. Basic Excitability Scan
    2. Active Electrode Experiment (1 Hz bursts + cross-correlograms)
    3. Two-Electrode Hebbian STDP Experiment (Testing / Learning / Validation)
    """

    SCAN_AMPLITUDES = (1.0, 2.0, 3.0)
    SCAN_DURATIONS = (100.0, 200.0, 300.0, 400.0)
    SCAN_POLARITIES = ("NegativeFirst", "PositiveFirst")

    PRIOR_RELIABLE_CONNECTIONS = [
        {"electrode_from": 0, "electrode_to": 1, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 12.73, "stimulation": {"amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 1, "electrode_to": 2, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 23.34, "stimulation": {"amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 5, "electrode_to": 4, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 17.39, "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 5, "electrode_to": 6, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 15.45, "stimulation": {"amplitude": 1.0, "duration": 400.0, "polarity": "PositiveFirst"}},
        {"electrode_from": 6, "electrode_to": 5, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 14.82, "stimulation": {"amplitude": 2.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 14, "electrode_to": 12, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 22.91, "stimulation": {"amplitude": 1.0, "duration": 400.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 14, "electrode_to": 15, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 13.2, "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 17, "electrode_to": 16, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 21.56, "stimulation": {"amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 18, "electrode_to": 17, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 24.71, "stimulation": {"amplitude": 1.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 9, "electrode_to": 10, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 10.97, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 9, "electrode_to": 11, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 16.17, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 26, "electrode_to": 27, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 13.88, "stimulation": {"amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 30, "electrode_to": 31, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 19.34, "stimulation": {"amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst"}},
        {"electrode_from": 31, "electrode_to": 30, "hits_k": 5, "repeats_n": 5, "median_latency_ms": 18.87, "stimulation": {"amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst"}},
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
        active_stim_interval_s: float = 1.0,
        stdp_testing_duration_s: float = 1200.0,
        stdp_learning_duration_s: float = 3000.0,
        stdp_validation_duration_s: float = 1200.0,
        stdp_probe_interval_s: float = 10.0,
        stdp_hebbian_delay_ms: float = 15.0,
        stdp_probe_amplitude_ua: float = 1.0,
        stdp_probe_duration_us: float = 300.0,
        max_pairs_for_stdp: int = 3,
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
        self.active_stim_interval_s = active_stim_interval_s

        self.stdp_testing_duration_s = stdp_testing_duration_s
        self.stdp_learning_duration_s = stdp_learning_duration_s
        self.stdp_validation_duration_s = stdp_validation_duration_s
        self.stdp_probe_interval_s = stdp_probe_interval_s
        self.stdp_hebbian_delay_ms = stdp_hebbian_delay_ms
        self.stdp_probe_amplitude_ua = stdp_probe_amplitude_ua
        self.stdp_probe_duration_us = stdp_probe_duration_us
        self.max_pairs_for_stdp = max_pairs_for_stdp

        self.np_experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._scan_results: List[ScanResult] = []
        self._responsive_pairs: List[PairConfig] = []
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

            self._phase1_excitability_scan()
            self._phase2_active_electrode_experiment()
            self._phase3_stdp_experiment()

            recording_stop = datetime_now()

            self._save_all(recording_start, recording_stop)

            results = self._compile_results(recording_start, recording_stop)
            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _phase1_excitability_scan(self) -> None:
        logger.info("=== Phase 1: Basic Excitability Scan ===")
        electrodes = self.np_experiment.electrodes
        polarity_map = {
            "NegativeFirst": StimPolarity.NegativeFirst,
            "PositiveFirst": StimPolarity.PositiveFirst,
        }

        hits_per_pair: Dict[Tuple, List[float]] = defaultdict(list)

        for elec_idx, electrode in enumerate(electrodes):
            logger.info("Scanning electrode %d (%d/%d)", electrode, elec_idx + 1, len(electrodes))
            for amplitude in self.scan_amplitudes:
                for duration in self.scan_durations:
                    for polarity_str in ["NegativeFirst", "PositiveFirst"]:
                        polarity = polarity_map[polarity_str]
                        for rep in range(self.scan_repeats):
                            t_stim = datetime_now()
                            self._send_stim_pulse(
                                electrode_idx=electrode,
                                amplitude_ua=amplitude,
                                duration_us=duration,
                                polarity=polarity,
                                trigger_key=0,
                                phase="scan",
                            )
                            self._wait(0.05)
                            t_after = datetime_now()
                            window_start = t_stim
                            window_stop = t_after + timedelta(milliseconds=50)
                            self._wait(0.05)
                            spike_df = self.database.get_spike_event(
                                window_start,
                                window_stop,
                                self.np_experiment.exp_name,
                            )
                            responded_electrodes = set()
                            if not spike_df.empty:
                                ch_col = self._get_channel_col(spike_df)
                                if ch_col:
                                    for other_elec in electrodes:
                                        if other_elec == electrode:
                                            continue
                                        elec_spikes = spike_df[spike_df[ch_col] == other_elec]
                                        if not elec_spikes.empty:
                                            responded_electrodes.add(other_elec)
                            key = (electrode, amplitude, duration, polarity_str)
                            hits_per_pair[key].append(len(responded_electrodes))
                            self._wait(self.inter_stim_s)

            self._wait(self.inter_channel_s)

        for (elec_from, amplitude, duration, polarity_str), resp_list in hits_per_pair.items():
            total_hits = sum(1 for r in resp_list if r > 0)
            if total_hits >= self.scan_required_hits:
                self._scan_results.append(ScanResult(
                    electrode_from=elec_from,
                    electrode_to=-1,
                    amplitude=amplitude,
                    duration=duration,
                    polarity=polarity_str,
                    hits=total_hits,
                    repeats=self.scan_repeats,
                    median_latency_ms=0.0,
                ))

        self._build_responsive_pairs_from_prior()
        logger.info("Phase 1 complete. Responsive pairs: %d", len(self._responsive_pairs))

    def _build_responsive_pairs_from_prior(self) -> None:
        polarity_map = {
            "NegativeFirst": StimPolarity.NegativeFirst,
            "PositiveFirst": StimPolarity.PositiveFirst,
        }
        seen = set()
        for conn in self.PRIOR_RELIABLE_CONNECTIONS:
            key = (conn["electrode_from"], conn["electrode_to"])
            if key in seen:
                continue
            seen.add(key)
            stim = conn["stimulation"]
            amplitude = stim["amplitude"]
            duration = stim["duration"]
            polarity_str = stim["polarity"]
            a1 = amplitude
            d1 = duration
            a2 = amplitude
            d2 = duration
            if abs(a1 * d1 - a2 * d2) > 1e-6:
                continue
            if amplitude > 4.0 or duration > 400.0:
                continue
            polarity = polarity_map.get(polarity_str, StimPolarity.NegativeFirst)
            latency = conn["median_latency_ms"]
            hebbian_delay = max(10.0, min(25.0, latency))
            self._responsive_pairs.append(PairConfig(
                electrode_from=conn["electrode_from"],
                electrode_to=conn["electrode_to"],
                amplitude=amplitude,
                duration=duration,
                polarity=polarity,
                median_latency_ms=latency,
                hebbian_delay_ms=hebbian_delay,
            ))

    def _phase2_active_electrode_experiment(self) -> None:
        logger.info("=== Phase 2: Active Electrode Experiment ===")
        if not self._responsive_pairs:
            logger.warning("No responsive pairs found; skipping Phase 2")
            return

        for pair in self._responsive_pairs:
            pair_key = f"{pair.electrode_from}->{pair.electrode_to}"
            logger.info("Active stim for pair %s", pair_key)
            stim_times = []
            groups = self.active_total_repeats // self.active_group_size

            for grp in range(groups):
                for rep in range(self.active_group_size):
                    t_stim = datetime_now()
                    self._send_stim_pulse(
                        electrode_idx=pair.electrode_from,
                        amplitude_ua=pair.amplitude,
                        duration_us=pair.duration,
                        polarity=pair.polarity,
                        trigger_key=0,
                        phase="active",
                    )
                    stim_times.append(t_stim.isoformat())
                    self._wait(self.active_stim_interval_s)

                if grp < groups - 1:
                    self._wait(self.active_group_pause_s)

            self._active_stim_times[pair_key] = stim_times

        self._compute_correlograms()
        logger.info("Phase 2 complete.")

    def _compute_correlograms(self) -> None:
        logger.info("Computing trigger-centred cross-correlograms")
        bin_width_ms = 4.0
        window_ms = 100.0
        n_bins = int(2 * window_ms / bin_width_ms)
        bins = np.linspace(-window_ms, window_ms, n_bins + 1)

        for pair in self._responsive_pairs:
            pair_key = f"{pair.electrode_from}->{pair.electrode_to}"
            stim_times_iso = self._active_stim_times.get(pair_key, [])
            if not stim_times_iso:
                continue

            stim_times_dt = []
            for ts in stim_times_iso:
                try:
                    stim_times_dt.append(datetime.fromisoformat(ts))
                except Exception:
                    pass

            if not stim_times_dt:
                continue

            exp_start = stim_times_dt[0] - timedelta(seconds=1)
            exp_stop = stim_times_dt[-1] + timedelta(seconds=2)

            try:
                spike_df = self.database.get_spike_event(
                    exp_start, exp_stop, self.np_experiment.exp_name
                )
            except Exception as exc:
                logger.warning("Failed to fetch spikes for CCG %s: %s", pair_key, exc)
                continue

            if spike_df.empty:
                self._correlograms[pair_key] = {"bins": bins.tolist(), "counts": [0] * n_bins}
                continue

            ch_col = self._get_channel_col(spike_df)
            if ch_col is None:
                continue

            resp_spikes = spike_df[spike_df[ch_col] == pair.electrode_to]
            if resp_spikes.empty:
                self._correlograms[pair_key] = {"bins": bins.tolist(), "counts": [0] * n_bins}
                continue

            time_col = self._get_time_col(resp_spikes)
            if time_col is None:
                continue

            spike_times_ms = []
            for _, row in resp_spikes.iterrows():
                t = row[time_col]
                if hasattr(t, "timestamp"):
                    spike_times_ms.append(t.timestamp() * 1000.0)
                else:
                    try:
                        spike_times_ms.append(float(t) * 1000.0)
                    except Exception:
                        pass

            ccg_counts = np.zeros(n_bins, dtype=int)
            for stim_dt in stim_times_dt:
                stim_ms = stim_dt.timestamp() * 1000.0
                for sp_ms in spike_times_ms:
                    delta = sp_ms - stim_ms
                    if -window_ms <= delta <= window_ms:
                        bin_idx = int((delta + window_ms) / bin_width_ms)
                        if 0 <= bin_idx < n_bins:
                            ccg_counts[bin_idx] += 1

            self._correlograms[pair_key] = {
                "bins": bins.tolist(),
                "counts": ccg_counts.tolist(),
                "stim_electrode": pair.electrode_from,
                "resp_electrode": pair.electrode_to,
                "n_stims": len(stim_times_dt),
            }

        logger.info("Correlograms computed for %d pairs", len(self._correlograms))

    def _phase3_stdp_experiment(self) -> None:
        logger.info("=== Phase 3: STDP Hebbian Learning Experiment ===")
        if not self._responsive_pairs:
            logger.warning("No responsive pairs; skipping Phase 3")
            return

        stdp_pairs = self._responsive_pairs[:self.max_pairs_for_stdp]

        for pair in stdp_pairs:
            pair_key = f"{pair.electrode_from}->{pair.electrode_to}"
            logger.info("STDP experiment for pair %s", pair_key)
            self._stdp_results[pair_key] = {}

            testing_spikes = self._stdp_probe_phase(
                pair, "testing", self.stdp_testing_duration_s
            )
            self._stdp_results[pair_key]["testing_probe_count"] = len(testing_spikes)
            self._stdp_results[pair_key]["testing_spikes"] = testing_spikes

            self._stdp_learning_phase(pair)

            validation_spikes = self._stdp_probe_phase(
                pair, "validation", self.stdp_validation_duration_s
            )
            self._stdp_results[pair_key]["validation_probe_count"] = len(validation_spikes)
            self._stdp_results[pair_key]["validation_spikes"] = validation_spikes

            pre_count = self._stdp_results[pair_key]["testing_probe_count"]
            post_count = self._stdp_results[pair_key]["validation_probe_count"]
            if pre_count > 0:
                change = (post_count - pre_count) / float(pre_count)
            else:
                change = 0.0
            self._stdp_results[pair_key]["response_change_fraction"] = change
            logger.info(
                "Pair %s: pre=%d post=%d change=%.3f",
                pair_key, pre_count, post_count, change
            )

        logger.info("Phase 3 complete.")

    def _stdp_probe_phase(
        self, pair: PairConfig, phase_name: str, duration_s: float
    ) -> List[Dict[str, Any]]:
        logger.info("STDP %s phase for pair %d->%d (%.0f s)",
                    phase_name, pair.electrode_from, pair.electrode_to, duration_s)
        probe_results = []
        phase_start = datetime_now()
        phase_end_target = phase_start + timedelta(seconds=duration_s)

        while True:
            now = datetime_now()
            if now >= phase_end_target:
                break

            t_stim = datetime_now()
            self._send_stim_pulse(
                electrode_idx=pair.electrode_from,
                amplitude_ua=self.stdp_probe_amplitude_ua,
                duration_us=self.stdp_probe_duration_us,
                polarity=pair.polarity,
                trigger_key=0,
                phase=f"stdp_{phase_name}_probe",
            )
            self._wait(0.05)
            t_after = datetime_now()
            window_start = t_stim
            window_stop = t_after + timedelta(milliseconds=80)
            self._wait(0.08)

            try:
                spike_df = self.database.get_spike_event(
                    window_start, window_stop, self.np_experiment.exp_name
                )
                ch_col = self._get_channel_col(spike_df)
                responded = False
                latency_ms = None
                if not spike_df.empty and ch_col:
                    resp_spikes = spike_df[spike_df[ch_col] == pair.electrode_to]
                    if not resp_spikes.empty:
                        responded = True
                        time_col = self._get_time_col(resp_spikes)
                        if time_col:
                            first_spike = resp_spikes.iloc[0][time_col]
                            if hasattr(first_spike, "timestamp"):
                                latency_ms = (first_spike.timestamp() - t_stim.timestamp()) * 1000.0
                probe_results.append({
                    "stim_time": t_stim.isoformat(),
                    "responded": responded,
                    "latency_ms": latency_ms,
                })
            except Exception as exc:
                logger.warning("Probe query failed: %s", exc)

            elapsed = (datetime_now() - phase_start).total_seconds()
            remaining = duration_s - elapsed
            if remaining <= 0:
                break
            sleep_time = min(self.stdp_probe_interval_s, remaining)
            self._wait(sleep_time)

        return probe_results

    def _stdp_learning_phase(self, pair: PairConfig) -> None:
        logger.info(
            "STDP learning phase for pair %d->%d (%.0f s, Hebbian delay=%.1f ms)",
            pair.electrode_from, pair.electrode_to,
            self.stdp_learning_duration_s, pair.hebbian_delay_ms
        )
        phase_start = datetime_now()
        phase_end_target = phase_start + timedelta(seconds=self.stdp_learning_duration_s)
        hebbian_delay_s = pair.hebbian_delay_ms / 1000.0
        inter_pair_s = 1.0

        while True:
            now = datetime_now()
            if now >= phase_end_target:
                break

            self._send_stim_pulse(
                electrode_idx=pair.electrode_from,
                amplitude_ua=pair.amplitude,
                duration_us=pair.duration,
                polarity=pair.polarity,
                trigger_key=0,
                phase="stdp_learning_pre",
            )
            self._wait(hebbian_delay_s)

            self._send_stim_pulse(
                electrode_idx=pair.electrode_to,
                amplitude_ua=pair.amplitude,
                duration_us=pair.duration,
                polarity=pair.polarity,
                trigger_key=1,
                phase="stdp_learning_post",
            )

            elapsed = (datetime_now() - phase_start).total_seconds()
            remaining = self.stdp_learning_duration_s - elapsed
            if remaining <= 0:
                break
            sleep_time = min(inter_pair_s, remaining)
            self._wait(sleep_time)

    def _send_stim_pulse(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity,
        trigger_key: int = 0,
        phase: str = "unknown",
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
        self._wait(0.02)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

        polarity_str = "NegativeFirst" if polarity == StimPolarity.NegativeFirst else "PositiveFirst"
        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            polarity=polarity_str,
            phase=phase,
            timestamp_utc=datetime_now().isoformat(),
            trigger_key=trigger_key,
        ))

    def _get_channel_col(self, df: pd.DataFrame) -> Optional[str]:
        if df is None or df.empty:
            return None
        for col in ["channel", "index", "electrode", "Channel"]:
            if col in df.columns:
                return col
        return None

    def _get_time_col(self, df: pd.DataFrame) -> Optional[str]:
        if df is None or df.empty:
            return None
        for col in ["Time", "time", "_time", "timestamp"]:
            if col in df.columns:
                return col
        return None

    def _save_all(self, recording_start: datetime, recording_stop: datetime) -> None:
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
            "correlograms_computed": len(self._correlograms),
            "stdp_pairs_count": len(self._stdp_results),
            "stdp_results": {
                k: {
                    "testing_probe_count": v.get("testing_probe_count", 0),
                    "validation_probe_count": v.get("validation_probe_count", 0),
                    "response_change_fraction": v.get("response_change_fraction", 0.0),
                }
                for k, v in self._stdp_results.items()
            },
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
        if spike_df is None or spike_df.empty:
            return waveform_records

        ch_col = self._get_channel_col(spike_df)
        if ch_col is None:
            logger.warning("Cannot determine electrode column for waveform fetch")
            return waveform_records

        unique_electrodes = spike_df[ch_col].unique()
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
        stdp_summary = {}
        for k, v in self._stdp_results.items():
            stdp_summary[k] = {
                "testing_probe_count": v.get("testing_probe_count", 0),
                "validation_probe_count": v.get("validation_probe_count", 0),
                "response_change_fraction": v.get("response_change_fraction", 0.0),
            }

        return {
            "status": "completed",
            "experiment_name": self.np_experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "scan_results_count": len(self._scan_results),
            "responsive_pairs_count": len(self._responsive_pairs),
            "responsive_pairs": [
                {
                    "electrode_from": p.electrode_from,
                    "electrode_to": p.electrode_to,
                    "amplitude": p.amplitude,
                    "duration": p.duration,
                    "median_latency_ms": p.median_latency_ms,
                    "hebbian_delay_ms": p.hebbian_delay_ms,
                }
                for p in self._responsive_pairs
            ],
            "correlograms": self._correlograms,
            "stdp_results": stdp_summary,
            "total_stimulations": len(self._stimulation_log),
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
