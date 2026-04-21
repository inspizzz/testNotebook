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
    phase: str
    stim_electrode: int
    resp_electrode: int
    trial_index: int
    stim_time_utc: str
    spike_count: int
    responded: bool
    latency_ms: float = 0.0


class DataSaver:
    def __init__(self, output_dir: Path, fs_name: str) -> None:
        self._dir = output_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime_now().strftime("%Y%m%dT%H%M%SZ")
        self._prefix = self._dir / f"{fs_name}_{timestamp}"

    def save_stimulation_log(self, stimulations: list) -> Path:
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
    Full neuronal plasticity experiment pipeline:
    Stage 1 - Basic Excitability Scan
    Stage 2 - Active Electrode Experiment (1 Hz stimulation + cross-correlograms)
    Stage 3 - Two-Electrode Hebbian/STDP Learning Experiment
    """

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
        scan_response_window_ms: float = 50.0,
        scan_required_hits: int = 3,
        active_total_repeats: int = 100,
        active_group_size: int = 10,
        active_group_pause_s: float = 5.0,
        active_isi_s: float = 1.0,
        ccg_window_ms: float = 100.0,
        ccg_bin_ms: float = 1.0,
        stdp_testing_duration_s: float = 1200.0,
        stdp_learning_duration_s: float = 3000.0,
        stdp_validation_duration_s: float = 1200.0,
        stdp_delta_t_ms: float = 10.0,
        stdp_conditioning_isi_s: float = 10.0,
        stdp_amplitude_ua: float = 3.0,
        stdp_duration_us: float = 400.0,
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
        self.scan_response_window_ms = scan_response_window_ms
        self.scan_required_hits = scan_required_hits

        self.active_total_repeats = active_total_repeats
        self.active_group_size = active_group_size
        self.active_group_pause_s = active_group_pause_s
        self.active_isi_s = active_isi_s
        self.ccg_window_ms = ccg_window_ms
        self.ccg_bin_ms = ccg_bin_ms

        self.stdp_testing_duration_s = stdp_testing_duration_s
        self.stdp_learning_duration_s = stdp_learning_duration_s
        self.stdp_validation_duration_s = stdp_validation_duration_s
        self.stdp_delta_t_ms = stdp_delta_t_ms
        self.stdp_conditioning_isi_s = stdp_conditioning_isi_s
        self.stdp_amplitude_ua = min(stdp_amplitude_ua, 4.0)
        self.stdp_duration_us = min(stdp_duration_us, 400.0)

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._trial_results: List[TrialResult] = []

        self._scan_responsive_pairs: List[Dict[str, Any]] = []
        self._active_stim_times: Dict[str, List[str]] = {}
        self._ccg_results: Dict[str, Any] = {}
        self._stdp_results: Dict[str, Any] = {}

        self._known_pairs = [
            {"stim": 7, "resp": 6, "amplitude": 3.0, "duration": 400.0, "polarity": StimPolarity.PositiveFirst, "median_latency_ms": 24.622},
            {"stim": 6, "resp": 7, "amplitude": 3.0, "duration": 400.0, "polarity": StimPolarity.PositiveFirst, "median_latency_ms": 19.294},
            {"stim": 17, "resp": 18, "amplitude": 3.0, "duration": 400.0, "polarity": StimPolarity.PositiveFirst, "median_latency_ms": 13.477},
            {"stim": 21, "resp": 19, "amplitude": 3.0, "duration": 400.0, "polarity": StimPolarity.PositiveFirst, "median_latency_ms": 18.979},
            {"stim": 21, "resp": 22, "amplitude": 3.0, "duration": 400.0, "polarity": StimPolarity.NegativeFirst, "median_latency_ms": 10.859},
            {"stim": 5, "resp": 4, "amplitude": 3.0, "duration": 400.0, "polarity": StimPolarity.PositiveFirst, "median_latency_ms": 14.634},
            {"stim": 13, "resp": 14, "amplitude": 3.0, "duration": 300.0, "polarity": StimPolarity.NegativeFirst, "median_latency_ms": 12.055},
        ]

    def _wait(self, seconds: float) -> None:
        wait(seconds)

    def run(self) -> Dict[str, Any]:
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

            recording_start = datetime_now()

            logger.info("=== Stage 1: Basic Excitability Scan ===")
            self._stage1_excitability_scan()

            logger.info("=== Stage 2: Active Electrode Experiment ===")
            self._stage2_active_electrode_experiment()

            logger.info("=== Stage 3: Hebbian STDP Learning Experiment ===")
            self._stage3_stdp_experiment()

            recording_stop = datetime_now()

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _stage1_excitability_scan(self) -> None:
        logger.info("Stage 1: Sweeping electrodes for excitability")
        electrodes = self.experiment.electrodes
        polarities = [StimPolarity.PositiveFirst, StimPolarity.NegativeFirst]
        responsive_pairs: List[Dict[str, Any]] = []

        for elec_idx in electrodes:
            for amplitude in self.scan_amplitudes:
                for duration in self.scan_durations:
                    for polarity in polarities:
                        hits = 0
                        latencies = []
                        for rep in range(self.scan_repeats):
                            stim_time = datetime_now()
                            self._send_single_stim(
                                electrode_idx=elec_idx,
                                amplitude_ua=amplitude,
                                duration_us=duration,
                                polarity=polarity,
                                trigger_key=0,
                                phase_label="scan",
                            )
                            self._wait(self.scan_inter_stim_s)
                            window_s = self.scan_response_window_ms / 1000.0
                            q_start = stim_time
                            q_stop = datetime_now()
                            spike_df = self.database.get_spike_event(
                                q_start, q_stop, self.experiment.exp_name
                            )
                            if not spike_df.empty:
                                hits += 1
                                if "Time" in spike_df.columns:
                                    for _, row in spike_df.iterrows():
                                        try:
                                            t_spike = pd.to_datetime(row["Time"], utc=True)
                                            t_stim = pd.to_datetime(stim_time)
                                            if t_stim.tzinfo is None:
                                                t_stim = t_stim.replace(tzinfo=timezone.utc)
                                            lat_ms = (t_spike - t_stim).total_seconds() * 1000.0
                                            if 0 < lat_ms < self.scan_response_window_ms:
                                                latencies.append(lat_ms)
                                        except Exception:
                                            pass

                        if hits >= self.scan_required_hits:
                            median_lat = float(np.median(latencies)) if latencies else 0.0
                            pair_info = {
                                "stim_electrode": elec_idx,
                                "hits": hits,
                                "repeats": self.scan_repeats,
                                "amplitude": amplitude,
                                "duration": duration,
                                "polarity": polarity.name,
                                "median_latency_ms": median_lat,
                            }
                            responsive_pairs.append(pair_info)
                            logger.info(
                                "Responsive electrode %d: hits=%d/%d amp=%.1f dur=%.0f pol=%s lat=%.2f ms",
                                elec_idx, hits, self.scan_repeats, amplitude, duration,
                                polarity.name, median_lat,
                            )

            self._wait(self.scan_inter_channel_s)

        self._scan_responsive_pairs = responsive_pairs
        logger.info("Stage 1 complete: %d responsive configurations found", len(responsive_pairs))

    def _stage2_active_electrode_experiment(self) -> None:
        logger.info("Stage 2: Active electrode experiment at 1 Hz")
        pairs_to_use = self._get_best_pairs_for_stage2()

        for pair in pairs_to_use:
            stim_elec = pair["stim"]
            resp_elec = pair["resp"]
            amplitude = pair["amplitude"]
            duration = pair["duration"]
            polarity = pair["polarity"]
            pair_key = f"{stim_elec}_to_{resp_elec}"
            stim_times = []

            logger.info("Stage 2 pair: %d -> %d", stim_elec, resp_elec)

            total_groups = self.active_total_repeats // self.active_group_size
            trial_idx = 0
            for group_i in range(total_groups):
                for pulse_i in range(self.active_group_size):
                    stim_time = datetime_now()
                    self._send_single_stim(
                        electrode_idx=stim_elec,
                        amplitude_ua=amplitude,
                        duration_us=duration,
                        polarity=polarity,
                        trigger_key=1,
                        phase_label="active",
                    )
                    stim_times.append(stim_time.isoformat())
                    self._wait(self.active_isi_s)
                    trial_idx += 1

                if group_i < total_groups - 1:
                    self._wait(self.active_group_pause_s)

            self._active_stim_times[pair_key] = stim_times
            logger.info("Stage 2 pair %s: %d stimulations delivered", pair_key, len(stim_times))

        logger.info("Stage 2: Computing cross-correlograms")
        self._compute_ccgs()
        logger.info("Stage 2 complete")

    def _compute_ccgs(self) -> None:
        pairs_to_use = self._get_best_pairs_for_stage2()
        ccg_results = {}

        for pair in pairs_to_use:
            stim_elec = pair["stim"]
            resp_elec = pair["resp"]
            pair_key = f"{stim_elec}_to_{resp_elec}"
            stim_times_iso = self._active_stim_times.get(pair_key, [])
            if not stim_times_iso:
                continue

            stim_times_dt = []
            for t_iso in stim_times_iso:
                try:
                    dt = datetime.fromisoformat(t_iso)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    stim_times_dt.append(dt)
                except Exception:
                    pass

            if not stim_times_dt:
                continue

            exp_start = stim_times_dt[0] - timedelta(seconds=1)
            exp_stop = stim_times_dt[-1] + timedelta(seconds=2)

            try:
                spike_df = self.database.get_spike_event(
                    exp_start, exp_stop, self.experiment.exp_name
                )
            except Exception as exc:
                logger.warning("CCG spike fetch failed for %s: %s", pair_key, exc)
                continue

            n_bins = int(self.ccg_window_ms / self.ccg_bin_ms)
            ccg_bins = np.zeros(n_bins, dtype=int)

            if not spike_df.empty and "Time" in spike_df.columns and "channel" in spike_df.columns:
                resp_spikes = spike_df[spike_df["channel"] == resp_elec]
                resp_times = []
                for _, row in resp_spikes.iterrows():
                    try:
                        t = pd.to_datetime(row["Time"], utc=True)
                        resp_times.append(t)
                    except Exception:
                        pass

                for stim_t in stim_times_dt:
                    for resp_t in resp_times:
                        lag_ms = (resp_t - stim_t).total_seconds() * 1000.0
                        if 0 <= lag_ms < self.ccg_window_ms:
                            bin_idx = int(lag_ms / self.ccg_bin_ms)
                            if 0 <= bin_idx < n_bins:
                                ccg_bins[bin_idx] += 1

            peak_bin = int(np.argmax(ccg_bins))
            peak_latency_ms = (peak_bin + 0.5) * self.ccg_bin_ms

            ccg_results[pair_key] = {
                "stim_electrode": stim_elec,
                "resp_electrode": resp_elec,
                "ccg_counts": ccg_bins.tolist(),
                "peak_latency_ms": peak_latency_ms,
                "total_stims": len(stim_times_dt),
            }
            logger.info("CCG %s: peak latency=%.2f ms", pair_key, peak_latency_ms)

        self._ccg_results = ccg_results

    def _stage3_stdp_experiment(self) -> None:
        logger.info("Stage 3: STDP Hebbian learning experiment")
        stdp_pairs = self._get_best_pairs_for_stdp()
        stdp_results = {}

        for pair in stdp_pairs:
            stim_elec = pair["stim"]
            resp_elec = pair["resp"]
            amplitude = min(pair["amplitude"], self.stdp_amplitude_ua)
            duration = min(pair["duration"], self.stdp_duration_us)
            polarity = pair["polarity"]
            pair_key = f"{stim_elec}_to_{resp_elec}"

            pair_key_ccg = f"{stim_elec}_to_{resp_elec}"
            if pair_key_ccg in self._ccg_results:
                hebbian_delay_ms = self._ccg_results[pair_key_ccg]["peak_latency_ms"]
            else:
                hebbian_delay_ms = pair.get("median_latency_ms", self.stdp_delta_t_ms)

            logger.info(
                "Stage 3 pair %s: Hebbian delay=%.2f ms", pair_key, hebbian_delay_ms
            )

            logger.info("Stage 3 pair %s: Testing phase (%.0f s)", pair_key, self.stdp_testing_duration_s)
            testing_spikes = self._stdp_passive_recording_phase(
                stim_elec, resp_elec, amplitude, duration, polarity,
                self.stdp_testing_duration_s, "stdp_testing"
            )

            logger.info("Stage 3 pair %s: Learning phase (%.0f s)", pair_key, self.stdp_learning_duration_s)
            learning_stim_times = self._stdp_learning_phase(
                stim_elec, resp_elec, amplitude, duration, polarity,
                hebbian_delay_ms, self.stdp_learning_duration_s
            )

            logger.info("Stage 3 pair %s: Validation phase (%.0f s)", pair_key, self.stdp_validation_duration_s)
            validation_spikes = self._stdp_passive_recording_phase(
                stim_elec, resp_elec, amplitude, duration, polarity,
                self.stdp_validation_duration_s, "stdp_validation"
            )

            pre_response_rate = self._compute_response_rate(testing_spikes)
            post_response_rate = self._compute_response_rate(validation_spikes)
            delta_response = post_response_rate - pre_response_rate

            stdp_results[pair_key] = {
                "stim_electrode": stim_elec,
                "resp_electrode": resp_elec,
                "hebbian_delay_ms": hebbian_delay_ms,
                "testing_spike_count": len(testing_spikes),
                "learning_stim_count": len(learning_stim_times),
                "validation_spike_count": len(validation_spikes),
                "pre_response_rate": pre_response_rate,
                "post_response_rate": post_response_rate,
                "delta_response_rate": delta_response,
            }
            logger.info(
                "Stage 3 pair %s: pre_rate=%.3f post_rate=%.3f delta=%.3f",
                pair_key, pre_response_rate, post_response_rate, delta_response
            )

        self._stdp_results = stdp_results
        logger.info("Stage 3 complete")

    def _stdp_passive_recording_phase(
        self,
        stim_elec: int,
        resp_elec: int,
        amplitude: float,
        duration: float,
        polarity: StimPolarity,
        phase_duration_s: float,
        phase_label: str,
    ) -> List[Dict[str, Any]]:
        phase_start = datetime_now()
        phase_end_target = phase_start + timedelta(seconds=phase_duration_s)
        spike_records = []
        probe_isi_s = 30.0
        next_probe = phase_start

        while datetime_now() < phase_end_target:
            now = datetime_now()
            if now >= next_probe:
                stim_time = datetime_now()
                self._send_single_stim(
                    electrode_idx=stim_elec,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    polarity=polarity,
                    trigger_key=2,
                    phase_label=phase_label,
                )
                self._wait(0.1)
                q_start = stim_time
                q_stop = datetime_now()
                try:
                    spike_df = self.database.get_spike_event(
                        q_start, q_stop, self.experiment.exp_name
                    )
                    if not spike_df.empty:
                        for _, row in spike_df.iterrows():
                            spike_records.append({
                                "phase": phase_label,
                                "stim_electrode": stim_elec,
                                "resp_electrode": resp_elec,
                                "stim_time": stim_time.isoformat(),
                                "amplitude": row.get("Amplitude", 0),
                            })
                except Exception as exc:
                    logger.warning("Passive phase spike fetch error: %s", exc)

                next_probe = datetime_now() + timedelta(seconds=probe_isi_s)

            remaining = (phase_end_target - datetime_now()).total_seconds()
            sleep_s = min(probe_isi_s, max(0.5, remaining))
            self._wait(sleep_s)

        return spike_records

    def _stdp_learning_phase(
        self,
        stim_elec: int,
        resp_elec: int,
        amplitude: float,
        duration: float,
        polarity: StimPolarity,
        hebbian_delay_ms: float,
        phase_duration_s: float,
    ) -> List[str]:
        phase_start = datetime_now()
        phase_end_target = phase_start + timedelta(seconds=phase_duration_s)
        stim_times = []
        delay_s = hebbian_delay_ms / 1000.0

        while datetime_now() < phase_end_target:
            stim_time = datetime_now()
            self._send_single_stim(
                electrode_idx=stim_elec,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=polarity,
                trigger_key=3,
                phase_label="stdp_learning_pre",
            )
            stim_times.append(stim_time.isoformat())

            self._wait(delay_s)

            self._send_single_stim(
                electrode_idx=resp_elec,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=polarity,
                trigger_key=4,
                phase_label="stdp_learning_post",
            )

            remaining = (phase_end_target - datetime_now()).total_seconds()
            if remaining <= 0:
                break
            sleep_s = min(self.stdp_conditioning_isi_s, remaining)
            self._wait(sleep_s)

        return stim_times

    def _compute_response_rate(self, spike_records: List[Dict[str, Any]]) -> float:
        if not spike_records:
            return 0.0
        return float(len(spike_records))

    def _get_best_pairs_for_stage2(self) -> List[Dict[str, Any]]:
        if self._scan_responsive_pairs:
            seen = set()
            pairs = []
            for p in self._scan_responsive_pairs:
                key = p["stim_electrode"]
                if key not in seen:
                    seen.add(key)
                    polarity = StimPolarity.PositiveFirst if p["polarity"] == "PositiveFirst" else StimPolarity.NegativeFirst
                    pairs.append({
                        "stim": p["stim_electrode"],
                        "resp": p.get("resp_electrode", p["stim_electrode"]),
                        "amplitude": p["amplitude"],
                        "duration": p["duration"],
                        "polarity": polarity,
                        "median_latency_ms": p.get("median_latency_ms", 20.0),
                    })
            if pairs:
                return pairs[:4]

        return [
            {
                "stim": p["stim"],
                "resp": p["resp"],
                "amplitude": p["amplitude"],
                "duration": p["duration"],
                "polarity": p["polarity"],
                "median_latency_ms": p["median_latency_ms"],
            }
            for p in self._known_pairs[:4]
        ]

    def _get_best_pairs_for_stdp(self) -> List[Dict[str, Any]]:
        priority_pairs = [
            p for p in self._known_pairs
            if (p["stim"] == 7 and p["resp"] == 6) or (p["stim"] == 17 and p["resp"] == 18)
        ]
        if priority_pairs:
            return priority_pairs[:2]
        return self._known_pairs[:2]

    def _send_single_stim(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity,
        trigger_key: int = 0,
        phase_label: str = "generic",
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
        self._wait(0.05)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            polarity=polarity.name,
            phase=phase_label,
            timestamp_utc=datetime_now().isoformat(),
            trigger_key=trigger_key,
        ))

    def _save_all(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> None:
        fs_name = getattr(self.experiment, "exp_name", "unknown")
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
            "scan_responsive_pairs": len(self._scan_responsive_pairs),
            "ccg_pairs_computed": len(self._ccg_results),
            "stdp_pairs_run": len(self._stdp_results),
            "ccg_results": self._ccg_results,
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
        fs_name = getattr(self.experiment, "exp_name", "unknown")
        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": fs_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "stage1_responsive_pairs": len(self._scan_responsive_pairs),
            "stage2_ccg_pairs": len(self._ccg_results),
            "stage3_stdp_pairs": len(self._stdp_results),
            "total_stimulations": len(self._stimulation_log),
            "ccg_results": self._ccg_results,
            "stdp_results": self._stdp_results,
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
