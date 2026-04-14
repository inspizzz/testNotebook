import numpy as np
import pandas as pd
import json
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
    TOP_PAIRS = [
        {"stim": 14, "resp": 12, "amplitude": 1.0, "duration": 400.0, "polarity": "NegativeFirst", "response_rate": 0.94},
        {"stim": 9,  "resp": 10, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "response_rate": 0.94},
        {"stim": 22, "resp": 21, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "response_rate": 0.93},
        {"stim": 5,  "resp": 4,  "amplitude": 1.0, "duration": 300.0, "polarity": "PositiveFirst", "response_rate": 0.93},
        {"stim": 17, "resp": 16, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "response_rate": 0.90},
    ]

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        default_amplitude: float = 2.0,
        default_duration: float = 200.0,
        num_scan_repeats: int = 3,
        ppf_trials: int = 5,
        io_trials: int = 5,
        freq_trials: int = 5,
        stdp_pairs: int = 20,
        rest_duration_s: float = 10.0,
        spontaneous_duration_s: float = 30.0,
        response_window_ms: float = 50.0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.default_amplitude = default_amplitude
        self.default_duration = default_duration
        self.num_scan_repeats = num_scan_repeats
        self.ppf_trials = ppf_trials
        self.io_trials = io_trials
        self.freq_trials = freq_trials
        self.stdp_pairs = stdp_pairs
        self.rest_duration_s = rest_duration_s
        self.spontaneous_duration_s = spontaneous_duration_s
        self.response_window_ms = response_window_ms

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []

        self._scan_results: Dict[str, Any] = {}
        self._top_pairs: List[Dict] = []
        self._ppf_results: Dict[str, Any] = {}
        self._io_results: Dict[str, Any] = {}
        self._freq_results: Dict[str, Any] = {}
        self._stdp_pre_post_results: Dict[str, Any] = {}
        self._stdp_post_pre_results: Dict[str, Any] = {}
        self._ltp_ltd_comparison: Dict[str, Any] = {}
        self._spontaneous_results: Dict[str, Any] = {}
        self._connectivity_before: Dict[str, Any] = {}
        self._connectivity_after: Dict[str, Any] = {}
        self._response_prob_bins: Dict[str, Any] = {}
        self._burst_rate_results: Dict[str, Any] = {}
        self._cross_correlograms: Dict[str, Any] = {}
        self._latency_distributions: Dict[str, Any] = {}
        self._biphasic_vs_mono_results: Dict[str, Any] = {}
        self._polarity_results: Dict[str, Any] = {}
        self._wasserstein_results: Dict[str, Any] = {}
        self._phase_data: Dict[str, Any] = {}

        self._scan_amplitudes = [1.0, 2.0, 3.0]
        self._ppf_intervals_ms = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
        self._io_amplitudes = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        self._frequencies_hz = [1.0, 2.0, 5.0, 10.0, 20.0]
        self._stdp_delays_ms = [10.0, -10.0]

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

            self._recording_start = datetime_now()

            logger.info("=== PHASE 1: Electrode Scan ===")
            self._phase_electrode_scan()
            self._save_phase_data("scan")
            self._wait(self.rest_duration_s)

            logger.info("=== PHASE 2: Identify Top 5 Pairs ===")
            self._identify_top_pairs()

            logger.info("=== PHASE 3: Spontaneous Activity (Baseline) ===")
            self._phase_spontaneous_recording("baseline")
            self._save_phase_data("spontaneous_baseline")
            self._wait(self.rest_duration_s)

            logger.info("=== PHASE 4: Connectivity Matrix Before Plasticity ===")
            self._phase_connectivity_matrix("before")
            self._save_phase_data("connectivity_before")
            self._wait(self.rest_duration_s)

            logger.info("=== PHASE 5: Paired-Pulse Facilitation ===")
            self._phase_paired_pulse_facilitation()
            self._save_phase_data("ppf")
            self._wait(self.rest_duration_s)

            logger.info("=== PHASE 6: Input-Output Curve ===")
            self._phase_input_output_curve()
            self._save_phase_data("io_curve")
            self._wait(self.rest_duration_s)

            logger.info("=== PHASE 7: Frequency Response ===")
            self._phase_frequency_response()
            self._save_phase_data("freq_response")
            self._wait(self.rest_duration_s)

            logger.info("=== PHASE 8: Biphasic vs Monophasic ===")
            self._phase_biphasic_vs_monophasic()
            self._save_phase_data("biphasic_vs_mono")
            self._wait(self.rest_duration_s)

            logger.info("=== PHASE 9: Polarity Comparison ===")
            self._phase_polarity_comparison()
            self._save_phase_data("polarity")
            self._wait(self.rest_duration_s)

            logger.info("=== PHASE 10: STDP Pre-Post Induction ===")
            self._phase_stdp_induction(mode="pre_post", delay_ms=10.0)
            self._save_phase_data("stdp_pre_post")
            self._wait(self.rest_duration_s)

            logger.info("=== PHASE 11: STDP Post-Pre Induction ===")
            self._phase_stdp_induction(mode="post_pre", delay_ms=-10.0)
            self._save_phase_data("stdp_post_pre")
            self._wait(self.rest_duration_s)

            logger.info("=== PHASE 12: LTP vs LTD Comparison ===")
            self._phase_ltp_ltd_comparison()
            self._save_phase_data("ltp_ltd")
            self._wait(self.rest_duration_s)

            logger.info("=== PHASE 13: Connectivity Matrix After Plasticity ===")
            self._phase_connectivity_matrix("after")
            self._save_phase_data("connectivity_after")
            self._wait(self.rest_duration_s)

            logger.info("=== PHASE 14: Spontaneous Activity (Post-Plasticity) ===")
            self._phase_spontaneous_recording("post_plasticity")
            self._save_phase_data("spontaneous_post")
            self._wait(self.rest_duration_s)

            logger.info("=== PHASE 15: Response Probability Over Time ===")
            self._phase_response_probability_bins()
            self._save_phase_data("response_prob")
            self._wait(self.rest_duration_s)

            logger.info("=== PHASE 16: Burst Rate Analysis ===")
            self._phase_burst_rate()
            self._save_phase_data("burst_rate")
            self._wait(self.rest_duration_s)

            logger.info("=== PHASE 17: Cross-Correlograms ===")
            self._phase_cross_correlograms()
            self._save_phase_data("cross_correlograms")
            self._wait(self.rest_duration_s)

            logger.info("=== PHASE 18: First-Spike Latency Distributions ===")
            self._phase_latency_distributions()
            self._save_phase_data("latency_distributions")
            self._wait(self.rest_duration_s)

            logger.info("=== PHASE 19: Wasserstein Distance ===")
            self._phase_wasserstein_distance()
            self._save_phase_data("wasserstein")

            self._recording_stop = datetime_now()

            results = self._compile_results(self._recording_start, self._recording_stop)

            self._save_all(self._recording_start, self._recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            try:
                self._recording_stop = datetime_now()
                self._save_all(self._recording_start, self._recording_stop)
            except Exception as save_exc:
                logger.error("Failed to save data on error: %s", save_exc)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _send_biphasic_pulse(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity = StimPolarity.NegativeFirst,
        trigger_key: int = 0,
        phase_label: str = "general",
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

        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            polarity=polarity.name,
            phase=phase_label,
            timestamp_utc=datetime_now().isoformat(),
            trigger_key=trigger_key,
        ))

    def _send_charge_balanced_asymmetric(
        self,
        electrode_idx: int,
        amplitude1: float,
        duration1: float,
        amplitude2: float,
        duration2: float,
        polarity: StimPolarity = StimPolarity.NegativeFirst,
        trigger_key: int = 0,
        phase_label: str = "asymmetric",
    ) -> None:
        amplitude1 = min(abs(amplitude1), 4.0)
        duration1 = min(abs(duration1), 400.0)
        amplitude2 = min(abs(amplitude2), 4.0)
        duration2 = min(abs(duration2), 400.0)

        charge1 = amplitude1 * duration1
        charge2 = amplitude2 * duration2
        if abs(charge1 - charge2) > 1e-6:
            logger.warning(
                "Charge imbalance detected: A1*D1=%.2f vs A2*D2=%.2f; adjusting D2",
                charge1, charge2
            )
            if amplitude2 > 0:
                duration2 = min(charge1 / amplitude2, 400.0)
            else:
                duration2 = duration1

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
        stim.phase_amplitude1 = amplitude1
        stim.phase_duration1 = duration1
        stim.phase_amplitude2 = amplitude2
        stim.phase_duration2 = duration2
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

        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=amplitude1,
            duration_us=duration1,
            polarity=polarity.name,
            phase=phase_label,
            timestamp_utc=datetime_now().isoformat(),
            trigger_key=trigger_key,
            extra={"amplitude2": amplitude2, "duration2": duration2},
        ))

    def _query_spikes(
        self,
        start: datetime,
        stop: datetime,
        fs_name: Optional[str] = None,
    ) -> pd.DataFrame:
        if fs_name is None:
            fs_name = self.experiment.exp_name
        try:
            df = self.database.get_spike_event(start, stop, fs_name)
            return df
        except Exception as exc:
            logger.warning("Spike query failed: %s", exc)
            return pd.DataFrame()

    def _count_spikes_in_window(
        self,
        electrode: int,
        window_start: datetime,
        window_stop: datetime,
    ) -> int:
        try:
            df = self.database.get_spike_event_electrode(window_start, window_stop, electrode)
            return len(df)
        except Exception:
            return 0

    def _phase_electrode_scan(self) -> None:
        scan_results = {}
        electrodes_to_scan = list(range(32))
        amplitudes = self._scan_amplitudes

        for elec in electrodes_to_scan:
            scan_results[elec] = {}
            for amp in amplitudes:
                duration = self.default_duration
                hits = 0
                for rep in range(self.num_scan_repeats):
                    t_before = datetime_now()
                    self._send_biphasic_pulse(
                        electrode_idx=elec,
                        amplitude_ua=amp,
                        duration_us=duration,
                        polarity=StimPolarity.NegativeFirst,
                        trigger_key=0,
                        phase_label="electrode_scan",
                    )
                    self._wait(0.3)
                    t_after = datetime_now()
                    n_spikes = self._count_spikes_in_window(elec, t_before, t_after)
                    if n_spikes > 0:
                        hits += 1
                    self._wait(0.2)
                scan_results[elec][amp] = {"hits": hits, "repeats": self.num_scan_repeats}
            self._wait(0.5)

        self._scan_results = scan_results
        logger.info("Electrode scan complete: %d electrodes scanned", len(electrodes_to_scan))

    def _identify_top_pairs(self) -> None:
        self._top_pairs = [dict(p) for p in self.TOP_PAIRS[:5]]
        logger.info("Top 5 pairs identified: %s", [(p["stim"], p["resp"]) for p in self._top_pairs])

    def _phase_spontaneous_recording(self, label: str = "baseline") -> None:
        logger.info("Recording spontaneous activity for %s seconds (label=%s)",
                    self.spontaneous_duration_s, label)
        t_start = datetime_now()
        self._wait(self.spontaneous_duration_s)
        t_stop = datetime_now()

        spike_df = self._query_spikes(t_start, t_stop)
        total_spikes = len(spike_df)

        spikes_per_electrode: Dict[int, int] = defaultdict(int)
        if not spike_df.empty and "channel" in spike_df.columns:
            for ch, grp in spike_df.groupby("channel"):
                spikes_per_electrode[int(ch)] = len(grp)

        burst_count = self._estimate_bursts(spike_df, t_start, t_stop)

        result = {
            "label": label,
            "duration_s": self.spontaneous_duration_s,
            "total_spikes": total_spikes,
            "spikes_per_electrode": dict(spikes_per_electrode),
            "burst_count": burst_count,
            "start_utc": t_start.isoformat(),
            "stop_utc": t_stop.isoformat(),
        }

        if label == "baseline":
            self._spontaneous_results["baseline"] = result
        else:
            self._spontaneous_results[label] = result

        logger.info("Spontaneous recording (%s): %d spikes, %d bursts", label, total_spikes, burst_count)

    def _estimate_bursts(self, spike_df: pd.DataFrame, t_start: datetime, t_stop: datetime) -> int:
        if spike_df.empty or "Time" not in spike_df.columns:
            return 0
        try:
            times = pd.to_datetime(spike_df["Time"]).sort_values()
            diffs = times.diff().dt.total_seconds().dropna()
            burst_threshold_s = 0.05
            in_burst = diffs < burst_threshold_s
            burst_starts = (~in_burst.shift(1, fill_value=False)) & in_burst
            return int(burst_starts.sum())
        except Exception:
            return 0

    def _phase_connectivity_matrix(self, phase: str) -> None:
        logger.info("Computing connectivity matrix (%s)", phase)
        n_electrodes = 32
        matrix = {}

        for pair in self._top_pairs:
            stim_e = pair["stim"]
            resp_e = pair["resp"]
            amp = pair["amplitude"]
            dur = pair["duration"]
            pol_str = pair["polarity"]
            polarity = StimPolarity.NegativeFirst if pol_str == "NegativeFirst" else StimPolarity.PositiveFirst

            hits = 0
            n_trials = 5
            for _ in range(n_trials):
                t_before = datetime_now()
                self._send_biphasic_pulse(
                    electrode_idx=stim_e,
                    amplitude_ua=amp,
                    duration_us=dur,
                    polarity=polarity,
                    trigger_key=0,
                    phase_label=f"connectivity_{phase}",
                )
                self._wait(0.1)
                t_after = datetime_now()
                n_spikes = self._count_spikes_in_window(resp_e, t_before, t_after)
                if n_spikes > 0:
                    hits += 1
                self._wait(0.3)

            key = f"{stim_e}->{resp_e}"
            matrix[key] = {
                "stim": stim_e,
                "resp": resp_e,
                "hits": hits,
                "trials": n_trials,
                "response_rate": hits / n_trials,
            }
            self._wait(0.5)

        if phase == "before":
            self._connectivity_before = matrix
        else:
            self._connectivity_after = matrix

        logger.info("Connectivity matrix (%s) computed for %d pairs", phase, len(matrix))

    def _phase_paired_pulse_facilitation(self) -> None:
        logger.info("Running paired-pulse facilitation at %d intervals", len(self._ppf_intervals_ms))
        ppf_results = {}

        if not self._top_pairs:
            logger.warning("No top pairs available for PPF")
            return

        pair = self._top_pairs[0]
        stim_e = pair["stim"]
        resp_e = pair["resp"]
        amp = self.default_amplitude
        dur = self.default_duration
        polarity = StimPolarity.NegativeFirst

        for isi_ms in self._ppf_intervals_ms:
            isi_s = isi_ms / 1000.0
            responses_p1 = []
            responses_p2 = []

            for trial in range(self.ppf_trials):
                t0 = datetime_now()
                self._send_biphasic_pulse(
                    electrode_idx=stim_e,
                    amplitude_ua=amp,
                    duration_us=dur,
                    polarity=polarity,
                    trigger_key=0,
                    phase_label=f"ppf_pulse1_isi{isi_ms}ms",
                )
                self._wait(isi_s)
                self._send_biphasic_pulse(
                    electrode_idx=stim_e,
                    amplitude_ua=amp,
                    duration_us=dur,
                    polarity=polarity,
                    trigger_key=0,
                    phase_label=f"ppf_pulse2_isi{isi_ms}ms",
                )
                self._wait(0.2)
                t1 = datetime_now()

                spike_df = self._query_spikes(t0, t1)
                if not spike_df.empty and "channel" in spike_df.columns:
                    resp_spikes = spike_df[spike_df["channel"] == resp_e]
                    n_resp = len(resp_spikes)
                else:
                    n_resp = 0

                responses_p1.append(1 if n_resp > 0 else 0)
                responses_p2.append(1 if n_resp > 1 else 0)
                self._wait(0.5)

            r1 = sum(responses_p1) / len(responses_p1) if responses_p1 else 0.0
            r2 = sum(responses_p2) / len(responses_p2) if responses_p2 else 0.0
            ppf_ratio = r2 / r1 if r1 > 0 else float("nan")

            ppf_results[f"isi_{isi_ms}ms"] = {
                "isi_ms": isi_ms,
                "response_rate_p1": r1,
                "response_rate_p2": r2,
                "ppf_ratio": ppf_ratio,
                "trials": self.ppf_trials,
            }
            logger.info("PPF ISI=%.0fms: r1=%.2f r2=%.2f ratio=%.2f", isi_ms, r1, r2, ppf_ratio)
            self._wait(1.0)

        self._ppf_results = ppf_results

    def _phase_input_output_curve(self) -> None:
        logger.info("Running input-output curve at %d amplitude levels", len(self._io_amplitudes))
        io_results = {}

        if not self._top_pairs:
            logger.warning("No top pairs available for I/O curve")
            return

        pair = self._top_pairs[0]
        stim_e = pair["stim"]
        resp_e = pair["resp"]
        dur = self.default_duration
        polarity = StimPolarity.NegativeFirst

        for amp in self._io_amplitudes:
            safe_amp = min(amp, 4.0)
            safe_dur = min(dur, 400.0)
            hits = 0
            latencies = []

            for trial in range(self.io_trials):
                t_before = datetime_now()
                self._send_biphasic_pulse(
                    electrode_idx=stim_e,
                    amplitude_ua=safe_amp,
                    duration_us=safe_dur,
                    polarity=polarity,
                    trigger_key=0,
                    phase_label=f"io_curve_amp{safe_amp}uA",
                )
                self._wait(0.15)
                t_after = datetime_now()

                spike_df = self._query_spikes(t_before, t_after)
                if not spike_df.empty and "channel" in spike_df.columns:
                    resp_spikes = spike_df[spike_df["channel"] == resp_e]
                    if len(resp_spikes) > 0:
                        hits += 1
                        if "Time" in resp_spikes.columns:
                            try:
                                t_stim = pd.Timestamp(t_before)
                                t_spike = pd.to_datetime(resp_spikes["Time"].iloc[0])
                                lat_ms = (t_spike - t_stim).total_seconds() * 1000.0
                                if 0 < lat_ms < 100:
                                    latencies.append(lat_ms)
                            except Exception:
                                pass
                self._wait(0.4)

            response_rate = hits / self.io_trials
            mean_lat = float(np.mean(latencies)) if latencies else float("nan")
            io_results[f"amp_{safe_amp}uA"] = {
                "amplitude_ua": safe_amp,
                "duration_us": safe_dur,
                "hits": hits,
                "trials": self.io_trials,
                "response_rate": response_rate,
                "mean_latency_ms": mean_lat,
            }
            logger.info("I/O amp=%.1fuA: response_rate=%.2f mean_lat=%.1fms",
                        safe_amp, response_rate, mean_lat)
            self._wait(0.5)

        self._io_results = io_results

    def _phase_frequency_response(self) -> None:
        logger.info("Running frequency response at %d frequencies", len(self._frequencies_hz))
        freq_results = {}

        if not self._top_pairs:
            logger.warning("No top pairs available for frequency response")
            return

        pair = self._top_pairs[0]
        stim_e = pair["stim"]
        resp_e = pair["resp"]
        amp = self.default_amplitude
        dur = self.default_duration
        polarity = StimPolarity.NegativeFirst
        n_pulses = 5

        for freq_hz in self._frequencies_hz:
            isi_s = 1.0 / freq_hz
            hits = 0
            total_pulses = 0

            for trial in range(self.freq_trials):
                for pulse_idx in range(n_pulses):
                    t_before = datetime_now()
                    self._send_biphasic_pulse(
                        electrode_idx=stim_e,
                        amplitude_ua=amp,
                        duration_us=dur,
                        polarity=polarity,
                        trigger_key=0,
                        phase_label=f"freq_{freq_hz}hz",
                    )
                    self._wait(0.05)
                    t_after = datetime_now()
                    n_spikes = self._count_spikes_in_window(resp_e, t_before, t_after)
                    if n_spikes > 0:
                        hits += 1
                    total_pulses += 1
                    if pulse_idx < n_pulses - 1:
                        self._wait(max(isi_s - 0.05, 0.01))
                self._wait(1.0)

            response_rate = hits / total_pulses if total_pulses > 0 else 0.0
            freq_results[f"freq_{freq_hz}hz"] = {
                "frequency_hz": freq_hz,
                "isi_s": isi_s,
                "hits": hits,
                "total_pulses": total_pulses,
                "response_rate": response_rate,
            }
            logger.info("Freq=%.1fHz: response_rate=%.2f (%d/%d)",
                        freq_hz, response_rate, hits, total_pulses)
            self._wait(1.0)

        self._freq_results = freq_results

    def _phase_biphasic_vs_monophasic(self) -> None:
        logger.info("Testing biphasic vs monophasic stimulation")
        results = {}

        if not self._top_pairs:
            logger.warning("No top pairs for biphasic vs monophasic test")
            return

        pair = self._top_pairs[0]
        stim_e = pair["stim"]
        resp_e = pair["resp"]
        amp = self.default_amplitude
        dur = self.default_duration
        n_trials = 5

        for stim_type in ["biphasic", "charge_balanced_asymmetric"]:
            hits = 0
            latencies = []
            for trial in range(n_trials):
                t_before = datetime_now()
                if stim_type == "biphasic":
                    self._send_biphasic_pulse(
                        electrode_idx=stim_e,
                        amplitude_ua=amp,
                        duration_us=dur,
                        polarity=StimPolarity.NegativeFirst,
                        trigger_key=0,
                        phase_label="biphasic_test",
                    )
                else:
                    a1 = amp
                    d1 = dur
                    a2 = amp * 2.0
                    d2 = d1 / 2.0
                    if a2 > 4.0:
                        a2 = 4.0
                        d2 = (a1 * d1) / a2
                    d2 = min(d2, 400.0)
                    self._send_charge_balanced_asymmetric(
                        electrode_idx=stim_e,
                        amplitude1=a1,
                        duration1=d1,
                        amplitude2=a2,
                        duration2=d2,
                        polarity=StimPolarity.NegativeFirst,
                        trigger_key=0,
                        phase_label="asymmetric_biphasic_test",
                    )
                self._wait(0.15)
                t_after = datetime_now()
                spike_df = self._query_spikes(t_before, t_after)
                if not spike_df.empty and "channel" in spike_df.columns:
                    resp_spikes = spike_df[spike_df["channel"] == resp_e]
                    if len(resp_spikes) > 0:
                        hits += 1
                        if "Time" in resp_spikes.columns:
                            try:
                                t_stim = pd.Timestamp(t_before)
                                t_spike = pd.to_datetime(resp_spikes["Time"].iloc[0])
                                lat_ms = (t_spike - t_stim).total_seconds() * 1000.0
                                if 0 < lat_ms < 100:
                                    latencies.append(lat_ms)
                            except Exception:
                                pass
                self._wait(0.5)

            results[stim_type] = {
                "hits": hits,
                "trials": n_trials,
                "response_rate": hits / n_trials,
                "mean_latency_ms": float(np.mean(latencies)) if latencies else float("nan"),
                "latencies_ms": latencies,
            }
            logger.info("Stim type=%s: response_rate=%.2f", stim_type, hits / n_trials)
            self._wait(1.0)

        self._biphasic_vs_mono_results = results

    def _phase_polarity_comparison(self) -> None:
        logger.info("Testing PositiveFirst vs NegativeFirst polarity")
        results = {}

        if not self._top_pairs:
            logger.warning("No top pairs for polarity comparison")
            return

        pair = self._top_pairs[0]
        stim_e = pair["stim"]
        resp_e = pair["resp"]
        amp = self.default_amplitude
        dur = self.default_duration
        n_trials = 5

        for pol_name, polarity in [("NegativeFirst", StimPolarity.NegativeFirst),
                                    ("PositiveFirst", StimPolarity.PositiveFirst)]:
            hits = 0
            latencies = []
            for trial in range(n_trials):
                t_before = datetime_now()
                self._send_biphasic_pulse(
                    electrode_idx=stim_e,
                    amplitude_ua=amp,
                    duration_us=dur,
                    polarity=polarity,
                    trigger_key=0,
                    phase_label=f"polarity_{pol_name}",
                )
                self._wait(0.15)
                t_after = datetime_now()
                spike_df = self._query_spikes(t_before, t_after)
                if not spike_df.empty and "channel" in spike_df.columns:
                    resp_spikes = spike_df[spike_df["channel"] == resp_e]
                    if len(resp_spikes) > 0:
                        hits += 1
                        if "Time" in resp_spikes.columns:
                            try:
                                t_stim = pd.Timestamp(t_before)
                                t_spike = pd.to_datetime(resp_spikes["Time"].iloc[0])
                                lat_ms = (t_spike - t_stim).total_seconds() * 1000.0
                                if 0 < lat_ms < 100:
                                    latencies.append(lat_ms)
                            except Exception:
                                pass
                self._wait(0.5)

            results[pol_name] = {
                "polarity": pol_name,
                "hits": hits,
                "trials": n_trials,
                "response_rate": hits / n_trials,
                "mean_latency_ms": float(np.mean(latencies)) if latencies else float("nan"),
                "latencies_ms": latencies,
            }
            logger.info("Polarity=%s: response_rate=%.2f", pol_name, hits / n_trials)
            self._wait(1.0)

        self._polarity_results = results

    def _phase_stdp_induction(self, mode: str, delay_ms: float) -> None:
        logger.info("STDP induction: mode=%s delay=%.1fms", mode, delay_ms)

        if not self._top_pairs:
            logger.warning("No top pairs for STDP induction")
            return

        pair = self._top_pairs[0]
        pre_e = pair["stim"]
        post_e = pair["resp"]
        amp = self.default_amplitude
        dur = self.default_duration
        polarity = StimPolarity.NegativeFirst
        delay_s = abs(delay_ms) / 1000.0

        results_per_trial = []

        for trial in range(self.stdp_pairs):
            t_before = datetime_now()

            if delay_ms >= 0:
                self._send_biphasic_pulse(
                    electrode_idx=pre_e,
                    amplitude_ua=amp,
                    duration_us=dur,
                    polarity=polarity,
                    trigger_key=0,
                    phase_label=f"stdp_{mode}_pre",
                )
                self._wait(delay_s)
                self._send_biphasic_pulse(
                    electrode_idx=post_e,
                    amplitude_ua=amp,
                    duration_us=dur,
                    polarity=polarity,
                    trigger_key=1,
                    phase_label=f"stdp_{mode}_post",
                )
            else:
                self._send_biphasic_pulse(
                    electrode_idx=post_e,
                    amplitude_ua=amp,
                    duration_us=dur,
                    polarity=polarity,
                    trigger_key=1,
                    phase_label=f"stdp_{mode}_post",
                )
                self._wait(delay_s)
                self._send_biphasic_pulse(
                    electrode_idx=pre_e,
                    amplitude_ua=amp,
                    duration_us=dur,
                    polarity=polarity,
                    trigger_key=0,
                    phase_label=f"stdp_{mode}_pre",
                )

            self._wait(0.2)
            t_after = datetime_now()

            spike_df = self._query_spikes(t_before, t_after)
            n_spikes = 0
            if not spike_df.empty and "channel" in spike_df.columns:
                n_spikes = len(spike_df[spike_df["channel"] == post_e])

            results_per_trial.append({
                "trial": trial,
                "n_spikes_post": n_spikes,
                "t_before": t_before.isoformat(),
                "t_after": t_after.isoformat(),
            })
            self._wait(0.5)

        summary = {
            "mode": mode,
            "delay_ms": delay_ms,
            "pre_electrode": pre_e,
            "post_electrode": post_e,
            "n_trials": self.stdp_pairs,
            "mean_post_spikes": float(np.mean([r["n_spikes_post"] for r in results_per_trial])),
            "trials": results_per_trial,
        }

        if mode == "pre_post":
            self._stdp_pre_post_results = summary
        else:
            self._stdp_post_pre_results = summary

        logger.info("STDP %s complete: mean_post_spikes=%.2f",
                    mode, summary["mean_post_spikes"])

    def _phase_ltp_ltd_comparison(self) -> None:
        logger.info("Comparing LTP vs LTD effects")

        pre_post_mean = self._stdp_pre_post_results.get("mean_post_spikes", 0.0)
        post_pre_mean = self._stdp_post_pre_results.get("mean_post_spikes", 0.0)

        ltp_effect = pre_post_mean
        ltd_effect = post_pre_mean
        delta = ltp_effect - ltd_effect

        self._ltp_ltd_comparison = {
            "ltp_mean_spikes": ltp_effect,
            "ltd_mean_spikes": ltd_effect,
            "delta_ltp_minus_ltd": delta,
            "ltp_stronger": delta > 0,
            "pre_post_delay_ms": self._stdp_pre_post_results.get("delay_ms", 10.0),
            "post_pre_delay_ms": self._stdp_post_pre_results.get("delay_ms", -10.0),
        }
        logger.info("LTP=%.2f LTD=%.2f delta=%.2f", ltp_effect, ltd_effect, delta)

    def _phase_response_probability_bins(self) -> None:
        logger.info("Tracking response probability over time in 1-minute bins")

        if not self._top_pairs:
            logger.warning("No top pairs for response probability tracking")
            return

        pair = self._top_pairs[0]
        stim_e = pair["stim"]
        resp_e = pair["resp"]
        amp = self.default_amplitude
        dur = self.default_duration
        polarity = StimPolarity.NegativeFirst

        n_bins = 3
        trials_per_bin = 5
        bin_duration_s = 20.0

        bins_results = []

        for bin_idx in range(n_bins):
            bin_start = datetime_now()
            hits = 0
            for trial in range(trials_per_bin):
                t_before = datetime_now()
                self._send_biphasic_pulse(
                    electrode_idx=stim_e,
                    amplitude_ua=amp,
                    duration_us=dur,
                    polarity=polarity,
                    trigger_key=0,
                    phase_label=f"resp_prob_bin{bin_idx}",
                )
                self._wait(0.15)
                t_after = datetime_now()
                n_spikes = self._count_spikes_in_window(resp_e, t_before, t_after)
                if n_spikes > 0:
                    hits += 1
                self._wait(bin_duration_s / trials_per_bin - 0.2)

            bin_stop = datetime_now()
            response_rate = hits / trials_per_bin
            bins_results.append({
                "bin_index": bin_idx,
                "bin_start_utc": bin_start.isoformat(),
                "bin_stop_utc": bin_stop.isoformat(),
                "hits": hits,
                "trials": trials_per_bin,
                "response_rate": response_rate,
            })
            logger.info("Response prob bin %d: %.2f", bin_idx, response_rate)

        self._response_prob_bins = {
            "stim_electrode": stim_e,
            "resp_electrode": resp_e,
            "bins": bins_results,
        }

    def _phase_burst_rate(self) -> None:
        logger.info("Measuring burst rate changes")

        baseline_bursts = self._spontaneous_results.get("baseline", {}).get("burst_count", 0)
        post_bursts = self._spontaneous_results.get("post_plasticity", {}).get("burst_count", 0)
        duration_s = self.spontaneous_duration_s

        baseline_rate = baseline_bursts / duration_s if duration_s > 0 else 0.0
        post_rate = post_bursts / duration_s if duration_s > 0 else 0.0
        delta_rate = post_rate - baseline_rate

        self._burst_rate_results = {
            "baseline_burst_count": baseline_bursts,
            "post_plasticity_burst_count": post_bursts,
            "recording_duration_s": duration_s,
            "baseline_burst_rate_per_s": baseline_rate,
            "post_burst_rate_per_s": post_rate,
            "delta_burst_rate": delta_rate,
        }
        logger.info("Burst rate: baseline=%.3f/s post=%.3f/s delta=%.3f/s",
                    baseline_rate, post_rate, delta_rate)

    def _phase_cross_correlograms(self) -> None:
        logger.info("Computing cross-correlograms for all top pairs")
        ccg_results = {}

        t_start = datetime_now()
        self._wait(10.0)
        t_stop = datetime_now()

        spike_df = self._query_spikes(t_start, t_stop)

        for pair in self._top_pairs:
            stim_e = pair["stim"]
            resp_e = pair["resp"]
            key = f"{stim_e}->{resp_e}"

            if spike_df.empty or "channel" not in spike_df.columns or "Time" not in spike_df.columns:
                ccg_results[key] = {"bins": [], "counts": [], "peak_lag_ms": float("nan")}
                continue

            try:
                spikes_a = pd.to_datetime(
                    spike_df[spike_df["channel"] == stim_e]["Time"]
                ).sort_values().values
                spikes_b = pd.to_datetime(
                    spike_df[spike_df["channel"] == resp_e]["Time"]
                ).sort_values().values

                max_lag_ms = 50.0
                bin_size_ms = 2.0
                n_bins = int(2 * max_lag_ms / bin_size_ms)
                bins = np.linspace(-max_lag_ms, max_lag_ms, n_bins + 1)
                counts = np.zeros(n_bins, dtype=int)

                for t_a in spikes_a:
                    for t_b in spikes_b:
                        lag_ms = (pd.Timestamp(t_b) - pd.Timestamp(t_a)).total_seconds() * 1000.0
                        if -max_lag_ms <= lag_ms <= max_lag_ms:
                            bin_idx = int((lag_ms + max_lag_ms) / bin_size_ms)
                            if 0 <= bin_idx < n_bins:
                                counts[bin_idx] += 1

                peak_idx = int(np.argmax(counts)) if counts.sum() > 0 else n_bins // 2
                peak_lag_ms = float(bins[peak_idx] + bin_size_ms / 2)

                ccg_results[key] = {
                    "stim": stim_e,
                    "resp": resp_e,
                    "bins_ms": bins.tolist(),
                    "counts": counts.tolist(),
                    "peak_lag_ms": peak_lag_ms,
                    "total_coincidences": int(counts.sum()),
                }
            except Exception as exc:
                logger.warning("CCG failed for pair %s: %s", key, exc)
                ccg_results[key] = {"bins": [], "counts": [], "peak_lag_ms": float("nan")}

        self._cross_correlograms = ccg_results
        logger.info("Cross-correlograms computed for %d pairs", len(ccg_results))

    def _phase_latency_distributions(self) -> None:
        logger.info("Measuring first-spike latency distributions")
        latency_results = {}

        for pair in self._top_pairs:
            stim_e = pair["stim"]
            resp_e = pair["resp"]
            amp = pair["amplitude"]
            dur = pair["duration"]
            pol_str = pair["polarity"]
            polarity = StimPolarity.NegativeFirst if pol_str == "NegativeFirst" else StimPolarity.PositiveFirst
            key = f"{stim_e}->{resp_e}"
            latencies = []
            n_trials = 8

            for trial in range(n_trials):
                t_before = datetime_now()
                self._send_biphasic_pulse(
                    electrode_idx=stim_e,
                    amplitude_ua=amp,
                    duration_us=dur,
                    polarity=polarity,
                    trigger_key=0,
                    phase_label=f"latency_dist_{key}",
                )
                self._wait(0.1)
                t_after = datetime_now()

                spike_df = self._query_spikes(t_before, t_after)
                if not spike_df.empty and "channel" in spike_df.columns and "Time" in spike_df.columns:
                    resp_spikes = spike_df[spike_df["channel"] == resp_e].copy()
                    if len(resp_spikes) > 0:
                        try:
                            resp_spikes = resp_spikes.sort_values("Time")
                            t_stim = pd.Timestamp(t_before)
                            t_first = pd.to_datetime(resp_spikes["Time"].iloc[0])
                            lat_ms = (t_first - t_stim).total_seconds() * 1000.0
                            if 0 < lat_ms < 100:
                                latencies.append(lat_ms)
                        except Exception:
                            pass
                self._wait(0.4)

            mean_lat = float(np.mean(latencies)) if latencies else float("nan")
            std_lat = float(np.std(latencies)) if len(latencies) > 1 else float("nan")
            median_lat = float(np.median(latencies)) if latencies else float("nan")

            latency_results[key] = {
                "stim": stim_e,
                "resp": resp_e,
                "n_trials": n_trials,
                "n_responses": len(latencies),
                "latencies_ms": latencies,
                "mean_latency_ms": mean_lat,
                "std_latency_ms": std_lat,
                "median_latency_ms": median_lat,
            }
            logger.info("Latency dist %s: mean=%.1fms std=%.1fms n=%d",
                        key, mean_lat, std_lat, len(latencies))
            self._wait(0.5)

        self._latency_distributions = latency_results

    def _phase_wasserstein_distance(self) -> None:
        logger.info("Computing Wasserstein distances for distribution comparison")
        wasserstein_results = {}

        for key, lat_data in self._latency_distributions.items():
            latencies = lat_data.get("latencies_ms", [])
            if len(latencies) < 2:
                wasserstein_results[key] = {"wasserstein_distance": float("nan"), "note": "insufficient_data"}
                continue

            n = len(latencies)
            half = n // 2
            dist_a = sorted(latencies[:half])
            dist_b = sorted(latencies[half:])

            if not dist_a or not dist_b:
                wasserstein_results[key] = {"wasserstein_distance": float("nan"), "note": "insufficient_data"}
                continue

            n_a = len(dist_a)
            n_b = len(dist_b)
            all_vals = sorted(set(dist_a + dist_b))
            cdf_a = []
            cdf_b = []
            cum_a = 0
            cum_b = 0
            for v in all_vals:
                cum_a += dist_a.count(v) / n_a
                cum_b += dist_b.count(v) / n_b
                cdf_a.append(cum_a)
                cdf_b.append(cum_b)

            w_dist = 0.0
            for i in range(len(all_vals) - 1):
                w_dist += abs(cdf_a[i] - cdf_b[i]) * (all_vals[i + 1] - all_vals[i])

            wasserstein_results[key] = {
                "pair": key,
                "n_samples_a": n_a,
                "n_samples_b": n_b,
                "wasserstein_distance": w_dist,
                "mean_a": float(np.mean(dist_a)),
                "mean_b": float(np.mean(dist_b)),
            }
            logger.info("Wasserstein %s: W=%.3f", key, w_dist)

        polarity_lats_neg = self._polarity_results.get("NegativeFirst", {}).get("latencies_ms", [])
        polarity_lats_pos = self._polarity_results.get("PositiveFirst", {}).get("latencies_ms", [])
        if polarity_lats_neg and polarity_lats_pos:
            dist_a = sorted(polarity_lats_neg)
            dist_b = sorted(polarity_lats_pos)
            n_a = len(dist_a)
            n_b = len(dist_b)
            all_vals = sorted(set(dist_a + dist_b))
            cdf_a = []
            cdf_b = []
            cum_a = 0.0
            cum_b = 0.0
            for v in all_vals:
                cum_a += dist_a.count(v) / n_a
                cum_b += dist_b.count(v) / n_b
                cdf_a.append(cum_a)
                cdf_b.append(cum_b)
            w_dist = 0.0
            for i in range(len(all_vals) - 1):
                w_dist += abs(cdf_a[i] - cdf_b[i]) * (all_vals[i + 1] - all_vals[i])
            wasserstein_results["polarity_NegFirst_vs_PosFirst"] = {
                "n_samples_a": n_a,
                "n_samples_b": n_b,
                "wasserstein_distance": w_dist,
                "mean_a": float(np.mean(dist_a)),
                "mean_b": float(np.mean(dist_b)),
            }
            logger.info("Wasserstein polarity comparison: W=%.3f", w_dist)

        self._wasserstein_results = wasserstein_results

    def _save_phase_data(self, phase_name: str) -> None:
        self._phase_data[phase_name] = {
            "saved_at_utc": datetime_now().isoformat(),
            "stimulation_count_so_far": len(self._stimulation_log),
        }
        logger.info("Phase data checkpoint saved: %s", phase_name)

    def _save_all(self, recording_start: datetime, recording_stop: datetime) -> None:
        fs_name = getattr(self.experiment, "exp_name", "unknown")
        saver = DataSaver(self._output_dir, fs_name)

        saver.save_stimulation_log(self._stimulation_log)

        spike_df = pd.DataFrame()
        try:
            spike_df = self.database.get_spike_event(recording_start, recording_stop, fs_name)
        except Exception as exc:
            logger.warning("Failed to fetch spike events: %s", exc)
        saver.save_spike_events(spike_df)

        trigger_df = pd.DataFrame()
        try:
            trigger_df = self.database.get_all_triggers(recording_start, recording_stop)
        except Exception as exc:
            logger.warning("Failed to fetch triggers: %s", exc)
        saver.save_triggers(trigger_df)

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "top_pairs": self._top_pairs,
            "scan_results_electrode_count": len(self._scan_results),
            "ppf_intervals_tested": list(self._ppf_results.keys()),
            "io_amplitudes_tested": list(self._io_results.keys()),
            "frequencies_tested": list(self._freq_results.keys()),
            "stdp_pre_post_mean_spikes": self._stdp_pre_post_results.get("mean_post_spikes", None),
            "stdp_post_pre_mean_spikes": self._stdp_post_pre_results.get("mean_post_spikes", None),
            "ltp_ltd_comparison": self._ltp_ltd_comparison,
            "connectivity_before": self._connectivity_before,
            "connectivity_after": self._connectivity_after,
            "burst_rate_results": self._burst_rate_results,
            "wasserstein_results": self._wasserstein_results,
            "polarity_results": {k: {kk: vv for kk, vv in v.items() if kk != "latencies_ms"}
                                  for k, v in self._polarity_results.items()},
            "biphasic_vs_mono": {k: {kk: vv for kk, vv in v.items() if kk != "latencies_ms"}
                                  for k, v in self._biphasic_vs_mono_results.items()},
            "response_prob_bins": self._response_prob_bins,
            "phase_data_checkpoints": self._phase_data,
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
        for col in ["channel", "index", "electrode"]:
            if col in spike_df.columns:
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
        fs_name = getattr(self.experiment, "exp_name", "unknown")

        connectivity_changes = {}
        for key in self._connectivity_before:
            before_rate = self._connectivity_before[key].get("response_rate", 0.0)
            after_rate = self._connectivity_after.get(key, {}).get("response_rate", 0.0)
            connectivity_changes[key] = {
                "before": before_rate,
                "after": after_rate,
                "delta": after_rate - before_rate,
            }

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": fs_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "total_stimulations": len(self._stimulation_log),
            "top_5_pairs": self._top_pairs,
            "electrode_scan_summary": {
                "electrodes_scanned": len(self._scan_results),
                "amplitudes_tested": self._scan_amplitudes,
            },
            "ppf_summary": {
                isi: {
                    "isi_ms": v["isi_ms"],
                    "ppf_ratio": v["ppf_ratio"],
                    "response_rate_p1": v["response_rate_p1"],
                    "response_rate_p2": v["response_rate_p2"],
                }
                for isi, v in self._ppf_results.items()
            },
            "io_curve_summary": {
                amp_key: {
                    "amplitude_ua": v["amplitude_ua"],
                    "response_rate": v["response_rate"],
                    "mean_latency_ms": v["mean_latency_ms"],
                }
                for amp_key, v in self._io_results.items()
            },
            "frequency_response_summary": {
                fk: {
                    "frequency_hz": v["frequency_hz"],
                    "response_rate": v["response_rate"],
                }
                for fk, v in self._freq_results.items()
            },
            "stdp_pre_post": {
                "delay_ms": self._stdp_pre_post_results.get("delay_ms"),
                "mean_post_spikes": self._stdp_pre_post_results.get("mean_post_spikes"),
                "n_trials": self._stdp_pre_post_results.get("n_trials"),
            },
            "stdp_post_pre": {
                "delay_ms": self._stdp_post_pre_results.get("delay_ms"),
                "mean_post_spikes": self._stdp_post_pre_results.get("mean_post_spikes"),
                "n_trials": self._stdp_post_pre_results.get("n_trials"),
            },
            "ltp_ltd_comparison": self._ltp_ltd_comparison,
            "connectivity_changes": connectivity_changes,
            "spontaneous_activity": {
                k: {kk: vv for kk, vv in v.items() if kk not in ["start_utc", "stop_utc"]}
                for k, v in self._spontaneous_results.items()
            },
            "response_probability_bins": self._response_prob_bins,
            "burst_rate_changes": self._burst_rate_results,
            "cross_correlograms_summary": {
                k: {"peak_lag_ms": v.get("peak_lag_ms"), "total_coincidences": v.get("total_coincidences", 0)}
                for k, v in self._cross_correlograms.items()
            },
            "latency_distributions_summary": {
                k: {
                    "mean_latency_ms": v["mean_latency_ms"],
                    "std_latency_ms": v["std_latency_ms"],
                    "median_latency_ms": v["median_latency_ms"],
                    "n_responses": v["n_responses"],
                }
                for k, v in self._latency_distributions.items()
            },
            "biphasic_vs_monophasic": {
                k: {"response_rate": v["response_rate"], "mean_latency_ms": v["mean_latency_ms"]}
                for k, v in self._biphasic_vs_mono_results.items()
            },
            "polarity_comparison": {
                k: {"response_rate": v["response_rate"], "mean_latency_ms": v["mean_latency_ms"]}
                for k, v in self._polarity_results.items()
            },
            "wasserstein_distances": self._wasserstein_results,
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
