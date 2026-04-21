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
        amplitude_ua: float = 2.0,
        duration_us: float = 200.0,
        scan_amplitudes: tuple = (1.0, 2.0, 3.0),
        io_amplitudes: tuple = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0),
        ppf_intervals_ms: tuple = (10.0, 20.0, 50.0, 100.0, 200.0, 500.0),
        freq_response_hz: tuple = (1.0, 2.0, 5.0, 10.0, 20.0),
        stdp_delays_ms: tuple = (5.0, 10.0, 20.0, -5.0, -10.0, -20.0),
        stdp_repeats: int = 20,
        scan_repeats: int = 5,
        io_repeats: int = 10,
        ppf_repeats: int = 10,
        freq_repeats: int = 10,
        spontaneous_duration_s: float = 60.0,
        rest_duration_s: float = 10.0,
        response_window_ms: float = 50.0,
        top_n_pairs: int = 5,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.amplitude_ua = amplitude_ua
        self.duration_us = duration_us
        self.scan_amplitudes = list(scan_amplitudes)
        self.io_amplitudes = list(io_amplitudes)
        self.ppf_intervals_ms = list(ppf_intervals_ms)
        self.freq_response_hz = list(freq_response_hz)
        self.stdp_delays_ms = list(stdp_delays_ms)
        self.stdp_repeats = stdp_repeats
        self.scan_repeats = scan_repeats
        self.io_repeats = io_repeats
        self.ppf_repeats = ppf_repeats
        self.freq_repeats = freq_repeats
        self.spontaneous_duration_s = spontaneous_duration_s
        self.rest_duration_s = rest_duration_s
        self.response_window_ms = response_window_ms
        self.top_n_pairs = top_n_pairs

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []

        self._scan_results: Dict[str, Any] = {}
        self._top_pairs: List[Tuple[int, int, float, float, str]] = []
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
        self._phase_spike_data: Dict[str, List[pd.DataFrame]] = defaultdict(list)

        self._known_pairs = [
            (17, 18, 3.0, 400.0, "PositiveFirst"),
            (21, 19, 3.0, 400.0, "PositiveFirst"),
            (21, 22, 3.0, 400.0, "NegativeFirst"),
            (7, 6, 3.0, 400.0, "PositiveFirst"),
            (6, 7, 3.0, 400.0, "PositiveFirst"),
            (5, 4, 3.0, 400.0, "PositiveFirst"),
            (13, 14, 3.0, 300.0, "NegativeFirst"),
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

            logger.info("Phase 1: Electrode scan")
            self._phase_electrode_scan()
            self._wait(self.rest_duration_s)

            logger.info("Phase 2: Identify top pairs")
            self._identify_top_pairs()
            self._wait(self.rest_duration_s)

            logger.info("Phase 3: Spontaneous activity baseline")
            self._phase_spontaneous_recording("baseline")
            self._wait(self.rest_duration_s)

            logger.info("Phase 4: Connectivity matrix before plasticity")
            self._phase_connectivity_matrix("before")
            self._wait(self.rest_duration_s)

            logger.info("Phase 5: Paired-pulse facilitation")
            self._phase_paired_pulse_facilitation()
            self._wait(self.rest_duration_s)

            logger.info("Phase 6: Input-output curve")
            self._phase_input_output_curve()
            self._wait(self.rest_duration_s)

            logger.info("Phase 7: Frequency response")
            self._phase_frequency_response()
            self._wait(self.rest_duration_s)

            logger.info("Phase 8: Biphasic vs monophasic")
            self._phase_biphasic_vs_monophasic()
            self._wait(self.rest_duration_s)

            logger.info("Phase 9: Polarity comparison")
            self._phase_polarity_comparison()
            self._wait(self.rest_duration_s)

            logger.info("Phase 10: STDP pre-post induction")
            self._phase_stdp_induction("pre_post")
            self._wait(self.rest_duration_s)

            logger.info("Phase 11: STDP post-pre induction")
            self._phase_stdp_induction("post_pre")
            self._wait(self.rest_duration_s)

            logger.info("Phase 12: Connectivity matrix after plasticity")
            self._phase_connectivity_matrix("after")
            self._wait(self.rest_duration_s)

            logger.info("Phase 13: LTP vs LTD comparison")
            self._compare_ltp_ltd()

            logger.info("Phase 14: Spontaneous activity post-plasticity")
            self._phase_spontaneous_recording("post_plasticity")
            self._wait(self.rest_duration_s)

            logger.info("Phase 15: Response probability over time")
            self._phase_response_probability_bins()
            self._wait(self.rest_duration_s)

            logger.info("Phase 16: Burst rate analysis")
            self._phase_burst_rate()
            self._wait(self.rest_duration_s)

            logger.info("Phase 17: Cross-correlograms")
            self._phase_cross_correlograms()
            self._wait(self.rest_duration_s)

            logger.info("Phase 18: First-spike latency distributions")
            self._phase_latency_distributions()

            recording_stop = datetime_now()

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _make_stim_param(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity = StimPolarity.NegativeFirst,
        trigger_key: int = 0,
        shape: StimShape = StimShape.Biphasic,
    ) -> StimParam:
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
        stim.stim_shape = shape
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
        return stim

    def _fire_trigger(self, trigger_key: int = 0) -> None:
        pattern = np.zeros(16, dtype=np.uint8)
        pattern[trigger_key] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.05)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

    def _stimulate_single(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity = StimPolarity.NegativeFirst,
        trigger_key: int = 0,
        post_stim_wait_s: float = 0.3,
        phase_label: str = "general",
        shape: StimShape = StimShape.Biphasic,
    ) -> pd.DataFrame:
        amplitude_ua = min(abs(amplitude_ua), 4.0)
        duration_us = min(abs(duration_us), 400.0)

        stim = self._make_stim_param(electrode_idx, amplitude_ua, duration_us, polarity, trigger_key, shape)
        self.intan.send_stimparam([stim])
        self._fire_trigger(trigger_key)

        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            polarity=polarity.name,
            phase=phase_label,
            timestamp_utc=datetime_now().isoformat(),
            trigger_key=trigger_key,
        ))

        self._wait(post_stim_wait_s)

        query_stop = datetime_now()
        query_start = query_stop - timedelta(seconds=post_stim_wait_s + self.response_window_ms / 1000.0 + 0.1)
        try:
            spike_df = self.database.get_spike_event(query_start, query_stop, self.experiment.exp_name)
        except Exception as exc:
            logger.warning("Spike query failed: %s", exc)
            spike_df = pd.DataFrame()
        return spike_df

    def _count_spikes_in_window(
        self,
        spike_df: pd.DataFrame,
        resp_electrode: int,
        window_ms: float = 50.0,
        stim_time: Optional[datetime] = None,
    ) -> int:
        if spike_df is None or spike_df.empty:
            return 0
        channel_col = None
        for col in spike_df.columns:
            if col.lower() in ("channel", "index", "electrode"):
                channel_col = col
                break
        if channel_col is None:
            return len(spike_df)
        sub = spike_df[spike_df[channel_col] == resp_electrode]
        return len(sub)

    def _get_polarity(self, polarity_str: str) -> StimPolarity:
        if polarity_str == "PositiveFirst":
            return StimPolarity.PositiveFirst
        return StimPolarity.NegativeFirst

    def _phase_electrode_scan(self) -> None:
        all_electrodes = list(range(32))
        scan_results = {}

        for elec in all_electrodes:
            elec_results = {}
            for amp in self.scan_amplitudes:
                dur = self.duration_us
                if amp * dur > 4.0 * 400.0:
                    dur = min(400.0, (4.0 * 400.0) / amp)
                spike_counts = []
                for rep in range(self.scan_repeats):
                    spike_df = self._stimulate_single(
                        electrode_idx=elec,
                        amplitude_ua=amp,
                        duration_us=dur,
                        polarity=StimPolarity.PositiveFirst,
                        post_stim_wait_s=0.5,
                        phase_label="electrode_scan",
                    )
                    count = len(spike_df) if not spike_df.empty else 0
                    spike_counts.append(count)
                    self._wait(0.2)
                elec_results[amp] = {
                    "mean_spikes": float(np.mean(spike_counts)),
                    "total_spikes": int(np.sum(spike_counts)),
                    "spike_counts": spike_counts,
                }
            scan_results[elec] = elec_results

        self._scan_results = scan_results
        logger.info("Electrode scan complete: %d electrodes scanned", len(all_electrodes))

    def _identify_top_pairs(self) -> None:
        pair_scores = []
        for stim_e, resp_e, amp, dur, pol_str in self._known_pairs:
            pol = self._get_polarity(pol_str)
            spike_counts = []
            for rep in range(self.scan_repeats):
                spike_df = self._stimulate_single(
                    electrode_idx=stim_e,
                    amplitude_ua=amp,
                    duration_us=dur,
                    polarity=pol,
                    post_stim_wait_s=0.5,
                    phase_label="pair_identification",
                )
                count = self._count_spikes_in_window(spike_df, resp_e)
                spike_counts.append(count)
                self._wait(0.2)
            score = float(np.mean(spike_counts))
            pair_scores.append((stim_e, resp_e, amp, dur, pol_str, score))

        pair_scores.sort(key=lambda x: x[5], reverse=True)
        self._top_pairs = [(s, r, a, d, p) for s, r, a, d, p, _ in pair_scores[:self.top_n_pairs]]
        logger.info("Top %d pairs identified: %s", self.top_n_pairs, self._top_pairs)

    def _phase_spontaneous_recording(self, label: str) -> None:
        logger.info("Spontaneous recording: %s for %.1f s", label, self.spontaneous_duration_s)
        start = datetime_now()
        self._wait(self.spontaneous_duration_s)
        stop = datetime_now()
        try:
            spike_df = self.database.get_spike_event(start, stop, self.experiment.exp_name)
        except Exception as exc:
            logger.warning("Spontaneous spike query failed: %s", exc)
            spike_df = pd.DataFrame()

        total_spikes = len(spike_df) if not spike_df.empty else 0
        duration_s = (stop - start).total_seconds()
        firing_rate = total_spikes / max(duration_s, 1.0)

        burst_rate = self._compute_burst_rate_from_df(spike_df, duration_s)

        self._spontaneous_results[label] = {
            "start_utc": start.isoformat(),
            "stop_utc": stop.isoformat(),
            "duration_s": duration_s,
            "total_spikes": total_spikes,
            "mean_firing_rate_hz": firing_rate,
            "burst_rate_per_min": burst_rate,
        }

    def _compute_burst_rate_from_df(self, spike_df: pd.DataFrame, duration_s: float) -> float:
        if spike_df is None or spike_df.empty:
            return 0.0
        time_col = None
        for col in spike_df.columns:
            if col.lower() in ("time", "_time", "timestamp"):
                time_col = col
                break
        if time_col is None:
            return 0.0
        try:
            times = pd.to_datetime(spike_df[time_col], utc=True)
            times_s = times.astype(np.int64) / 1e9
            times_s = np.sort(times_s.values)
            if len(times_s) < 2:
                return 0.0
            isi = np.diff(times_s)
            burst_threshold = 0.1
            burst_starts = np.sum(isi < burst_threshold)
            burst_rate_per_min = burst_starts / max(duration_s / 60.0, 1.0)
            return float(burst_rate_per_min)
        except Exception:
            return 0.0

    def _phase_connectivity_matrix(self, label: str) -> None:
        pairs = self._top_pairs if self._top_pairs else [(s, r, a, d, p) for s, r, a, d, p in self._known_pairs[:5]]
        matrix = {}
        for stim_e, resp_e, amp, dur, pol_str in pairs:
            pol = self._get_polarity(pol_str)
            spike_counts = []
            for rep in range(self.scan_repeats):
                spike_df = self._stimulate_single(
                    electrode_idx=stim_e,
                    amplitude_ua=amp,
                    duration_us=dur,
                    polarity=pol,
                    post_stim_wait_s=0.5,
                    phase_label=f"connectivity_{label}",
                )
                count = self._count_spikes_in_window(spike_df, resp_e)
                spike_counts.append(count)
                self._wait(0.2)
            key = f"{stim_e}->{resp_e}"
            matrix[key] = {
                "mean_response": float(np.mean(spike_counts)),
                "response_rate": float(np.mean([1 if c > 0 else 0 for c in spike_counts])),
            }

        if label == "before":
            self._connectivity_before = matrix
        else:
            self._connectivity_after = matrix
        logger.info("Connectivity matrix (%s) computed for %d pairs", label, len(matrix))

    def _phase_paired_pulse_facilitation(self) -> None:
        if not self._top_pairs:
            pairs_to_use = [(s, r, a, d, p) for s, r, a, d, p in self._known_pairs[:3]]
        else:
            pairs_to_use = self._top_pairs[:3]

        ppf_results = {}
        for stim_e, resp_e, amp, dur, pol_str in pairs_to_use:
            pol = self._get_polarity(pol_str)
            pair_key = f"{stim_e}->{resp_e}"
            ppf_results[pair_key] = {}

            for isi_ms in self.ppf_intervals_ms:
                isi_s = isi_ms / 1000.0
                responses_p1 = []
                responses_p2 = []

                for rep in range(self.ppf_repeats):
                    stim1 = self._make_stim_param(stim_e, amp, dur, pol, trigger_key=0)
                    self.intan.send_stimparam([stim1])
                    self._fire_trigger(0)
                    self._stimulation_log.append(StimulationRecord(
                        electrode_idx=stim_e,
                        amplitude_ua=amp,
                        duration_us=dur,
                        polarity=pol.name,
                        phase="ppf_pulse1",
                        timestamp_utc=datetime_now().isoformat(),
                        trigger_key=0,
                        extra={"isi_ms": isi_ms},
                    ))

                    self._wait(isi_s)

                    stim2 = self._make_stim_param(stim_e, amp, dur, pol, trigger_key=1)
                    self.intan.send_stimparam([stim2])
                    self._fire_trigger(1)
                    self._stimulation_log.append(StimulationRecord(
                        electrode_idx=stim_e,
                        amplitude_ua=amp,
                        duration_us=dur,
                        polarity=pol.name,
                        phase="ppf_pulse2",
                        timestamp_utc=datetime_now().isoformat(),
                        trigger_key=1,
                        extra={"isi_ms": isi_ms},
                    ))

                    self._wait(0.5)

                    query_stop = datetime_now()
                    query_start = query_stop - timedelta(seconds=0.5 + isi_s + 0.2)
                    try:
                        spike_df = self.database.get_spike_event(query_start, query_stop, self.experiment.exp_name)
                    except Exception:
                        spike_df = pd.DataFrame()

                    c1 = self._count_spikes_in_window(spike_df, resp_e)
                    c2 = self._count_spikes_in_window(spike_df, resp_e)
                    responses_p1.append(c1)
                    responses_p2.append(c2)
                    self._wait(1.0)

                ratio = float(np.mean(responses_p2)) / max(float(np.mean(responses_p1)), 0.001)
                ppf_results[pair_key][isi_ms] = {
                    "mean_p1": float(np.mean(responses_p1)),
                    "mean_p2": float(np.mean(responses_p2)),
                    "ppf_ratio": ratio,
                }

        self._ppf_results = ppf_results
        logger.info("PPF complete for %d pairs x %d ISIs", len(pairs_to_use), len(self.ppf_intervals_ms))

    def _phase_input_output_curve(self) -> None:
        if not self._top_pairs:
            pairs_to_use = [(s, r, a, d, p) for s, r, a, d, p in self._known_pairs[:2]]
        else:
            pairs_to_use = self._top_pairs[:2]

        io_results = {}
        for stim_e, resp_e, _, _, pol_str in pairs_to_use:
            pol = self._get_polarity(pol_str)
            pair_key = f"{stim_e}->{resp_e}"
            io_results[pair_key] = {}

            for amp in self.io_amplitudes:
                dur = self.duration_us
                if amp * dur > 4.0 * 400.0:
                    dur = min(400.0, (4.0 * 400.0) / amp)
                spike_counts = []
                latencies = []

                for rep in range(self.io_repeats):
                    spike_df = self._stimulate_single(
                        electrode_idx=stim_e,
                        amplitude_ua=amp,
                        duration_us=dur,
                        polarity=pol,
                        post_stim_wait_s=0.5,
                        phase_label="io_curve",
                    )
                    count = self._count_spikes_in_window(spike_df, resp_e)
                    spike_counts.append(count)
                    self._wait(0.5)

                io_results[pair_key][amp] = {
                    "mean_spikes": float(np.mean(spike_counts)),
                    "response_rate": float(np.mean([1 if c > 0 else 0 for c in spike_counts])),
                    "duration_us": dur,
                }

        self._io_results = io_results
        logger.info("IO curve complete for %d pairs x %d amplitudes", len(pairs_to_use), len(self.io_amplitudes))

    def _phase_frequency_response(self) -> None:
        if not self._top_pairs:
            pairs_to_use = [(s, r, a, d, p) for s, r, a, d, p in self._known_pairs[:2]]
        else:
            pairs_to_use = self._top_pairs[:2]

        freq_results = {}
        for stim_e, resp_e, amp, dur, pol_str in pairs_to_use:
            pol = self._get_polarity(pol_str)
            pair_key = f"{stim_e}->{resp_e}"
            freq_results[pair_key] = {}

            for freq_hz in self.freq_response_hz:
                isi_s = 1.0 / freq_hz
                spike_counts = []

                for rep in range(self.freq_repeats):
                    spike_df = self._stimulate_single(
                        electrode_idx=stim_e,
                        amplitude_ua=amp,
                        duration_us=dur,
                        polarity=pol,
                        post_stim_wait_s=max(isi_s * 0.8, 0.05),
                        phase_label="freq_response",
                    )
                    count = self._count_spikes_in_window(spike_df, resp_e)
                    spike_counts.append(count)
                    remaining = max(isi_s - 0.05 - max(isi_s * 0.8, 0.05), 0.0)
                    self._wait(remaining)

                freq_results[pair_key][freq_hz] = {
                    "mean_spikes": float(np.mean(spike_counts)),
                    "response_rate": float(np.mean([1 if c > 0 else 0 for c in spike_counts])),
                }

        self._freq_results = freq_results
        logger.info("Frequency response complete")

    def _phase_biphasic_vs_monophasic(self) -> None:
        if not self._top_pairs:
            pairs_to_use = [(s, r, a, d, p) for s, r, a, d, p in self._known_pairs[:2]]
        else:
            pairs_to_use = self._top_pairs[:2]

        results = {}
        for stim_e, resp_e, amp, dur, pol_str in pairs_to_use:
            pol = self._get_polarity(pol_str)
            pair_key = f"{stim_e}->{resp_e}"
            results[pair_key] = {}

            for shape_name, shape in [("biphasic", StimShape.Biphasic), ("biphasic_interphase", StimShape.BiphasicWithInterphaseDelay)]:
                spike_counts = []
                for rep in range(self.scan_repeats):
                    spike_df = self._stimulate_single(
                        electrode_idx=stim_e,
                        amplitude_ua=amp,
                        duration_us=dur,
                        polarity=pol,
                        post_stim_wait_s=0.5,
                        phase_label=f"biphasic_vs_mono_{shape_name}",
                        shape=shape,
                    )
                    count = self._count_spikes_in_window(spike_df, resp_e)
                    spike_counts.append(count)
                    self._wait(0.3)

                results[pair_key][shape_name] = {
                    "mean_spikes": float(np.mean(spike_counts)),
                    "response_rate": float(np.mean([1 if c > 0 else 0 for c in spike_counts])),
                }

        self._biphasic_vs_mono_results = results
        logger.info("Biphasic vs monophasic comparison complete")

    def _phase_polarity_comparison(self) -> None:
        if not self._top_pairs:
            pairs_to_use = [(s, r, a, d, p) for s, r, a, d, p in self._known_pairs[:2]]
        else:
            pairs_to_use = self._top_pairs[:2]

        results = {}
        for stim_e, resp_e, amp, dur, _ in pairs_to_use:
            pair_key = f"{stim_e}->{resp_e}"
            results[pair_key] = {}

            for pol_name, pol in [("PositiveFirst", StimPolarity.PositiveFirst), ("NegativeFirst", StimPolarity.NegativeFirst)]:
                spike_counts = []
                for rep in range(self.scan_repeats):
                    spike_df = self._stimulate_single(
                        electrode_idx=stim_e,
                        amplitude_ua=amp,
                        duration_us=dur,
                        polarity=pol,
                        post_stim_wait_s=0.5,
                        phase_label=f"polarity_{pol_name}",
                    )
                    count = self._count_spikes_in_window(spike_df, resp_e)
                    spike_counts.append(count)
                    self._wait(0.3)

                results[pair_key][pol_name] = {
                    "mean_spikes": float(np.mean(spike_counts)),
                    "response_rate": float(np.mean([1 if c > 0 else 0 for c in spike_counts])),
                }

        self._polarity_results = results
        logger.info("Polarity comparison complete")

    def _phase_stdp_induction(self, mode: str) -> None:
        if not self._top_pairs:
            pairs_to_use = [(s, r, a, d, p) for s, r, a, d, p in self._known_pairs[:2]]
        else:
            pairs_to_use = self._top_pairs[:2]

        results = {}
        for stim_e, resp_e, amp, dur, pol_str in pairs_to_use:
            pol = self._get_polarity(pol_str)
            pair_key = f"{stim_e}->{resp_e}"
            results[pair_key] = {}

            for delay_ms in self.stdp_delays_ms:
                delay_s = abs(delay_ms) / 1000.0
                spike_counts_pre = []
                spike_counts_post = []

                for rep in range(self.stdp_repeats):
                    if mode == "pre_post":
                        pre_e = stim_e
                        post_e = resp_e
                        if delay_ms > 0:
                            spike_df_pre = self._stimulate_single(
                                electrode_idx=pre_e,
                                amplitude_ua=amp,
                                duration_us=dur,
                                polarity=pol,
                                post_stim_wait_s=delay_s,
                                phase_label=f"stdp_{mode}_pre",
                            )
                            spike_df_post = self._stimulate_single(
                                electrode_idx=post_e,
                                amplitude_ua=amp,
                                duration_us=dur,
                                polarity=pol,
                                post_stim_wait_s=0.3,
                                phase_label=f"stdp_{mode}_post",
                            )
                        else:
                            spike_df_post = self._stimulate_single(
                                electrode_idx=post_e,
                                amplitude_ua=amp,
                                duration_us=dur,
                                polarity=pol,
                                post_stim_wait_s=delay_s,
                                phase_label=f"stdp_{mode}_post",
                            )
                            spike_df_pre = self._stimulate_single(
                                electrode_idx=pre_e,
                                amplitude_ua=amp,
                                duration_us=dur,
                                polarity=pol,
                                post_stim_wait_s=0.3,
                                phase_label=f"stdp_{mode}_pre",
                            )
                    else:
                        pre_e = resp_e
                        post_e = stim_e
                        if delay_ms > 0:
                            spike_df_pre = self._stimulate_single(
                                electrode_idx=pre_e,
                                amplitude_ua=amp,
                                duration_us=dur,
                                polarity=pol,
                                post_stim_wait_s=delay_s,
                                phase_label=f"stdp_{mode}_pre",
                            )
                            spike_df_post = self._stimulate_single(
                                electrode_idx=post_e,
                                amplitude_ua=amp,
                                duration_us=dur,
                                polarity=pol,
                                post_stim_wait_s=0.3,
                                phase_label=f"stdp_{mode}_post",
                            )
                        else:
                            spike_df_post = self._stimulate_single(
                                electrode_idx=post_e,
                                amplitude_ua=amp,
                                duration_us=dur,
                                polarity=pol,
                                post_stim_wait_s=delay_s,
                                phase_label=f"stdp_{mode}_post",
                            )
                            spike_df_pre = self._stimulate_single(
                                electrode_idx=pre_e,
                                amplitude_ua=amp,
                                duration_us=dur,
                                polarity=pol,
                                post_stim_wait_s=0.3,
                                phase_label=f"stdp_{mode}_pre",
                            )

                    c_pre = self._count_spikes_in_window(spike_df_pre, pre_e)
                    c_post = self._count_spikes_in_window(spike_df_post, post_e)
                    spike_counts_pre.append(c_pre)
                    spike_counts_post.append(c_post)
                    self._wait(0.5)

                results[pair_key][delay_ms] = {
                    "mean_pre_spikes": float(np.mean(spike_counts_pre)),
                    "mean_post_spikes": float(np.mean(spike_counts_post)),
                    "mode": mode,
                }

        if mode == "pre_post":
            self._stdp_pre_post_results = results
        else:
            self._stdp_post_pre_results = results
        logger.info("STDP induction (%s) complete", mode)

    def _compare_ltp_ltd(self) -> None:
        comparison = {}
        positive_delays = [d for d in self.stdp_delays_ms if d > 0]
        negative_delays = [d for d in self.stdp_delays_ms if d < 0]

        for pair_key in self._stdp_pre_post_results:
            pre_post = self._stdp_pre_post_results.get(pair_key, {})
            post_pre = self._stdp_post_pre_results.get(pair_key, {})

            ltp_scores = []
            for d in positive_delays:
                if d in pre_post:
                    ltp_scores.append(pre_post[d].get("mean_post_spikes", 0.0))

            ltd_scores = []
            for d in negative_delays:
                if d in pre_post:
                    ltd_scores.append(pre_post[d].get("mean_post_spikes", 0.0))

            mean_ltp = float(np.mean(ltp_scores)) if ltp_scores else 0.0
            mean_ltd = float(np.mean(ltd_scores)) if ltd_scores else 0.0

            wasserstein = self._compute_wasserstein(ltp_scores, ltd_scores)

            comparison[pair_key] = {
                "mean_ltp_response": mean_ltp,
                "mean_ltd_response": mean_ltd,
                "ltp_minus_ltd": mean_ltp - mean_ltd,
                "wasserstein_distance": wasserstein,
                "ltp_scores": ltp_scores,
                "ltd_scores": ltd_scores,
            }

        self._ltp_ltd_comparison = comparison
        logger.info("LTP vs LTD comparison complete for %d pairs", len(comparison))

    def _compute_wasserstein(self, dist1: List[float], dist2: List[float]) -> float:
        if not dist1 or not dist2:
            return 0.0
        a = sorted(dist1)
        b = sorted(dist2)
        n = max(len(a), len(b))
        if n == 0:
            return 0.0
        a_interp = [a[int(i * (len(a) - 1) / max(n - 1, 1))] for i in range(n)]
        b_interp = [b[int(i * (len(b) - 1) / max(n - 1, 1))] for i in range(n)]
        return float(np.mean(np.abs(np.array(a_interp) - np.array(b_interp))))

    def _phase_response_probability_bins(self) -> None:
        if not self._top_pairs:
            pairs_to_use = [(s, r, a, d, p) for s, r, a, d, p in self._known_pairs[:2]]
        else:
            pairs_to_use = self._top_pairs[:2]

        bin_duration_s = 60.0
        n_bins = 3
        results = {}

        for stim_e, resp_e, amp, dur, pol_str in pairs_to_use:
            pol = self._get_polarity(pol_str)
            pair_key = f"{stim_e}->{resp_e}"
            bin_results = []

            for bin_idx in range(n_bins):
                bin_start = datetime_now()
                spike_counts = []
                stims_per_bin = 5
                for rep in range(stims_per_bin):
                    spike_df = self._stimulate_single(
                        electrode_idx=stim_e,
                        amplitude_ua=amp,
                        duration_us=dur,
                        polarity=pol,
                        post_stim_wait_s=0.5,
                        phase_label=f"resp_prob_bin{bin_idx}",
                    )
                    count = self._count_spikes_in_window(spike_df, resp_e)
                    spike_counts.append(count)
                    remaining = (bin_duration_s / stims_per_bin) - 0.5
                    self._wait(max(remaining, 0.1))

                bin_end = datetime_now()
                bin_results.append({
                    "bin_index": bin_idx,
                    "start_utc": bin_start.isoformat(),
                    "stop_utc": bin_end.isoformat(),
                    "response_probability": float(np.mean([1 if c > 0 else 0 for c in spike_counts])),
                    "mean_spikes": float(np.mean(spike_counts)),
                })

            results[pair_key] = bin_results

        self._response_prob_bins = results
        logger.info("Response probability bins complete")

    def _phase_burst_rate(self) -> None:
        duration_s = 30.0
        start = datetime_now()
        self._wait(duration_s)
        stop = datetime_now()

        try:
            spike_df = self.database.get_spike_event(start, stop, self.experiment.exp_name)
        except Exception as exc:
            logger.warning("Burst rate spike query failed: %s", exc)
            spike_df = pd.DataFrame()

        burst_rate = self._compute_burst_rate_from_df(spike_df, duration_s)
        self._burst_rate_results = {
            "start_utc": start.isoformat(),
            "stop_utc": stop.isoformat(),
            "duration_s": duration_s,
            "burst_rate_per_min": burst_rate,
            "total_spikes": len(spike_df) if not spike_df.empty else 0,
        }
        logger.info("Burst rate: %.2f bursts/min", burst_rate)

    def _phase_cross_correlograms(self) -> None:
        if not self._top_pairs:
            pairs_to_use = [(s, r, a, d, p) for s, r, a, d, p in self._known_pairs[:3]]
        else:
            pairs_to_use = self._top_pairs[:3]

        ccg_results = {}
        window_s = 30.0
        start = datetime_now()
        self._wait(window_s)
        stop = datetime_now()

        try:
            spike_df = self.database.get_spike_event(start, stop, self.experiment.exp_name)
        except Exception as exc:
            logger.warning("CCG spike query failed: %s", exc)
            spike_df = pd.DataFrame()

        for stim_e, resp_e, _, _, _ in pairs_to_use:
            pair_key = f"{stim_e}->{resp_e}"
            ccg = self._compute_ccg(spike_df, stim_e, resp_e)
            ccg_results[pair_key] = ccg

        self._cross_correlograms = ccg_results
        logger.info("Cross-correlograms computed for %d pairs", len(pairs_to_use))

    def _compute_ccg(self, spike_df: pd.DataFrame, elec_a: int, elec_b: int, max_lag_ms: float = 100.0, bin_ms: float = 5.0) -> Dict[str, Any]:
        if spike_df is None or spike_df.empty:
            return {"bins": [], "counts": [], "peak_lag_ms": None}

        channel_col = None
        time_col = None
        for col in spike_df.columns:
            if col.lower() in ("channel", "index", "electrode"):
                channel_col = col
            if col.lower() in ("time", "_time", "timestamp"):
                time_col = col

        if channel_col is None or time_col is None:
            return {"bins": [], "counts": [], "peak_lag_ms": None}

        try:
            times_a = pd.to_datetime(spike_df[spike_df[channel_col] == elec_a][time_col], utc=True)
            times_b = pd.to_datetime(spike_df[spike_df[channel_col] == elec_b][time_col], utc=True)
            times_a_s = np.sort(times_a.astype(np.int64).values / 1e9)
            times_b_s = np.sort(times_b.astype(np.int64).values / 1e9)

            if len(times_a_s) == 0 or len(times_b_s) == 0:
                return {"bins": [], "counts": [], "peak_lag_ms": None}

            max_lag_s = max_lag_ms / 1000.0
            bin_s = bin_ms / 1000.0
            bins = np.arange(-max_lag_s, max_lag_s + bin_s, bin_s)
            counts = np.zeros(len(bins) - 1, dtype=int)

            for t_a in times_a_s:
                lags = times_b_s - t_a
                lags = lags[(lags >= -max_lag_s) & (lags <= max_lag_s)]
                hist, _ = np.histogram(lags, bins=bins)
                counts += hist

            peak_idx = int(np.argmax(counts))
            peak_lag_ms = float((bins[peak_idx] + bins[peak_idx + 1]) / 2.0 * 1000.0)

            return {
                "bins_ms": [float(b * 1000.0) for b in bins.tolist()],
                "counts": counts.tolist(),
                "peak_lag_ms": peak_lag_ms,
            }
        except Exception as exc:
            logger.warning("CCG computation failed: %s", exc)
            return {"bins": [], "counts": [], "peak_lag_ms": None}

    def _phase_latency_distributions(self) -> None:
        if not self._top_pairs:
            pairs_to_use = [(s, r, a, d, p) for s, r, a, d, p in self._known_pairs[:3]]
        else:
            pairs_to_use = self._top_pairs[:3]

        latency_results = {}
        n_trials = 10

        for stim_e, resp_e, amp, dur, pol_str in pairs_to_use:
            pol = self._get_polarity(pol_str)
            pair_key = f"{stim_e}->{resp_e}"
            latencies_ms = []

            for rep in range(n_trials):
                stim_time = datetime_now()
                spike_df = self._stimulate_single(
                    electrode_idx=stim_e,
                    amplitude_ua=amp,
                    duration_us=dur,
                    polarity=pol,
                    post_stim_wait_s=0.1,
                    phase_label="latency_dist",
                )
                self._wait(0.3)

                if spike_df is not None and not spike_df.empty:
                    channel_col = None
                    time_col = None
                    for col in spike_df.columns:
                        if col.lower() in ("channel", "index", "electrode"):
                            channel_col = col
                        if col.lower() in ("time", "_time", "timestamp"):
                            time_col = col

                    if channel_col is not None and time_col is not None:
                        sub = spike_df[spike_df[channel_col] == resp_e]
                        if not sub.empty:
                            try:
                                t_spikes = pd.to_datetime(sub[time_col], utc=True)
                                stim_ts = stim_time.timestamp()
                                for t in t_spikes:
                                    lat_ms = (t.timestamp() - stim_ts) * 1000.0
                                    if 0 < lat_ms < self.response_window_ms:
                                        latencies_ms.append(lat_ms)
                            except Exception:
                                pass

            if latencies_ms:
                latency_arr = np.array(latencies_ms)
                mean_lat = float(np.mean(latency_arr))
                std_lat = float(np.std(latency_arr))
                median_lat = float(np.median(latency_arr))
                wasserstein_vs_uniform = self._compute_wasserstein(
                    latencies_ms,
                    list(np.linspace(0, self.response_window_ms, len(latencies_ms)))
                )
            else:
                mean_lat = 0.0
                std_lat = 0.0
                median_lat = 0.0
                wasserstein_vs_uniform = 0.0

            latency_results[pair_key] = {
                "n_trials": n_trials,
                "n_spikes": len(latencies_ms),
                "mean_latency_ms": mean_lat,
                "std_latency_ms": std_lat,
                "median_latency_ms": median_lat,
                "latencies_ms": latencies_ms,
                "wasserstein_vs_uniform": wasserstein_vs_uniform,
            }

        self._latency_distributions = latency_results
        logger.info("Latency distributions computed for %d pairs", len(pairs_to_use))

    def _save_all(self, recording_start: datetime, recording_stop: datetime) -> None:
        fs_name = getattr(self.experiment, "exp_name", "unknown")
        saver = DataSaver(self._output_dir, fs_name)

        saver.save_stimulation_log(self._stimulation_log)

        try:
            spike_df = self.database.get_spike_event(recording_start, recording_stop, fs_name)
        except Exception as exc:
            logger.warning("Failed to fetch spike events: %s", exc)
            spike_df = pd.DataFrame()
        saver.save_spike_events(spike_df)

        try:
            trigger_df = self.database.get_all_triggers(recording_start, recording_stop)
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
            "total_spike_events": len(spike_df) if not spike_df.empty else 0,
            "total_triggers": len(trigger_df) if not trigger_df.empty else 0,
            "top_pairs": self._top_pairs,
            "scan_results_electrodes": len(self._scan_results),
            "ppf_pairs": len(self._ppf_results),
            "io_pairs": len(self._io_results),
            "freq_pairs": len(self._freq_results),
            "stdp_pre_post_pairs": len(self._stdp_pre_post_results),
            "stdp_post_pre_pairs": len(self._stdp_post_pre_results),
            "ltp_ltd_pairs": len(self._ltp_ltd_comparison),
            "connectivity_before_pairs": len(self._connectivity_before),
            "connectivity_after_pairs": len(self._connectivity_after),
            "latency_pairs": len(self._latency_distributions),
            "ccg_pairs": len(self._cross_correlograms),
            "spontaneous_phases": list(self._spontaneous_results.keys()),
        }
        saver.save_summary(summary)

        waveform_records = self._fetch_spike_waveforms(fs_name, spike_df, trigger_df, recording_start, recording_stop)
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
        if spike_df is None or spike_df.empty:
            return waveform_records

        channel_col = None
        for col in spike_df.columns:
            if col.lower() in ("channel", "index", "electrode"):
                channel_col = col
                break

        if channel_col is None:
            logger.warning("Cannot determine electrode column for waveform fetch")
            return waveform_records

        unique_electrodes = spike_df[channel_col].unique()
        for electrode_idx in unique_electrodes:
            try:
                raw_df = self.database.get_raw_spike(recording_start, recording_stop, int(electrode_idx))
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
        duration_s = (recording_stop - recording_start).total_seconds()

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": duration_s,
            "total_stimulations": len(self._stimulation_log),
            "top_pairs": self._top_pairs,
            "scan_results": {str(k): v for k, v in self._scan_results.items()},
            "ppf_results": self._ppf_results,
            "io_results": self._io_results,
            "freq_results": self._freq_results,
            "stdp_pre_post_results": self._stdp_pre_post_results,
            "stdp_post_pre_results": self._stdp_post_pre_results,
            "ltp_ltd_comparison": self._ltp_ltd_comparison,
            "spontaneous_results": self._spontaneous_results,
            "connectivity_before": self._connectivity_before,
            "connectivity_after": self._connectivity_after,
            "response_prob_bins": self._response_prob_bins,
            "burst_rate_results": self._burst_rate_results,
            "cross_correlograms": self._cross_correlograms,
            "latency_distributions": self._latency_distributions,
            "biphasic_vs_mono_results": self._biphasic_vs_mono_results,
            "polarity_results": self._polarity_results,
        }

        conn_changes = {}
        for key in self._connectivity_before:
            if key in self._connectivity_after:
                before_rate = self._connectivity_before[key].get("response_rate", 0.0)
                after_rate = self._connectivity_after[key].get("response_rate", 0.0)
                conn_changes[key] = {
                    "before": before_rate,
                    "after": after_rate,
                    "delta": after_rate - before_rate,
                }
        summary["connectivity_changes"] = conn_changes

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
