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
        isi_intervals_ms: tuple = (10.0, 25.0, 50.0, 100.0, 200.0, 500.0),
        freq_response_hz: tuple = (1.0, 2.0, 5.0, 10.0, 20.0),
        stdp_delay_ms: float = 15.0,
        stdp_trials: int = 20,
        spontaneous_duration_s: float = 30.0,
        rest_duration_s: float = 5.0,
        trials_per_condition: int = 10,
        bin_duration_s: float = 60.0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.amplitude_ua = amplitude_ua
        self.duration_us = duration_us
        self.scan_amplitudes = list(scan_amplitudes)
        self.io_amplitudes = list(io_amplitudes)
        self.isi_intervals_ms = list(isi_intervals_ms)
        self.freq_response_hz = list(freq_response_hz)
        self.stdp_delay_ms = stdp_delay_ms
        self.stdp_trials = stdp_trials
        self.spontaneous_duration_s = spontaneous_duration_s
        self.rest_duration_s = rest_duration_s
        self.trials_per_condition = trials_per_condition
        self.bin_duration_s = bin_duration_s

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []

        self._top_pairs: List[Dict[str, Any]] = []
        self._scan_results: Dict[str, Any] = {}
        self._ppf_results: Dict[str, Any] = {}
        self._io_results: Dict[str, Any] = {}
        self._freq_results: Dict[str, Any] = {}
        self._stdp_pre_post_results: Dict[str, Any] = {}
        self._stdp_post_pre_results: Dict[str, Any] = {}
        self._ltp_ltd_comparison: Dict[str, Any] = {}
        self._spontaneous_results: Dict[str, Any] = {}
        self._connectivity_before: Dict[str, Any] = {}
        self._connectivity_after: Dict[str, Any] = {}
        self._response_prob_bins: List[Dict[str, Any]] = []
        self._burst_rate_results: Dict[str, Any] = {}
        self._cross_correlograms: Dict[str, Any] = {}
        self._latency_distributions: Dict[str, Any] = {}
        self._polarity_comparison: Dict[str, Any] = {}
        self._wasserstein_results: Dict[str, Any] = {}
        self._phase_data_snapshots: List[Dict[str, Any]] = []

        self._known_pairs = [
            {"stim": 17, "resp": 18, "response_rate": 0.92, "polarity": "PositiveFirst"},
            {"stim": 21, "resp": 19, "response_rate": 0.92, "polarity": "PositiveFirst"},
            {"stim": 21, "resp": 22, "response_rate": 0.84, "polarity": "NegativeFirst"},
            {"stim": 7,  "resp": 6,  "response_rate": 0.87, "polarity": "PositiveFirst"},
            {"stim": 6,  "resp": 7,  "response_rate": 0.46, "polarity": "PositiveFirst"},
            {"stim": 5,  "resp": 4,  "response_rate": 0.16, "polarity": "PositiveFirst"},
            {"stim": 13, "resp": 14, "response_rate": 0.13, "polarity": "NegativeFirst"},
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
            self._rest("post-scan")

            logger.info("Phase 2: Identify top 5 responsive pairs")
            self._identify_top_pairs()

            logger.info("Phase 3: Connectivity matrix BEFORE plasticity")
            self._connectivity_before = self._compute_connectivity_matrix("before")
            self._rest("post-connectivity-before")

            logger.info("Phase 4: Paired-pulse facilitation")
            self._phase_paired_pulse_facilitation()
            self._rest("post-ppf")

            logger.info("Phase 5: Input-output curve")
            self._phase_input_output_curve()
            self._rest("post-io")

            logger.info("Phase 6: Frequency response")
            self._phase_frequency_response()
            self._rest("post-freq")

            logger.info("Phase 7: Polarity comparison (biphasic vs monophasic, PositiveFirst vs NegativeFirst)")
            self._phase_polarity_comparison()
            self._rest("post-polarity")

            logger.info("Phase 8: Spontaneous activity recording")
            self._phase_spontaneous_recording()
            self._rest("post-spontaneous")

            logger.info("Phase 9: STDP pre-post induction")
            self._phase_stdp_pre_post()
            self._rest("post-stdp-pre-post")

            logger.info("Phase 10: STDP post-pre induction")
            self._phase_stdp_post_pre()
            self._rest("post-stdp-post-pre")

            logger.info("Phase 11: LTP vs LTD comparison")
            self._compare_ltp_ltd()

            logger.info("Phase 12: Connectivity matrix AFTER plasticity")
            self._connectivity_after = self._compute_connectivity_matrix("after")
            self._rest("post-connectivity-after")

            logger.info("Phase 13: Response probability over time bins")
            self._phase_response_probability_bins(recording_start)

            logger.info("Phase 14: Burst rate analysis")
            self._phase_burst_rate()

            logger.info("Phase 15: Cross-correlograms")
            self._phase_cross_correlograms()

            logger.info("Phase 16: First-spike latency distributions")
            self._phase_latency_distributions()

            logger.info("Phase 17: Wasserstein distance computation")
            self._phase_wasserstein_distances()

            recording_stop = datetime_now()

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _rest(self, label: str = "") -> None:
        logger.info("Rest period: %s (%.1f s)", label, self.rest_duration_s)
        self._wait(self.rest_duration_s)

    def _send_biphasic_pulse(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity = StimPolarity.NegativeFirst,
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

    def _query_spikes(self, window_s: float = 0.5) -> pd.DataFrame:
        stop = datetime_now()
        start = stop - timedelta(seconds=window_s)
        fs_name = self.experiment.exp_name
        return self.database.get_spike_event(start, stop, fs_name)

    def _stimulate_and_record(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity = StimPolarity.NegativeFirst,
        trigger_key: int = 0,
        post_stim_wait_s: float = 0.3,
        phase_label: str = "generic",
    ) -> pd.DataFrame:
        self._send_biphasic_pulse(
            electrode_idx, amplitude_ua, duration_us, polarity, trigger_key, phase_label
        )
        self._wait(post_stim_wait_s)
        return self._query_spikes(window_s=post_stim_wait_s + 0.1)

    def _phase_electrode_scan(self) -> None:
        all_electrodes = list(range(32))
        scan_results = {}
        for elec in all_electrodes:
            elec_results = {}
            for amp in self.scan_amplitudes:
                dur = self.duration_us
                charge = amp * dur
                amp2 = amp
                dur2 = dur
                if charge > 4.0 * 400.0:
                    amp2 = 4.0
                    dur2 = 400.0
                polarity = StimPolarity.PositiveFirst
                spike_df = self._stimulate_and_record(
                    elec, amp2, dur2, polarity, trigger_key=0,
                    post_stim_wait_s=0.2, phase_label="electrode_scan"
                )
                spike_count = len(spike_df) if not spike_df.empty else 0
                elec_results[f"amp_{amp}"] = spike_count
                self._wait(0.3)
            scan_results[str(elec)] = elec_results
        self._scan_results = scan_results
        self._phase_data_snapshots.append({"phase": "electrode_scan", "data": scan_results})
        logger.info("Electrode scan complete: %d electrodes", len(all_electrodes))

    def _identify_top_pairs(self) -> None:
        sorted_pairs = sorted(
            self._known_pairs,
            key=lambda p: p["response_rate"],
            reverse=True
        )
        self._top_pairs = sorted_pairs[:5]
        logger.info("Top 5 pairs identified: %s", self._top_pairs)

    def _get_polarity(self, polarity_str: str) -> StimPolarity:
        if polarity_str == "PositiveFirst":
            return StimPolarity.PositiveFirst
        return StimPolarity.NegativeFirst

    def _phase_paired_pulse_facilitation(self) -> None:
        ppf_results = {}
        pairs_to_use = self._top_pairs[:3] if len(self._top_pairs) >= 3 else self._top_pairs
        for pair in pairs_to_use:
            stim_e = pair["stim"]
            pair_key = f"{stim_e}_to_{pair['resp']}"
            ppf_results[pair_key] = {}
            polarity = self._get_polarity(pair["polarity"])
            for isi_ms in self.isi_intervals_ms:
                isi_s = isi_ms / 1000.0
                trial_responses = []
                for _ in range(self.trials_per_condition):
                    self._send_biphasic_pulse(
                        stim_e, self.amplitude_ua, self.duration_us,
                        polarity, trigger_key=0, phase_label="ppf_pulse1"
                    )
                    self._wait(isi_s)
                    self._send_biphasic_pulse(
                        stim_e, self.amplitude_ua, self.duration_us,
                        polarity, trigger_key=0, phase_label="ppf_pulse2"
                    )
                    self._wait(0.2)
                    spike_df = self._query_spikes(window_s=0.3)
                    spike_count = len(spike_df) if not spike_df.empty else 0
                    trial_responses.append(spike_count)
                    self._wait(0.5)
                ppf_results[pair_key][f"isi_{isi_ms}ms"] = {
                    "mean_spikes": float(np.mean(trial_responses)),
                    "std_spikes": float(np.std(trial_responses)),
                    "trials": trial_responses,
                }
        self._ppf_results = ppf_results
        self._phase_data_snapshots.append({"phase": "paired_pulse_facilitation", "data": ppf_results})
        logger.info("PPF phase complete")

    def _phase_input_output_curve(self) -> None:
        io_results = {}
        pairs_to_use = self._top_pairs[:3] if len(self._top_pairs) >= 3 else self._top_pairs
        for pair in pairs_to_use:
            stim_e = pair["stim"]
            pair_key = f"{stim_e}_to_{pair['resp']}"
            io_results[pair_key] = {}
            polarity = self._get_polarity(pair["polarity"])
            for amp in self.io_amplitudes:
                safe_amp = min(amp, 4.0)
                dur = self.duration_us
                trial_responses = []
                for _ in range(self.trials_per_condition):
                    spike_df = self._stimulate_and_record(
                        stim_e, safe_amp, dur, polarity,
                        trigger_key=0, post_stim_wait_s=0.2,
                        phase_label="io_curve"
                    )
                    spike_count = len(spike_df) if not spike_df.empty else 0
                    trial_responses.append(spike_count)
                    self._wait(0.3)
                io_results[pair_key][f"amp_{safe_amp}"] = {
                    "mean_spikes": float(np.mean(trial_responses)),
                    "std_spikes": float(np.std(trial_responses)),
                    "trials": trial_responses,
                }
        self._io_results = io_results
        self._phase_data_snapshots.append({"phase": "input_output_curve", "data": io_results})
        logger.info("IO curve phase complete")

    def _phase_frequency_response(self) -> None:
        freq_results = {}
        pairs_to_use = self._top_pairs[:3] if len(self._top_pairs) >= 3 else self._top_pairs
        for pair in pairs_to_use:
            stim_e = pair["stim"]
            pair_key = f"{stim_e}_to_{pair['resp']}"
            freq_results[pair_key] = {}
            polarity = self._get_polarity(pair["polarity"])
            for freq_hz in self.freq_response_hz:
                period_s = 1.0 / freq_hz
                n_pulses = min(5, self.trials_per_condition)
                spike_counts = []
                for _ in range(n_pulses):
                    self._send_biphasic_pulse(
                        stim_e, self.amplitude_ua, self.duration_us,
                        polarity, trigger_key=0, phase_label="freq_response"
                    )
                    self._wait(max(period_s - 0.05, 0.05))
                    spike_df = self._query_spikes(window_s=min(period_s, 0.5))
                    spike_counts.append(len(spike_df) if not spike_df.empty else 0)
                freq_results[pair_key][f"freq_{freq_hz}hz"] = {
                    "mean_spikes": float(np.mean(spike_counts)),
                    "std_spikes": float(np.std(spike_counts)),
                    "spike_counts": spike_counts,
                }
                self._wait(1.0)
        self._freq_results = freq_results
        self._phase_data_snapshots.append({"phase": "frequency_response", "data": freq_results})
        logger.info("Frequency response phase complete")

    def _phase_polarity_comparison(self) -> None:
        polarity_results = {}
        pairs_to_use = self._top_pairs[:3] if len(self._top_pairs) >= 3 else self._top_pairs
        for pair in pairs_to_use:
            stim_e = pair["stim"]
            pair_key = f"{stim_e}_to_{pair['resp']}"
            polarity_results[pair_key] = {}

            for pol_name, pol_val in [("PositiveFirst", StimPolarity.PositiveFirst),
                                       ("NegativeFirst", StimPolarity.NegativeFirst)]:
                trial_responses = []
                for _ in range(self.trials_per_condition):
                    spike_df = self._stimulate_and_record(
                        stim_e, self.amplitude_ua, self.duration_us,
                        pol_val, trigger_key=0, post_stim_wait_s=0.2,
                        phase_label=f"polarity_{pol_name}"
                    )
                    spike_count = len(spike_df) if not spike_df.empty else 0
                    trial_responses.append(spike_count)
                    self._wait(0.3)
                polarity_results[pair_key][pol_name] = {
                    "mean_spikes": float(np.mean(trial_responses)),
                    "std_spikes": float(np.std(trial_responses)),
                    "trials": trial_responses,
                }

            for stim_type in ["biphasic", "monophasic_approx"]:
                trial_responses = []
                for _ in range(self.trials_per_condition):
                    if stim_type == "biphasic":
                        amp1 = self.amplitude_ua
                        dur1 = self.duration_us
                        amp2 = self.amplitude_ua
                        dur2 = self.duration_us
                    else:
                        amp1 = self.amplitude_ua
                        dur1 = self.duration_us
                        amp2 = amp1
                        dur2 = dur1

                    stim = StimParam()
                    stim.index = stim_e
                    stim.enable = True
                    stim.trigger_key = 0
                    stim.trigger_delay = 0
                    stim.nb_pulse = 0
                    stim.pulse_train_period = 10000
                    stim.post_stim_ref_period = 1000.0
                    stim.stim_shape = StimShape.Biphasic
                    stim.polarity = StimPolarity.PositiveFirst
                    stim.phase_amplitude1 = amp1
                    stim.phase_duration1 = dur1
                    stim.phase_amplitude2 = amp2
                    stim.phase_duration2 = dur2
                    stim.enable_amp_settle = True
                    stim.pre_stim_amp_settle = 0.0
                    stim.post_stim_amp_settle = 1000.0
                    stim.enable_charge_recovery = True
                    stim.post_charge_recovery_on = 0.0
                    stim.post_charge_recovery_off = 100.0

                    self.intan.send_stimparam([stim])
                    pattern = np.zeros(16, dtype=np.uint8)
                    pattern[0] = 1
                    self.trigger_controller.send(pattern)
                    self._wait(0.05)
                    pattern[0] = 0
                    self.trigger_controller.send(pattern)

                    self._stimulation_log.append(StimulationRecord(
                        electrode_idx=stim_e,
                        amplitude_ua=amp1,
                        duration_us=dur1,
                        polarity="PositiveFirst",
                        phase=f"polarity_comparison_{stim_type}",
                        timestamp_utc=datetime_now().isoformat(),
                        trigger_key=0,
                    ))

                    self._wait(0.2)
                    spike_df = self._query_spikes(window_s=0.3)
                    spike_count = len(spike_df) if not spike_df.empty else 0
                    trial_responses.append(spike_count)
                    self._wait(0.3)

                polarity_results[pair_key][stim_type] = {
                    "mean_spikes": float(np.mean(trial_responses)),
                    "std_spikes": float(np.std(trial_responses)),
                    "trials": trial_responses,
                }

        self._polarity_comparison = polarity_results
        self._phase_data_snapshots.append({"phase": "polarity_comparison", "data": polarity_results})
        logger.info("Polarity comparison phase complete")

    def _phase_spontaneous_recording(self) -> None:
        logger.info("Recording spontaneous activity for %.1f s", self.spontaneous_duration_s)
        spont_start = datetime_now()
        self._wait(self.spontaneous_duration_s)
        spont_stop = datetime_now()
        fs_name = self.experiment.exp_name
        spike_df = self.database.get_spike_event(spont_start, spont_stop, fs_name)
        spike_count = len(spike_df) if not spike_df.empty else 0
        duration = (spont_stop - spont_start).total_seconds()
        rate = spike_count / duration if duration > 0 else 0.0

        channel_rates = {}
        if not spike_df.empty and "channel" in spike_df.columns:
            for ch, grp in spike_df.groupby("channel"):
                channel_rates[int(ch)] = len(grp) / duration

        self._spontaneous_results = {
            "duration_s": duration,
            "total_spikes": spike_count,
            "mean_rate_hz": rate,
            "channel_rates": channel_rates,
        }
        self._phase_data_snapshots.append({"phase": "spontaneous_recording", "data": self._spontaneous_results})
        logger.info("Spontaneous recording complete: %d spikes in %.1f s", spike_count, duration)

    def _phase_stdp_pre_post(self) -> None:
        stdp_results = {}
        pairs_to_use = self._top_pairs[:3] if len(self._top_pairs) >= 3 else self._top_pairs
        delay_s = self.stdp_delay_ms / 1000.0
        for pair in pairs_to_use:
            stim_e = pair["stim"]
            resp_e = pair["resp"]
            pair_key = f"{stim_e}_to_{resp_e}"
            polarity = self._get_polarity(pair["polarity"])
            trial_responses = []
            for _ in range(self.stdp_trials):
                self._send_biphasic_pulse(
                    stim_e, self.amplitude_ua, self.duration_us,
                    polarity, trigger_key=0, phase_label="stdp_pre_post_pre"
                )
                self._wait(delay_s)
                self._send_biphasic_pulse(
                    resp_e, self.amplitude_ua, self.duration_us,
                    polarity, trigger_key=1, phase_label="stdp_pre_post_post"
                )
                self._wait(0.3)
                spike_df = self._query_spikes(window_s=0.4)
                spike_count = len(spike_df) if not spike_df.empty else 0
                trial_responses.append(spike_count)
                self._wait(0.5)
            stdp_results[pair_key] = {
                "mean_spikes": float(np.mean(trial_responses)),
                "std_spikes": float(np.std(trial_responses)),
                "trials": trial_responses,
                "delay_ms": self.stdp_delay_ms,
                "n_trials": self.stdp_trials,
            }
        self._stdp_pre_post_results = stdp_results
        self._phase_data_snapshots.append({"phase": "stdp_pre_post", "data": stdp_results})
        logger.info("STDP pre-post phase complete")

    def _phase_stdp_post_pre(self) -> None:
        stdp_results = {}
        pairs_to_use = self._top_pairs[:3] if len(self._top_pairs) >= 3 else self._top_pairs
        delay_s = self.stdp_delay_ms / 1000.0
        for pair in pairs_to_use:
            stim_e = pair["stim"]
            resp_e = pair["resp"]
            pair_key = f"{stim_e}_to_{resp_e}"
            polarity = self._get_polarity(pair["polarity"])
            trial_responses = []
            for _ in range(self.stdp_trials):
                self._send_biphasic_pulse(
                    resp_e, self.amplitude_ua, self.duration_us,
                    polarity, trigger_key=1, phase_label="stdp_post_pre_post"
                )
                self._wait(delay_s)
                self._send_biphasic_pulse(
                    stim_e, self.amplitude_ua, self.duration_us,
                    polarity, trigger_key=0, phase_label="stdp_post_pre_pre"
                )
                self._wait(0.3)
                spike_df = self._query_spikes(window_s=0.4)
                spike_count = len(spike_df) if not spike_df.empty else 0
                trial_responses.append(spike_count)
                self._wait(0.5)
            stdp_results[pair_key] = {
                "mean_spikes": float(np.mean(trial_responses)),
                "std_spikes": float(np.std(trial_responses)),
                "trials": trial_responses,
                "delay_ms": self.stdp_delay_ms,
                "n_trials": self.stdp_trials,
            }
        self._stdp_post_pre_results = stdp_results
        self._phase_data_snapshots.append({"phase": "stdp_post_pre", "data": stdp_results})
        logger.info("STDP post-pre phase complete")

    def _compare_ltp_ltd(self) -> None:
        comparison = {}
        all_keys = set(list(self._stdp_pre_post_results.keys()) + list(self._stdp_post_pre_results.keys()))
        for key in all_keys:
            pre_post = self._stdp_pre_post_results.get(key, {})
            post_pre = self._stdp_post_pre_results.get(key, {})
            mean_pp = pre_post.get("mean_spikes", 0.0)
            mean_ppr = post_pre.get("mean_spikes", 0.0)
            delta = mean_pp - mean_ppr
            effect = "LTP" if delta > 0 else ("LTD" if delta < 0 else "neutral")
            comparison[key] = {
                "pre_post_mean": mean_pp,
                "post_pre_mean": mean_ppr,
                "delta": delta,
                "inferred_effect": effect,
            }
        self._ltp_ltd_comparison = comparison
        self._phase_data_snapshots.append({"phase": "ltp_ltd_comparison", "data": comparison})
        logger.info("LTP vs LTD comparison complete")

    def _compute_connectivity_matrix(self, label: str) -> Dict[str, Any]:
        matrix = {}
        pairs_to_use = self._top_pairs if self._top_pairs else self._known_pairs[:5]
        for pair in pairs_to_use:
            stim_e = pair["stim"]
            resp_e = pair["resp"]
            pair_key = f"{stim_e}_to_{resp_e}"
            polarity = self._get_polarity(pair["polarity"])
            trial_responses = []
            for _ in range(self.trials_per_condition):
                spike_df = self._stimulate_and_record(
                    stim_e, self.amplitude_ua, self.duration_us,
                    polarity, trigger_key=0, post_stim_wait_s=0.2,
                    phase_label=f"connectivity_{label}"
                )
                spike_count = len(spike_df) if not spike_df.empty else 0
                trial_responses.append(spike_count)
                self._wait(0.3)
            mean_resp = float(np.mean(trial_responses)) if trial_responses else 0.0
            matrix[pair_key] = {
                "mean_response": mean_resp,
                "trials": trial_responses,
                "label": label,
            }
        logger.info("Connectivity matrix (%s) computed: %d pairs", label, len(matrix))
        return matrix

    def _phase_response_probability_bins(self, recording_start: datetime) -> None:
        now = datetime_now()
        elapsed_s = (now - recording_start).total_seconds()
        n_bins = max(1, int(elapsed_s / self.bin_duration_s))
        bins = []
        bin_start = recording_start
        for i in range(n_bins):
            bin_end = bin_start + timedelta(seconds=self.bin_duration_s)
            if bin_end > now:
                bin_end = now
            fs_name = self.experiment.exp_name
            try:
                spike_df = self.database.get_spike_event(bin_start, bin_end, fs_name)
                spike_count = len(spike_df) if not spike_df.empty else 0
            except Exception as exc:
                logger.warning("Bin %d spike query failed: %s", i, exc)
                spike_count = 0
            bin_duration = (bin_end - bin_start).total_seconds()
            rate = spike_count / bin_duration if bin_duration > 0 else 0.0
            bins.append({
                "bin_index": i,
                "bin_start": bin_start.isoformat(),
                "bin_end": bin_end.isoformat(),
                "spike_count": spike_count,
                "rate_hz": rate,
            })
            bin_start = bin_end
        self._response_prob_bins = bins
        self._phase_data_snapshots.append({"phase": "response_probability_bins", "data": bins})
        logger.info("Response probability bins computed: %d bins", len(bins))

    def _phase_burst_rate(self) -> None:
        burst_results = {}
        pairs_to_use = self._top_pairs[:3] if len(self._top_pairs) >= 3 else self._top_pairs
        for pair in pairs_to_use:
            stim_e = pair["stim"]
            pair_key = f"{stim_e}_to_{pair['resp']}"
            polarity = self._get_polarity(pair["polarity"])
            burst_window_s = 1.0
            n_windows = 5
            burst_counts = []
            for _ in range(n_windows):
                self._send_biphasic_pulse(
                    stim_e, self.amplitude_ua, self.duration_us,
                    polarity, trigger_key=0, phase_label="burst_rate"
                )
                self._wait(burst_window_s)
                spike_df = self._query_spikes(window_s=burst_window_s)
                if not spike_df.empty and "Time" in spike_df.columns:
                    times = pd.to_datetime(spike_df["Time"]).sort_values()
                    if len(times) > 1:
                        diffs = times.diff().dt.total_seconds().dropna()
                        burst_threshold_s = 0.1
                        bursts = int((diffs < burst_threshold_s).sum())
                    else:
                        bursts = 0
                else:
                    bursts = 0
                burst_counts.append(bursts)
                self._wait(0.5)
            burst_results[pair_key] = {
                "mean_burst_count": float(np.mean(burst_counts)),
                "std_burst_count": float(np.std(burst_counts)),
                "burst_counts": burst_counts,
            }
        self._burst_rate_results = burst_results
        self._phase_data_snapshots.append({"phase": "burst_rate", "data": burst_results})
        logger.info("Burst rate analysis complete")

    def _phase_cross_correlograms(self) -> None:
        ccg_results = {}
        pairs_to_use = self._top_pairs if self._top_pairs else self._known_pairs[:5]
        window_s = 2.0
        for pair in pairs_to_use:
            stim_e = pair["stim"]
            resp_e = pair["resp"]
            pair_key = f"{stim_e}_to_{resp_e}"
            polarity = self._get_polarity(pair["polarity"])
            self._send_biphasic_pulse(
                stim_e, self.amplitude_ua, self.duration_us,
                polarity, trigger_key=0, phase_label="cross_correlogram"
            )
            self._wait(window_s)
            query_stop = datetime_now()
            query_start = query_stop - timedelta(seconds=window_s)
            fs_name = self.experiment.exp_name
            try:
                spike_df = self.database.get_spike_event(query_start, query_stop, fs_name)
            except Exception as exc:
                logger.warning("CCG spike query failed for %s: %s", pair_key, exc)
                spike_df = pd.DataFrame()

            stim_times = []
            resp_times = []
            if not spike_df.empty and "channel" in spike_df.columns and "Time" in spike_df.columns:
                stim_spikes = spike_df[spike_df["channel"] == stim_e]
                resp_spikes = spike_df[spike_df["channel"] == resp_e]
                stim_times = pd.to_datetime(stim_spikes["Time"]).tolist()
                resp_times = pd.to_datetime(resp_spikes["Time"]).tolist()

            lags_ms = []
            for st in stim_times:
                for rt in resp_times:
                    lag = (rt - st).total_seconds() * 1000.0
                    if -100.0 <= lag <= 100.0:
                        lags_ms.append(lag)

            if lags_ms:
                mean_lag = float(np.mean(lags_ms))
                std_lag = float(np.std(lags_ms))
                peak_lag = float(lags_ms[int(np.argmin(np.abs(lags_ms)))])
            else:
                mean_lag = 0.0
                std_lag = 0.0
                peak_lag = 0.0

            ccg_results[pair_key] = {
                "n_lags": len(lags_ms),
                "mean_lag_ms": mean_lag,
                "std_lag_ms": std_lag,
                "peak_lag_ms": peak_lag,
                "lags_ms": lags_ms[:50],
            }
            self._wait(0.5)

        self._cross_correlograms = ccg_results
        self._phase_data_snapshots.append({"phase": "cross_correlograms", "data": ccg_results})
        logger.info("Cross-correlograms complete: %d pairs", len(ccg_results))

    def _phase_latency_distributions(self) -> None:
        latency_results = {}
        pairs_to_use = self._top_pairs if self._top_pairs else self._known_pairs[:5]
        for pair in pairs_to_use:
            stim_e = pair["stim"]
            resp_e = pair["resp"]
            pair_key = f"{stim_e}_to_{resp_e}"
            polarity = self._get_polarity(pair["polarity"])
            latencies_ms = []
            for _ in range(self.trials_per_condition):
                t_stim = datetime_now()
                self._send_biphasic_pulse(
                    stim_e, self.amplitude_ua, self.duration_us,
                    polarity, trigger_key=0, phase_label="latency_distribution"
                )
                self._wait(0.1)
                t_query_stop = datetime_now()
                fs_name = self.experiment.exp_name
                try:
                    spike_df = self.database.get_spike_event(t_stim, t_query_stop, fs_name)
                except Exception as exc:
                    logger.warning("Latency query failed: %s", exc)
                    spike_df = pd.DataFrame()

                if not spike_df.empty and "channel" in spike_df.columns and "Time" in spike_df.columns:
                    resp_spikes = spike_df[spike_df["channel"] == resp_e]
                    if not resp_spikes.empty:
                        first_time = pd.to_datetime(resp_spikes["Time"]).min()
                        latency_ms = (first_time - t_stim).total_seconds() * 1000.0
                        if 0 < latency_ms < 100.0:
                            latencies_ms.append(latency_ms)
                self._wait(0.3)

            if latencies_ms:
                mean_lat = float(np.mean(latencies_ms))
                std_lat = float(np.std(latencies_ms))
                median_lat = float(np.median(latencies_ms))
                q25 = float(np.percentile(latencies_ms, 25))
                q75 = float(np.percentile(latencies_ms, 75))
                iqr = q75 - q25
                n = len(latencies_ms)
                skew = float(
                    (sum((x - mean_lat) ** 3 for x in latencies_ms) / n) /
                    (std_lat ** 3 + 1e-12)
                ) if n > 1 else 0.0
            else:
                mean_lat = std_lat = median_lat = iqr = skew = 0.0

            latency_results[pair_key] = {
                "latencies_ms": latencies_ms,
                "mean_ms": mean_lat,
                "std_ms": std_lat,
                "median_ms": median_lat,
                "iqr_ms": iqr,
                "skew": skew,
                "n_responses": len(latencies_ms),
            }

        self._latency_distributions = latency_results
        self._phase_data_snapshots.append({"phase": "latency_distributions", "data": latency_results})
        logger.info("Latency distributions complete: %d pairs", len(latency_results))

    def _wasserstein_distance(self, dist_a: List[float], dist_b: List[float]) -> float:
        if not dist_a or not dist_b:
            return 0.0
        sorted_a = sorted(dist_a)
        sorted_b = sorted(dist_b)
        all_vals = sorted(set(sorted_a + sorted_b))
        n_a = len(sorted_a)
        n_b = len(sorted_b)
        cdf_a = []
        cdf_b = []
        ia = 0
        ib = 0
        for v in all_vals:
            while ia < n_a and sorted_a[ia] <= v:
                ia += 1
            while ib < n_b and sorted_b[ib] <= v:
                ib += 1
            cdf_a.append(ia / n_a)
            cdf_b.append(ib / n_b)
        w_dist = 0.0
        for i in range(len(all_vals) - 1):
            w_dist += abs(cdf_a[i] - cdf_b[i]) * (all_vals[i + 1] - all_vals[i])
        return float(w_dist)

    def _phase_wasserstein_distances(self) -> None:
        wasserstein_results = {}
        for key, lat_data in self._latency_distributions.items():
            lats = lat_data.get("latencies_ms", [])
            if len(lats) < 2:
                wasserstein_results[key] = {"wasserstein_pre_post_vs_post_pre": 0.0}
                continue
            half = len(lats) // 2
            dist_a = lats[:half]
            dist_b = lats[half:]
            w = self._wasserstein_distance(dist_a, dist_b)
            wasserstein_results[key] = {
                "wasserstein_first_half_vs_second_half": w,
                "n_first_half": len(dist_a),
                "n_second_half": len(dist_b),
            }

        for key in self._stdp_pre_post_results:
            pre_post_trials = self._stdp_pre_post_results[key].get("trials", [])
            post_pre_trials = self._stdp_post_pre_results.get(key, {}).get("trials", [])
            pre_post_f = [float(x) for x in pre_post_trials]
            post_pre_f = [float(x) for x in post_pre_trials]
            w = self._wasserstein_distance(pre_post_f, post_pre_f)
            if key not in wasserstein_results:
                wasserstein_results[key] = {}
            wasserstein_results[key]["wasserstein_pre_post_vs_post_pre"] = w

        self._wasserstein_results = wasserstein_results
        self._phase_data_snapshots.append({"phase": "wasserstein_distances", "data": wasserstein_results})
        logger.info("Wasserstein distances computed: %d pairs", len(wasserstein_results))

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
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "top_pairs": self._top_pairs,
            "scan_results_electrodes": len(self._scan_results),
            "ppf_conditions": len(self._ppf_results),
            "io_amplitudes_tested": len(self.io_amplitudes),
            "freq_response_conditions": len(self._freq_results),
            "stdp_pre_post_pairs": len(self._stdp_pre_post_results),
            "stdp_post_pre_pairs": len(self._stdp_post_pre_results),
            "ltp_ltd_comparison": self._ltp_ltd_comparison,
            "connectivity_before_pairs": len(self._connectivity_before),
            "connectivity_after_pairs": len(self._connectivity_after),
            "response_prob_bins": len(self._response_prob_bins),
            "burst_rate_pairs": len(self._burst_rate_results),
            "cross_correlogram_pairs": len(self._cross_correlograms),
            "latency_distribution_pairs": len(self._latency_distributions),
            "polarity_comparison_pairs": len(self._polarity_comparison),
            "wasserstein_pairs": len(self._wasserstein_results),
            "phase_snapshots": len(self._phase_data_snapshots),
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
        fs_name = getattr(self.experiment, "exp_name", "unknown")

        connectivity_delta = {}
        for key in self._connectivity_before:
            before_val = self._connectivity_before[key].get("mean_response", 0.0)
            after_val = self._connectivity_after.get(key, {}).get("mean_response", 0.0)
            connectivity_delta[key] = {
                "before": before_val,
                "after": after_val,
                "delta": after_val - before_val,
            }

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": fs_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": duration_s,
            "total_stimulations": len(self._stimulation_log),
            "top_5_pairs": self._top_pairs,
            "electrode_scan_electrodes": len(self._scan_results),
            "ppf_results_summary": {
                k: {isi: v["mean_spikes"] for isi, v in cond.items()}
                for k, cond in self._ppf_results.items()
            },
            "io_curve_summary": {
                k: {amp: v["mean_spikes"] for amp, v in cond.items()}
                for k, cond in self._io_results.items()
            },
            "freq_response_summary": {
                k: {f: v["mean_spikes"] for f, v in cond.items()}
                for k, cond in self._freq_results.items()
            },
            "stdp_pre_post_summary": {
                k: v["mean_spikes"] for k, v in self._stdp_pre_post_results.items()
            },
            "stdp_post_pre_summary": {
                k: v["mean_spikes"] for k, v in self._stdp_post_pre_results.items()
            },
            "ltp_ltd_comparison": self._ltp_ltd_comparison,
            "spontaneous_activity": self._spontaneous_results,
            "connectivity_delta": connectivity_delta,
            "response_probability_bins": self._response_prob_bins,
            "burst_rate_summary": {
                k: v["mean_burst_count"] for k, v in self._burst_rate_results.items()
            },
            "cross_correlogram_summary": {
                k: {"peak_lag_ms": v["peak_lag_ms"], "n_lags": v["n_lags"]}
                for k, v in self._cross_correlograms.items()
            },
            "latency_distribution_summary": {
                k: {"mean_ms": v["mean_ms"], "std_ms": v["std_ms"], "n": v["n_responses"]}
                for k, v in self._latency_distributions.items()
            },
            "polarity_comparison_summary": {
                k: {cond: v["mean_spikes"] for cond, v in conds.items()}
                for k, conds in self._polarity_comparison.items()
            },
            "wasserstein_distances": self._wasserstein_results,
            "phase_snapshots_count": len(self._phase_data_snapshots),
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
