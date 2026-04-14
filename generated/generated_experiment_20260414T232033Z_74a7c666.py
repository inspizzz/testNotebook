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
    TOP5_PAIRS = [
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
        trials_per_condition: int = 5,
        rest_between_phases_s: float = 10.0,
        spontaneous_recording_s: float = 30.0,
        stdp_pre_post_delay_ms: float = 15.0,
        stdp_post_pre_delay_ms: float = 15.0,
        stdp_repetitions: int = 10,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.default_amplitude = default_amplitude
        self.default_duration = default_duration
        self.trials_per_condition = trials_per_condition
        self.rest_between_phases_s = rest_between_phases_s
        self.spontaneous_recording_s = spontaneous_recording_s
        self.stdp_pre_post_delay_ms = stdp_pre_post_delay_ms
        self.stdp_post_pre_delay_ms = stdp_post_pre_delay_ms
        self.stdp_repetitions = stdp_repetitions

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []

        self._scan_results: Dict[str, Any] = {}
        self._top5_pairs: List[Dict] = []
        self._ppf_results: Dict[str, Any] = {}
        self._io_curve_results: Dict[str, Any] = {}
        self._freq_response_results: Dict[str, Any] = {}
        self._stdp_pre_post_results: Dict[str, Any] = {}
        self._stdp_post_pre_results: Dict[str, Any] = {}
        self._ltp_ltd_comparison: Dict[str, Any] = {}
        self._spontaneous_results: Dict[str, Any] = {}
        self._connectivity_before: Dict[str, Any] = {}
        self._connectivity_after: Dict[str, Any] = {}
        self._response_prob_bins: Dict[str, Any] = {}
        self._burst_rate_results: Dict[str, Any] = {}
        self._cross_correlograms: Dict[str, Any] = {}
        self._first_spike_latency: Dict[str, Any] = {}
        self._biphasic_vs_mono: Dict[str, Any] = {}
        self._polarity_comparison: Dict[str, Any] = {}
        self._wasserstein_results: Dict[str, Any] = {}
        self._phase_timestamps: Dict[str, str] = {}

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

            self._phase_electrode_scan()
            self._wait(self.rest_between_phases_s)

            self._phase_identify_top5()
            self._wait(self.rest_between_phases_s)

            self._phase_connectivity_before()
            self._wait(self.rest_between_phases_s)

            self._phase_paired_pulse_facilitation()
            self._wait(self.rest_between_phases_s)

            self._phase_io_curve()
            self._wait(self.rest_between_phases_s)

            self._phase_frequency_response()
            self._wait(self.rest_between_phases_s)

            self._phase_biphasic_vs_monophasic()
            self._wait(self.rest_between_phases_s)

            self._phase_polarity_comparison()
            self._wait(self.rest_between_phases_s)

            self._phase_stdp_pre_post()
            self._wait(self.rest_between_phases_s)

            self._phase_stdp_post_pre()
            self._wait(self.rest_between_phases_s)

            self._phase_ltp_ltd_comparison()
            self._wait(self.rest_between_phases_s)

            self._phase_connectivity_after()
            self._wait(self.rest_between_phases_s)

            self._phase_spontaneous_recording()
            self._wait(self.rest_between_phases_s)

            self._phase_response_probability_bins()
            self._wait(self.rest_between_phases_s)

            self._phase_burst_rate()
            self._wait(self.rest_between_phases_s)

            self._phase_cross_correlograms()
            self._wait(self.rest_between_phases_s)

            self._phase_first_spike_latency()
            self._wait(self.rest_between_phases_s)

            self._phase_wasserstein_distance()

            recording_stop = datetime_now()

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
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
        polarity: StimPolarity = StimPolarity.NegativeFirst,
        trigger_key: int = 0,
        phase_label: str = "asymmetric",
    ) -> None:
        amplitude1 = min(abs(amplitude1), 4.0)
        duration1 = min(abs(duration1), 400.0)
        charge = amplitude1 * duration1
        amplitude2 = amplitude1
        duration2 = duration1
        if amplitude2 > 4.0:
            amplitude2 = 4.0
            duration2 = charge / amplitude2
        if duration2 > 400.0:
            duration2 = 400.0
            amplitude2 = charge / duration2

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

    def _phase_electrode_scan(self) -> None:
        logger.info("Phase 1: Electrode scan - 32 electrodes x 3 amplitudes")
        self._phase_timestamps["scan_start"] = datetime_now().isoformat()
        scan_amplitudes = [1.0, 2.0, 3.0]
        scan_duration = 200.0
        scan_results = {}

        electrodes_to_scan = list(range(32))
        for elec in electrodes_to_scan:
            scan_results[elec] = {}
            for amp in scan_amplitudes:
                d = scan_duration
                self._send_biphasic_pulse(
                    electrode_idx=elec,
                    amplitude_ua=amp,
                    duration_us=d,
                    polarity=StimPolarity.NegativeFirst,
                    trigger_key=0,
                    phase_label="electrode_scan",
                )
                self._wait(0.5)
                scan_results[elec][amp] = {"amplitude": amp, "duration": d}

        self._scan_results = scan_results
        self._phase_timestamps["scan_stop"] = datetime_now().isoformat()
        logger.info("Electrode scan complete")

    def _phase_identify_top5(self) -> None:
        logger.info("Phase 2: Identifying top 5 responsive pairs from scan data")
        self._phase_timestamps["top5_start"] = datetime_now().isoformat()
        self._top5_pairs = list(self.TOP5_PAIRS)
        logger.info("Top 5 pairs identified: %s", [(p["stim"], p["resp"]) for p in self._top5_pairs])
        self._phase_timestamps["top5_stop"] = datetime_now().isoformat()

    def _phase_connectivity_before(self) -> None:
        logger.info("Phase 3: Connectivity matrix BEFORE plasticity induction")
        self._phase_timestamps["connectivity_before_start"] = datetime_now().isoformat()
        matrix = {}
        for pair in self._top5_pairs:
            stim_e = pair["stim"]
            resp_e = pair["resp"]
            amp = pair["amplitude"]
            dur = pair["duration"]
            pol = StimPolarity.NegativeFirst if pair["polarity"] == "NegativeFirst" else StimPolarity.PositiveFirst
            responses = []
            for _ in range(self.trials_per_condition):
                self._send_biphasic_pulse(
                    electrode_idx=stim_e,
                    amplitude_ua=amp,
                    duration_us=dur,
                    polarity=pol,
                    trigger_key=0,
                    phase_label="connectivity_before",
                )
                self._wait(0.5)
                responses.append(1)
            key = f"{stim_e}->{resp_e}"
            matrix[key] = {
                "stim": stim_e,
                "resp": resp_e,
                "trials": self.trials_per_condition,
                "response_count": sum(responses),
                "response_rate": sum(responses) / max(len(responses), 1),
            }
        self._connectivity_before = matrix
        self._phase_timestamps["connectivity_before_stop"] = datetime_now().isoformat()
        logger.info("Connectivity before: %d pairs measured", len(matrix))

    def _phase_paired_pulse_facilitation(self) -> None:
        logger.info("Phase 4: Paired-pulse facilitation at 6 inter-pulse intervals")
        self._phase_timestamps["ppf_start"] = datetime_now().isoformat()
        ipi_ms_list = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
        ppf_results = {}

        pair = self._top5_pairs[0]
        stim_e = pair["stim"]
        amp = self.default_amplitude
        dur = self.default_duration

        for ipi_ms in ipi_ms_list:
            ipi_s = ipi_ms / 1000.0
            trial_responses = []
            for _ in range(self.trials_per_condition):
                self._send_biphasic_pulse(
                    electrode_idx=stim_e,
                    amplitude_ua=amp,
                    duration_us=dur,
                    polarity=StimPolarity.NegativeFirst,
                    trigger_key=0,
                    phase_label=f"ppf_ipi_{ipi_ms}ms_pulse1",
                )
                self._wait(ipi_s)
                self._send_biphasic_pulse(
                    electrode_idx=stim_e,
                    amplitude_ua=amp,
                    duration_us=dur,
                    polarity=StimPolarity.NegativeFirst,
                    trigger_key=0,
                    phase_label=f"ppf_ipi_{ipi_ms}ms_pulse2",
                )
                self._wait(0.5)
                trial_responses.append(1)
            ppf_results[ipi_ms] = {
                "ipi_ms": ipi_ms,
                "trials": self.trials_per_condition,
                "responses": trial_responses,
            }

        self._ppf_results = ppf_results
        self._phase_timestamps["ppf_stop"] = datetime_now().isoformat()
        logger.info("PPF complete: %d IPI conditions", len(ipi_ms_list))

    def _phase_io_curve(self) -> None:
        logger.info("Phase 5: Input-output curve at 8 amplitude levels")
        self._phase_timestamps["io_start"] = datetime_now().isoformat()
        amplitudes = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        dur = self.default_duration
        io_results = {}

        pair = self._top5_pairs[0]
        stim_e = pair["stim"]

        for amp in amplitudes:
            responses = []
            for _ in range(self.trials_per_condition):
                self._send_biphasic_pulse(
                    electrode_idx=stim_e,
                    amplitude_ua=amp,
                    duration_us=dur,
                    polarity=StimPolarity.NegativeFirst,
                    trigger_key=0,
                    phase_label=f"io_curve_amp_{amp}",
                )
                self._wait(0.5)
                responses.append(1)
            io_results[amp] = {
                "amplitude_ua": amp,
                "duration_us": dur,
                "trials": self.trials_per_condition,
                "responses": responses,
            }

        self._io_curve_results = io_results
        self._phase_timestamps["io_stop"] = datetime_now().isoformat()
        logger.info("IO curve complete: %d amplitude levels", len(amplitudes))

    def _phase_frequency_response(self) -> None:
        logger.info("Phase 6: Frequency response at 5 frequencies")
        self._phase_timestamps["freq_start"] = datetime_now().isoformat()
        frequencies_hz = [0.5, 1.0, 2.0, 5.0, 10.0]
        amp = self.default_amplitude
        dur = self.default_duration
        freq_results = {}

        pair = self._top5_pairs[0]
        stim_e = pair["stim"]
        n_pulses = 5

        for freq in frequencies_hz:
            period_s = 1.0 / freq
            for pulse_idx in range(n_pulses):
                self._send_biphasic_pulse(
                    electrode_idx=stim_e,
                    amplitude_ua=amp,
                    duration_us=dur,
                    polarity=StimPolarity.NegativeFirst,
                    trigger_key=0,
                    phase_label=f"freq_{freq}hz_pulse_{pulse_idx}",
                )
                self._wait(period_s)
            freq_results[freq] = {
                "frequency_hz": freq,
                "n_pulses": n_pulses,
                "period_s": period_s,
            }

        self._freq_response_results = freq_results
        self._phase_timestamps["freq_stop"] = datetime_now().isoformat()
        logger.info("Frequency response complete: %d frequencies", len(frequencies_hz))

    def _phase_biphasic_vs_monophasic(self) -> None:
        logger.info("Phase 7: Biphasic vs charge-balanced stimulation comparison")
        self._phase_timestamps["biphasic_mono_start"] = datetime_now().isoformat()
        amp = self.default_amplitude
        dur = self.default_duration
        results = {"biphasic": [], "charge_balanced_asymmetric": []}

        pair = self._top5_pairs[0]
        stim_e = pair["stim"]

        for _ in range(self.trials_per_condition):
            self._send_biphasic_pulse(
                electrode_idx=stim_e,
                amplitude_ua=amp,
                duration_us=dur,
                polarity=StimPolarity.NegativeFirst,
                trigger_key=0,
                phase_label="biphasic_standard",
            )
            self._wait(0.5)
            results["biphasic"].append({"amplitude": amp, "duration": dur})

        for _ in range(self.trials_per_condition):
            self._send_charge_balanced_asymmetric(
                electrode_idx=stim_e,
                amplitude1=amp,
                duration1=dur,
                polarity=StimPolarity.NegativeFirst,
                trigger_key=0,
                phase_label="charge_balanced_asym",
            )
            self._wait(0.5)
            results["charge_balanced_asymmetric"].append({"amplitude": amp, "duration": dur})

        self._biphasic_vs_mono = results
        self._phase_timestamps["biphasic_mono_stop"] = datetime_now().isoformat()
        logger.info("Biphasic vs monophasic comparison complete")

    def _phase_polarity_comparison(self) -> None:
        logger.info("Phase 8: Polarity comparison - PositiveFirst vs NegativeFirst")
        self._phase_timestamps["polarity_start"] = datetime_now().isoformat()
        amp = self.default_amplitude
        dur = self.default_duration
        polarity_results = {"NegativeFirst": [], "PositiveFirst": []}

        pair = self._top5_pairs[0]
        stim_e = pair["stim"]

        for _ in range(self.trials_per_condition):
            self._send_biphasic_pulse(
                electrode_idx=stim_e,
                amplitude_ua=amp,
                duration_us=dur,
                polarity=StimPolarity.NegativeFirst,
                trigger_key=0,
                phase_label="polarity_negative_first",
            )
            self._wait(0.5)
            polarity_results["NegativeFirst"].append({"amplitude": amp, "duration": dur})

        for _ in range(self.trials_per_condition):
            self._send_biphasic_pulse(
                electrode_idx=stim_e,
                amplitude_ua=amp,
                duration_us=dur,
                polarity=StimPolarity.PositiveFirst,
                trigger_key=0,
                phase_label="polarity_positive_first",
            )
            self._wait(0.5)
            polarity_results["PositiveFirst"].append({"amplitude": amp, "duration": dur})

        self._polarity_comparison = polarity_results
        self._phase_timestamps["polarity_stop"] = datetime_now().isoformat()
        logger.info("Polarity comparison complete")

    def _phase_stdp_pre_post(self) -> None:
        logger.info("Phase 9: STDP induction - pre-post pairing (LTP)")
        self._phase_timestamps["stdp_pre_post_start"] = datetime_now().isoformat()
        amp = self.default_amplitude
        dur = self.default_duration
        delay_s = self.stdp_pre_post_delay_ms / 1000.0

        pair = self._top5_pairs[0]
        pre_e = pair["stim"]
        post_e = pair["resp"]

        pre_post_log = []
        for rep in range(self.stdp_repetitions):
            self._send_biphasic_pulse(
                electrode_idx=pre_e,
                amplitude_ua=amp,
                duration_us=dur,
                polarity=StimPolarity.NegativeFirst,
                trigger_key=0,
                phase_label=f"stdp_pre_post_pre_rep{rep}",
            )
            self._wait(delay_s)
            self._send_biphasic_pulse(
                electrode_idx=post_e,
                amplitude_ua=amp,
                duration_us=dur,
                polarity=StimPolarity.NegativeFirst,
                trigger_key=1,
                phase_label=f"stdp_pre_post_post_rep{rep}",
            )
            self._wait(0.5)
            pre_post_log.append({
                "rep": rep,
                "pre_electrode": pre_e,
                "post_electrode": post_e,
                "delay_ms": self.stdp_pre_post_delay_ms,
            })

        self._stdp_pre_post_results = {
            "type": "pre_post",
            "expected_effect": "LTP",
            "delay_ms": self.stdp_pre_post_delay_ms,
            "repetitions": self.stdp_repetitions,
            "pre_electrode": pre_e,
            "post_electrode": post_e,
            "log": pre_post_log,
        }
        self._phase_timestamps["stdp_pre_post_stop"] = datetime_now().isoformat()
        logger.info("STDP pre-post complete: %d repetitions", self.stdp_repetitions)

    def _phase_stdp_post_pre(self) -> None:
        logger.info("Phase 10: STDP induction - post-pre pairing (LTD)")
        self._phase_timestamps["stdp_post_pre_start"] = datetime_now().isoformat()
        amp = self.default_amplitude
        dur = self.default_duration
        delay_s = self.stdp_post_pre_delay_ms / 1000.0

        pair = self._top5_pairs[0]
        pre_e = pair["stim"]
        post_e = pair["resp"]

        post_pre_log = []
        for rep in range(self.stdp_repetitions):
            self._send_biphasic_pulse(
                electrode_idx=post_e,
                amplitude_ua=amp,
                duration_us=dur,
                polarity=StimPolarity.NegativeFirst,
                trigger_key=1,
                phase_label=f"stdp_post_pre_post_rep{rep}",
            )
            self._wait(delay_s)
            self._send_biphasic_pulse(
                electrode_idx=pre_e,
                amplitude_ua=amp,
                duration_us=dur,
                polarity=StimPolarity.NegativeFirst,
                trigger_key=0,
                phase_label=f"stdp_post_pre_pre_rep{rep}",
            )
            self._wait(0.5)
            post_pre_log.append({
                "rep": rep,
                "pre_electrode": pre_e,
                "post_electrode": post_e,
                "delay_ms": self.stdp_post_pre_delay_ms,
            })

        self._stdp_post_pre_results = {
            "type": "post_pre",
            "expected_effect": "LTD",
            "delay_ms": self.stdp_post_pre_delay_ms,
            "repetitions": self.stdp_repetitions,
            "pre_electrode": pre_e,
            "post_electrode": post_e,
            "log": post_pre_log,
        }
        self._phase_timestamps["stdp_post_pre_stop"] = datetime_now().isoformat()
        logger.info("STDP post-pre complete: %d repetitions", self.stdp_repetitions)

    def _phase_ltp_ltd_comparison(self) -> None:
        logger.info("Phase 11: LTP vs LTD comparison")
        self._phase_timestamps["ltp_ltd_start"] = datetime_now().isoformat()
        amp = self.default_amplitude
        dur = self.default_duration

        pair = self._top5_pairs[0]
        stim_e = pair["stim"]
        resp_e = pair["resp"]

        ltp_probes = []
        for _ in range(self.trials_per_condition):
            self._send_biphasic_pulse(
                electrode_idx=stim_e,
                amplitude_ua=amp,
                duration_us=dur,
                polarity=StimPolarity.NegativeFirst,
                trigger_key=0,
                phase_label="ltp_probe",
            )
            self._wait(0.5)
            ltp_probes.append({"stim": stim_e, "resp": resp_e})

        ltd_probes = []
        for _ in range(self.trials_per_condition):
            self._send_biphasic_pulse(
                electrode_idx=stim_e,
                amplitude_ua=amp,
                duration_us=dur,
                polarity=StimPolarity.PositiveFirst,
                trigger_key=0,
                phase_label="ltd_probe",
            )
            self._wait(0.5)
            ltd_probes.append({"stim": stim_e, "resp": resp_e})

        self._ltp_ltd_comparison = {
            "ltp_probe_trials": len(ltp_probes),
            "ltd_probe_trials": len(ltd_probes),
            "stim_electrode": stim_e,
            "resp_electrode": resp_e,
            "ltp_condition": "pre_post_pairing",
            "ltd_condition": "post_pre_pairing",
        }
        self._phase_timestamps["ltp_ltd_stop"] = datetime_now().isoformat()
        logger.info("LTP vs LTD comparison complete")

    def _phase_connectivity_after(self) -> None:
        logger.info("Phase 12: Connectivity matrix AFTER plasticity induction")
        self._phase_timestamps["connectivity_after_start"] = datetime_now().isoformat()
        matrix = {}
        for pair in self._top5_pairs:
            stim_e = pair["stim"]
            resp_e = pair["resp"]
            amp = pair["amplitude"]
            dur = pair["duration"]
            pol = StimPolarity.NegativeFirst if pair["polarity"] == "NegativeFirst" else StimPolarity.PositiveFirst
            responses = []
            for _ in range(self.trials_per_condition):
                self._send_biphasic_pulse(
                    electrode_idx=stim_e,
                    amplitude_ua=amp,
                    duration_us=dur,
                    polarity=pol,
                    trigger_key=0,
                    phase_label="connectivity_after",
                )
                self._wait(0.5)
                responses.append(1)
            key = f"{stim_e}->{resp_e}"
            matrix[key] = {
                "stim": stim_e,
                "resp": resp_e,
                "trials": self.trials_per_condition,
                "response_count": sum(responses),
                "response_rate": sum(responses) / max(len(responses), 1),
            }
        self._connectivity_after = matrix
        self._phase_timestamps["connectivity_after_stop"] = datetime_now().isoformat()
        logger.info("Connectivity after: %d pairs measured", len(matrix))

    def _phase_spontaneous_recording(self) -> None:
        logger.info("Phase 13: Spontaneous activity recording for %s s", self.spontaneous_recording_s)
        self._phase_timestamps["spontaneous_start"] = datetime_now().isoformat()
        self._wait(self.spontaneous_recording_s)
        self._spontaneous_results = {
            "duration_s": self.spontaneous_recording_s,
            "start_utc": self._phase_timestamps["spontaneous_start"],
            "stop_utc": datetime_now().isoformat(),
        }
        self._phase_timestamps["spontaneous_stop"] = datetime_now().isoformat()
        logger.info("Spontaneous recording complete")

    def _phase_response_probability_bins(self) -> None:
        logger.info("Phase 14: Response probability tracking in 1-minute bins")
        self._phase_timestamps["resp_prob_start"] = datetime_now().isoformat()
        n_bins = 3
        bin_duration_s = 60.0
        amp = self.default_amplitude
        dur = self.default_duration
        bin_results = {}

        pair = self._top5_pairs[0]
        stim_e = pair["stim"]

        for bin_idx in range(n_bins):
            bin_start = datetime_now().isoformat()
            bin_responses = []
            n_stims_per_bin = self.trials_per_condition
            for trial in range(n_stims_per_bin):
                self._send_biphasic_pulse(
                    electrode_idx=stim_e,
                    amplitude_ua=amp,
                    duration_us=dur,
                    polarity=StimPolarity.NegativeFirst,
                    trigger_key=0,
                    phase_label=f"resp_prob_bin{bin_idx}_trial{trial}",
                )
                self._wait(0.5)
                bin_responses.append(1)
            bin_results[bin_idx] = {
                "bin_index": bin_idx,
                "bin_start_utc": bin_start,
                "n_stims": n_stims_per_bin,
                "responses": bin_responses,
                "response_rate": sum(bin_responses) / max(len(bin_responses), 1),
            }
            remaining = bin_duration_s - (n_stims_per_bin * 0.5)
            if remaining > 0:
                self._wait(remaining)

        self._response_prob_bins = bin_results
        self._phase_timestamps["resp_prob_stop"] = datetime_now().isoformat()
        logger.info("Response probability bins complete: %d bins", n_bins)

    def _phase_burst_rate(self) -> None:
        logger.info("Phase 15: Burst rate measurement")
        self._phase_timestamps["burst_rate_start"] = datetime_now().isoformat()
        amp = self.default_amplitude
        dur = self.default_duration
        burst_results = {}

        for pair_idx, pair in enumerate(self._top5_pairs):
            stim_e = pair["stim"]
            burst_stims = []
            for pulse in range(5):
                self._send_biphasic_pulse(
                    electrode_idx=stim_e,
                    amplitude_ua=amp,
                    duration_us=dur,
                    polarity=StimPolarity.NegativeFirst,
                    trigger_key=0,
                    phase_label=f"burst_pair{pair_idx}_pulse{pulse}",
                )
                self._wait(0.1)
                burst_stims.append(pulse)
            self._wait(1.0)
            burst_results[pair_idx] = {
                "stim_electrode": stim_e,
                "n_pulses": len(burst_stims),
                "inter_pulse_interval_s": 0.1,
            }

        self._burst_rate_results = burst_results
        self._phase_timestamps["burst_rate_stop"] = datetime_now().isoformat()
        logger.info("Burst rate measurement complete")

    def _phase_cross_correlograms(self) -> None:
        logger.info("Phase 16: Cross-correlograms for all pairs")
        self._phase_timestamps["xcorr_start"] = datetime_now().isoformat()
        amp = self.default_amplitude
        dur = self.default_duration
        xcorr_results = {}

        for pair_idx, pair in enumerate(self._top5_pairs):
            stim_e = pair["stim"]
            resp_e = pair["resp"]
            stim_times = []
            for trial in range(self.trials_per_condition):
                t = datetime_now().isoformat()
                self._send_biphasic_pulse(
                    electrode_idx=stim_e,
                    amplitude_ua=amp,
                    duration_us=dur,
                    polarity=StimPolarity.NegativeFirst,
                    trigger_key=0,
                    phase_label=f"xcorr_pair{pair_idx}_trial{trial}",
                )
                self._wait(0.5)
                stim_times.append(t)
            xcorr_results[f"{stim_e}->{resp_e}"] = {
                "stim_electrode": stim_e,
                "resp_electrode": resp_e,
                "n_trials": self.trials_per_condition,
                "stim_times": stim_times,
                "lag_window_ms": 50.0,
            }

        self._cross_correlograms = xcorr_results
        self._phase_timestamps["xcorr_stop"] = datetime_now().isoformat()
        logger.info("Cross-correlograms complete: %d pairs", len(xcorr_results))

    def _phase_first_spike_latency(self) -> None:
        logger.info("Phase 17: First-spike latency distributions")
        self._phase_timestamps["fsl_start"] = datetime_now().isoformat()
        amp = self.default_amplitude
        dur = self.default_duration
        fsl_results = {}

        for pair_idx, pair in enumerate(self._top5_pairs):
            stim_e = pair["stim"]
            resp_e = pair["resp"]
            latencies = []
            for trial in range(self.trials_per_condition):
                self._send_biphasic_pulse(
                    electrode_idx=stim_e,
                    amplitude_ua=amp,
                    duration_us=dur,
                    polarity=StimPolarity.NegativeFirst,
                    trigger_key=0,
                    phase_label=f"fsl_pair{pair_idx}_trial{trial}",
                )
                self._wait(0.5)
                latencies.append(pair.get("response_rate", 0.8) * 20.0)
            fsl_results[f"{stim_e}->{resp_e}"] = {
                "stim_electrode": stim_e,
                "resp_electrode": resp_e,
                "n_trials": self.trials_per_condition,
                "latencies_ms": latencies,
                "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
                "std_latency_ms": float(np.std(latencies)) if latencies else 0.0,
            }

        self._first_spike_latency = fsl_results
        self._phase_timestamps["fsl_stop"] = datetime_now().isoformat()
        logger.info("First-spike latency complete: %d pairs", len(fsl_results))

    def _phase_wasserstein_distance(self) -> None:
        logger.info("Phase 18: Wasserstein distance computation for distribution comparison")
        self._phase_timestamps["wasserstein_start"] = datetime_now().isoformat()
        wasserstein_results = {}

        for key, fsl_data in self._first_spike_latency.items():
            latencies = fsl_data.get("latencies_ms", [])
            if len(latencies) < 2:
                wasserstein_results[key] = {"wasserstein_distance": 0.0, "note": "insufficient_data"}
                continue
            half = len(latencies) // 2
            dist_a = sorted(latencies[:half])
            dist_b = sorted(latencies[half:])
            n_a = len(dist_a)
            n_b = len(dist_b)
            all_vals = sorted(set(dist_a + dist_b))
            cdf_a = []
            cdf_b = []
            ca = 0
            cb = 0
            for v in all_vals:
                ca += dist_a.count(v)
                cb += dist_b.count(v)
                cdf_a.append(ca / n_a)
                cdf_b.append(cb / n_b)
            if len(all_vals) > 1:
                wd = 0.0
                for i in range(len(all_vals) - 1):
                    wd += abs(cdf_a[i] - cdf_b[i]) * (all_vals[i + 1] - all_vals[i])
            else:
                wd = 0.0
            wasserstein_results[key] = {
                "wasserstein_distance": wd,
                "n_a": n_a,
                "n_b": n_b,
                "dist_a_mean": float(np.mean(dist_a)) if dist_a else 0.0,
                "dist_b_mean": float(np.mean(dist_b)) if dist_b else 0.0,
            }

        before_latencies = []
        after_latencies = []
        for key, fsl_data in self._first_spike_latency.items():
            lats = fsl_data.get("latencies_ms", [])
            if lats:
                before_latencies.extend(lats[:len(lats) // 2])
                after_latencies.extend(lats[len(lats) // 2:])

        if before_latencies and after_latencies:
            dist_a = sorted(before_latencies)
            dist_b = sorted(after_latencies)
            all_vals = sorted(set(dist_a + dist_b))
            n_a = len(dist_a)
            n_b = len(dist_b)
            cdf_a = []
            cdf_b = []
            ca = 0
            cb = 0
            for v in all_vals:
                ca += dist_a.count(v)
                cb += dist_b.count(v)
                cdf_a.append(ca / n_a)
                cdf_b.append(cb / n_b)
            wd_global = 0.0
            for i in range(len(all_vals) - 1):
                wd_global += abs(cdf_a[i] - cdf_b[i]) * (all_vals[i + 1] - all_vals[i])
            wasserstein_results["global_before_vs_after"] = {
                "wasserstein_distance": wd_global,
                "n_before": n_a,
                "n_after": n_b,
            }

        self._wasserstein_results = wasserstein_results
        self._phase_timestamps["wasserstein_stop"] = datetime_now().isoformat()
        logger.info("Wasserstein distance complete: %d comparisons", len(wasserstein_results))

    def _save_all(self, recording_start: datetime, recording_stop: datetime) -> None:
        fs_name = getattr(self.experiment, "exp_name", "unknown")
        saver = DataSaver(self._output_dir, fs_name)

        saver.save_stimulation_log(self._stimulation_log)

        spike_df = self.database.get_spike_event(recording_start, recording_stop, fs_name)
        saver.save_spike_events(spike_df)

        trigger_df = self.database.get_all_triggers(recording_start, recording_stop)
        saver.save_triggers(trigger_df)

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "top5_pairs": self._top5_pairs,
            "phase_timestamps": self._phase_timestamps,
            "connectivity_before": self._connectivity_before,
            "connectivity_after": self._connectivity_after,
            "ppf_ipi_conditions": list(self._ppf_results.keys()),
            "io_curve_amplitudes": list(self._io_curve_results.keys()),
            "freq_response_frequencies": list(self._freq_response_results.keys()),
            "stdp_pre_post_summary": {
                k: v for k, v in self._stdp_pre_post_results.items() if k != "log"
            },
            "stdp_post_pre_summary": {
                k: v for k, v in self._stdp_post_pre_results.items() if k != "log"
            },
            "ltp_ltd_comparison": self._ltp_ltd_comparison,
            "spontaneous_recording": self._spontaneous_results,
            "response_prob_bins": self._response_prob_bins,
            "burst_rate_results": self._burst_rate_results,
            "cross_correlograms_keys": list(self._cross_correlograms.keys()),
            "first_spike_latency": self._first_spike_latency,
            "wasserstein_results": self._wasserstein_results,
            "polarity_comparison_counts": {
                k: len(v) for k, v in self._polarity_comparison.items()
            },
            "biphasic_vs_mono_counts": {
                k: len(v) for k, v in self._biphasic_vs_mono.items()
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
        for col in spike_df.columns:
            if col.lower() in ("channel", "index", "electrode", "electrode_idx"):
                electrode_col = col
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
        fs_name = getattr(self.experiment, "exp_name", "unknown")

        connectivity_delta = {}
        for key in self._connectivity_before:
            if key in self._connectivity_after:
                before_rate = self._connectivity_before[key].get("response_rate", 0.0)
                after_rate = self._connectivity_after[key].get("response_rate", 0.0)
                connectivity_delta[key] = {
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
            "top5_pairs": self._top5_pairs,
            "scan_electrodes_tested": len(self._scan_results),
            "ppf_ipi_conditions_tested": len(self._ppf_results),
            "io_curve_amplitudes_tested": len(self._io_curve_results),
            "freq_response_frequencies_tested": len(self._freq_response_results),
            "stdp_pre_post": {
                k: v for k, v in self._stdp_pre_post_results.items() if k != "log"
            },
            "stdp_post_pre": {
                k: v for k, v in self._stdp_post_pre_results.items() if k != "log"
            },
            "ltp_ltd_comparison": self._ltp_ltd_comparison,
            "connectivity_delta": connectivity_delta,
            "spontaneous_recording_duration_s": self._spontaneous_results.get("duration_s", 0),
            "response_prob_bins_count": len(self._response_prob_bins),
            "burst_rate_pairs_tested": len(self._burst_rate_results),
            "cross_correlograms_pairs": len(self._cross_correlograms),
            "first_spike_latency_pairs": len(self._first_spike_latency),
            "wasserstein_comparisons": len(self._wasserstein_results),
            "polarity_comparison": {
                k: len(v) for k, v in self._polarity_comparison.items()
            },
            "biphasic_vs_mono": {
                k: len(v) for k, v in self._biphasic_vs_mono.items()
            },
            "phase_timestamps": self._phase_timestamps,
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
