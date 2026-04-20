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
    """
    STDP-inspired plasticity experiment on FinalSpark NeuroPlatform.

    Phase 1 (Baseline): Probe responsive electrode pairs with multiple
    amplitude levels to map which spots light up most.

    Phase 2 (Conditioning): Deliver paired stimulation to two electrodes
    with a short inter-stimulus interval (STDP window ~10 ms) to attempt
    Hebbian-like plasticity induction.

    Phase 3 (Post-conditioning probe): Re-probe the same pairs to assess
    whether the conditioning changed evoked response rates.
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        probe_amplitudes: tuple = (1.0, 2.0, 3.0),
        probe_duration_us: float = 400.0,
        probe_trials_per_condition: int = 10,
        conditioning_pre_electrode: int = 9,
        conditioning_post_electrode: int = 10,
        conditioning_amplitude_ua: float = 1.5,
        conditioning_duration_us: float = 200.0,
        conditioning_isi_ms: float = 10.0,
        conditioning_num_pairs: int = 50,
        conditioning_iti_s: float = 0.6,
        post_probe_trials: int = 10,
        inter_stim_wait_s: float = 0.5,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.probe_amplitudes = list(probe_amplitudes)
        self.probe_duration_us = float(probe_duration_us)
        self.probe_trials_per_condition = int(probe_trials_per_condition)

        self.conditioning_pre_electrode = int(conditioning_pre_electrode)
        self.conditioning_post_electrode = int(conditioning_post_electrode)
        self.conditioning_amplitude_ua = float(conditioning_amplitude_ua)
        self.conditioning_duration_us = float(conditioning_duration_us)
        self.conditioning_isi_ms = float(conditioning_isi_ms)
        self.conditioning_num_pairs = int(conditioning_num_pairs)
        self.conditioning_iti_s = float(conditioning_iti_s)
        self.post_probe_trials = int(post_probe_trials)
        self.inter_stim_wait_s = float(inter_stim_wait_s)

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []

        self._baseline_results: Dict[str, Any] = {}
        self._conditioning_results: Dict[str, Any] = {}
        self._post_results: Dict[str, Any] = {}

        self._probe_pairs: List[Tuple[int, int]] = [
            (9, 10),
            (9, 11),
            (14, 15),
            (14, 12),
            (17, 16),
            (5, 4),
            (5, 6),
            (0, 1),
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

            logger.info("=== PHASE 1: Baseline amplitude sweep ===")
            self._phase_baseline_sweep()

            logger.info("=== PHASE 2: STDP-style paired conditioning ===")
            self._phase_conditioning()

            logger.info("=== PHASE 3: Post-conditioning probe ===")
            self._phase_post_probe()

            recording_stop = datetime_now()

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _phase_baseline_sweep(self) -> None:
        """
        Probe each electrode pair at multiple amplitudes to identify
        which spots light up most. Uses charge-balanced biphasic pulses.
        """
        logger.info("Baseline sweep: %d pairs x %d amplitudes x %d trials",
                    len(self._probe_pairs), len(self.probe_amplitudes),
                    self.probe_trials_per_condition)

        pair_responses: Dict[str, List[int]] = {}

        for (stim_elec, resp_elec) in self._probe_pairs:
            for amplitude in self.probe_amplitudes:
                duration = self.probe_duration_us
                amplitude = min(amplitude, 4.0)
                duration = min(duration, 400.0)

                condition_key = f"e{stim_elec}_to_e{resp_elec}_amp{amplitude}"
                spike_counts = []

                for trial in range(self.probe_trials_per_condition):
                    t_before = datetime_now()
                    self._send_single_pulse(
                        electrode_idx=stim_elec,
                        amplitude_ua=amplitude,
                        duration_us=duration,
                        polarity=StimPolarity.NegativeFirst,
                        trigger_key=0,
                        phase_label="baseline",
                    )
                    self._wait(self.inter_stim_wait_s)
                    t_after = datetime_now()

                    spike_df = self.database.get_spike_event_electrode(
                        t_before, t_after, resp_elec
                    )
                    n_spikes = len(spike_df) if not spike_df.empty else 0
                    spike_counts.append(n_spikes)

                    logger.info(
                        "Baseline pair (%d->%d) amp=%.1f trial=%d spikes=%d",
                        stim_elec, resp_elec, amplitude, trial + 1, n_spikes
                    )

                pair_responses[condition_key] = spike_counts

        self._baseline_results = {
            "pair_responses": pair_responses,
            "probe_pairs": self._probe_pairs,
            "amplitudes": self.probe_amplitudes,
            "trials_per_condition": self.probe_trials_per_condition,
        }
        logger.info("Baseline sweep complete. %d conditions tested.", len(pair_responses))

    def _phase_conditioning(self) -> None:
        """
        Deliver paired stimulation: pre-electrode pulse followed by
        post-electrode pulse with a short ISI (STDP window ~10 ms).
        Charge balance: A1*D1 == A2*D2 enforced by equal phases.
        """
        pre_elec = self.conditioning_pre_electrode
        post_elec = self.conditioning_post_electrode
        amplitude = min(self.conditioning_amplitude_ua, 4.0)
        duration = min(self.conditioning_duration_us, 400.0)
        isi_s = self.conditioning_isi_ms / 1000.0

        logger.info(
            "Conditioning: pre=%d post=%d amp=%.2f uA dur=%.0f us ISI=%.1f ms pairs=%d",
            pre_elec, post_elec, amplitude, duration,
            self.conditioning_isi_ms, self.conditioning_num_pairs
        )

        pair_count = 0
        for pair_idx in range(self.conditioning_num_pairs):
            self._send_single_pulse(
                electrode_idx=pre_elec,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=StimPolarity.NegativeFirst,
                trigger_key=0,
                phase_label="conditioning_pre",
            )

            self._wait(isi_s)

            self._send_single_pulse(
                electrode_idx=post_elec,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=StimPolarity.NegativeFirst,
                trigger_key=1,
                phase_label="conditioning_post",
            )

            pair_count += 1

            self._wait(self.conditioning_iti_s)

            if (pair_idx + 1) % 10 == 0:
                logger.info("Conditioning progress: %d / %d pairs delivered",
                            pair_idx + 1, self.conditioning_num_pairs)

        self._conditioning_results = {
            "pre_electrode": pre_elec,
            "post_electrode": post_elec,
            "amplitude_ua": amplitude,
            "duration_us": duration,
            "isi_ms": self.conditioning_isi_ms,
            "pairs_delivered": pair_count,
            "iti_s": self.conditioning_iti_s,
        }
        logger.info("Conditioning complete: %d pairs delivered.", pair_count)

    def _phase_post_probe(self) -> None:
        """
        Re-probe the same electrode pairs used in baseline to assess
        whether conditioning changed evoked response rates.
        """
        logger.info("Post-conditioning probe: %d pairs x %d trials",
                    len(self._probe_pairs), self.post_probe_trials)

        post_pair_responses: Dict[str, List[int]] = {}

        for (stim_elec, resp_elec) in self._probe_pairs:
            amplitude = 2.0
            duration = self.probe_duration_us
            amplitude = min(amplitude, 4.0)
            duration = min(duration, 400.0)

            condition_key = f"e{stim_elec}_to_e{resp_elec}_amp{amplitude}"
            spike_counts = []

            for trial in range(self.post_probe_trials):
                t_before = datetime_now()
                self._send_single_pulse(
                    electrode_idx=stim_elec,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    polarity=StimPolarity.NegativeFirst,
                    trigger_key=2,
                    phase_label="post_probe",
                )
                self._wait(self.inter_stim_wait_s)
                t_after = datetime_now()

                spike_df = self.database.get_spike_event_electrode(
                    t_before, t_after, resp_elec
                )
                n_spikes = len(spike_df) if not spike_df.empty else 0
                spike_counts.append(n_spikes)

                logger.info(
                    "Post-probe pair (%d->%d) amp=%.1f trial=%d spikes=%d",
                    stim_elec, resp_elec, amplitude, trial + 1, n_spikes
                )

            post_pair_responses[condition_key] = spike_counts

        self._post_results = {
            "pair_responses": post_pair_responses,
            "probe_pairs": self._probe_pairs,
            "amplitude": 2.0,
            "trials_per_pair": self.post_probe_trials,
        }
        logger.info("Post-conditioning probe complete.")

    def _send_single_pulse(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity = StimPolarity.NegativeFirst,
        trigger_key: int = 0,
        phase_label: str = "stim",
    ) -> None:
        """
        Configure and fire a single charge-balanced biphasic pulse.
        Charge balance: A1*D1 == A2*D2 (equal amplitude and duration on both phases).
        """
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
            extra={
                "charge_balance_check": round(amplitude_ua * duration_us, 4),
            },
        ))

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

        baseline_summary = {}
        post_summary = {}
        if self._baseline_results.get("pair_responses"):
            for key, counts in self._baseline_results["pair_responses"].items():
                total = sum(counts)
                baseline_summary[key] = {"total_spikes": total, "trials": len(counts)}
        if self._post_results.get("pair_responses"):
            for key, counts in self._post_results["pair_responses"].items():
                total = sum(counts)
                post_summary[key] = {"total_spikes": total, "trials": len(counts)}

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "baseline_summary": baseline_summary,
            "conditioning_results": self._conditioning_results,
            "post_conditioning_summary": post_summary,
            "probe_pairs": self._probe_pairs,
            "probe_amplitudes": self.probe_amplitudes,
            "conditioning_pre_electrode": self.conditioning_pre_electrode,
            "conditioning_post_electrode": self.conditioning_post_electrode,
            "conditioning_amplitude_ua": self.conditioning_amplitude_ua,
            "conditioning_duration_us": self.conditioning_duration_us,
            "conditioning_isi_ms": self.conditioning_isi_ms,
            "conditioning_num_pairs": self.conditioning_num_pairs,
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

        baseline_totals: Dict[str, int] = {}
        if self._baseline_results.get("pair_responses"):
            for key, counts in self._baseline_results["pair_responses"].items():
                baseline_totals[key] = sum(counts)

        post_totals: Dict[str, int] = {}
        if self._post_results.get("pair_responses"):
            for key, counts in self._post_results["pair_responses"].items():
                post_totals[key] = sum(counts)

        plasticity_delta: Dict[str, float] = {}
        for key in post_totals:
            if key in baseline_totals and baseline_totals[key] > 0:
                delta = (post_totals[key] - baseline_totals[key]) / float(baseline_totals[key])
                plasticity_delta[key] = round(delta, 4)
            elif key in baseline_totals:
                plasticity_delta[key] = float("nan") if baseline_totals[key] == 0 else 0.0

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "total_stimulations": len(self._stimulation_log),
            "baseline_spike_totals": baseline_totals,
            "post_conditioning_spike_totals": post_totals,
            "plasticity_delta_fraction": plasticity_delta,
            "conditioning_pairs_delivered": self._conditioning_results.get("pairs_delivered", 0),
            "conditioning_pre_electrode": self.conditioning_pre_electrode,
            "conditioning_post_electrode": self.conditioning_post_electrode,
            "conditioning_isi_ms": self.conditioning_isi_ms,
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
