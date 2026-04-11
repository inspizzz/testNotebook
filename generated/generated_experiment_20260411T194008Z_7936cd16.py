import numpy as np
import pandas as pd
import json
import time
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
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StimulationRecord:
    electrode_idx: int
    amplitude_ua: float
    duration_us: float
    timestamp_utc: str
    trigger_key: int = 0
    phase: str = ""
    condition: str = ""
    trial: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialResult:
    electrode_idx: int
    phase: str
    condition: str
    trial: int
    amplitude_ua: float
    duration_us: float
    baseline_spike_count: int
    post_stim_spike_count: int
    spike_count_change: int
    baseline_window_s: float = 0.0
    post_stim_window_s: float = 0.0
    timestamp_utc: str = ""


class DataSaver:
    def __init__(self, output_dir: Path, fs_name: str) -> None:
        self._dir = output_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
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

    def save_trial_results(self, trials: List[TrialResult]) -> Path:
        path = Path(f"{self._prefix}_trial_results.json")
        records = [asdict(t) for t in trials]
        path.write_text(json.dumps(records, indent=2, default=str))
        logger.info("Saved trial results -> %s  (%d records)", path, len(records))
        return path


class Experiment:
    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        num_electrodes: int = 4,
        amplitudes_ua: Optional[List[float]] = None,
        durations_us: Optional[List[float]] = None,
        baseline_window_s: float = 5.0,
        post_stim_window_s: float = 5.0,
        inter_trial_interval_s: float = 3.0,
        trials_per_condition: int = 5,
        excitation_amplitude_ua: float = 3.0,
        excitation_duration_us: float = 200.0,
        inhibition_amplitude_ua: float = 1.0,
        inhibition_duration_us: float = 100.0,
        sham_trials: int = 3,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.num_electrodes = num_electrodes
        self.amplitudes_ua = amplitudes_ua if amplitudes_ua is not None else [1.0, 2.0, 3.0, 4.0]
        self.durations_us = durations_us if durations_us is not None else [100.0, 200.0, 300.0]
        self.baseline_window_s = baseline_window_s
        self.post_stim_window_s = post_stim_window_s
        self.inter_trial_interval_s = inter_trial_interval_s
        self.trials_per_condition = trials_per_condition
        self.excitation_amplitude_ua = min(excitation_amplitude_ua, 4.0)
        self.excitation_duration_us = min(excitation_duration_us, 400.0)
        self.inhibition_amplitude_ua = min(inhibition_amplitude_ua, 4.0)
        self.inhibition_duration_us = min(inhibition_duration_us, 400.0)
        self.sham_trials = sham_trials

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._trial_results: List[TrialResult] = []
        self._active_electrodes: List[int] = []
        self._baseline_rates: Dict[int, float] = {}

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

            recording_start = datetime.now(timezone.utc)

            self._phase_identify_active_electrodes()

            if not self._active_electrodes:
                logger.warning("No active electrodes found, using first %d electrodes", self.num_electrodes)
                self._active_electrodes = self.experiment.electrodes[:self.num_electrodes]

            self._phase_baseline_characterization()

            self._phase_excitation_protocol()

            self._phase_inhibition_protocol()

            self._phase_sham_control()

            self._phase_dose_response_sweep()

            recording_stop = datetime.now(timezone.utc)

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _phase_identify_active_electrodes(self) -> None:
        logger.info("Phase: Identifying active electrodes via spike counting")

        all_electrodes = self.experiment.electrodes
        electrode_spike_counts: Dict[int, int] = {}

        scan_window_s = 5.0
        scan_start = datetime.now(timezone.utc)
        time.sleep(scan_window_s)
        scan_stop = datetime.now(timezone.utc)

        fs_name = self.experiment.exp_name

        spike_df = self.database.get_spike_event(scan_start, scan_stop, fs_name)

        if not spike_df.empty:
            electrode_col = self._find_electrode_column(spike_df)
            if electrode_col is not None:
                for elec in all_electrodes:
                    count = len(spike_df[spike_df[electrode_col] == elec])
                    electrode_spike_counts[elec] = count
            else:
                for elec in all_electrodes:
                    elec_df = self.database.get_spike_event_electrode(scan_start, scan_stop, elec)
                    electrode_spike_counts[elec] = len(elec_df)
        else:
            for elec in all_electrodes:
                elec_df = self.database.get_spike_event_electrode(scan_start, scan_stop, elec)
                electrode_spike_counts[elec] = len(elec_df)

        sorted_electrodes = sorted(electrode_spike_counts.items(), key=lambda x: x[1], reverse=True)
        logger.info("Electrode activity ranking: %s", sorted_electrodes)

        self._active_electrodes = [
            elec for elec, count in sorted_electrodes[:self.num_electrodes]
            if count > 0
        ]

        if not self._active_electrodes and sorted_electrodes:
            self._active_electrodes = [elec for elec, _ in sorted_electrodes[:self.num_electrodes]]

        logger.info("Selected active electrodes: %s", self._active_electrodes)

    def _phase_baseline_characterization(self) -> None:
        logger.info("Phase: Baseline characterization of active electrodes")

        baseline_start = datetime.now(timezone.utc)
        time.sleep(self.baseline_window_s)
        baseline_stop = datetime.now(timezone.utc)

        for elec in self._active_electrodes:
            spike_df = self.database.get_spike_event_electrode(baseline_start, baseline_stop, elec)
            spike_count = len(spike_df)
            rate = spike_count / self.baseline_window_s if self.baseline_window_s > 0 else 0.0
            self._baseline_rates[elec] = rate
            logger.info("Electrode %d baseline rate: %.2f spikes/s (%d spikes in %.1fs)",
                        elec, rate, spike_count, self.baseline_window_s)

    def _phase_excitation_protocol(self) -> None:
        logger.info("Phase: Excitation protocol - attempting to INCREASE activity")

        amplitude = self.excitation_amplitude_ua
        duration = self.excitation_duration_us

        for trial in range(self.trials_per_condition):
            for elec in self._active_electrodes:
                pre_start = datetime.now(timezone.utc)
                time.sleep(self.baseline_window_s)
                pre_stop = datetime.now(timezone.utc)
                pre_spikes = self.database.get_spike_event_electrode(pre_start, pre_stop, elec)
                pre_count = len(pre_spikes)

                self._stimulate_electrode(
                    electrode_idx=elec,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    polarity=StimPolarity.NegativeFirst,
                    trigger_key=0,
                    phase="excitation",
                    condition=f"amp{amplitude}_dur{duration}",
                    trial=trial,
                )

                time.sleep(self.post_stim_window_s)
                post_stop = datetime.now(timezone.utc)
                post_start = post_stop - timedelta(seconds=self.post_stim_window_s)
                post_spikes = self.database.get_spike_event_electrode(post_start, post_stop, elec)
                post_count = len(post_spikes)

                result = TrialResult(
                    electrode_idx=elec,
                    phase="excitation",
                    condition=f"amp{amplitude}_dur{duration}",
                    trial=trial,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    baseline_spike_count=pre_count,
                    post_stim_spike_count=post_count,
                    spike_count_change=post_count - pre_count,
                    baseline_window_s=self.baseline_window_s,
                    post_stim_window_s=self.post_stim_window_s,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                )
                self._trial_results.append(result)
                logger.info("Excitation trial %d elec %d: pre=%d post=%d change=%d",
                            trial, elec, pre_count, post_count, post_count - pre_count)

                time.sleep(self.inter_trial_interval_s)

    def _phase_inhibition_protocol(self) -> None:
        logger.info("Phase: Inhibition protocol - attempting to DECREASE activity")

        amplitude = self.inhibition_amplitude_ua
        duration = self.inhibition_duration_us

        for trial in range(self.trials_per_condition):
            for elec in self._active_electrodes:
                pre_start = datetime.now(timezone.utc)
                time.sleep(self.baseline_window_s)
                pre_stop = datetime.now(timezone.utc)
                pre_spikes = self.database.get_spike_event_electrode(pre_start, pre_stop, elec)
                pre_count = len(pre_spikes)

                self._stimulate_electrode(
                    electrode_idx=elec,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    polarity=StimPolarity.PositiveFirst,
                    trigger_key=1,
                    phase="inhibition",
                    condition=f"amp{amplitude}_dur{duration}",
                    trial=trial,
                )

                time.sleep(self.post_stim_window_s)
                post_stop = datetime.now(timezone.utc)
                post_start = post_stop - timedelta(seconds=self.post_stim_window_s)
                post_spikes = self.database.get_spike_event_electrode(post_start, post_stop, elec)
                post_count = len(post_spikes)

                result = TrialResult(
                    electrode_idx=elec,
                    phase="inhibition",
                    condition=f"amp{amplitude}_dur{duration}",
                    trial=trial,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    baseline_spike_count=pre_count,
                    post_stim_spike_count=post_count,
                    spike_count_change=post_count - pre_count,
                    baseline_window_s=self.baseline_window_s,
                    post_stim_window_s=self.post_stim_window_s,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                )
                self._trial_results.append(result)
                logger.info("Inhibition trial %d elec %d: pre=%d post=%d change=%d",
                            trial, elec, pre_count, post_count, post_count - pre_count)

                time.sleep(self.inter_trial_interval_s)

    def _phase_sham_control(self) -> None:
        logger.info("Phase: Sham control - no stimulation, measure spontaneous variability")

        for trial in range(self.sham_trials):
            for elec in self._active_electrodes:
                pre_start = datetime.now(timezone.utc)
                time.sleep(self.baseline_window_s)
                pre_stop = datetime.now(timezone.utc)
                pre_spikes = self.database.get_spike_event_electrode(pre_start, pre_stop, elec)
                pre_count = len(pre_spikes)

                time.sleep(0.5)

                time.sleep(self.post_stim_window_s)
                post_stop = datetime.now(timezone.utc)
                post_start = post_stop - timedelta(seconds=self.post_stim_window_s)
                post_spikes = self.database.get_spike_event_electrode(post_start, post_stop, elec)
                post_count = len(post_spikes)

                result = TrialResult(
                    electrode_idx=elec,
                    phase="sham",
                    condition="no_stim",
                    trial=trial,
                    amplitude_ua=0.0,
                    duration_us=0.0,
                    baseline_spike_count=pre_count,
                    post_stim_spike_count=post_count,
                    spike_count_change=post_count - pre_count,
                    baseline_window_s=self.baseline_window_s,
                    post_stim_window_s=self.post_stim_window_s,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                )
                self._trial_results.append(result)
                logger.info("Sham trial %d elec %d: pre=%d post=%d change=%d",
                            trial, elec, pre_count, post_count, post_count - pre_count)

                time.sleep(self.inter_trial_interval_s)

    def _phase_dose_response_sweep(self) -> None:
        logger.info("Phase: Dose-response sweep across amplitudes and durations")

        if not self._active_electrodes:
            logger.warning("No active electrodes for dose-response sweep")
            return

        target_elec = self._active_electrodes[0]

        for amplitude in self.amplitudes_ua:
            amplitude = min(abs(amplitude), 4.0)
            for duration in self.durations_us:
                duration = min(abs(duration), 400.0)

                pre_start = datetime.now(timezone.utc)
                time.sleep(self.baseline_window_s)
                pre_stop = datetime.now(timezone.utc)
                pre_spikes = self.database.get_spike_event_electrode(pre_start, pre_stop, target_elec)
                pre_count = len(pre_spikes)

                self._stimulate_electrode(
                    electrode_idx=target_elec,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    polarity=StimPolarity.NegativeFirst,
                    trigger_key=2,
                    phase="dose_response",
                    condition=f"amp{amplitude}_dur{duration}",
                    trial=0,
                )

                time.sleep(self.post_stim_window_s)
                post_stop = datetime.now(timezone.utc)
                post_start = post_stop - timedelta(seconds=self.post_stim_window_s)
                post_spikes = self.database.get_spike_event_electrode(post_start, post_stop, target_elec)
                post_count = len(post_spikes)

                result = TrialResult(
                    electrode_idx=target_elec,
                    phase="dose_response",
                    condition=f"amp{amplitude}_dur{duration}",
                    trial=0,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    baseline_spike_count=pre_count,
                    post_stim_spike_count=post_count,
                    spike_count_change=post_count - pre_count,
                    baseline_window_s=self.baseline_window_s,
                    post_stim_window_s=self.post_stim_window_s,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                )
                self._trial_results.append(result)
                logger.info("Dose-response amp=%.1f dur=%.0f elec %d: pre=%d post=%d change=%d",
                            amplitude, duration, target_elec, pre_count, post_count, post_count - pre_count)

                time.sleep(self.inter_trial_interval_s)

    def _stimulate_electrode(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity = StimPolarity.NegativeFirst,
        trigger_key: int = 0,
        phase: str = "",
        condition: str = "",
        trial: int = 0,
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
        time.sleep(0.05)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            trigger_key=trigger_key,
            phase=phase,
            condition=condition,
            trial=trial,
        ))

    def _find_electrode_column(self, df: pd.DataFrame) -> Optional[str]:
        if "channel" in df.columns:
            return "channel"
        if "index" in df.columns:
            return "index"
        for col in df.columns:
            if "electrode" in col.lower() or "idx" in col.lower() or "channel" in col.lower():
                return col
        return None

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

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "active_electrodes": self._active_electrodes,
            "baseline_rates": {str(k): v for k, v in self._baseline_rates.items()},
            "total_trials": len(self._trial_results),
            "phases_completed": list(set(t.phase for t in self._trial_results)),
            "excitation_params": {
                "amplitude_ua": self.excitation_amplitude_ua,
                "duration_us": self.excitation_duration_us,
            },
            "inhibition_params": {
                "amplitude_ua": self.inhibition_amplitude_ua,
                "duration_us": self.inhibition_duration_us,
            },
        }
        saver.save_summary(summary)

        waveform_records = self._fetch_spike_waveforms(
            fs_name, spike_df, trigger_df, recording_start, recording_stop
        )
        saver.save_spike_waveforms(waveform_records)

        saver.save_trial_results(self._trial_results)

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

        electrode_col = self._find_electrode_column(spike_df)
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

        excitation_results = [t for t in self._trial_results if t.phase == "excitation"]
        inhibition_results = [t for t in self._trial_results if t.phase == "inhibition"]
        sham_results = [t for t in self._trial_results if t.phase == "sham"]
        dose_response_results = [t for t in self._trial_results if t.phase == "dose_response"]

        def compute_stats(trials: List[TrialResult]) -> Dict[str, Any]:
            if not trials:
                return {"count": 0, "mean_change": 0.0, "std_change": 0.0, "increases": 0, "decreases": 0, "no_change": 0}
            changes = [t.spike_count_change for t in trials]
            mean_change = sum(changes) / len(changes) if changes else 0.0
            variance = sum((c - mean_change) ** 2 for c in changes) / len(changes) if len(changes) > 1 else 0.0
            std_change = math.sqrt(variance)
            increases = sum(1 for c in changes if c > 0)
            decreases = sum(1 for c in changes if c < 0)
            no_change = sum(1 for c in changes if c == 0)
            return {
                "count": len(trials),
                "mean_change": round(mean_change, 3),
                "std_change": round(std_change, 3),
                "increases": increases,
                "decreases": decreases,
                "no_change": no_change,
            }

        per_electrode_excitation: Dict[str, Any] = {}
        for elec in self._active_electrodes:
            elec_trials = [t for t in excitation_results if t.electrode_idx == elec]
            per_electrode_excitation[str(elec)] = compute_stats(elec_trials)

        per_electrode_inhibition: Dict[str, Any] = {}
        for elec in self._active_electrodes:
            elec_trials = [t for t in inhibition_results if t.electrode_idx == elec]
            per_electrode_inhibition[str(elec)] = compute_stats(elec_trials)

        dose_response_summary: List[Dict[str, Any]] = []
        for t in dose_response_results:
            dose_response_summary.append({
                "electrode": t.electrode_idx,
                "amplitude_ua": t.amplitude_ua,
                "duration_us": t.duration_us,
                "baseline_count": t.baseline_spike_count,
                "post_count": t.post_stim_spike_count,
                "change": t.spike_count_change,
            })

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "active_electrodes": self._active_electrodes,
            "baseline_rates_spikes_per_s": {str(k): round(v, 3) for k, v in self._baseline_rates.items()},
            "excitation_summary": compute_stats(excitation_results),
            "inhibition_summary": compute_stats(inhibition_results),
            "sham_summary": compute_stats(sham_results),
            "per_electrode_excitation": per_electrode_excitation,
            "per_electrode_inhibition": per_electrode_inhibition,
            "dose_response": dose_response_summary,
            "total_stimulations": len(self._stimulation_log),
            "total_trials": len(self._trial_results),
            "hypothesis_context": {
                "primary_hypothesis": "Uniform weak electric field stimulation at ~1 V/m will fail to reliably induce network state transitions in organoids, requiring 2-5x higher thresholds.",
                "approach": "Target active electrodes with excitation (high amplitude, negative-first) and inhibition (low amplitude, positive-first) protocols to modulate activity.",
                "citations": ["arxiv:1906.00676", "arxiv:2210.15957", "arxiv:2402.05886"],
            },
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
