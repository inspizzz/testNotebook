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
    """
    Experiment: amplitude sweep to identify most responsive electrodes,
    followed by paired STDP-style conditioning (two electrodes stimulated
    close together in time) and a post-conditioning probe to assess plasticity.

    Phases:
      1. Baseline recording (30 s quiet observation)
      2. Amplitude sweep across known-responsive electrode pairs at multiple
         power levels to rank responsiveness
      3. STDP conditioning: stimulate pre-electrode, then post-electrode with
         a short delay (10 ms) for a fixed number of paired trials
      4. Post-conditioning probe: repeat the amplitude sweep to detect changes
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        sweep_amplitudes: tuple = (1.0, 2.0, 3.0),
        sweep_duration_us: float = 400.0,
        sweep_trials_per_condition: int = 10,
        stdp_pre_electrode: int = 7,
        stdp_post_electrode: int = 6,
        stdp_amplitude_ua: float = 3.0,
        stdp_duration_us: float = 400.0,
        stdp_delay_ms: float = 10.0,
        stdp_num_pairs: int = 50,
        stdp_isi_s: float = 1.0,
        probe_trials: int = 10,
        baseline_duration_s: float = 30.0,
        inter_stim_wait_s: float = 0.5,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.sweep_amplitudes = list(sweep_amplitudes)
        self.sweep_duration_us = sweep_duration_us
        self.sweep_trials_per_condition = sweep_trials_per_condition

        self.stdp_pre_electrode = stdp_pre_electrode
        self.stdp_post_electrode = stdp_post_electrode
        self.stdp_amplitude_ua = stdp_amplitude_ua
        self.stdp_duration_us = stdp_duration_us
        self.stdp_delay_ms = stdp_delay_ms
        self.stdp_num_pairs = stdp_num_pairs
        self.stdp_isi_s = stdp_isi_s

        self.probe_trials = probe_trials
        self.baseline_duration_s = baseline_duration_s
        self.inter_stim_wait_s = inter_stim_wait_s

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []

        self._sweep_results_pre: Dict[str, Any] = {}
        self._sweep_results_post: Dict[str, Any] = {}
        self._stdp_results: List[Dict[str, Any]] = []

        self._electrode_pairs = [
            {"from": 17, "to": 18, "polarity": StimPolarity.PositiveFirst},
            {"from": 21, "to": 19, "polarity": StimPolarity.PositiveFirst},
            {"from": 21, "to": 22, "polarity": StimPolarity.NegativeFirst},
            {"from": 7,  "to": 6,  "polarity": StimPolarity.PositiveFirst},
            {"from": 6,  "to": 7,  "polarity": StimPolarity.PositiveFirst},
            {"from": 5,  "to": 4,  "polarity": StimPolarity.PositiveFirst},
            {"from": 13, "to": 14, "polarity": StimPolarity.NegativeFirst},
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

            self._phase_baseline()
            self._phase_amplitude_sweep(label="pre")
            self._phase_stdp_conditioning()
            self._phase_amplitude_sweep(label="post")

            recording_stop = datetime_now()

            self._save_all(recording_start, recording_stop)

            results = self._compile_results(recording_start, recording_stop)
            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _phase_baseline(self) -> None:
        logger.info("Phase 1: Baseline recording for %.1f s", self.baseline_duration_s)
        self._wait(self.baseline_duration_s)
        logger.info("Baseline recording complete")

    def _phase_amplitude_sweep(self, label: str = "pre") -> None:
        logger.info("Phase: Amplitude sweep (%s-conditioning)", label)
        results: Dict[str, Any] = {}

        for pair in self._electrode_pairs:
            stim_elec = pair["from"]
            resp_elec = pair["to"]
            polarity = pair["polarity"]
            pair_key = f"e{stim_elec}_to_e{resp_elec}"
            results[pair_key] = {}

            for amplitude in self.sweep_amplitudes:
                duration = self.sweep_duration_us
                amplitude_clamped = min(amplitude, 4.0)
                duration_clamped = min(duration, 400.0)

                trial_spike_counts = []
                for trial_idx in range(self.sweep_trials_per_condition):
                    spike_df = self._stimulate_and_record(
                        electrode_idx=stim_elec,
                        amplitude_ua=amplitude_clamped,
                        duration_us=duration_clamped,
                        polarity=polarity,
                        trigger_key=0,
                        post_stim_wait_s=0.1,
                        recording_window_s=0.1,
                        phase_label=label,
                    )
                    n_spikes = len(spike_df) if not spike_df.empty else 0
                    trial_spike_counts.append(n_spikes)
                    self._wait(self.inter_stim_wait_s)

                mean_spikes = float(np.mean(trial_spike_counts)) if trial_spike_counts else 0.0
                results[pair_key][f"amp_{amplitude_clamped}"] = {
                    "mean_spike_count": mean_spikes,
                    "trial_counts": trial_spike_counts,
                }
                logger.info(
                    "[%s] Pair %s amp=%.1f uA -> mean spikes=%.2f",
                    label, pair_key, amplitude_clamped, mean_spikes,
                )

        if label == "pre":
            self._sweep_results_pre = results
        else:
            self._sweep_results_post = results

        logger.info("Amplitude sweep (%s) complete", label)

    def _phase_stdp_conditioning(self) -> None:
        logger.info(
            "Phase: STDP conditioning - pre=%d post=%d delay=%.1f ms pairs=%d",
            self.stdp_pre_electrode,
            self.stdp_post_electrode,
            self.stdp_delay_ms,
            self.stdp_num_pairs,
        )

        pre_elec = self.stdp_pre_electrode
        post_elec = self.stdp_post_electrode
        amplitude = min(self.stdp_amplitude_ua, 4.0)
        duration = min(self.stdp_duration_us, 400.0)
        delay_s = self.stdp_delay_ms / 1000.0

        for pair_idx in range(self.stdp_num_pairs):
            t_pre = datetime_now()

            self._send_single_pulse(
                electrode_idx=pre_elec,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=StimPolarity.PositiveFirst,
                trigger_key=0,
                phase_label="stdp_pre",
            )

            self._wait(delay_s)

            self._send_single_pulse(
                electrode_idx=post_elec,
                amplitude_ua=amplitude,
                duration_us=duration,
                polarity=StimPolarity.PositiveFirst,
                trigger_key=1,
                phase_label="stdp_post",
            )

            t_post = datetime_now()
            self._stdp_results.append({
                "pair_idx": pair_idx,
                "pre_electrode": pre_elec,
                "post_electrode": post_elec,
                "amplitude_ua": amplitude,
                "duration_us": duration,
                "delay_ms": self.stdp_delay_ms,
                "t_pre_utc": t_pre.isoformat(),
                "t_post_utc": t_post.isoformat(),
            })

            self._wait(self.stdp_isi_s)

            if (pair_idx + 1) % 10 == 0:
                logger.info("STDP conditioning: %d / %d pairs done", pair_idx + 1, self.stdp_num_pairs)

        logger.info("STDP conditioning complete")

    def _send_single_pulse(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity,
        trigger_key: int,
        phase_label: str,
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

    def _stimulate_and_record(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity = StimPolarity.NegativeFirst,
        trigger_key: int = 0,
        post_stim_wait_s: float = 0.1,
        recording_window_s: float = 0.1,
        phase_label: str = "sweep",
    ) -> pd.DataFrame:
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

        self._wait(post_stim_wait_s)

        query_stop = datetime_now()
        query_start = query_stop - timedelta(seconds=post_stim_wait_s + recording_window_s)
        try:
            spike_df = self.database.get_spike_event_electrode(
                query_start, query_stop, electrode_idx
            )
        except Exception as exc:
            logger.warning("Spike query failed for electrode %d: %s", electrode_idx, exc)
            spike_df = pd.DataFrame()

        return spike_df

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
            "stdp_pairs_delivered": len(self._stdp_results),
            "stdp_pre_electrode": self.stdp_pre_electrode,
            "stdp_post_electrode": self.stdp_post_electrode,
            "stdp_delay_ms": self.stdp_delay_ms,
            "sweep_amplitudes": self.sweep_amplitudes,
            "sweep_results_pre": self._sweep_results_pre,
            "sweep_results_post": self._sweep_results_post,
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
                    electrode_idx, exc,
                )

        return waveform_records

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        logger.info("Compiling results")

        responsiveness_delta: Dict[str, Any] = {}
        for pair_key in self._sweep_results_pre:
            if pair_key in self._sweep_results_post:
                responsiveness_delta[pair_key] = {}
                for amp_key in self._sweep_results_pre[pair_key]:
                    pre_mean = self._sweep_results_pre[pair_key][amp_key]["mean_spike_count"]
                    post_mean = self._sweep_results_post[pair_key].get(amp_key, {}).get("mean_spike_count", 0.0)
                    delta = post_mean - pre_mean
                    responsiveness_delta[pair_key][amp_key] = {
                        "pre_mean": pre_mean,
                        "post_mean": post_mean,
                        "delta": delta,
                    }

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "total_stimulations": len(self._stimulation_log),
            "stdp_pairs_delivered": len(self._stdp_results),
            "stdp_pre_electrode": self.stdp_pre_electrode,
            "stdp_post_electrode": self.stdp_post_electrode,
            "stdp_delay_ms": self.stdp_delay_ms,
            "sweep_results_pre": self._sweep_results_pre,
            "sweep_results_post": self._sweep_results_post,
            "responsiveness_delta": responsiveness_delta,
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
