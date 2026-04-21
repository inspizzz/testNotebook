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
    timestamp_utc: str
    phase: str
    trigger_key: int = 0
    train_index: int = -1
    stim_index_in_train: int = -1
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialResult:
    trial_index: int
    phase: str
    timestamp_utc: str
    stim_electrode: int
    resp_electrode: int
    responded: bool
    spike_count: int
    latency_ms: float


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
    LTP induction experiment on the FinalSpark NeuroPlatform.

    Phase 1 - Baseline (15 min): single stimulations at 0.1 Hz, record response probability.
    Phase 2 - Tetanic stimulation: 4 trains of 100 stimulations at 5 Hz, trains separated by 30 s.
    Phase 3 - Post-tetanus monitoring (30 min): resume 0.1 Hz single stimulation, track response probability.

    Top responsive electrode pair from scan: electrode 17 (stim) -> electrode 18 (resp),
    response_rate=0.92, PositiveFirst polarity, 3 uA / 300 us.
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        stim_electrode: int = 17,
        resp_electrode: int = 18,
        amplitude_ua: float = 3.0,
        duration_us: float = 300.0,
        polarity: str = "PositiveFirst",
        baseline_duration_s: float = 900.0,
        baseline_freq_hz: float = 0.1,
        tetanus_n_trains: int = 4,
        tetanus_stims_per_train: int = 100,
        tetanus_freq_hz: float = 5.0,
        inter_train_interval_s: float = 30.0,
        post_duration_s: float = 1800.0,
        post_freq_hz: float = 0.1,
        response_window_ms: float = 50.0,
        trigger_key: int = 0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.stim_electrode = stim_electrode
        self.resp_electrode = resp_electrode
        self.amplitude_ua = min(abs(amplitude_ua), 4.0)
        self.duration_us = min(abs(duration_us), 400.0)
        self.polarity_str = polarity
        self.polarity = StimPolarity.PositiveFirst if polarity == "PositiveFirst" else StimPolarity.NegativeFirst

        self.baseline_duration_s = baseline_duration_s
        self.baseline_freq_hz = baseline_freq_hz
        self.tetanus_n_trains = tetanus_n_trains
        self.tetanus_stims_per_train = tetanus_stims_per_train
        self.tetanus_freq_hz = tetanus_freq_hz
        self.inter_train_interval_s = inter_train_interval_s
        self.post_duration_s = post_duration_s
        self.post_freq_hz = post_freq_hz
        self.response_window_ms = response_window_ms
        self.trigger_key = trigger_key

        # Charge balance: A1*D1 == A2*D2
        # We use equal amplitudes and equal durations on both phases.
        self.phase_amplitude1 = self.amplitude_ua
        self.phase_duration1 = self.duration_us
        self.phase_amplitude2 = self.amplitude_ua
        self.phase_duration2 = self.duration_us

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._trial_results: List[TrialResult] = []

        self._baseline_response_count = 0
        self._baseline_trial_count = 0
        self._post_response_count = 0
        self._post_trial_count = 0

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
            logger.info(
                "Stim electrode: %d -> Resp electrode: %d",
                self.stim_electrode,
                self.resp_electrode,
            )

            if not self.experiment.start():
                logger.error("Failed to start experiment")
                return {"status": "failed", "error": "Could not start experiment"}

            recording_start = datetime_now()

            self._configure_stim_params()

            logger.info("=== Phase 1: Baseline (%.0f s at %.2f Hz) ===", self.baseline_duration_s, self.baseline_freq_hz)
            self._phase_baseline()

            logger.info("=== Phase 2: Tetanic Stimulation ===")
            self._phase_tetanus()

            logger.info("=== Phase 3: Post-Tetanus Monitoring (%.0f s at %.2f Hz) ===", self.post_duration_s, self.post_freq_hz)
            self._phase_post_tetanus()

            recording_stop = datetime_now()

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _configure_stim_params(self) -> None:
        stim = self._build_stim_param(self.stim_electrode, self.trigger_key)
        self.intan.send_stimparam([stim])
        logger.info(
            "Stim params configured: electrode=%d, A=%.1f uA, D=%.0f us, polarity=%s",
            self.stim_electrode,
            self.amplitude_ua,
            self.duration_us,
            self.polarity_str,
        )

    def _build_stim_param(self, electrode_idx: int, trigger_key: int = 0) -> StimParam:
        stim = StimParam()
        stim.index = electrode_idx
        stim.enable = True
        stim.trigger_key = trigger_key
        stim.trigger_delay = 0
        stim.nb_pulse = 0
        stim.pulse_train_period = 10000
        stim.post_stim_ref_period = 1000.0
        stim.stim_shape = StimShape.Biphasic
        stim.polarity = self.polarity
        stim.phase_amplitude1 = self.phase_amplitude1
        stim.phase_duration1 = self.phase_duration1
        stim.phase_amplitude2 = self.phase_amplitude2
        stim.phase_duration2 = self.phase_duration2
        stim.enable_amp_settle = True
        stim.pre_stim_amp_settle = 0.0
        stim.post_stim_amp_settle = 1000.0
        stim.enable_charge_recovery = True
        stim.post_charge_recovery_on = 0.0
        stim.post_charge_recovery_off = 100.0
        stim.interphase_delay = 0.0
        return stim

    def _fire_trigger(self) -> None:
        pattern = np.zeros(16, dtype=np.uint8)
        pattern[self.trigger_key] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.02)
        pattern[self.trigger_key] = 0
        self.trigger_controller.send(pattern)

    def _single_stimulation(self, phase: str, trial_index: int, train_index: int = -1, stim_index_in_train: int = -1) -> bool:
        ts = datetime_now()
        self._fire_trigger()

        self._stimulation_log.append(StimulationRecord(
            electrode_idx=self.stim_electrode,
            amplitude_ua=self.amplitude_ua,
            duration_us=self.duration_us,
            polarity=self.polarity_str,
            timestamp_utc=ts.isoformat(),
            phase=phase,
            trigger_key=self.trigger_key,
            train_index=train_index,
            stim_index_in_train=stim_index_in_train,
        ))

        response_window_s = self.response_window_ms / 1000.0
        self._wait(response_window_s + 0.01)

        query_stop = datetime_now()
        query_start = query_stop - timedelta(seconds=response_window_s + 0.05)

        try:
            spike_df = self.database.get_spike_event(
                query_start, query_stop, self.experiment.exp_name
            )
            if not spike_df.empty:
                resp_spikes = spike_df[spike_df["channel"] == self.resp_electrode]
                responded = len(resp_spikes) > 0
                spike_count = len(resp_spikes)
                if responded and "Time" in resp_spikes.columns:
                    first_spike_time = pd.to_datetime(resp_spikes["Time"].iloc[0], utc=True)
                    latency_ms = (first_spike_time - ts).total_seconds() * 1000.0
                else:
                    latency_ms = float("nan")
            else:
                responded = False
                spike_count = 0
                latency_ms = float("nan")
        except Exception as exc:
            logger.warning("Spike query failed for trial %d: %s", trial_index, exc)
            responded = False
            spike_count = 0
            latency_ms = float("nan")

        self._trial_results.append(TrialResult(
            trial_index=trial_index,
            phase=phase,
            timestamp_utc=ts.isoformat(),
            stim_electrode=self.stim_electrode,
            resp_electrode=self.resp_electrode,
            responded=responded,
            spike_count=spike_count,
            latency_ms=latency_ms,
        ))

        return responded

    def _phase_baseline(self) -> None:
        inter_stim_interval_s = 1.0 / self.baseline_freq_hz
        n_stims = int(self.baseline_duration_s * self.baseline_freq_hz)
        logger.info("Baseline: %d stimulations at %.2f Hz", n_stims, self.baseline_freq_hz)

        for i in range(n_stims):
            responded = self._single_stimulation(phase="baseline", trial_index=i)
            if responded:
                self._baseline_response_count += 1
            self._baseline_trial_count += 1

            if (i + 1) % 10 == 0:
                prob = self._baseline_response_count / self._baseline_trial_count if self._baseline_trial_count > 0 else 0.0
                logger.info(
                    "Baseline trial %d/%d | Response prob: %.3f",
                    i + 1, n_stims, prob
                )

            wait_time = inter_stim_interval_s - (self.response_window_ms / 1000.0) - 0.03
            if wait_time > 0:
                self._wait(wait_time)

        baseline_prob = self._baseline_response_count / self._baseline_trial_count if self._baseline_trial_count > 0 else 0.0
        logger.info(
            "Baseline complete: %d/%d responded (prob=%.3f)",
            self._baseline_response_count,
            self._baseline_trial_count,
            baseline_prob,
        )

    def _phase_tetanus(self) -> None:
        inter_stim_interval_s = 1.0 / self.tetanus_freq_hz
        logger.info(
            "Tetanus: %d trains x %d stims at %.1f Hz, inter-train interval=%.0f s",
            self.tetanus_n_trains,
            self.tetanus_stims_per_train,
            self.tetanus_freq_hz,
            self.inter_train_interval_s,
        )

        global_trial_index = self._baseline_trial_count

        for train_idx in range(self.tetanus_n_trains):
            logger.info("Tetanus train %d/%d", train_idx + 1, self.tetanus_n_trains)
            for stim_idx in range(self.tetanus_stims_per_train):
                self._single_stimulation(
                    phase="tetanus",
                    trial_index=global_trial_index,
                    train_index=train_idx,
                    stim_index_in_train=stim_idx,
                )
                global_trial_index += 1

                wait_time = inter_stim_interval_s - (self.response_window_ms / 1000.0) - 0.03
                if wait_time > 0:
                    self._wait(wait_time)

            if train_idx < self.tetanus_n_trains - 1:
                logger.info(
                    "Inter-train pause: %.0f s", self.inter_train_interval_s
                )
                self._wait(self.inter_train_interval_s)

        logger.info("Tetanus complete")

    def _phase_post_tetanus(self) -> None:
        inter_stim_interval_s = 1.0 / self.post_freq_hz
        n_stims = int(self.post_duration_s * self.post_freq_hz)
        logger.info("Post-tetanus: %d stimulations at %.2f Hz", n_stims, self.post_freq_hz)

        global_trial_index = self._baseline_trial_count + self.tetanus_n_trains * self.tetanus_stims_per_train

        for i in range(n_stims):
            responded = self._single_stimulation(phase="post_tetanus", trial_index=global_trial_index + i)
            if responded:
                self._post_response_count += 1
            self._post_trial_count += 1

            if (i + 1) % 10 == 0:
                prob = self._post_response_count / self._post_trial_count if self._post_trial_count > 0 else 0.0
                logger.info(
                    "Post-tetanus trial %d/%d | Response prob: %.3f",
                    i + 1, n_stims, prob
                )

            wait_time = inter_stim_interval_s - (self.response_window_ms / 1000.0) - 0.03
            if wait_time > 0:
                self._wait(wait_time)

        post_prob = self._post_response_count / self._post_trial_count if self._post_trial_count > 0 else 0.0
        logger.info(
            "Post-tetanus complete: %d/%d responded (prob=%.3f)",
            self._post_response_count,
            self._post_trial_count,
            post_prob,
        )

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        logger.info("Compiling results")

        baseline_prob = (
            self._baseline_response_count / self._baseline_trial_count
            if self._baseline_trial_count > 0 else 0.0
        )
        post_prob = (
            self._post_response_count / self._post_trial_count
            if self._post_trial_count > 0 else 0.0
        )
        ltp_delta = post_prob - baseline_prob

        tetanus_trials = [r for r in self._trial_results if r.phase == "tetanus"]
        tetanus_responded = sum(1 for r in tetanus_trials if r.responded)

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "stim_electrode": self.stim_electrode,
            "resp_electrode": self.resp_electrode,
            "amplitude_ua": self.amplitude_ua,
            "duration_us": self.duration_us,
            "polarity": self.polarity_str,
            "baseline_trials": self._baseline_trial_count,
            "baseline_responses": self._baseline_response_count,
            "baseline_response_probability": baseline_prob,
            "tetanus_trains": self.tetanus_n_trains,
            "tetanus_stims_per_train": self.tetanus_stims_per_train,
            "tetanus_total_stims": len(tetanus_trials),
            "tetanus_responses": tetanus_responded,
            "post_tetanus_trials": self._post_trial_count,
            "post_tetanus_responses": self._post_response_count,
            "post_tetanus_response_probability": post_prob,
            "ltp_delta_probability": ltp_delta,
            "total_stimulations": len(self._stimulation_log),
        }

        logger.info(
            "Baseline prob=%.3f | Post-tetanus prob=%.3f | Delta=%.3f",
            baseline_prob, post_prob, ltp_delta
        )

        return summary

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

        baseline_prob = (
            self._baseline_response_count / self._baseline_trial_count
            if self._baseline_trial_count > 0 else 0.0
        )
        post_prob = (
            self._post_response_count / self._post_trial_count
            if self._post_trial_count > 0 else 0.0
        )

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "stim_electrode": self.stim_electrode,
            "resp_electrode": self.resp_electrode,
            "amplitude_ua": self.amplitude_ua,
            "duration_us": self.duration_us,
            "polarity": self.polarity_str,
            "baseline_trials": self._baseline_trial_count,
            "baseline_responses": self._baseline_response_count,
            "baseline_response_probability": baseline_prob,
            "post_tetanus_trials": self._post_trial_count,
            "post_tetanus_responses": self._post_response_count,
            "post_tetanus_response_probability": post_prob,
            "ltp_delta_probability": post_prob - baseline_prob,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "trial_results": [asdict(r) for r in self._trial_results],
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
