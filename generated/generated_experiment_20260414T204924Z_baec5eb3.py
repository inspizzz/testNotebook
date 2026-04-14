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
    trial_index: int
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
    LTP induction experiment using electrode pair 14->15 (highest response rate 0.94
    from deep scan). Three phases:
      Phase 1 - Baseline (15 min): 0.1 Hz single stimulations
      Phase 2 - Tetanic: 4 trains x 100 stimulations at 5 Hz, trains separated by 30 s
      Phase 3 - Post-tetanus (30 min): 0.1 Hz single stimulations

    Stimulation: 3 uA amplitude, 300 us duration, charge-balanced biphasic.
    Charge balance: A1*D1 = 3.0*300 = 900; A2*D2 = 3.0*300 = 900. Balanced.
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        stim_electrode: int = 14,
        resp_electrode: int = 15,
        amplitude_ua: float = 3.0,
        duration_us: float = 300.0,
        polarity: str = "NegativeFirst",
        baseline_duration_s: float = 900.0,
        post_tetanus_duration_s: float = 1800.0,
        baseline_freq_hz: float = 0.1,
        tetanic_trains: int = 4,
        tetanic_stims_per_train: int = 100,
        tetanic_freq_hz: float = 5.0,
        inter_train_interval_s: float = 30.0,
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
        self.polarity = StimPolarity.NegativeFirst if polarity == "NegativeFirst" else StimPolarity.PositiveFirst

        self.baseline_duration_s = baseline_duration_s
        self.post_tetanus_duration_s = post_tetanus_duration_s
        self.baseline_freq_hz = baseline_freq_hz
        self.tetanic_trains = tetanic_trains
        self.tetanic_stims_per_train = tetanic_stims_per_train
        self.tetanic_freq_hz = tetanic_freq_hz
        self.inter_train_interval_s = inter_train_interval_s
        self.trigger_key = trigger_key

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []

        self._baseline_results: List[Dict[str, Any]] = []
        self._tetanic_results: List[Dict[str, Any]] = []
        self._post_tetanus_results: List[Dict[str, Any]] = []

        self._phase1_start: Optional[datetime] = None
        self._phase1_stop: Optional[datetime] = None
        self._phase2_start: Optional[datetime] = None
        self._phase2_stop: Optional[datetime] = None
        self._phase3_start: Optional[datetime] = None
        self._phase3_stop: Optional[datetime] = None

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
            logger.info("Electrodes available: %s", self.experiment.electrodes)
            logger.info(
                "LTP experiment: stim electrode %d -> resp electrode %d",
                self.stim_electrode, self.resp_electrode
            )
            logger.info(
                "Stimulation params: %.1f uA, %.1f us, %s",
                self.amplitude_ua, self.duration_us, self.polarity_str
            )

            if not self.experiment.start():
                logger.error("Failed to start experiment")
                return {"status": "failed", "error": "Could not start experiment"}

            recording_start = datetime_now()

            self._configure_stim_param()

            logger.info("=== Phase 1: Baseline (%.0f s at %.2f Hz) ===", self.baseline_duration_s, self.baseline_freq_hz)
            self._phase1_start = datetime_now()
            self._phase_baseline()
            self._phase1_stop = datetime_now()

            logger.info("=== Phase 2: Tetanic Stimulation ===")
            self._phase2_start = datetime_now()
            self._phase_tetanic()
            self._phase2_stop = datetime_now()

            logger.info("=== Phase 3: Post-Tetanus Monitoring (%.0f s at %.2f Hz) ===", self.post_tetanus_duration_s, self.baseline_freq_hz)
            self._phase3_start = datetime_now()
            self._phase_post_tetanus()
            self._phase3_stop = datetime_now()

            recording_stop = datetime_now()

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _configure_stim_param(self) -> None:
        stim = self._build_stim_param(nb_pulse=0)
        self.intan.send_stimparam([stim])
        logger.info("Stimulation parameters configured on electrode %d", self.stim_electrode)

    def _build_stim_param(self, nb_pulse: int = 0, pulse_train_period_us: int = 10000) -> StimParam:
        stim = StimParam()
        stim.index = self.stim_electrode
        stim.enable = True
        stim.trigger_key = self.trigger_key
        stim.trigger_delay = 0
        stim.nb_pulse = nb_pulse
        stim.pulse_train_period = pulse_train_period_us
        stim.post_stim_ref_period = 1000.0
        stim.stim_shape = StimShape.Biphasic
        stim.polarity = self.polarity
        stim.phase_amplitude1 = self.amplitude_ua
        stim.phase_duration1 = self.duration_us
        stim.phase_amplitude2 = self.amplitude_ua
        stim.phase_duration2 = self.duration_us
        stim.enable_amp_settle = True
        stim.pre_stim_amp_settle = 0.0
        stim.post_stim_amp_settle = 1000.0
        stim.enable_charge_recovery = True
        stim.post_charge_recovery_on = 0.0
        stim.post_charge_recovery_off = 100.0
        stim.interphase_delay = 0.0
        return stim

    def _fire_single_trigger(self) -> None:
        pattern = np.zeros(16, dtype=np.uint8)
        pattern[self.trigger_key] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.02)
        pattern[self.trigger_key] = 0
        self.trigger_controller.send(pattern)

    def _stimulate_single(self, trial_index: int, phase: str) -> None:
        self._fire_single_trigger()
        ts = datetime_now().isoformat()
        self._stimulation_log.append(StimulationRecord(
            electrode_idx=self.stim_electrode,
            amplitude_ua=self.amplitude_ua,
            duration_us=self.duration_us,
            polarity=self.polarity_str,
            phase=phase,
            trial_index=trial_index,
            timestamp_utc=ts,
            trigger_key=self.trigger_key,
        ))

    def _phase_baseline(self) -> None:
        inter_stim_interval_s = 1.0 / self.baseline_freq_hz
        num_stims = int(self.baseline_duration_s * self.baseline_freq_hz)
        logger.info("Baseline: %d stimulations, ISI=%.1f s", num_stims, inter_stim_interval_s)

        for i in range(num_stims):
            stim_time = datetime_now()
            self._stimulate_single(trial_index=i, phase="baseline")
            self._baseline_results.append({
                "trial": i,
                "stim_time": stim_time.isoformat(),
            })
            if i < num_stims - 1:
                self._wait(inter_stim_interval_s - 0.02)

        logger.info("Baseline phase complete: %d stimulations delivered", num_stims)

    def _phase_tetanic(self) -> None:
        inter_stim_interval_s = 1.0 / self.tetanic_freq_hz
        logger.info(
            "Tetanic: %d trains x %d stims at %.1f Hz, inter-train interval=%.1f s",
            self.tetanic_trains, self.tetanic_stims_per_train,
            self.tetanic_freq_hz, self.inter_train_interval_s
        )

        for train_idx in range(self.tetanic_trains):
            logger.info("Tetanic train %d/%d", train_idx + 1, self.tetanic_trains)
            train_start = datetime_now()

            for stim_idx in range(self.tetanic_stims_per_train):
                global_trial = train_idx * self.tetanic_stims_per_train + stim_idx
                self._stimulate_single(trial_index=global_trial, phase="tetanic")
                self._tetanic_results.append({
                    "train": train_idx,
                    "stim_in_train": stim_idx,
                    "global_trial": global_trial,
                    "stim_time": datetime_now().isoformat(),
                })
                if stim_idx < self.tetanic_stims_per_train - 1:
                    self._wait(inter_stim_interval_s - 0.02)

            train_stop = datetime_now()
            logger.info(
                "Train %d complete in %.2f s",
                train_idx + 1,
                (train_stop - train_start).total_seconds()
            )

            if train_idx < self.tetanic_trains - 1:
                logger.info("Inter-train interval: %.1f s", self.inter_train_interval_s)
                self._wait(self.inter_train_interval_s)

        logger.info("Tetanic phase complete")

    def _phase_post_tetanus(self) -> None:
        inter_stim_interval_s = 1.0 / self.baseline_freq_hz
        num_stims = int(self.post_tetanus_duration_s * self.baseline_freq_hz)
        logger.info("Post-tetanus: %d stimulations, ISI=%.1f s", num_stims, inter_stim_interval_s)

        for i in range(num_stims):
            stim_time = datetime_now()
            self._stimulate_single(trial_index=i, phase="post_tetanus")
            self._post_tetanus_results.append({
                "trial": i,
                "stim_time": stim_time.isoformat(),
            })
            if i < num_stims - 1:
                self._wait(inter_stim_interval_s - 0.02)

        logger.info("Post-tetanus phase complete: %d stimulations delivered", num_stims)

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        logger.info("Compiling results")

        total_baseline = len(self._baseline_results)
        total_tetanic = len(self._tetanic_results)
        total_post = len(self._post_tetanus_results)

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
            "charge_balance_check": self.amplitude_ua * self.duration_us,
            "phase1_baseline": {
                "start": self._phase1_start.isoformat() if self._phase1_start else None,
                "stop": self._phase1_stop.isoformat() if self._phase1_stop else None,
                "total_stimulations": total_baseline,
                "target_freq_hz": self.baseline_freq_hz,
                "duration_s": self.baseline_duration_s,
            },
            "phase2_tetanic": {
                "start": self._phase2_start.isoformat() if self._phase2_start else None,
                "stop": self._phase2_stop.isoformat() if self._phase2_stop else None,
                "total_stimulations": total_tetanic,
                "trains": self.tetanic_trains,
                "stims_per_train": self.tetanic_stims_per_train,
                "freq_hz": self.tetanic_freq_hz,
                "inter_train_interval_s": self.inter_train_interval_s,
            },
            "phase3_post_tetanus": {
                "start": self._phase3_start.isoformat() if self._phase3_start else None,
                "stop": self._phase3_stop.isoformat() if self._phase3_stop else None,
                "total_stimulations": total_post,
                "target_freq_hz": self.baseline_freq_hz,
                "duration_s": self.post_tetanus_duration_s,
            },
            "total_stimulations": len(self._stimulation_log),
        }

        return summary

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
        if spike_df is None or (hasattr(spike_df, 'empty') and spike_df.empty):
            spike_df = pd.DataFrame()
        saver.save_spike_events(spike_df)

        trigger_df = self.database.get_all_triggers(
            recording_start, recording_stop
        )
        if trigger_df is None or (hasattr(trigger_df, 'empty') and trigger_df.empty):
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
            "stim_electrode": self.stim_electrode,
            "resp_electrode": self.resp_electrode,
            "amplitude_ua": self.amplitude_ua,
            "duration_us": self.duration_us,
            "polarity": self.polarity_str,
            "baseline_stimulations": len(self._baseline_results),
            "tetanic_stimulations": len(self._tetanic_results),
            "post_tetanus_stimulations": len(self._post_tetanus_results),
            "phase1_start": self._phase1_start.isoformat() if self._phase1_start else None,
            "phase1_stop": self._phase1_stop.isoformat() if self._phase1_stop else None,
            "phase2_start": self._phase2_start.isoformat() if self._phase2_start else None,
            "phase2_stop": self._phase2_stop.isoformat() if self._phase2_stop else None,
            "phase3_start": self._phase3_start.isoformat() if self._phase3_start else None,
            "phase3_stop": self._phase3_stop.isoformat() if self._phase3_stop else None,
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
        if spike_df is None or spike_df.empty:
            return waveform_records

        electrode_col = None
        for candidate in ["channel", "index", "electrode", "electrode_idx"]:
            if candidate in spike_df.columns:
                electrode_col = candidate
                break

        if electrode_col is None:
            for col in spike_df.columns:
                if "electrode" in col.lower() or "channel" in col.lower() or "idx" in col.lower():
                    electrode_col = col
                    break

        if electrode_col is None:
            logger.warning("Cannot determine electrode column for waveform fetch")
            electrodes_to_fetch = [self.stim_electrode, self.resp_electrode]
        else:
            electrodes_to_fetch = list(spike_df[electrode_col].unique())

        for electrode_idx in electrodes_to_fetch:
            try:
                raw_df = self.database.get_raw_spike(
                    recording_start, recording_stop, int(electrode_idx)
                )
                if raw_df is not None and not raw_df.empty:
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
