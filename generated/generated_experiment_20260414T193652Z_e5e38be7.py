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
    amplitude_level_index: int
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
    Input-output curve experiment using escalating amplitude from 1.0 to 4.0 uA
    in 0.5 uA steps, with 400 us duration and PositiveFirst polarity.
    At each amplitude level, 20 stimulations are delivered at 1 Hz on the most
    responsive electrode (electrode 14, based on parameter scan showing highest
    response rate of 0.94 to electrode 12).
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        stim_electrode: int = 14,
        resp_electrode: int = 12,
        phase_duration_us: float = 400.0,
        polarity: str = "PositiveFirst",
        amp_start_ua: float = 1.0,
        amp_stop_ua: float = 4.0,
        amp_step_ua: float = 0.5,
        trials_per_level: int = 20,
        isi_s: float = 1.0,
        inter_level_wait_s: float = 3.0,
        trigger_key: int = 0,
        post_stim_wait_s: float = 0.5,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.stim_electrode = stim_electrode
        self.resp_electrode = resp_electrode
        self.phase_duration_us = min(abs(phase_duration_us), 400.0)
        self.polarity_str = polarity
        self.polarity = StimPolarity.PositiveFirst if polarity == "PositiveFirst" else StimPolarity.NegativeFirst
        self.amp_start_ua = amp_start_ua
        self.amp_stop_ua = amp_stop_ua
        self.amp_step_ua = amp_step_ua
        self.trials_per_level = trials_per_level
        self.isi_s = isi_s
        self.inter_level_wait_s = inter_level_wait_s
        self.trigger_key = trigger_key
        self.post_stim_wait_s = post_stim_wait_s

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._io_curve: List[Dict[str, Any]] = []

        n_steps = round((amp_stop_ua - amp_start_ua) / amp_step_ua) + 1
        self._amplitude_levels = [
            round(amp_start_ua + i * amp_step_ua, 4)
            for i in range(n_steps)
            if round(amp_start_ua + i * amp_step_ua, 4) <= amp_stop_ua + 1e-9
        ]
        self._amplitude_levels = [a for a in self._amplitude_levels if a <= 4.0]

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
            logger.info("Stim electrode: %d, Resp electrode: %d", self.stim_electrode, self.resp_electrode)
            logger.info("Amplitude levels: %s", self._amplitude_levels)
            logger.info("Trials per level: %d, ISI: %.2f s", self.trials_per_level, self.isi_s)

            if not self.experiment.start():
                logger.error("Failed to start experiment")
                return {"status": "failed", "error": "Could not start experiment"}

            recording_start = datetime_now()

            self._phase_io_curve()

            recording_stop = datetime_now()

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _phase_io_curve(self) -> None:
        logger.info("Phase: input-output curve, %d amplitude levels x %d trials",
                    len(self._amplitude_levels), self.trials_per_level)

        for level_idx, amplitude in enumerate(self._amplitude_levels):
            logger.info("Amplitude level %d/%d: %.2f uA",
                        level_idx + 1, len(self._amplitude_levels), amplitude)

            level_spike_counts = []

            self._configure_stim_param(amplitude)

            for trial_idx in range(self.trials_per_level):
                trial_start = datetime_now()

                self._fire_trigger()

                ts = datetime_now().isoformat()
                self._stimulation_log.append(StimulationRecord(
                    electrode_idx=self.stim_electrode,
                    amplitude_ua=amplitude,
                    duration_us=self.phase_duration_us,
                    polarity=self.polarity_str,
                    amplitude_level_index=level_idx,
                    trial_index=trial_idx,
                    timestamp_utc=ts,
                    trigger_key=self.trigger_key,
                ))

                self._wait(self.post_stim_wait_s)

                query_stop = datetime_now()
                query_start = query_stop - timedelta(seconds=self.post_stim_wait_s + 0.1)
                try:
                    spike_df = self.database.get_spike_event(
                        query_start, query_stop, self.experiment.exp_name
                    )
                    if not spike_df.empty and "channel" in spike_df.columns:
                        resp_spikes = spike_df[spike_df["channel"] == self.resp_electrode]
                        n_spikes = len(resp_spikes)
                    else:
                        n_spikes = 0
                except Exception as exc:
                    logger.warning("Spike query failed for trial %d: %s", trial_idx, exc)
                    n_spikes = 0

                level_spike_counts.append(n_spikes)
                logger.debug("  Trial %d/%d: %d spikes on electrode %d",
                             trial_idx + 1, self.trials_per_level, n_spikes, self.resp_electrode)

                elapsed = (datetime_now() - trial_start).total_seconds()
                remaining_wait = self.isi_s - elapsed
                if remaining_wait > 0:
                    self._wait(remaining_wait)

            mean_spikes = float(np.mean(level_spike_counts)) if level_spike_counts else 0.0
            response_rate = float(np.mean([1 if s > 0 else 0 for s in level_spike_counts]))

            self._io_curve.append({
                "amplitude_level_index": level_idx,
                "amplitude_ua": amplitude,
                "duration_us": self.phase_duration_us,
                "polarity": self.polarity_str,
                "stim_electrode": self.stim_electrode,
                "resp_electrode": self.resp_electrode,
                "trials_n": self.trials_per_level,
                "spike_counts": level_spike_counts,
                "mean_spikes_per_trial": mean_spikes,
                "response_rate": response_rate,
            })

            logger.info("  Level %.2f uA: mean_spikes=%.2f, response_rate=%.2f",
                        amplitude, mean_spikes, response_rate)

            if level_idx < len(self._amplitude_levels) - 1:
                self._wait(self.inter_level_wait_s)

        logger.info("IO curve phase complete. %d levels recorded.", len(self._io_curve))

    def _configure_stim_param(self, amplitude_ua: float) -> None:
        amplitude_ua = min(abs(amplitude_ua), 4.0)
        duration_us = self.phase_duration_us

        stim = StimParam()
        stim.index = self.stim_electrode
        stim.enable = True
        stim.trigger_key = self.trigger_key
        stim.trigger_delay = 0
        stim.nb_pulse = 0
        stim.pulse_train_period = 10000
        stim.post_stim_ref_period = 1000.0
        stim.stim_shape = StimShape.Biphasic
        stim.polarity = self.polarity

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

    def _fire_trigger(self) -> None:
        pattern = np.zeros(16, dtype=np.uint8)
        pattern[self.trigger_key] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.02)
        pattern[self.trigger_key] = 0
        self.trigger_controller.send(pattern)

    def _save_all(self, recording_start: datetime, recording_stop: datetime) -> None:
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
            "stim_electrode": self.stim_electrode,
            "resp_electrode": self.resp_electrode,
            "polarity": self.polarity_str,
            "phase_duration_us": self.phase_duration_us,
            "amplitude_levels": self._amplitude_levels,
            "trials_per_level": self.trials_per_level,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df) if not spike_df.empty else 0,
            "total_triggers": len(trigger_df) if not trigger_df.empty else 0,
            "io_curve": self._io_curve,
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
            if col.lower() in ("channel", "index", "electrode", "electrode_idx"):
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

        io_summary = []
        for entry in self._io_curve:
            io_summary.append({
                "amplitude_ua": entry["amplitude_ua"],
                "mean_spikes_per_trial": entry["mean_spikes_per_trial"],
                "response_rate": entry["response_rate"],
                "trials_n": entry["trials_n"],
            })

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "stim_electrode": self.stim_electrode,
            "resp_electrode": self.resp_electrode,
            "polarity": self.polarity_str,
            "phase_duration_us": self.phase_duration_us,
            "amplitude_levels_tested": self._amplitude_levels,
            "trials_per_level": self.trials_per_level,
            "total_stimulations": len(self._stimulation_log),
            "io_curve_summary": io_summary,
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
