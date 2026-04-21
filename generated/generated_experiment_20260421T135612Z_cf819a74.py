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
    timestamp_utc: str
    trigger_key: int = 0
    amplitude_level_index: int = 0
    trial_index: int = 0
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
    Input-output curve experiment using escalating amplitude stimulation.

    The most responsive electrode from the parameter scan is electrode 17
    (stimulating to electrode 18), which showed the highest response rate
    (0.92) and perfect temporal stability (1.0) at 3 uA / 400 us /
    PositiveFirst. We use electrode 17 as the stimulating electrode and
    deliver 20 stimulations at 1 Hz for each amplitude level from 1.0 to
    4.0 uA in 0.5 uA steps (amplitude is capped at 4.0 uA per hardware
    limits, so the range 1.0 to 4.0 in 0.5 uA steps gives 7 levels).

    Charge balance: A1 * D1 == A2 * D2. Both phases use the same amplitude
    and duration (symmetric biphasic), so balance is guaranteed.
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        stim_electrode: int = 17,
        resp_electrode: int = 18,
        stim_duration_us: float = 400.0,
        trials_per_level: int = 20,
        isi_s: float = 1.0,
        amplitude_min_ua: float = 1.0,
        amplitude_max_ua: float = 4.0,
        amplitude_step_ua: float = 0.5,
        inter_level_wait_s: float = 5.0,
        trigger_key: int = 0,
        post_stim_wait_s: float = 0.5,
        recording_window_s: float = 0.1,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.stim_electrode = stim_electrode
        self.resp_electrode = resp_electrode
        self.stim_duration_us = min(abs(stim_duration_us), 400.0)
        self.trials_per_level = trials_per_level
        self.isi_s = isi_s
        self.inter_level_wait_s = inter_level_wait_s
        self.trigger_key = trigger_key
        self.post_stim_wait_s = post_stim_wait_s
        self.recording_window_s = recording_window_s

        # Build amplitude levels list
        n_steps = round((amplitude_max_ua - amplitude_min_ua) / amplitude_step_ua)
        self.amplitude_levels: List[float] = [
            round(amplitude_min_ua + i * amplitude_step_ua, 6)
            for i in range(n_steps + 1)
            if round(amplitude_min_ua + i * amplitude_step_ua, 6) <= 4.0
        ]

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._io_curve: List[Dict[str, Any]] = []

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
            logger.info("Amplitude levels (uA): %s", self.amplitude_levels)
            logger.info("Trials per level: %d", self.trials_per_level)
            logger.info("Stimulation electrode: %d", self.stim_electrode)
            logger.info("Response electrode: %d", self.resp_electrode)

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
        logger.info("Phase: input-output curve sweep")

        for level_idx, amplitude_ua in enumerate(self.amplitude_levels):
            logger.info(
                "Amplitude level %d/%d: %.2f uA",
                level_idx + 1,
                len(self.amplitude_levels),
                amplitude_ua,
            )

            level_spike_counts = []

            for trial_idx in range(self.trials_per_level):
                spike_count = self._stimulate_and_record(
                    electrode_idx=self.stim_electrode,
                    amplitude_ua=amplitude_ua,
                    duration_us=self.stim_duration_us,
                    polarity=StimPolarity.PositiveFirst,
                    trigger_key=self.trigger_key,
                    post_stim_wait_s=self.post_stim_wait_s,
                    recording_window_s=self.recording_window_s,
                    level_idx=level_idx,
                    trial_idx=trial_idx,
                )
                level_spike_counts.append(spike_count)
                logger.info(
                    "  Trial %d/%d: %d spikes detected",
                    trial_idx + 1,
                    self.trials_per_level,
                    spike_count,
                )

                # 1 Hz inter-stimulus interval: subtract time already spent
                remaining = self.isi_s - self.post_stim_wait_s
                if remaining > 0:
                    self._wait(remaining)

            level_summary = {
                "amplitude_ua": amplitude_ua,
                "duration_us": self.stim_duration_us,
                "polarity": "PositiveFirst",
                "stim_electrode": self.stim_electrode,
                "resp_electrode": self.resp_electrode,
                "trials_n": self.trials_per_level,
                "total_spikes": sum(level_spike_counts),
                "mean_spikes_per_trial": (
                    sum(level_spike_counts) / len(level_spike_counts)
                    if level_spike_counts else 0.0
                ),
                "responding_trials": sum(1 for c in level_spike_counts if c > 0),
                "response_rate": (
                    sum(1 for c in level_spike_counts if c > 0) / len(level_spike_counts)
                    if level_spike_counts else 0.0
                ),
                "spike_counts_per_trial": level_spike_counts,
            }
            self._io_curve.append(level_summary)
            logger.info(
                "  Level summary: response_rate=%.2f, mean_spikes=%.2f",
                level_summary["response_rate"],
                level_summary["mean_spikes_per_trial"],
            )

            if level_idx < len(self.amplitude_levels) - 1:
                logger.info("Inter-level pause: %.1f s", self.inter_level_wait_s)
                self._wait(self.inter_level_wait_s)

    def _stimulate_and_record(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity = StimPolarity.PositiveFirst,
        trigger_key: int = 0,
        post_stim_wait_s: float = 0.5,
        recording_window_s: float = 0.1,
        level_idx: int = 0,
        trial_idx: int = 0,
    ) -> int:
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

        # Charge-balanced: A1*D1 == A2*D2 (symmetric biphasic)
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

        stim_time = datetime_now()

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
            timestamp_utc=stim_time.isoformat(),
            trigger_key=trigger_key,
            amplitude_level_index=level_idx,
            trial_index=trial_idx,
        ))

        self._wait(post_stim_wait_s)

        query_stop = datetime_now()
        query_start = query_stop - timedelta(seconds=post_stim_wait_s + recording_window_s)

        try:
            spike_df = self.database.get_spike_event(
                query_start, query_stop, self.experiment.exp_name
            )
            if spike_df.empty:
                return 0
            # Count spikes on the responding electrode within the window
            if "channel" in spike_df.columns:
                resp_spikes = spike_df[spike_df["channel"] == self.resp_electrode]
            else:
                resp_spikes = spike_df
            return len(resp_spikes)
        except Exception as exc:
            logger.warning("Spike query failed: %s", exc)
            return 0

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
            "stim_electrode": self.stim_electrode,
            "resp_electrode": self.resp_electrode,
            "amplitude_levels_ua": self.amplitude_levels,
            "trials_per_level": self.trials_per_level,
            "stim_duration_us": self.stim_duration_us,
            "polarity": "PositiveFirst",
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
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
            if col.lower() in ("channel", "electrode", "index", "idx"):
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
                "response_rate": entry["response_rate"],
                "mean_spikes_per_trial": entry["mean_spikes_per_trial"],
                "total_spikes": entry["total_spikes"],
                "responding_trials": entry["responding_trials"],
            })

        return {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "stim_electrode": self.stim_electrode,
            "resp_electrode": self.resp_electrode,
            "amplitude_levels_ua": self.amplitude_levels,
            "trials_per_level": self.trials_per_level,
            "stim_duration_us": self.stim_duration_us,
            "polarity": "PositiveFirst",
            "total_stimulations": len(self._stimulation_log),
            "io_curve_summary": io_summary,
        }

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
