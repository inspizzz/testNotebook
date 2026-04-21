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


@dataclass
class AmplitudeLevelResult:
    amplitude_ua: float
    duration_us: float
    n_stimulations: int
    n_responding_trials: int
    response_probability: float
    mean_spike_count: float
    total_spikes: int
    spike_counts_per_trial: List[int] = field(default_factory=list)


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
    Input-output curve experiment on the FinalSpark NeuroPlatform.

    Uses electrode pair (stim=17, resp=18) identified as most responsive
    from the parameter scan (response_rate=0.92, highest hits_k=5).
    Sweeps amplitude from 0.5 to 4.0 uA in 0.5 uA steps (8 levels),
    300 us duration, PositiveFirst polarity, 30 stimulations at 1 Hz per level,
    10 s inter-level wait.
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        stim_electrode: int = 17,
        resp_electrode: int = 18,
        amplitudes_ua: Tuple = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0),
        stim_duration_us: float = 300.0,
        n_stimulations_per_level: int = 30,
        isi_s: float = 1.0,
        inter_level_wait_s: float = 10.0,
        response_window_ms: float = 100.0,
        trigger_key: int = 0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.stim_electrode = stim_electrode
        self.resp_electrode = resp_electrode
        self.amplitudes_ua = list(amplitudes_ua)
        self.stim_duration_us = stim_duration_us
        self.n_stimulations_per_level = n_stimulations_per_level
        self.isi_s = isi_s
        self.inter_level_wait_s = inter_level_wait_s
        self.response_window_ms = response_window_ms
        self.trigger_key = trigger_key

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._amplitude_results: List[AmplitudeLevelResult] = []

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

            self._configure_stimulation()
            self._run_io_curve()

            recording_stop = datetime_now()

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _configure_stimulation(self) -> None:
        logger.info(
            "Configuring stimulation: electrode=%d, duration=%.1f us, polarity=PositiveFirst",
            self.stim_electrode,
            self.stim_duration_us,
        )
        stim = self._build_stim_param(
            electrode_idx=self.stim_electrode,
            amplitude_ua=self.amplitudes_ua[0],
            duration_us=self.stim_duration_us,
            polarity=StimPolarity.PositiveFirst,
            trigger_key=self.trigger_key,
        )
        self.intan.send_stimparam([stim])
        logger.info("Initial stim param sent")

    def _build_stim_param(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity,
        trigger_key: int,
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

        return stim

    def _fire_trigger(self) -> None:
        pattern = np.zeros(16, dtype=np.uint8)
        pattern[self.trigger_key] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.02)
        pattern[self.trigger_key] = 0
        self.trigger_controller.send(pattern)

    def _run_io_curve(self) -> None:
        logger.info(
            "Starting I/O curve: %d amplitude levels, %d stimulations each at %.1f Hz",
            len(self.amplitudes_ua),
            self.n_stimulations_per_level,
            1.0 / self.isi_s,
        )

        for level_idx, amplitude in enumerate(self.amplitudes_ua):
            logger.info(
                "Amplitude level %d/%d: %.2f uA",
                level_idx + 1,
                len(self.amplitudes_ua),
                amplitude,
            )

            stim = self._build_stim_param(
                electrode_idx=self.stim_electrode,
                amplitude_ua=amplitude,
                duration_us=self.stim_duration_us,
                polarity=StimPolarity.PositiveFirst,
                trigger_key=self.trigger_key,
            )
            self.intan.send_stimparam([stim])

            spike_counts_per_trial: List[int] = []

            for trial_idx in range(self.n_stimulations_per_level):
                stim_time = datetime_now()

                self._fire_trigger()

                self._stimulation_log.append(StimulationRecord(
                    electrode_idx=self.stim_electrode,
                    amplitude_ua=amplitude,
                    duration_us=self.stim_duration_us,
                    polarity="PositiveFirst",
                    timestamp_utc=stim_time.isoformat(),
                    trigger_key=self.trigger_key,
                    amplitude_level_index=level_idx,
                    trial_index=trial_idx,
                ))

                response_wait_s = self.response_window_ms / 1000.0
                self._wait(response_wait_s)

                query_start = stim_time
                query_stop = datetime_now()

                try:
                    spike_df = self.database.get_spike_event(
                        query_start,
                        query_stop,
                        self.experiment.exp_name,
                    )
                    if spike_df.empty:
                        n_spikes = 0
                    else:
                        if "channel" in spike_df.columns:
                            resp_spikes = spike_df[spike_df["channel"] == self.resp_electrode]
                        else:
                            resp_spikes = spike_df
                        n_spikes = len(resp_spikes)
                except Exception as exc:
                    logger.warning("Spike query failed for trial %d: %s", trial_idx, exc)
                    n_spikes = 0

                spike_counts_per_trial.append(n_spikes)

                remaining_wait = self.isi_s - response_wait_s - 0.02
                if remaining_wait > 0:
                    self._wait(remaining_wait)

                logger.debug(
                    "Level %d, trial %d/%d: %d spikes",
                    level_idx + 1,
                    trial_idx + 1,
                    self.n_stimulations_per_level,
                    n_spikes,
                )

            n_responding = sum(1 for c in spike_counts_per_trial if c > 0)
            total_spikes = sum(spike_counts_per_trial)
            response_prob = n_responding / self.n_stimulations_per_level if self.n_stimulations_per_level > 0 else 0.0
            mean_spike_count = total_spikes / self.n_stimulations_per_level if self.n_stimulations_per_level > 0 else 0.0

            level_result = AmplitudeLevelResult(
                amplitude_ua=amplitude,
                duration_us=self.stim_duration_us,
                n_stimulations=self.n_stimulations_per_level,
                n_responding_trials=n_responding,
                response_probability=response_prob,
                mean_spike_count=mean_spike_count,
                total_spikes=total_spikes,
                spike_counts_per_trial=spike_counts_per_trial,
            )
            self._amplitude_results.append(level_result)

            logger.info(
                "Level %.2f uA complete: response_prob=%.3f, mean_spikes=%.3f",
                amplitude,
                response_prob,
                mean_spike_count,
            )

            if level_idx < len(self.amplitudes_ua) - 1:
                logger.info("Waiting %.1f s before next amplitude level", self.inter_level_wait_s)
                self._wait(self.inter_level_wait_s)

        logger.info("I/O curve complete")

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        logger.info("Compiling results")

        io_curve = []
        for r in self._amplitude_results:
            io_curve.append({
                "amplitude_ua": r.amplitude_ua,
                "duration_us": r.duration_us,
                "n_stimulations": r.n_stimulations,
                "n_responding_trials": r.n_responding_trials,
                "response_probability": r.response_probability,
                "mean_spike_count": r.mean_spike_count,
                "total_spikes": r.total_spikes,
            })

        summary = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "stim_electrode": self.stim_electrode,
            "resp_electrode": self.resp_electrode,
            "stim_duration_us": self.stim_duration_us,
            "polarity": "PositiveFirst",
            "n_amplitude_levels": len(self.amplitudes_ua),
            "n_stimulations_per_level": self.n_stimulations_per_level,
            "total_stimulations": len(self._stimulation_log),
            "io_curve": io_curve,
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

        io_curve = []
        for r in self._amplitude_results:
            io_curve.append({
                "amplitude_ua": r.amplitude_ua,
                "duration_us": r.duration_us,
                "n_stimulations": r.n_stimulations,
                "n_responding_trials": r.n_responding_trials,
                "response_probability": r.response_probability,
                "mean_spike_count": r.mean_spike_count,
                "total_spikes": r.total_spikes,
                "spike_counts_per_trial": r.spike_counts_per_trial,
            })

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "stim_electrode": self.stim_electrode,
            "resp_electrode": self.resp_electrode,
            "stim_duration_us": self.stim_duration_us,
            "polarity": "PositiveFirst",
            "n_amplitude_levels": len(self.amplitudes_ua),
            "n_stimulations_per_level": self.n_stimulations_per_level,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "io_curve": io_curve,
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
            electrodes_to_fetch = [self.stim_electrode, self.resp_electrode]
        else:
            if "channel" in spike_df.columns:
                electrode_col = "channel"
            else:
                electrode_col = None
                for col in spike_df.columns:
                    if "electrode" in col.lower() or "idx" in col.lower() or "index" in col.lower():
                        electrode_col = col
                        break

            if electrode_col is not None:
                electrodes_to_fetch = list(spike_df[electrode_col].unique())
            else:
                electrodes_to_fetch = [self.stim_electrode, self.resp_electrode]

        for electrode_idx in electrodes_to_fetch:
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
