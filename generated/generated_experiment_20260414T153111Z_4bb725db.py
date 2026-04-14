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
    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        amplitudes_ua: List[float] = None,
        stim_duration_us: float = 300.0,
        num_trials: int = 10,
        inter_stim_interval_s: float = 1.5,
        response_window_ms: float = 50.0,
        electrodes: List[int] = None,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.amplitudes_ua = amplitudes_ua if amplitudes_ua is not None else [1.0, 2.0, 3.0]
        self.stim_duration_us = stim_duration_us
        self.num_trials = num_trials
        self.inter_stim_interval_s = inter_stim_interval_s
        self.response_window_ms = response_window_ms

        # 8 electrodes selected from scan results based on highest response rates
        # From deep_scan_pair_summaries: electrodes 14, 9, 22, 5, 18, 30, 0, 6
        self.scan_electrodes = electrodes if electrodes is not None else [14, 9, 22, 5, 18, 30, 0, 6]

        self.np_experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []

        # Results storage: {electrode: {amplitude: [bool, ...]}}
        self._response_results: Dict[int, Dict[float, List[bool]]] = defaultdict(lambda: defaultdict(list))
        self._response_rates: Dict[int, Dict[float, float]] = {}
        self._best_electrode: Optional[int] = None
        self._best_response_rate: float = 0.0

    def _wait(self, seconds: float) -> None:
        wait(seconds)

    def run(self) -> Dict[str, Any]:
        try:
            logger.info("Initialising hardware connections")

            from neuroplatform import Experiment as NeuroPlatformExperiment

            self.np_experiment = NeuroPlatformExperiment(self.token)
            self.trigger_controller = TriggerController(email=self.booking_email)
            self.intan = IntanSofware()
            self.database = Database()

            logger.info("Experiment: %s", self.np_experiment.exp_name)
            logger.info("Electrodes available: %s", self.np_experiment.electrodes)

            if not self.np_experiment.start():
                logger.error("Failed to start experiment")
                return {"status": "failed", "error": "Could not start experiment"}

            recording_start = datetime_now()

            self._phase_excitability_scan()

            recording_stop = datetime_now()

            self._compute_response_rates()
            self._find_best_electrode()

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _phase_excitability_scan(self) -> None:
        logger.info("Phase: multi-electrode excitability scan")
        logger.info("Electrodes: %s", self.scan_electrodes)
        logger.info("Amplitudes: %s uA", self.amplitudes_ua)
        logger.info("Duration: %s us, Trials per condition: %d", self.stim_duration_us, self.num_trials)

        polarity = StimPolarity.PositiveFirst

        # Charge balance: A1*D1 = A2*D2
        # With equal amplitudes and equal durations, balance is guaranteed
        d1 = self.stim_duration_us
        d2 = self.stim_duration_us

        for electrode_idx in self.scan_electrodes:
            logger.info("Scanning electrode %d", electrode_idx)
            for amplitude in self.amplitudes_ua:
                a1 = amplitude
                a2 = amplitude
                # Verify charge balance
                assert abs(a1 * d1 - a2 * d2) < 1e-9, f"Charge balance violated: {a1}*{d1} != {a2}*{d2}"

                logger.info("  Amplitude: %.1f uA, Duration: %.1f us", amplitude, d1)

                stim = StimParam()
                stim.index = electrode_idx
                stim.enable = True
                stim.trigger_key = 0
                stim.trigger_delay = 0
                stim.nb_pulse = 0
                stim.pulse_train_period = 10000
                stim.post_stim_ref_period = 1000.0
                stim.stim_shape = StimShape.Biphasic
                stim.polarity = polarity
                stim.phase_amplitude1 = a1
                stim.phase_duration1 = d1
                stim.phase_amplitude2 = a2
                stim.phase_duration2 = d2
                stim.enable_amp_settle = True
                stim.pre_stim_amp_settle = 0.0
                stim.post_stim_amp_settle = 1000.0
                stim.enable_charge_recovery = True
                stim.post_charge_recovery_on = 0.0
                stim.post_charge_recovery_off = 100.0
                stim.interphase_delay = 0.0

                self.intan.send_stimparam([stim])

                for trial in range(self.num_trials):
                    stim_time = datetime_now()

                    pattern = np.zeros(16, dtype=np.uint8)
                    pattern[0] = 1
                    self.trigger_controller.send(pattern)
                    self._wait(0.05)
                    pattern[0] = 0
                    self.trigger_controller.send(pattern)

                    self._stimulation_log.append(StimulationRecord(
                        electrode_idx=electrode_idx,
                        amplitude_ua=amplitude,
                        duration_us=d1,
                        polarity=polarity.name,
                        timestamp_utc=stim_time.isoformat(),
                        trigger_key=0,
                        trial_index=trial,
                    ))

                    # Wait for response window
                    response_window_s = self.response_window_ms / 1000.0
                    self._wait(response_window_s)

                    # Query spikes in the response window
                    query_stop = datetime_now()
                    query_start = stim_time

                    try:
                        spike_df = self.database.get_spike_event(
                            query_start, query_stop, self.np_experiment.exp_name
                        )
                        has_response = not spike_df.empty and len(spike_df) > 0
                    except Exception as exc:
                        logger.warning("Spike query failed for electrode %d trial %d: %s", electrode_idx, trial, exc)
                        has_response = False

                    self._response_results[electrode_idx][amplitude].append(has_response)

                    logger.debug(
                        "Electrode %d, amp %.1f uA, trial %d: response=%s",
                        electrode_idx, amplitude, trial, has_response
                    )

                    # Inter-stimulation interval (minus the response window already waited)
                    remaining_wait = self.inter_stim_interval_s - response_window_s - 0.05
                    if remaining_wait > 0:
                        self._wait(remaining_wait)

    def _compute_response_rates(self) -> None:
        logger.info("Computing per-electrode response rates")
        for electrode_idx in self.scan_electrodes:
            self._response_rates[electrode_idx] = {}
            for amplitude in self.amplitudes_ua:
                trials = self._response_results[electrode_idx][amplitude]
                if len(trials) > 0:
                    rate = sum(trials) / len(trials)
                else:
                    rate = 0.0
                self._response_rates[electrode_idx][amplitude] = rate
                logger.info(
                    "Electrode %d, amplitude %.1f uA: response rate = %.2f (%d/%d)",
                    electrode_idx, amplitude, rate, sum(trials), len(trials)
                )

    def _find_best_electrode(self) -> None:
        logger.info("Finding electrode with highest overall response rate")
        best_electrode = None
        best_rate = -1.0

        for electrode_idx in self.scan_electrodes:
            amp_rates = self._response_rates.get(electrode_idx, {})
            if amp_rates:
                avg_rate = sum(amp_rates.values()) / len(amp_rates)
            else:
                avg_rate = 0.0

            logger.info("Electrode %d average response rate: %.3f", electrode_idx, avg_rate)

            if avg_rate > best_rate:
                best_rate = avg_rate
                best_electrode = electrode_idx

        self._best_electrode = best_electrode
        self._best_response_rate = best_rate
        logger.info("Best electrode: %d with average response rate %.3f", best_electrode, best_rate)

    def _save_all(self, recording_start: datetime, recording_stop: datetime) -> None:
        fs_name = getattr(self.np_experiment, "exp_name", "unknown")
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

        response_rates_serializable = {
            str(elec): {str(amp): rate for amp, rate in amp_dict.items()}
            for elec, amp_dict in self._response_rates.items()
        }

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "electrodes_scanned": self.scan_electrodes,
            "amplitudes_ua": self.amplitudes_ua,
            "stim_duration_us": self.stim_duration_us,
            "num_trials_per_condition": self.num_trials,
            "response_window_ms": self.response_window_ms,
            "response_rates": response_rates_serializable,
            "best_electrode": self._best_electrode,
            "best_electrode_avg_response_rate": self._best_response_rate,
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

        response_rates_serializable = {
            str(elec): {str(amp): rate for amp, rate in amp_dict.items()}
            for elec, amp_dict in self._response_rates.items()
        }

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.np_experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "electrodes_scanned": self.scan_electrodes,
            "amplitudes_ua": self.amplitudes_ua,
            "stim_duration_us": self.stim_duration_us,
            "num_trials_per_condition": self.num_trials,
            "response_window_ms": self.response_window_ms,
            "response_rates": response_rates_serializable,
            "best_electrode": self._best_electrode,
            "best_electrode_avg_response_rate": self._best_response_rate,
            "total_stimulations": len(self._stimulation_log),
        }

        return summary

    def _cleanup(self) -> None:
        logger.info("Cleaning up resources")

        if self.np_experiment is not None:
            try:
                self.np_experiment.stop()
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
