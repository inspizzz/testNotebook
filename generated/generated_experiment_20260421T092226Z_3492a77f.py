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
    trigger_key: int = 0
    trial_index: int = 0
    amplitude_level: int = 0
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
        polarity: str = "PositiveFirst",
        num_trials_per_condition: int = 10,
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
        self.polarity_str = polarity
        self.polarity = StimPolarity.PositiveFirst if polarity == "PositiveFirst" else StimPolarity.NegativeFirst
        self.num_trials = num_trials_per_condition
        self.inter_stim_interval_s = inter_stim_interval_s
        self.response_window_ms = response_window_ms

        # 8 electrodes selected from scan results based on highest response rates
        # From deep_scan_pair_summaries: electrodes 14, 9, 22, 5, 18, 30, 0, 6 are top stimulators
        self.selected_electrodes = electrodes if electrodes is not None else [14, 9, 22, 5, 18, 30, 0, 6]

        self.np_experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []

        # Results storage: {electrode: {amplitude: [bool, ...]}}
        self._response_results: Dict[int, Dict[float, List[bool]]] = defaultdict(lambda: defaultdict(list))

        # Per-electrode overall response rate (across all amplitudes)
        self._electrode_response_rates: Dict[int, float] = {}

        # Best electrode
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
            logger.info("Recording started at %s", recording_start.isoformat())

            self._phase_excitability_scan()

            self._compute_response_rates()

            recording_stop = datetime_now()
            logger.info("Recording stopped at %s", recording_stop.isoformat())

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
        logger.info("Electrodes: %s", self.selected_electrodes)
        logger.info("Amplitudes: %s uA", self.amplitudes_ua)
        logger.info("Duration: %s us, Polarity: %s", self.stim_duration_us, self.polarity_str)
        logger.info("Trials per condition: %d, ISI: %.1f s", self.num_trials, self.inter_stim_interval_s)

        # Validate charge balance: A1*D1 == A2*D2
        # We use equal amplitude and duration on both phases
        # So A1*D1 = A2*D2 is guaranteed when A1==A2 and D1==D2

        for electrode_idx in self.selected_electrodes:
            logger.info("--- Electrode %d ---", electrode_idx)
            for amplitude in self.amplitudes_ua:
                logger.info("  Amplitude: %.1f uA", amplitude)
                for trial in range(self.num_trials):
                    stim_time = datetime_now()
                    self._send_stimulation(electrode_idx, amplitude, self.stim_duration_us, self.polarity, trigger_key=0)

                    # Wait for response window
                    self._wait(self.response_window_ms / 1000.0)

                    # Query spikes in response window
                    query_start = stim_time
                    query_stop = datetime_now()
                    spike_df = self._query_spikes(query_start, query_stop, electrode_idx)

                    has_response = not spike_df.empty and len(spike_df) > 0
                    self._response_results[electrode_idx][amplitude].append(has_response)

                    logger.info(
                        "    Trial %d/%d: electrode=%d amp=%.1f -> response=%s",
                        trial + 1, self.num_trials, electrode_idx, amplitude, has_response
                    )

                    # Wait for remainder of ISI
                    remaining_wait = self.inter_stim_interval_s - (self.response_window_ms / 1000.0)
                    if remaining_wait > 0:
                        self._wait(remaining_wait)

    def _send_stimulation(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity,
        trigger_key: int = 0,
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

        # Charge balance: A1*D1 == A2*D2 (equal amplitudes and durations)
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
            timestamp_utc=datetime_now().isoformat(),
            trigger_key=trigger_key,
            trial_index=len(self._stimulation_log),
            amplitude_level=self.amplitudes_ua.index(amplitude_ua) if amplitude_ua in self.amplitudes_ua else -1,
        ))

    def _query_spikes(
        self,
        start: datetime,
        stop: datetime,
        electrode_idx: int,
    ) -> pd.DataFrame:
        try:
            spike_df = self.database.get_spike_event_electrode(start, stop, electrode_idx)
            return spike_df
        except Exception as exc:
            logger.warning("Failed to query spikes for electrode %d: %s", electrode_idx, exc)
            return pd.DataFrame()

    def _compute_response_rates(self) -> None:
        logger.info("Computing per-electrode response rates")

        for electrode_idx in self.selected_electrodes:
            all_responses = []
            amp_rates = {}
            for amplitude in self.amplitudes_ua:
                trials = self._response_results[electrode_idx][amplitude]
                if trials:
                    rate = sum(trials) / len(trials)
                    amp_rates[amplitude] = rate
                    all_responses.extend(trials)
                else:
                    amp_rates[amplitude] = 0.0

            overall_rate = sum(all_responses) / len(all_responses) if all_responses else 0.0
            self._electrode_response_rates[electrode_idx] = overall_rate

            logger.info(
                "Electrode %d: overall_rate=%.3f, per_amplitude=%s",
                electrode_idx, overall_rate,
                {a: f"{r:.3f}" for a, r in amp_rates.items()}
            )

        if self._electrode_response_rates:
            best_elec = max(self._electrode_response_rates, key=self._electrode_response_rates.get)
            self._best_electrode = best_elec
            self._best_response_rate = self._electrode_response_rates[best_elec]
            logger.info(
                "Best electrode: %d with overall response rate %.3f",
                self._best_electrode, self._best_response_rate
            )

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        logger.info("Compiling results")

        per_electrode_detail = {}
        for electrode_idx in self.selected_electrodes:
            amp_detail = {}
            for amplitude in self.amplitudes_ua:
                trials = self._response_results[electrode_idx][amplitude]
                rate = sum(trials) / len(trials) if trials else 0.0
                amp_detail[str(amplitude)] = {
                    "response_rate": rate,
                    "num_trials": len(trials),
                    "num_responses": sum(trials),
                }
            per_electrode_detail[str(electrode_idx)] = {
                "overall_response_rate": self._electrode_response_rates.get(electrode_idx, 0.0),
                "per_amplitude": amp_detail,
            }

        summary = {
            "status": "completed",
            "experiment_name": self.np_experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "electrodes_scanned": self.selected_electrodes,
            "amplitudes_ua": self.amplitudes_ua,
            "stim_duration_us": self.stim_duration_us,
            "polarity": self.polarity_str,
            "num_trials_per_condition": self.num_trials,
            "inter_stim_interval_s": self.inter_stim_interval_s,
            "response_window_ms": self.response_window_ms,
            "total_stimulations": len(self._stimulation_log),
            "best_electrode": self._best_electrode,
            "best_electrode_response_rate": self._best_response_rate,
            "electrode_response_rates": {str(k): v for k, v in self._electrode_response_rates.items()},
            "per_electrode_detail": per_electrode_detail,
        }

        return summary

    def _save_all(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> None:
        fs_name = getattr(self.np_experiment, "exp_name", "unknown")
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

        per_electrode_detail = {}
        for electrode_idx in self.selected_electrodes:
            amp_detail = {}
            for amplitude in self.amplitudes_ua:
                trials = self._response_results[electrode_idx][amplitude]
                rate = sum(trials) / len(trials) if trials else 0.0
                amp_detail[str(amplitude)] = {
                    "response_rate": rate,
                    "num_trials": len(trials),
                    "num_responses": sum(trials),
                }
            per_electrode_detail[str(electrode_idx)] = {
                "overall_response_rate": self._electrode_response_rates.get(electrode_idx, 0.0),
                "per_amplitude": amp_detail,
            }

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "electrodes_scanned": self.selected_electrodes,
            "amplitudes_ua": self.amplitudes_ua,
            "stim_duration_us": self.stim_duration_us,
            "polarity": self.polarity_str,
            "num_trials_per_condition": self.num_trials,
            "inter_stim_interval_s": self.inter_stim_interval_s,
            "response_window_ms": self.response_window_ms,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "best_electrode": self._best_electrode,
            "best_electrode_response_rate": self._best_response_rate,
            "electrode_response_rates": {str(k): v for k, v in self._electrode_response_rates.items()},
            "per_electrode_detail": per_electrode_detail,
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

        electrodes_to_fetch = list(self.selected_electrodes)

        if not spike_df.empty:
            electrode_col = None
            for col in spike_df.columns:
                if col.lower() in ("channel", "index", "electrode", "electrode_idx"):
                    electrode_col = col
                    break
            if electrode_col is not None:
                extra_electrodes = [int(e) for e in spike_df[electrode_col].unique()]
                for e in extra_electrodes:
                    if e not in electrodes_to_fetch:
                        electrodes_to_fetch.append(e)

        for electrode_idx in electrodes_to_fetch:
            try:
                raw_df = self.database.get_raw_spike(recording_start, recording_stop, int(electrode_idx))
                if not raw_df.empty:
                    waveform_records.append({
                        "electrode_idx": int(electrode_idx),
                        "num_waveforms": len(raw_df),
                        "waveform_samples": raw_df.values.tolist(),
                    })
            except Exception as exc:
                logger.warning("Failed to fetch waveforms for electrode %d: %s", electrode_idx, exc)

        return waveform_records

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
