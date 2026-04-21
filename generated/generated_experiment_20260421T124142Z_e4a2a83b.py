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
        electrodes: tuple = (5, 6, 7, 13, 17, 21, 22, 19),
        amplitudes_ua: tuple = (1.0, 2.0, 3.0),
        stim_duration_us: float = 300.0,
        trials_per_condition: int = 10,
        inter_stim_interval_s: float = 1.5,
        response_window_ms: float = 50.0,
        trigger_key: int = 0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.electrodes = list(electrodes)
        self.amplitudes_ua = list(amplitudes_ua)
        self.stim_duration_us = stim_duration_us
        self.trials_per_condition = trials_per_condition
        self.inter_stim_interval_s = inter_stim_interval_s
        self.response_window_ms = response_window_ms
        self.trigger_key = trigger_key

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []

        # Results storage: keyed by (electrode_idx, amplitude_ua)
        # Each value is a list of booleans (True = at least one spike in window)
        self._trial_responses: Dict[Tuple[int, float], List[bool]] = defaultdict(list)

        # Timestamps of each stimulation for spike window queries
        self._stim_timestamps: List[Tuple[int, float, datetime]] = []

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

            if not self.experiment.start():
                logger.error("Failed to start experiment")
                return {"status": "failed", "error": "Could not start experiment"}

            recording_start = datetime_now()

            self._phase_excitability_scan()

            recording_stop = datetime_now()

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
        logger.info(
            "Electrodes: %s | Amplitudes: %s uA | Duration: %s us | Trials: %d | ISI: %.1f s",
            self.electrodes,
            self.amplitudes_ua,
            self.stim_duration_us,
            self.trials_per_condition,
            self.inter_stim_interval_s,
        )

        # Charge balance: A1*D1 = A2*D2
        # With equal amplitudes and equal durations this is trivially satisfied.
        # phase_amplitude2 = amplitude_ua * stim_duration_us / stim_duration_us = amplitude_ua
        # We keep both phases identical.

        for electrode_idx in self.electrodes:
            for amplitude_ua in self.amplitudes_ua:
                logger.info(
                    "Scanning electrode %d at %.1f uA", electrode_idx, amplitude_ua
                )
                for trial in range(self.trials_per_condition):
                    stim_time = self._deliver_stimulation(
                        electrode_idx=electrode_idx,
                        amplitude_ua=amplitude_ua,
                        duration_us=self.stim_duration_us,
                        polarity=StimPolarity.PositiveFirst,
                        trigger_key=self.trigger_key,
                        trial_index=trial,
                    )
                    self._stim_timestamps.append((electrode_idx, amplitude_ua, stim_time))

                    # Wait for response window to elapse before querying
                    response_window_s = self.response_window_ms / 1000.0
                    self._wait(response_window_s)

                    # Query spikes in the response window
                    query_start = stim_time
                    query_stop = stim_time + timedelta(milliseconds=self.response_window_ms)
                    try:
                        spike_df = self.database.get_spike_event(
                            query_start, query_stop, self.experiment.exp_name
                        )
                        has_response = not spike_df.empty
                    except Exception as exc:
                        logger.warning(
                            "Spike query failed for electrode %d trial %d: %s",
                            electrode_idx, trial, exc
                        )
                        has_response = False

                    self._trial_responses[(electrode_idx, amplitude_ua)].append(has_response)

                    logger.debug(
                        "Electrode %d | Amp %.1f uA | Trial %d | Response: %s",
                        electrode_idx, amplitude_ua, trial, has_response
                    )

                    # Inter-stimulation interval (minus the response window already waited)
                    remaining_isi = self.inter_stim_interval_s - response_window_s
                    if remaining_isi > 0:
                        self._wait(remaining_isi)

    def _deliver_stimulation(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity,
        trigger_key: int,
        trial_index: int,
    ) -> datetime:
        # Safety clamps
        amplitude_ua = min(abs(amplitude_ua), 4.0)
        duration_us = min(abs(duration_us), 400.0)

        # Charge balance: A1*D1 = A2*D2
        # Both phases have same amplitude and duration -> balanced
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

        # Fire trigger
        pattern = np.zeros(16, dtype=np.uint8)
        pattern[trigger_key] = 1
        self.trigger_controller.send(pattern)

        stim_time = datetime_now()

        self._wait(0.02)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            polarity=polarity.name,
            timestamp_utc=stim_time.isoformat(),
            trigger_key=trigger_key,
            trial_index=trial_index,
        ))

        return stim_time

    def _compute_response_rates(self) -> Dict[str, Any]:
        response_rates: Dict[str, Any] = {}

        # Per electrode, per amplitude
        per_electrode_per_amp: Dict[int, Dict[float, float]] = defaultdict(dict)
        for (electrode_idx, amplitude_ua), responses in self._trial_responses.items():
            if len(responses) == 0:
                rate = 0.0
            else:
                rate = sum(responses) / len(responses)
            per_electrode_per_amp[electrode_idx][amplitude_ua] = rate

        # Per electrode: average response rate across all amplitudes
        per_electrode_avg: Dict[int, float] = {}
        for electrode_idx, amp_rates in per_electrode_per_amp.items():
            if amp_rates:
                per_electrode_avg[electrode_idx] = sum(amp_rates.values()) / len(amp_rates)
            else:
                per_electrode_avg[electrode_idx] = 0.0

        # Find best electrode
        best_electrode = max(per_electrode_avg, key=lambda e: per_electrode_avg[e]) if per_electrode_avg else None
        best_rate = per_electrode_avg[best_electrode] if best_electrode is not None else 0.0

        response_rates["per_electrode_per_amplitude"] = {
            str(elec): {str(amp): rate for amp, rate in amp_dict.items()}
            for elec, amp_dict in per_electrode_per_amp.items()
        }
        response_rates["per_electrode_avg"] = {
            str(elec): rate for elec, rate in per_electrode_avg.items()
        }
        response_rates["best_electrode"] = best_electrode
        response_rates["best_electrode_avg_response_rate"] = best_rate

        logger.info("Best electrode: %s with avg response rate %.3f", best_electrode, best_rate)
        return response_rates

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

        response_rates = self._compute_response_rates()

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "electrodes_scanned": self.electrodes,
            "amplitudes_ua": self.amplitudes_ua,
            "stim_duration_us": self.stim_duration_us,
            "trials_per_condition": self.trials_per_condition,
            "inter_stim_interval_s": self.inter_stim_interval_s,
            "response_window_ms": self.response_window_ms,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "response_rates": response_rates,
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

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        logger.info("Compiling results")

        response_rates = self._compute_response_rates()

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "electrodes_scanned": self.electrodes,
            "amplitudes_ua": self.amplitudes_ua,
            "stim_duration_us": self.stim_duration_us,
            "trials_per_condition": self.trials_per_condition,
            "total_stimulations": len(self._stimulation_log),
            "response_rates": response_rates,
            "best_electrode": response_rates.get("best_electrode"),
            "best_electrode_avg_response_rate": response_rates.get("best_electrode_avg_response_rate"),
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
