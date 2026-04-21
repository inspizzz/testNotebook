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
    amplitude_level_idx: int = 0
    trial_idx: int = 0
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
        num_trials: int = 10,
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
        self.num_trials = num_trials
        self.inter_stim_interval_s = inter_stim_interval_s
        self.response_window_ms = response_window_ms
        self.trigger_key = trigger_key

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []

        # Results storage: keyed by (electrode_idx, amplitude_ua) -> list of bool (responded or not)
        self._response_matrix: Dict[Tuple[int, float], List[bool]] = defaultdict(list)

        # Per-electrode response rates aggregated across all amplitudes
        self._electrode_response_rates: Dict[int, float] = {}

        # Per (electrode, amplitude) response rates
        self._condition_response_rates: Dict[Tuple[int, float], float] = {}

        # Best electrode
        self._best_electrode: Optional[int] = None
        self._best_response_rate: float = 0.0

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

            self._compute_response_rates()

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
            "Electrodes: %s | Amplitudes: %s uA | Duration: %s us | Trials: %d",
            self.electrodes,
            self.amplitudes_ua,
            self.stim_duration_us,
            self.num_trials,
        )

        # Validate charge balance: A1*D1 == A2*D2
        # Since we use equal amplitudes and equal durations on both phases, balance is guaranteed.
        # Also verify duration does not exceed 400 us
        duration_us = min(self.stim_duration_us, 400.0)

        for electrode_idx in self.electrodes:
            for amplitude_ua in self.amplitudes_ua:
                # Clamp amplitude
                amp = min(abs(amplitude_ua), 4.0)

                logger.info(
                    "Scanning electrode %d at %.1f uA (%d trials)",
                    electrode_idx, amp, self.num_trials,
                )

                for trial in range(self.num_trials):
                    stim_time, spike_df = self._stimulate_and_record(
                        electrode_idx=electrode_idx,
                        amplitude_ua=amp,
                        duration_us=duration_us,
                        polarity=StimPolarity.PositiveFirst,
                        trigger_key=self.trigger_key,
                        post_stim_wait_s=self.response_window_ms / 1000.0 + 0.05,
                        amplitude_level_idx=self.amplitudes_ua.index(amplitude_ua),
                        trial_idx=trial,
                    )

                    responded = self._check_response(
                        spike_df=spike_df,
                        stim_time=stim_time,
                        response_window_ms=self.response_window_ms,
                    )
                    self._response_matrix[(electrode_idx, amp)].append(responded)

                    logger.debug(
                        "  Electrode %d | %.1f uA | Trial %d | Responded: %s",
                        electrode_idx, amp, trial, responded,
                    )

                    # Inter-stimulation interval
                    self._wait(self.inter_stim_interval_s)

        logger.info("Excitability scan complete")

    def _stimulate_and_record(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity = StimPolarity.PositiveFirst,
        trigger_key: int = 0,
        post_stim_wait_s: float = 0.1,
        amplitude_level_idx: int = 0,
        trial_idx: int = 0,
    ) -> Tuple[datetime, pd.DataFrame]:
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

        # Charge-balanced: A1*D1 == A2*D2 (equal amplitudes, equal durations)
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

        # Fire trigger
        pattern = np.zeros(16, dtype=np.uint8)
        pattern[trigger_key] = 1
        self.trigger_controller.send(pattern)

        stim_time = datetime_now()

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
            amplitude_level_idx=amplitude_level_idx,
            trial_idx=trial_idx,
        ))

        self._wait(post_stim_wait_s)

        query_stop = datetime_now()
        query_start = stim_time - timedelta(seconds=0.01)

        spike_df = self.database.get_spike_event(
            query_start, query_stop, self.experiment.exp_name
        )
        return stim_time, spike_df

    def _check_response(
        self,
        spike_df: pd.DataFrame,
        stim_time: datetime,
        response_window_ms: float,
    ) -> bool:
        if spike_df is None or spike_df.empty:
            return False

        time_col = None
        for col in spike_df.columns:
            if col.lower() in ("time", "_time", "timestamp"):
                time_col = col
                break

        if time_col is None:
            return False

        window_end = stim_time + timedelta(milliseconds=response_window_ms)

        times = pd.to_datetime(spike_df[time_col], utc=True)
        stim_ts = pd.Timestamp(stim_time)
        window_end_ts = pd.Timestamp(window_end)

        mask = (times >= stim_ts) & (times <= window_end_ts)
        return bool(mask.any())

    def _compute_response_rates(self) -> None:
        logger.info("Computing per-electrode response rates")

        electrode_all_responses: Dict[int, List[bool]] = defaultdict(list)

        for (electrode_idx, amplitude_ua), responses in self._response_matrix.items():
            n = len(responses)
            rate = sum(responses) / n if n > 0 else 0.0
            self._condition_response_rates[(electrode_idx, amplitude_ua)] = rate
            electrode_all_responses[electrode_idx].extend(responses)
            logger.info(
                "  Electrode %d | %.1f uA | Response rate: %.3f (%d/%d)",
                electrode_idx, amplitude_ua, rate, sum(responses), n,
            )

        for electrode_idx, all_responses in electrode_all_responses.items():
            n = len(all_responses)
            rate = sum(all_responses) / n if n > 0 else 0.0
            self._electrode_response_rates[electrode_idx] = rate

        if self._electrode_response_rates:
            best_elec = max(
                self._electrode_response_rates,
                key=lambda e: self._electrode_response_rates[e],
            )
            self._best_electrode = best_elec
            self._best_response_rate = self._electrode_response_rates[best_elec]
            logger.info(
                "Best electrode: %d with overall response rate %.3f",
                self._best_electrode, self._best_response_rate,
            )

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        logger.info("Compiling results")

        condition_rates_serializable = {
            f"electrode_{e}_amplitude_{a}": rate
            for (e, a), rate in self._condition_response_rates.items()
        }

        electrode_rates_serializable = {
            f"electrode_{e}": rate
            for e, rate in self._electrode_response_rates.items()
        }

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "electrodes_scanned": self.electrodes,
            "amplitudes_ua": self.amplitudes_ua,
            "stim_duration_us": self.stim_duration_us,
            "num_trials_per_condition": self.num_trials,
            "response_window_ms": self.response_window_ms,
            "total_stimulations": len(self._stimulation_log),
            "per_condition_response_rates": condition_rates_serializable,
            "per_electrode_response_rates": electrode_rates_serializable,
            "best_electrode": self._best_electrode,
            "best_electrode_response_rate": self._best_response_rate,
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
        saver.save_spike_events(spike_df)

        trigger_df = self.database.get_all_triggers(
            recording_start, recording_stop
        )
        saver.save_triggers(trigger_df)

        condition_rates_serializable = {
            f"electrode_{e}_amplitude_{a}": rate
            for (e, a), rate in self._condition_response_rates.items()
        }
        electrode_rates_serializable = {
            f"electrode_{e}": rate
            for e, rate in self._electrode_response_rates.items()
        }

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "electrodes_scanned": self.electrodes,
            "amplitudes_ua": self.amplitudes_ua,
            "stim_duration_us": self.stim_duration_us,
            "num_trials_per_condition": self.num_trials,
            "response_window_ms": self.response_window_ms,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "per_condition_response_rates": condition_rates_serializable,
            "per_electrode_response_rates": electrode_rates_serializable,
            "best_electrode": self._best_electrode,
            "best_electrode_response_rate": self._best_response_rate,
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
        if spike_df is None or spike_df.empty:
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
                if raw_df is not None and not raw_df.empty:
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
