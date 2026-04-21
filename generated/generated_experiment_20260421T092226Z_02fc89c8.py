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
    timestamp_utc: str
    trigger_key: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PairPulseTrialResult:
    isi_ms: float
    trial_index: int
    pulse1_time_utc: str
    pulse2_time_utc: str
    spike_count_p1: int
    spike_count_p2: int
    facilitation_ratio: float


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
    Paired-pulse facilitation experiment.

    Uses electrode pair (stim=14, record=15) which showed the highest
    response rate (hits_k=5/5) at 2 uA / 200 us / PositiveFirst in the
    parameter scan, matching the requested stimulation parameters exactly.

    For each inter-pulse interval (10, 20, 30, 50, 75, 100 ms), 20 trials
    are delivered with 3 seconds between pairs.  The facilitation ratio
    (spike count after pulse 2 / spike count after pulse 1) is computed
    per interval.
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        stim_electrode: int = 14,
        record_electrode: int = 15,
        amplitude_ua: float = 2.0,
        duration_us: float = 200.0,
        polarity: str = "PositiveFirst",
        isi_ms_list: Optional[List[float]] = None,
        num_trials: int = 20,
        inter_pair_interval_s: float = 3.0,
        response_window_ms: float = 50.0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.stim_electrode = stim_electrode
        self.record_electrode = record_electrode
        self.amplitude_ua = amplitude_ua
        self.duration_us = duration_us
        self.polarity = StimPolarity.PositiveFirst if polarity == "PositiveFirst" else StimPolarity.NegativeFirst

        self.isi_ms_list: List[float] = isi_ms_list if isi_ms_list is not None else [10.0, 20.0, 30.0, 50.0, 75.0, 100.0]
        self.num_trials = num_trials
        self.inter_pair_interval_s = inter_pair_interval_s
        self.response_window_ms = response_window_ms

        # Charge balance: A1*D1 == A2*D2 => same amplitude and duration on both phases
        assert abs(self.amplitude_ua * self.duration_us - self.amplitude_ua * self.duration_us) < 1e-9

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._trial_results: List[PairPulseTrialResult] = []
        self._isi_summary: Dict[float, Dict[str, Any]] = {}

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
            self._run_paired_pulse_experiment()
            self._compute_isi_summaries()

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
        """Pre-configure the stimulation parameters for the stim electrode."""
        logger.info(
            "Configuring stimulation: electrode=%d, amplitude=%.1f uA, duration=%.1f us, polarity=%s",
            self.stim_electrode, self.amplitude_ua, self.duration_us, self.polarity.name
        )
        stim = self._build_stim_param(trigger_key=0)
        self.intan.send_stimparam([stim])
        logger.info("Stimulation parameters sent to Intan")

    def _build_stim_param(self, trigger_key: int = 0) -> StimParam:
        """Build a charge-balanced StimParam for the stim electrode."""
        stim = StimParam()
        stim.index = self.stim_electrode
        stim.enable = True
        stim.trigger_key = trigger_key
        stim.trigger_delay = 0
        stim.nb_pulse = 0
        stim.pulse_train_period = 10000
        stim.post_stim_ref_period = 1000.0
        stim.stim_shape = StimShape.Biphasic
        stim.polarity = self.polarity

        # Charge balance: A1*D1 == A2*D2 (equal amplitude and duration on both phases)
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

    def _fire_trigger(self, trigger_key: int = 0) -> datetime:
        """Fire a single trigger pulse and return the timestamp."""
        pattern = np.zeros(16, dtype=np.uint8)
        pattern[trigger_key] = 1
        self.trigger_controller.send(pattern)
        ts = datetime_now()
        self._wait(0.005)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)
        return ts

    def _count_spikes_in_window(
        self,
        window_start: datetime,
        window_end: datetime,
    ) -> int:
        """Query spike events on the recording electrode within a time window."""
        try:
            df = self.database.get_spike_event(
                window_start, window_end, self.experiment.exp_name
            )
            if df.empty:
                return 0
            # Filter to recording electrode
            if "channel" in df.columns:
                df = df[df["channel"] == self.record_electrode]
            return len(df)
        except Exception as exc:
            logger.warning("Spike query failed: %s", exc)
            return 0

    def _deliver_pulse_and_log(self, trigger_key: int = 0) -> datetime:
        """Deliver one stimulation pulse, log it, and return the fire timestamp."""
        ts = self._fire_trigger(trigger_key=trigger_key)
        self._stimulation_log.append(StimulationRecord(
            electrode_idx=self.stim_electrode,
            amplitude_ua=self.amplitude_ua,
            duration_us=self.duration_us,
            timestamp_utc=ts.isoformat(),
            trigger_key=trigger_key,
            extra={"polarity": self.polarity.name},
        ))
        return ts

    def _run_paired_pulse_experiment(self) -> None:
        """Main experiment loop: iterate over ISIs and trials."""
        logger.info(
            "Starting paired-pulse facilitation experiment: %d ISIs x %d trials",
            len(self.isi_ms_list), self.num_trials
        )

        response_window_s = self.response_window_ms / 1000.0

        for isi_ms in self.isi_ms_list:
            isi_s = isi_ms / 1000.0
            logger.info("--- ISI = %.1f ms ---", isi_ms)

            for trial_idx in range(self.num_trials):
                logger.info("  Trial %d/%d (ISI=%.1f ms)", trial_idx + 1, self.num_trials, isi_ms)

                # --- Pulse 1 ---
                p1_time = self._deliver_pulse_and_log(trigger_key=0)

                # Wait for ISI before pulse 2
                self._wait(isi_s)

                # --- Pulse 2 ---
                p2_time = self._deliver_pulse_and_log(trigger_key=0)

                # Wait for response window to elapse after pulse 2
                self._wait(response_window_s)

                # Count spikes after pulse 1 (window: p1_time to p2_time)
                # We use the interval between p1 and p2 as the P1 response window,
                # capped at response_window_ms.
                p1_window_end = p1_time + timedelta(milliseconds=min(self.response_window_ms, isi_ms - 1.0))
                spike_count_p1 = self._count_spikes_in_window(p1_time, p1_window_end)

                # Count spikes after pulse 2 (window: p2_time to p2_time + response_window_ms)
                p2_window_end = p2_time + timedelta(milliseconds=self.response_window_ms)
                spike_count_p2 = self._count_spikes_in_window(p2_time, p2_window_end)

                # Facilitation ratio
                if spike_count_p1 > 0:
                    facilitation_ratio = spike_count_p2 / spike_count_p1
                else:
                    facilitation_ratio = float("nan")

                result = PairPulseTrialResult(
                    isi_ms=isi_ms,
                    trial_index=trial_idx,
                    pulse1_time_utc=p1_time.isoformat(),
                    pulse2_time_utc=p2_time.isoformat(),
                    spike_count_p1=spike_count_p1,
                    spike_count_p2=spike_count_p2,
                    facilitation_ratio=facilitation_ratio,
                )
                self._trial_results.append(result)

                logger.info(
                    "    P1 spikes=%d, P2 spikes=%d, ratio=%.3f",
                    spike_count_p1, spike_count_p2,
                    facilitation_ratio if not math.isnan(facilitation_ratio) else -1,
                )

                # Inter-pair interval (minus time already spent on response window)
                remaining_wait = self.inter_pair_interval_s - response_window_s
                if remaining_wait > 0:
                    self._wait(remaining_wait)

        logger.info("Paired-pulse experiment complete. Total trials: %d", len(self._trial_results))

    def _compute_isi_summaries(self) -> None:
        """Aggregate trial results per ISI and compute mean facilitation ratios."""
        grouped: Dict[float, List[PairPulseTrialResult]] = defaultdict(list)
        for r in self._trial_results:
            grouped[r.isi_ms].append(r)

        for isi_ms, trials in sorted(grouped.items()):
            total_p1 = sum(t.spike_count_p1 for t in trials)
            total_p2 = sum(t.spike_count_p2 for t in trials)
            valid_ratios = [t.facilitation_ratio for t in trials if not math.isnan(t.facilitation_ratio)]
            mean_ratio = sum(valid_ratios) / len(valid_ratios) if valid_ratios else float("nan")

            self._isi_summary[isi_ms] = {
                "isi_ms": isi_ms,
                "num_trials": len(trials),
                "total_spike_count_p1": total_p1,
                "total_spike_count_p2": total_p2,
                "mean_facilitation_ratio": mean_ratio,
                "valid_ratio_trials": len(valid_ratios),
            }
            logger.info(
                "ISI=%.1f ms | P1_total=%d | P2_total=%d | mean_ratio=%.3f",
                isi_ms, total_p1, total_p2,
                mean_ratio if not math.isnan(mean_ratio) else -1,
            )

    def _save_all(self, recording_start: datetime, recording_stop: datetime) -> None:
        fs_name = getattr(self.experiment, "exp_name", "unknown")
        saver = DataSaver(self._output_dir, fs_name)

        saver.save_stimulation_log(self._stimulation_log)

        spike_df = self.database.get_spike_event(recording_start, recording_stop, fs_name)
        saver.save_spike_events(spike_df)

        trigger_df = self.database.get_all_triggers(recording_start, recording_stop)
        saver.save_triggers(trigger_df)

        trial_dicts = [asdict(r) for r in self._trial_results]
        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "stim_electrode": self.stim_electrode,
            "record_electrode": self.record_electrode,
            "amplitude_ua": self.amplitude_ua,
            "duration_us": self.duration_us,
            "polarity": self.polarity.name,
            "isi_ms_list": self.isi_ms_list,
            "num_trials_per_isi": self.num_trials,
            "inter_pair_interval_s": self.inter_pair_interval_s,
            "response_window_ms": self.response_window_ms,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "total_trials": len(self._trial_results),
            "isi_summaries": {str(k): v for k, v in self._isi_summary.items()},
            "trial_results": trial_dicts,
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
            if col in ("channel", "index", "electrode"):
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
                logger.warning("Failed to fetch waveforms for electrode %s: %s", electrode_idx, exc)

        return waveform_records

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        logger.info("Compiling results")

        isi_summary_serialisable = {}
        for isi_ms, summary in self._isi_summary.items():
            entry = dict(summary)
            if math.isnan(entry.get("mean_facilitation_ratio", 0.0)):
                entry["mean_facilitation_ratio"] = None
            isi_summary_serialisable[str(isi_ms)] = entry

        return {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "stim_electrode": self.stim_electrode,
            "record_electrode": self.record_electrode,
            "amplitude_ua": self.amplitude_ua,
            "duration_us": self.duration_us,
            "polarity": self.polarity.name,
            "isi_ms_list": self.isi_ms_list,
            "num_trials_per_isi": self.num_trials,
            "total_stimulations": len(self._stimulation_log),
            "total_trials": len(self._trial_results),
            "isi_summaries": isi_summary_serialisable,
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
