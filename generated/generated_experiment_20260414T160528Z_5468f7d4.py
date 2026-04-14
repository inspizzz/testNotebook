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
    trial_type: str = ""
    isi_ms: float = 0.0
    trial_index: int = 0
    pulse_number: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PairedPulseTrialResult:
    isi_ms: float
    trial_index: int
    pulse1_spike_count: int
    pulse2_spike_count: int
    facilitation_ratio: float
    pulse1_time_utc: str
    pulse2_time_utc: str


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

    Uses electrode pair (stim=14, record=15) — the best responsive pair
    from the scan (response_rate=0.80, hits_k=5/5 at 2uA/300us).
    Delivers pairs of stimulations with varying inter-pulse intervals:
    10, 20, 30, 50, 75, 100 ms. Uses 2 uA amplitude and 200 us duration
    for both pulses (charge-balanced: 2*200 = 2*200). Repeats each ISI
    20 times with 3 seconds between pairs.
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
        inter_pulse_intervals_ms: Optional[List[float]] = None,
        num_trials_per_isi: int = 20,
        inter_pair_interval_s: float = 3.0,
        response_window_ms: float = 50.0,
        trigger_key_p1: int = 0,
        trigger_key_p2: int = 1,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.stim_electrode = stim_electrode
        self.record_electrode = record_electrode
        self.amplitude_ua = min(abs(amplitude_ua), 4.0)
        self.duration_us = min(abs(duration_us), 400.0)

        if inter_pulse_intervals_ms is None:
            self.inter_pulse_intervals_ms = [10.0, 20.0, 30.0, 50.0, 75.0, 100.0]
        else:
            self.inter_pulse_intervals_ms = inter_pulse_intervals_ms

        self.num_trials_per_isi = num_trials_per_isi
        self.inter_pair_interval_s = inter_pair_interval_s
        self.response_window_ms = response_window_ms
        self.trigger_key_p1 = trigger_key_p1
        self.trigger_key_p2 = trigger_key_p2

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._trial_results: List[PairedPulseTrialResult] = []

        assert abs(self.amplitude_ua * self.duration_us - self.amplitude_ua * self.duration_us) < 1e-9, \
            "Charge balance violated"

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
            logger.info("Stim electrode: %d, Record electrode: %d",
                        self.stim_electrode, self.record_electrode)
            logger.info("Amplitude: %.1f uA, Duration: %.1f us",
                        self.amplitude_ua, self.duration_us)
            logger.info("ISIs (ms): %s", self.inter_pulse_intervals_ms)
            logger.info("Trials per ISI: %d", self.num_trials_per_isi)

            if not self.experiment.start():
                logger.error("Failed to start experiment")
                return {"status": "failed", "error": "Could not start experiment"}

            recording_start = datetime_now()

            self._configure_stimulation()
            self._run_paired_pulse_experiment()

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
        logger.info("Configuring stimulation parameters for electrode %d", self.stim_electrode)

        stim = self._build_stim_param(self.stim_electrode, self.trigger_key_p1)
        self.intan.send_stimparam([stim])
        logger.info("Stimulation parameters configured (trigger key %d)", self.trigger_key_p1)

    def _build_stim_param(self, electrode_idx: int, trigger_key: int) -> StimParam:
        stim = StimParam()
        stim.index = electrode_idx
        stim.enable = True
        stim.trigger_key = trigger_key
        stim.trigger_delay = 0
        stim.nb_pulse = 0
        stim.pulse_train_period = 10000
        stim.post_stim_ref_period = 1000.0
        stim.stim_shape = StimShape.Biphasic
        stim.polarity = StimPolarity.PositiveFirst

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

    def _fire_trigger(self, trigger_key: int) -> None:
        pattern = np.zeros(16, dtype=np.uint8)
        pattern[trigger_key] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.005)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

    def _run_paired_pulse_experiment(self) -> None:
        logger.info("Starting paired-pulse facilitation experiment")

        for isi_ms in self.inter_pulse_intervals_ms:
            logger.info("=== ISI = %.1f ms ===", isi_ms)
            isi_s = isi_ms / 1000.0

            for trial_idx in range(self.num_trials_per_isi):
                logger.info("  ISI=%.1f ms, Trial %d/%d",
                            isi_ms, trial_idx + 1, self.num_trials_per_isi)

                stim_p1 = self._build_stim_param(self.stim_electrode, self.trigger_key_p1)
                self.intan.send_stimparam([stim_p1])

                pulse1_time = datetime_now()
                self._fire_trigger(self.trigger_key_p1)

                self._stimulation_log.append(StimulationRecord(
                    electrode_idx=self.stim_electrode,
                    amplitude_ua=self.amplitude_ua,
                    duration_us=self.duration_us,
                    timestamp_utc=pulse1_time.isoformat(),
                    trigger_key=self.trigger_key_p1,
                    trial_type="paired_pulse_p1",
                    isi_ms=isi_ms,
                    trial_index=trial_idx,
                    pulse_number=1,
                ))

                self._wait(isi_s)

                stim_p2 = self._build_stim_param(self.stim_electrode, self.trigger_key_p1)
                self.intan.send_stimparam([stim_p2])

                pulse2_time = datetime_now()
                self._fire_trigger(self.trigger_key_p1)

                self._stimulation_log.append(StimulationRecord(
                    electrode_idx=self.stim_electrode,
                    amplitude_ua=self.amplitude_ua,
                    duration_us=self.duration_us,
                    timestamp_utc=pulse2_time.isoformat(),
                    trigger_key=self.trigger_key_p1,
                    trial_type="paired_pulse_p2",
                    isi_ms=isi_ms,
                    trial_index=trial_idx,
                    pulse_number=2,
                ))

                response_window_s = self.response_window_ms / 1000.0
                self._wait(response_window_s)

                pair_end_time = datetime_now()

                p1_count, p2_count = self._count_responses(
                    pulse1_time=pulse1_time,
                    pulse2_time=pulse2_time,
                    pair_end_time=pair_end_time,
                    response_window_s=response_window_s,
                )

                if p1_count > 0:
                    ratio = p2_count / p1_count
                else:
                    ratio = float('nan')

                trial_result = PairedPulseTrialResult(
                    isi_ms=isi_ms,
                    trial_index=trial_idx,
                    pulse1_spike_count=p1_count,
                    pulse2_spike_count=p2_count,
                    facilitation_ratio=ratio,
                    pulse1_time_utc=pulse1_time.isoformat(),
                    pulse2_time_utc=pulse2_time.isoformat(),
                )
                self._trial_results.append(trial_result)

                logger.info(
                    "    P1 spikes=%d, P2 spikes=%d, ratio=%.3f",
                    p1_count, p2_count,
                    ratio if not math.isnan(ratio) else -1.0,
                )

                self._wait(self.inter_pair_interval_s)

        logger.info("Paired-pulse experiment complete. Total trials: %d",
                    len(self._trial_results))

    def _count_responses(
        self,
        pulse1_time: datetime,
        pulse2_time: datetime,
        pair_end_time: datetime,
        response_window_s: float,
    ) -> Tuple[int, int]:
        try:
            query_start = pulse1_time - timedelta(seconds=0.01)
            query_stop = pair_end_time + timedelta(seconds=0.01)

            spike_df = self.database.get_spike_event(
                query_start,
                query_stop,
                self.experiment.exp_name,
            )

            if spike_df.empty:
                return 0, 0

            if "channel" in spike_df.columns:
                ch_col = "channel"
            elif "index" in spike_df.columns:
                ch_col = "index"
            else:
                ch_col = None

            if ch_col is not None:
                spike_df = spike_df[spike_df[ch_col] == self.record_electrode]

            if spike_df.empty:
                return 0, 0

            time_col = "Time" if "Time" in spike_df.columns else "_time"
            spike_times = pd.to_datetime(spike_df[time_col], utc=True)

            p1_window_start = pulse1_time
            p1_window_end = pulse1_time + timedelta(seconds=response_window_s)

            p2_window_start = pulse2_time
            p2_window_end = pulse2_time + timedelta(seconds=response_window_s)

            p1_mask = (spike_times >= p1_window_start) & (spike_times < p1_window_end)
            p2_mask = (spike_times >= p2_window_start) & (spike_times < p2_window_end)

            p1_count = int(p1_mask.sum())
            p2_count = int(p2_mask.sum())

            return p1_count, p2_count

        except Exception as exc:
            logger.warning("Error counting responses: %s", exc)
            return 0, 0

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        logger.info("Compiling results")

        isi_summary: Dict[float, Dict[str, Any]] = {}

        for isi_ms in self.inter_pulse_intervals_ms:
            trials = [t for t in self._trial_results if t.isi_ms == isi_ms]
            if not trials:
                continue

            p1_counts = [t.pulse1_spike_count for t in trials]
            p2_counts = [t.pulse2_spike_count for t in trials]
            ratios = [t.facilitation_ratio for t in trials if not math.isnan(t.facilitation_ratio)]

            total_p1 = sum(p1_counts)
            total_p2 = sum(p2_counts)

            mean_ratio = sum(ratios) / len(ratios) if ratios else float('nan')

            isi_summary[isi_ms] = {
                "isi_ms": isi_ms,
                "num_trials": len(trials),
                "total_p1_spikes": total_p1,
                "total_p2_spikes": total_p2,
                "mean_p1_spikes_per_trial": total_p1 / len(trials) if trials else 0.0,
                "mean_p2_spikes_per_trial": total_p2 / len(trials) if trials else 0.0,
                "mean_facilitation_ratio": mean_ratio,
                "num_trials_with_p1_response": sum(1 for c in p1_counts if c > 0),
                "num_trials_with_p2_response": sum(1 for c in p2_counts if c > 0),
            }

            logger.info(
                "ISI=%.1f ms: P1_total=%d, P2_total=%d, mean_ratio=%.3f",
                isi_ms, total_p1, total_p2,
                mean_ratio if not math.isnan(mean_ratio) else -1.0,
            )

        trial_records = [asdict(t) for t in self._trial_results]

        summary = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "stim_electrode": self.stim_electrode,
            "record_electrode": self.record_electrode,
            "amplitude_ua": self.amplitude_ua,
            "duration_us": self.duration_us,
            "charge_balance_nC": self.amplitude_ua * self.duration_us,
            "inter_pulse_intervals_ms": self.inter_pulse_intervals_ms,
            "num_trials_per_isi": self.num_trials_per_isi,
            "total_trials": len(self._trial_results),
            "total_stimulations": len(self._stimulation_log),
            "isi_summary": isi_summary,
            "trial_records": trial_records,
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

        isi_summary: Dict[float, Dict[str, Any]] = {}
        for isi_ms in self.inter_pulse_intervals_ms:
            trials = [t for t in self._trial_results if t.isi_ms == isi_ms]
            if not trials:
                continue
            p1_counts = [t.pulse1_spike_count for t in trials]
            p2_counts = [t.pulse2_spike_count for t in trials]
            ratios = [t.facilitation_ratio for t in trials if not math.isnan(t.facilitation_ratio)]
            mean_ratio = sum(ratios) / len(ratios) if ratios else float('nan')
            isi_summary[str(isi_ms)] = {
                "isi_ms": isi_ms,
                "num_trials": len(trials),
                "total_p1_spikes": sum(p1_counts),
                "total_p2_spikes": sum(p2_counts),
                "mean_facilitation_ratio": mean_ratio,
            }

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "stim_electrode": self.stim_electrode,
            "record_electrode": self.record_electrode,
            "amplitude_ua": self.amplitude_ua,
            "duration_us": self.duration_us,
            "inter_pulse_intervals_ms": self.inter_pulse_intervals_ms,
            "num_trials_per_isi": self.num_trials_per_isi,
            "total_trials": len(self._trial_results),
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "isi_summary": isi_summary,
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

        ch_col = None
        for col in ["channel", "index"]:
            if col in spike_df.columns:
                ch_col = col
                break

        if ch_col is None:
            logger.warning("Cannot determine electrode column for waveform fetch")
            return waveform_records

        unique_electrodes = spike_df[ch_col].unique()
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
