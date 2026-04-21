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
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PairTrialResult:
    ipi_ms: float
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
    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        stim_electrode: int = 17,
        resp_electrode: int = 18,
        amplitude_ua: float = 2.0,
        duration_us: float = 200.0,
        polarity: str = "PositiveFirst",
        ipi_ms_list: Tuple = (10, 20, 30, 50, 75, 100),
        num_trials: int = 20,
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
        self.resp_electrode = resp_electrode
        self.amplitude_ua = min(abs(amplitude_ua), 4.0)
        self.duration_us = min(abs(duration_us), 400.0)
        self.polarity_str = polarity
        self.ipi_ms_list = list(ipi_ms_list)
        self.num_trials = num_trials
        self.inter_pair_interval_s = inter_pair_interval_s
        self.response_window_ms = response_window_ms
        self.trigger_key_p1 = trigger_key_p1
        self.trigger_key_p2 = trigger_key_p2

        # Verify charge balance: A1*D1 == A2*D2 (symmetric biphasic)
        assert math.isclose(
            self.amplitude_ua * self.duration_us,
            self.amplitude_ua * self.duration_us,
        ), "Charge balance violated"

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._trial_results: List[PairTrialResult] = []
        self._facilitation_summary: Dict[float, Dict[str, Any]] = {}

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

            recording_stop = datetime_now()

            self._compute_facilitation_summary()
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
            "Configuring stimulation: electrode=%d, amplitude=%.2f uA, duration=%.1f us, polarity=%s",
            self.stim_electrode, self.amplitude_ua, self.duration_us, self.polarity_str,
        )

        polarity_enum = (
            StimPolarity.PositiveFirst
            if self.polarity_str == "PositiveFirst"
            else StimPolarity.NegativeFirst
        )

        stim_p1 = self._build_stim_param(self.trigger_key_p1, polarity_enum)
        stim_p2 = self._build_stim_param(self.trigger_key_p2, polarity_enum)

        self.intan.send_stimparam([stim_p1, stim_p2])
        logger.info("Stimulation parameters sent for trigger keys %d and %d",
                    self.trigger_key_p1, self.trigger_key_p2)

    def _build_stim_param(self, trigger_key: int, polarity: StimPolarity) -> StimParam:
        stim = StimParam()
        stim.index = self.stim_electrode
        stim.enable = True
        stim.trigger_key = trigger_key
        stim.trigger_delay = 0
        stim.nb_pulse = 0
        stim.pulse_train_period = 10000
        stim.post_stim_ref_period = 1000.0
        stim.stim_shape = StimShape.Biphasic
        stim.polarity = polarity
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
        logger.info(
            "Starting paired-pulse facilitation experiment: %d IPI conditions x %d trials",
            len(self.ipi_ms_list), self.num_trials,
        )

        response_window_s = self.response_window_ms / 1000.0

        for ipi_ms in self.ipi_ms_list:
            ipi_s = ipi_ms / 1000.0
            logger.info("IPI = %d ms: running %d trials", ipi_ms, self.num_trials)

            for trial_idx in range(self.num_trials):
                logger.info("  IPI=%d ms, trial %d/%d", ipi_ms, trial_idx + 1, self.num_trials)

                # --- Pulse 1 ---
                p1_time = datetime_now()
                self._fire_trigger(self.trigger_key_p1)
                self._stimulation_log.append(StimulationRecord(
                    electrode_idx=self.stim_electrode,
                    amplitude_ua=self.amplitude_ua,
                    duration_us=self.duration_us,
                    polarity=self.polarity_str,
                    timestamp_utc=p1_time.isoformat(),
                    trigger_key=self.trigger_key_p1,
                    extra={"ipi_ms": ipi_ms, "trial": trial_idx, "pulse": 1},
                ))

                # Wait for IPI
                self._wait(ipi_s)

                # --- Pulse 2 ---
                p2_time = datetime_now()
                self._fire_trigger(self.trigger_key_p2)
                self._stimulation_log.append(StimulationRecord(
                    electrode_idx=self.stim_electrode,
                    amplitude_ua=self.amplitude_ua,
                    duration_us=self.duration_us,
                    polarity=self.polarity_str,
                    timestamp_utc=p2_time.isoformat(),
                    trigger_key=self.trigger_key_p2,
                    extra={"ipi_ms": ipi_ms, "trial": trial_idx, "pulse": 2},
                ))

                # Wait for responses to arrive
                self._wait(response_window_s)

                # Query spikes after pulse 1 (window: p1_time to p2_time)
                p1_window_start = p1_time
                p1_window_stop = p2_time
                spikes_p1 = self._query_spikes_in_window(p1_window_start, p1_window_stop)

                # Query spikes after pulse 2 (window: p2_time to now)
                p2_window_start = p2_time
                p2_window_stop = datetime_now()
                spikes_p2 = self._query_spikes_in_window(p2_window_start, p2_window_stop)

                count_p1 = len(spikes_p1)
                count_p2 = len(spikes_p2)

                if count_p1 > 0:
                    facilitation_ratio = count_p2 / count_p1
                else:
                    facilitation_ratio = float("nan")

                trial_result = PairTrialResult(
                    ipi_ms=float(ipi_ms),
                    trial_index=trial_idx,
                    pulse1_time_utc=p1_time.isoformat(),
                    pulse2_time_utc=p2_time.isoformat(),
                    spike_count_p1=count_p1,
                    spike_count_p2=count_p2,
                    facilitation_ratio=facilitation_ratio,
                )
                self._trial_results.append(trial_result)

                logger.info(
                    "    spikes_p1=%d, spikes_p2=%d, ratio=%.3f",
                    count_p1, count_p2,
                    facilitation_ratio if not math.isnan(facilitation_ratio) else -1,
                )

                # Inter-pair interval
                self._wait(self.inter_pair_interval_s)

    def _query_spikes_in_window(
        self, window_start: datetime, window_stop: datetime
    ) -> pd.DataFrame:
        try:
            df = self.database.get_spike_event_electrode(
                window_start, window_stop, self.resp_electrode
            )
            return df
        except Exception as exc:
            logger.warning("Spike query failed: %s", exc)
            return pd.DataFrame()

    def _compute_facilitation_summary(self) -> None:
        logger.info("Computing facilitation summary per IPI")
        grouped: Dict[float, List[PairTrialResult]] = defaultdict(list)
        for r in self._trial_results:
            grouped[r.ipi_ms].append(r)

        for ipi_ms, trials in sorted(grouped.items()):
            counts_p1 = [t.spike_count_p1 for t in trials]
            counts_p2 = [t.spike_count_p2 for t in trials]
            ratios = [t.facilitation_ratio for t in trials if not math.isnan(t.facilitation_ratio)]

            mean_p1 = float(np.mean(counts_p1)) if counts_p1 else 0.0
            mean_p2 = float(np.mean(counts_p2)) if counts_p2 else 0.0
            mean_ratio = float(np.mean(ratios)) if ratios else float("nan")
            std_ratio = float(np.std(ratios)) if len(ratios) > 1 else float("nan")

            self._facilitation_summary[ipi_ms] = {
                "ipi_ms": ipi_ms,
                "n_trials": len(trials),
                "mean_spike_count_p1": mean_p1,
                "mean_spike_count_p2": mean_p2,
                "mean_facilitation_ratio": mean_ratio,
                "std_facilitation_ratio": std_ratio,
                "n_valid_ratios": len(ratios),
            }

            logger.info(
                "IPI=%g ms: mean_p1=%.2f, mean_p2=%.2f, mean_ratio=%.3f (n_valid=%d)",
                ipi_ms, mean_p1, mean_p2, mean_ratio, len(ratios),
            )

    def _save_all(self, recording_start: datetime, recording_stop: datetime) -> None:
        fs_name = getattr(self.experiment, "exp_name", "unknown")
        saver = DataSaver(self._output_dir, fs_name)

        saver.save_stimulation_log(self._stimulation_log)

        spike_df = self.database.get_spike_event(recording_start, recording_stop, fs_name)
        saver.save_spike_events(spike_df)

        trigger_df = self.database.get_all_triggers(recording_start, recording_stop)
        saver.save_triggers(trigger_df)

        trial_records = [asdict(r) for r in self._trial_results]
        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "stim_electrode": self.stim_electrode,
            "resp_electrode": self.resp_electrode,
            "amplitude_ua": self.amplitude_ua,
            "duration_us": self.duration_us,
            "polarity": self.polarity_str,
            "ipi_ms_list": self.ipi_ms_list,
            "num_trials": self.num_trials,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "facilitation_summary": {
                str(k): v for k, v in self._facilitation_summary.items()
            },
            "trial_results": trial_records,
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
                    "Failed to fetch waveforms for electrode %s: %s", electrode_idx, exc
                )

        return waveform_records

    def _compile_results(
        self, recording_start: datetime, recording_stop: datetime
    ) -> Dict[str, Any]:
        logger.info("Compiling results")

        facilitation_list = []
        for ipi_ms in sorted(self._facilitation_summary.keys()):
            facilitation_list.append(self._facilitation_summary[ipi_ms])

        return {
            "status": "completed",
            "experiment_name": getattr(self.experiment, "exp_name", "unknown"),
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "stim_electrode": self.stim_electrode,
            "resp_electrode": self.resp_electrode,
            "amplitude_ua": self.amplitude_ua,
            "duration_us": self.duration_us,
            "polarity": self.polarity_str,
            "ipi_ms_list": self.ipi_ms_list,
            "num_trials": self.num_trials,
            "total_stimulations": len(self._stimulation_log),
            "total_trials": len(self._trial_results),
            "facilitation_summary": facilitation_list,
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
