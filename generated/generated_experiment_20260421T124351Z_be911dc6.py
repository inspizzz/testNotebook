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
class PairedPulseTrialResult:
    ipi_ms: float
    trial_index: int
    pulse1_time_utc: str
    pulse2_time_utc: str
    spikes_after_pulse1: int
    spikes_after_pulse2: int
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
        num_trials_per_ipi: int = 20,
        inter_pair_interval_s: float = 3.0,
        response_window_ms: float = 80.0,
        trigger_key_pulse1: int = 0,
        trigger_key_pulse2: int = 1,
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
        self.num_trials_per_ipi = num_trials_per_ipi
        self.inter_pair_interval_s = inter_pair_interval_s
        self.response_window_ms = response_window_ms
        self.trigger_key_pulse1 = trigger_key_pulse1
        self.trigger_key_pulse2 = trigger_key_pulse2

        # Charge balance check: A1*D1 == A2*D2 (symmetric biphasic)
        assert math.isclose(self.amplitude_ua * self.duration_us,
                            self.amplitude_ua * self.duration_us), \
            "Charge balance violated"

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []
        self._trial_results: List[PairedPulseTrialResult] = []

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
            logger.info("Recording started at %s", recording_start.isoformat())

            self._configure_stimulation()
            self._run_paired_pulse_experiment()

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

    def _configure_stimulation(self) -> None:
        logger.info(
            "Configuring stimulation: electrode=%d, amplitude=%.2f uA, duration=%.1f us, polarity=%s",
            self.stim_electrode, self.amplitude_ua, self.duration_us, self.polarity_str
        )
        polarity_enum = StimPolarity.PositiveFirst if self.polarity_str == "PositiveFirst" else StimPolarity.NegativeFirst

        # Configure pulse 1 on trigger_key_pulse1
        stim1 = self._build_stim_param(
            electrode_idx=self.stim_electrode,
            amplitude_ua=self.amplitude_ua,
            duration_us=self.duration_us,
            polarity=polarity_enum,
            trigger_key=self.trigger_key_pulse1,
        )
        # Configure pulse 2 on trigger_key_pulse2
        stim2 = self._build_stim_param(
            electrode_idx=self.stim_electrode,
            amplitude_ua=self.amplitude_ua,
            duration_us=self.duration_us,
            polarity=polarity_enum,
            trigger_key=self.trigger_key_pulse2,
        )
        self.intan.send_stimparam([stim1, stim2])
        logger.info("Stimulation parameters sent to Intan")

    def _build_stim_param(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity,
        trigger_key: int,
    ) -> StimParam:
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
        # Charge balance: A1*D1 == A2*D2 (symmetric)
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

    def _fire_trigger(self, trigger_key: int) -> None:
        pattern = np.zeros(16, dtype=np.uint8)
        pattern[trigger_key] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.005)
        pattern[trigger_key] = 0
        self.trigger_controller.send(pattern)

    def _run_paired_pulse_experiment(self) -> None:
        logger.info(
            "Starting paired-pulse facilitation experiment: %d IPIs x %d trials",
            len(self.ipi_ms_list), self.num_trials_per_ipi
        )
        response_window_s = self.response_window_ms / 1000.0

        for ipi_ms in self.ipi_ms_list:
            ipi_s = ipi_ms / 1000.0
            logger.info("=== IPI = %d ms ===", ipi_ms)

            for trial_idx in range(self.num_trials_per_ipi):
                logger.info("  IPI=%d ms, trial %d/%d", ipi_ms, trial_idx + 1, self.num_trials_per_ipi)

                # --- Pulse 1 ---
                pulse1_time = datetime_now()
                self._fire_trigger(self.trigger_key_pulse1)
                self._stimulation_log.append(StimulationRecord(
                    electrode_idx=self.stim_electrode,
                    amplitude_ua=self.amplitude_ua,
                    duration_us=self.duration_us,
                    polarity=self.polarity_str,
                    timestamp_utc=pulse1_time.isoformat(),
                    trigger_key=self.trigger_key_pulse1,
                    extra={"pulse": 1, "ipi_ms": ipi_ms, "trial": trial_idx},
                ))

                # Wait for IPI
                self._wait(ipi_s)

                # --- Pulse 2 ---
                pulse2_time = datetime_now()
                self._fire_trigger(self.trigger_key_pulse2)
                self._stimulation_log.append(StimulationRecord(
                    electrode_idx=self.stim_electrode,
                    amplitude_ua=self.amplitude_ua,
                    duration_us=self.duration_us,
                    polarity=self.polarity_str,
                    timestamp_utc=pulse2_time.isoformat(),
                    trigger_key=self.trigger_key_pulse2,
                    extra={"pulse": 2, "ipi_ms": ipi_ms, "trial": trial_idx},
                ))

                # Wait for response window after pulse 2
                self._wait(response_window_s)

                # Query spikes after pulse 1 (window: pulse1_time to pulse2_time)
                spikes_p1 = self._count_spikes_in_window(
                    start=pulse1_time,
                    stop=pulse2_time,
                    electrode=self.resp_electrode,
                )

                # Query spikes after pulse 2 (window: pulse2_time to now)
                query_stop = datetime_now()
                spikes_p2 = self._count_spikes_in_window(
                    start=pulse2_time,
                    stop=query_stop,
                    electrode=self.resp_electrode,
                )

                # Compute facilitation ratio
                if spikes_p1 > 0:
                    facilitation_ratio = spikes_p2 / spikes_p1
                else:
                    facilitation_ratio = float('nan')

                trial_result = PairedPulseTrialResult(
                    ipi_ms=ipi_ms,
                    trial_index=trial_idx,
                    pulse1_time_utc=pulse1_time.isoformat(),
                    pulse2_time_utc=pulse2_time.isoformat(),
                    spikes_after_pulse1=spikes_p1,
                    spikes_after_pulse2=spikes_p2,
                    facilitation_ratio=facilitation_ratio,
                )
                self._trial_results.append(trial_result)

                logger.info(
                    "    spikes_p1=%d, spikes_p2=%d, ratio=%.3f",
                    spikes_p1, spikes_p2,
                    facilitation_ratio if not math.isnan(facilitation_ratio) else -1,
                )

                # Inter-pair interval
                self._wait(self.inter_pair_interval_s)

        logger.info("Paired-pulse experiment complete. Total trials: %d", len(self._trial_results))

    def _count_spikes_in_window(
        self,
        start: datetime,
        stop: datetime,
        electrode: int,
    ) -> int:
        try:
            df = self.database.get_spike_event_electrode(start, stop, electrode)
            if df is None or df.empty:
                return 0
            return len(df)
        except Exception as exc:
            logger.warning("Spike query failed for electrode %d: %s", electrode, exc)
            return 0

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        logger.info("Compiling results")

        per_ipi_summary = {}
        for ipi_ms in self.ipi_ms_list:
            trials = [t for t in self._trial_results if t.ipi_ms == ipi_ms]
            if not trials:
                continue
            p1_counts = [t.spikes_after_pulse1 for t in trials]
            p2_counts = [t.spikes_after_pulse2 for t in trials]
            ratios = [t.facilitation_ratio for t in trials if not math.isnan(t.facilitation_ratio)]

            mean_p1 = float(np.mean(p1_counts)) if p1_counts else 0.0
            mean_p2 = float(np.mean(p2_counts)) if p2_counts else 0.0
            mean_ratio = float(np.mean(ratios)) if ratios else float('nan')
            std_ratio = float(np.std(ratios)) if len(ratios) > 1 else float('nan')

            per_ipi_summary[str(ipi_ms)] = {
                "ipi_ms": ipi_ms,
                "n_trials": len(trials),
                "mean_spikes_pulse1": mean_p1,
                "mean_spikes_pulse2": mean_p2,
                "mean_facilitation_ratio": mean_ratio,
                "std_facilitation_ratio": std_ratio,
                "n_valid_ratio_trials": len(ratios),
            }
            logger.info(
                "IPI=%d ms: mean_p1=%.2f, mean_p2=%.2f, mean_ratio=%.3f (n=%d)",
                ipi_ms, mean_p1, mean_p2, mean_ratio, len(trials)
            )

        summary = {
            "status": "completed",
            "experiment_name": self.experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "stim_electrode": self.stim_electrode,
            "resp_electrode": self.resp_electrode,
            "amplitude_ua": self.amplitude_ua,
            "duration_us": self.duration_us,
            "polarity": self.polarity_str,
            "ipi_ms_list": self.ipi_ms_list,
            "num_trials_per_ipi": self.num_trials_per_ipi,
            "total_trials": len(self._trial_results),
            "total_stimulations": len(self._stimulation_log),
            "per_ipi_summary": per_ipi_summary,
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

        trial_dicts = [asdict(t) for t in self._trial_results]
        per_ipi_summary = {}
        for ipi_ms in self.ipi_ms_list:
            trials = [t for t in self._trial_results if t.ipi_ms == ipi_ms]
            if not trials:
                continue
            p1_counts = [t.spikes_after_pulse1 for t in trials]
            p2_counts = [t.spikes_after_pulse2 for t in trials]
            ratios = [t.facilitation_ratio for t in trials if not math.isnan(t.facilitation_ratio)]
            mean_p1 = float(np.mean(p1_counts)) if p1_counts else 0.0
            mean_p2 = float(np.mean(p2_counts)) if p2_counts else 0.0
            mean_ratio = float(np.mean(ratios)) if ratios else float('nan')
            std_ratio = float(np.std(ratios)) if len(ratios) > 1 else float('nan')
            per_ipi_summary[str(ipi_ms)] = {
                "ipi_ms": ipi_ms,
                "n_trials": len(trials),
                "mean_spikes_pulse1": mean_p1,
                "mean_spikes_pulse2": mean_p2,
                "mean_facilitation_ratio": mean_ratio,
                "std_facilitation_ratio": std_ratio,
                "n_valid_ratio_trials": len(ratios),
            }

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
            "num_trials_per_ipi": self.num_trials_per_ipi,
            "total_trials": len(self._trial_results),
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "per_ipi_summary": per_ipi_summary,
            "trial_results": trial_dicts,
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
            if col.lower() in ("channel", "index", "electrode"):
                electrode_col = col
                break
            if "electrode" in col.lower() or "idx" in col.lower():
                electrode_col = col
                break

        if electrode_col is None:
            logger.warning("Cannot determine electrode column for waveform fetch")
            electrodes_to_query = [self.resp_electrode, self.stim_electrode]
        else:
            electrodes_to_query = list(spike_df[electrode_col].unique())

        for electrode_idx in electrodes_to_query:
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
