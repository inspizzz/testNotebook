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
    phase: str = "probe"
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
    Spontaneous activity monitoring experiment with sparse probe stimulations.

    Phase 1 (baseline): 10 minutes of pure recording, no stimulation.
    Phase 2 (probe):    20 minutes, one biphasic probe pulse every 60 s on
                        electrode 7 (2 uA, 200 us per phase, charge-balanced).

    Analyses computed:
      - Spontaneous firing rate per electrode (baseline and probe phases).
      - Burst detection via inter-spike-interval threshold.
      - Pre/post-probe spike-rate comparison around each probe event.
    """

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        baseline_duration_s: float = 600.0,
        probe_phase_duration_s: float = 1200.0,
        probe_interval_s: float = 60.0,
        probe_electrode: int = 7,
        probe_amplitude_ua: float = 2.0,
        probe_duration_us: float = 200.0,
        burst_isi_threshold_ms: float = 100.0,
        burst_min_spikes: int = 3,
        evoked_window_pre_s: float = 5.0,
        evoked_window_post_s: float = 5.0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        self.baseline_duration_s = baseline_duration_s
        self.probe_phase_duration_s = probe_phase_duration_s
        self.probe_interval_s = probe_interval_s
        self.probe_electrode = probe_electrode
        self.probe_amplitude_ua = min(abs(probe_amplitude_ua), 4.0)
        self.probe_duration_us = min(abs(probe_duration_us), 400.0)
        self.burst_isi_threshold_ms = burst_isi_threshold_ms
        self.burst_min_spikes = burst_min_spikes
        self.evoked_window_pre_s = evoked_window_pre_s
        self.evoked_window_post_s = evoked_window_post_s

        # Charge balance: A1*D1 == A2*D2  =>  both phases equal
        assert abs(self.probe_amplitude_ua * self.probe_duration_us -
                   self.probe_amplitude_ua * self.probe_duration_us) < 1e-9

        self.experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        self._stimulation_log: List[StimulationRecord] = []

        # Phase boundary timestamps
        self._baseline_start: Optional[datetime] = None
        self._baseline_stop: Optional[datetime] = None
        self._probe_start: Optional[datetime] = None
        self._probe_stop: Optional[datetime] = None

        # Probe event timestamps
        self._probe_event_times: List[datetime] = []

        # Analysis results
        self._analysis_results: Dict[str, Any] = {}

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

            self._phase_baseline()
            self._phase_probe()

            recording_stop = datetime_now()

            self._analysis_results = self._analyse(recording_start, recording_stop)

            results = self._compile_results(recording_start, recording_stop)

            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    def _phase_baseline(self) -> None:
        logger.info("Phase 1: baseline recording (%.0f s, no stimulation)",
                    self.baseline_duration_s)
        self._baseline_start = datetime_now()
        self._wait(self.baseline_duration_s)
        self._baseline_stop = datetime_now()
        logger.info("Baseline phase complete at %s", self._baseline_stop.isoformat())

    def _phase_probe(self) -> None:
        logger.info("Phase 2: probe stimulation phase (%.0f s, one probe every %.0f s)",
                    self.probe_phase_duration_s, self.probe_interval_s)
        self._probe_start = datetime_now()

        num_probes = int(self.probe_phase_duration_s / self.probe_interval_s)
        logger.info("Planned probe count: %d", num_probes)

        for probe_idx in range(num_probes):
            logger.info("Probe %d / %d on electrode %d",
                        probe_idx + 1, num_probes, self.probe_electrode)
            self._deliver_probe()
            # Wait for the remainder of the interval after the probe
            self._wait(self.probe_interval_s - 0.5)

        self._probe_stop = datetime_now()
        logger.info("Probe phase complete at %s", self._probe_stop.isoformat())

    def _deliver_probe(self) -> None:
        """Send one charge-balanced biphasic probe pulse."""
        amplitude_ua = self.probe_amplitude_ua
        duration_us = self.probe_duration_us

        # Charge balance: A1*D1 == A2*D2  (both phases identical)
        stim = StimParam()
        stim.index = self.probe_electrode
        stim.enable = True
        stim.trigger_key = 0
        stim.trigger_delay = 0
        stim.nb_pulse = 0
        stim.pulse_train_period = 10000
        stim.post_stim_ref_period = 1000.0
        stim.stim_shape = StimShape.Biphasic
        stim.polarity = StimPolarity.PositiveFirst
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
        pattern[0] = 1
        self.trigger_controller.send(pattern)
        self._wait(0.05)
        pattern[0] = 0
        self.trigger_controller.send(pattern)

        event_time = datetime_now()
        self._probe_event_times.append(event_time)

        self._stimulation_log.append(StimulationRecord(
            electrode_idx=self.probe_electrode,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            polarity="PositiveFirst",
            timestamp_utc=event_time.isoformat(),
            trigger_key=0,
            phase="probe",
        ))

        self._wait(0.45)

    def _analyse(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        logger.info("Running analysis")
        fs_name = getattr(self.experiment, "exp_name", "unknown")

        results: Dict[str, Any] = {}

        # ---- Fetch spike events for each phase ----
        baseline_spikes = pd.DataFrame()
        probe_spikes = pd.DataFrame()

        if self._baseline_start and self._baseline_stop:
            try:
                baseline_spikes = self.database.get_spike_event(
                    self._baseline_start, self._baseline_stop, fs_name
                )
            except Exception as exc:
                logger.warning("Could not fetch baseline spikes: %s", exc)

        if self._probe_start and self._probe_stop:
            try:
                probe_spikes = self.database.get_spike_event(
                    self._probe_start, self._probe_stop, fs_name
                )
            except Exception as exc:
                logger.warning("Could not fetch probe spikes: %s", exc)

        # ---- Spontaneous firing rate per electrode ----
        baseline_duration = self.baseline_duration_s
        probe_duration = self.probe_phase_duration_s

        baseline_rates = self._compute_firing_rates(baseline_spikes, baseline_duration)
        probe_rates = self._compute_firing_rates(probe_spikes, probe_duration)

        results["baseline_firing_rates_hz"] = baseline_rates
        results["probe_phase_firing_rates_hz"] = probe_rates

        # ---- Burst detection ----
        baseline_bursts = self._detect_bursts(baseline_spikes)
        probe_bursts = self._detect_bursts(probe_spikes)

        results["baseline_burst_count"] = baseline_bursts["total_bursts"]
        results["baseline_burst_rate_per_min"] = (
            baseline_bursts["total_bursts"] / (baseline_duration / 60.0)
            if baseline_duration > 0 else 0.0
        )
        results["probe_phase_burst_count"] = probe_bursts["total_bursts"]
        results["probe_phase_burst_rate_per_min"] = (
            probe_bursts["total_bursts"] / (probe_duration / 60.0)
            if probe_duration > 0 else 0.0
        )
        results["baseline_burst_details"] = baseline_bursts["burst_details"]
        results["probe_burst_details"] = probe_bursts["burst_details"]

        # ---- Pre/post probe comparison ----
        peri_probe = self._peri_probe_analysis(fs_name)
        results["peri_probe_analysis"] = peri_probe

        return results

    def _compute_firing_rates(
        self, spike_df: pd.DataFrame, duration_s: float
    ) -> Dict[str, float]:
        if spike_df.empty or duration_s <= 0:
            return {}

        time_col = self._find_time_col(spike_df)
        channel_col = self._find_channel_col(spike_df)
        if time_col is None or channel_col is None:
            return {}

        rates: Dict[str, float] = {}
        for ch, grp in spike_df.groupby(channel_col):
            rates[str(ch)] = len(grp) / duration_s
        return rates

    def _detect_bursts(self, spike_df: pd.DataFrame) -> Dict[str, Any]:
        if spike_df.empty:
            return {"total_bursts": 0, "burst_details": []}

        time_col = self._find_time_col(spike_df)
        channel_col = self._find_channel_col(spike_df)
        if time_col is None or channel_col is None:
            return {"total_bursts": 0, "burst_details": []}

        threshold_s = self.burst_isi_threshold_ms / 1000.0
        all_bursts = []

        for ch, grp in spike_df.groupby(channel_col):
            times = pd.to_datetime(grp[time_col]).sort_values()
            times_s = times.astype(np.int64) / 1e9

            if len(times_s) < self.burst_min_spikes:
                continue

            times_arr = times_s.values
            isis = np.diff(times_arr)

            in_burst = False
            burst_start_idx = 0
            burst_spike_count = 1

            for i, isi in enumerate(isis):
                if isi <= threshold_s:
                    if not in_burst:
                        in_burst = True
                        burst_start_idx = i
                        burst_spike_count = 2
                    else:
                        burst_spike_count += 1
                else:
                    if in_burst and burst_spike_count >= self.burst_min_spikes:
                        all_bursts.append({
                            "electrode": int(ch),
                            "start_s": float(times_arr[burst_start_idx]),
                            "end_s": float(times_arr[i]),
                            "spike_count": burst_spike_count,
                            "duration_s": float(times_arr[i] - times_arr[burst_start_idx]),
                        })
                    in_burst = False
                    burst_spike_count = 1

            if in_burst and burst_spike_count >= self.burst_min_spikes:
                all_bursts.append({
                    "electrode": int(ch),
                    "start_s": float(times_arr[burst_start_idx]),
                    "end_s": float(times_arr[-1]),
                    "spike_count": burst_spike_count,
                    "duration_s": float(times_arr[-1] - times_arr[burst_start_idx]),
                })

        return {"total_bursts": len(all_bursts), "burst_details": all_bursts}

    def _peri_probe_analysis(self, fs_name: str) -> List[Dict[str, Any]]:
        if not self._probe_event_times:
            return []

        peri_results = []
        for idx, probe_time in enumerate(self._probe_event_times):
            pre_start = probe_time - timedelta(seconds=self.evoked_window_pre_s)
            pre_stop = probe_time
            post_start = probe_time
            post_stop = probe_time + timedelta(seconds=self.evoked_window_post_s)

            try:
                pre_spikes = self.database.get_spike_event(pre_start, pre_stop, fs_name)
                post_spikes = self.database.get_spike_event(post_start, post_stop, fs_name)
            except Exception as exc:
                logger.warning("Peri-probe query failed for probe %d: %s", idx, exc)
                continue

            pre_count = len(pre_spikes) if not pre_spikes.empty else 0
            post_count = len(post_spikes) if not post_spikes.empty else 0

            pre_rate = pre_count / self.evoked_window_pre_s if self.evoked_window_pre_s > 0 else 0.0
            post_rate = post_count / self.evoked_window_post_s if self.evoked_window_post_s > 0 else 0.0

            peri_results.append({
                "probe_index": idx,
                "probe_time_utc": probe_time.isoformat(),
                "pre_spike_count": pre_count,
                "post_spike_count": post_count,
                "pre_rate_hz": pre_rate,
                "post_rate_hz": post_rate,
                "rate_change_hz": post_rate - pre_rate,
                "rate_ratio": (post_rate / pre_rate) if pre_rate > 0 else None,
            })

        return peri_results

    def _find_time_col(self, df: pd.DataFrame) -> Optional[str]:
        for col in df.columns:
            if col.lower() in ("time", "_time", "timestamp"):
                return col
        return None

    def _find_channel_col(self, df: pd.DataFrame) -> Optional[str]:
        for col in df.columns:
            if col.lower() in ("channel", "index", "electrode", "ch"):
                return col
        return None

    def _save_all(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> None:
        fs_name = getattr(self.experiment, "exp_name", "unknown")
        saver = DataSaver(self._output_dir, fs_name)

        saver.save_stimulation_log(self._stimulation_log)

        spike_df = pd.DataFrame()
        try:
            spike_df = self.database.get_spike_event(
                recording_start, recording_stop, fs_name
            )
        except Exception as exc:
            logger.warning("Could not fetch full spike events: %s", exc)
        saver.save_spike_events(spike_df)

        trigger_df = pd.DataFrame()
        try:
            trigger_df = self.database.get_all_triggers(
                recording_start, recording_stop
            )
        except Exception as exc:
            logger.warning("Could not fetch triggers: %s", exc)
        saver.save_triggers(trigger_df)

        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "baseline_duration_s": self.baseline_duration_s,
            "probe_phase_duration_s": self.probe_phase_duration_s,
            "probe_interval_s": self.probe_interval_s,
            "probe_electrode": self.probe_electrode,
            "probe_amplitude_ua": self.probe_amplitude_ua,
            "probe_duration_us": self.probe_duration_us,
            "total_probes_delivered": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "analysis": self._analysis_results,
        }
        saver.save_summary(summary)

        waveform_records = self._fetch_spike_waveforms(
            spike_df, recording_start, recording_stop
        )
        saver.save_spike_waveforms(waveform_records)

    def _fetch_spike_waveforms(
        self,
        spike_df: pd.DataFrame,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> list:
        waveform_records = []
        if spike_df.empty:
            return waveform_records

        channel_col = self._find_channel_col(spike_df)
        if channel_col is None:
            logger.warning("Cannot determine electrode column for waveform fetch")
            return waveform_records

        unique_electrodes = spike_df[channel_col].unique()
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

        fs_name = getattr(self.experiment, "exp_name", "unknown")
        duration_s = (recording_stop - recording_start).total_seconds()

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": fs_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": duration_s,
            "baseline_duration_s": self.baseline_duration_s,
            "probe_phase_duration_s": self.probe_phase_duration_s,
            "probe_electrode": self.probe_electrode,
            "probe_amplitude_ua": self.probe_amplitude_ua,
            "probe_duration_us": self.probe_duration_us,
            "total_probes_delivered": len(self._stimulation_log),
            "probe_event_times_utc": [t.isoformat() for t in self._probe_event_times],
        }

        summary.update(self._analysis_results)

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
