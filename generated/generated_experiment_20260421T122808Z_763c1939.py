"""
FinalSpark STDP Plasticity Experiment
======================================
Three-phase pipeline:
  1. Basic Excitability Scan
  2. Active Electrode Experiment (1 Hz, cross-correlograms)
  3. Two-Electrode Hebbian Learning (STDP: Testing / Learning / Validation)
"""

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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StimulationRecord:
    electrode_idx: int
    amplitude_ua: float
    duration_us: float
    polarity: str
    timestamp_utc: str
    trigger_key: int = 0
    phase: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScanResult:
    electrode_from: int
    electrode_to: int
    amplitude: float
    duration: float
    polarity: str
    hits: int
    repeats: int
    median_latency_ms: float


@dataclass
class CrossCorrelogramResult:
    electrode_from: int
    electrode_to: int
    bins_ms: List[float]
    counts: List[int]
    peak_lag_ms: float
    hebbian_delay_ms: float


@dataclass
class STDPTrialResult:
    phase: str
    trial_index: int
    stim_electrode: int
    resp_electrode: int
    timestamp_utc: str
    spike_count: int
    latency_ms: Optional[float]


# ---------------------------------------------------------------------------
# Data persistence
# ---------------------------------------------------------------------------

class DataSaver:
    """Handles persistence of stimulation records, spike events, and triggers."""

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

    def save_extra(self, name: str, data: Any) -> Path:
        path = Path(f"{self._prefix}_{name}.json")
        path.write_text(json.dumps(data, indent=2, default=str))
        logger.info("Saved extra data -> %s", path)
        return path


# ---------------------------------------------------------------------------
# Main experiment class
# ---------------------------------------------------------------------------

class Experiment:
    """
    Full STDP plasticity pipeline for FinalSpark NeuroPlatform.

    Stages:
      1. Basic Excitability Scan
      2. Active Electrode Experiment (1 Hz, cross-correlograms)
      3. Two-Electrode Hebbian Learning (STDP)
    """

    # Known responsive pairs from prior parameter scan
    KNOWN_PAIRS = [
        {"electrode_from": 17, "electrode_to": 18, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 13.477},
        {"electrode_from": 21, "electrode_to": 19, "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 18.979},
        {"electrode_from": 21, "electrode_to": 22, "amplitude": 3.0, "duration": 400.0, "polarity": "NegativeFirst", "median_latency_ms": 10.859},
        {"electrode_from": 7,  "electrode_to": 6,  "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 24.622},
        {"electrode_from": 6,  "electrode_to": 7,  "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 19.294},
        {"electrode_from": 5,  "electrode_to": 4,  "amplitude": 3.0, "duration": 400.0, "polarity": "PositiveFirst", "median_latency_ms": 14.634},
        {"electrode_from": 13, "electrode_to": 14, "amplitude": 3.0, "duration": 300.0, "polarity": "NegativeFirst", "median_latency_ms": 12.055},
    ]

    def __init__(
        self,
        token: str = "CVQE258EEE",
        booking_email: str = "ww414@exeter.ac.uk",
        testing: bool = False,
        output_dir: str = "experiment_output",
        # Scan parameters
        scan_amplitudes: tuple = (1.0, 2.0, 3.0),
        scan_durations: tuple = (100.0, 200.0, 300.0, 400.0),
        scan_repeats: int = 5,
        scan_hits_required: int = 3,
        scan_isi_s: float = 1.0,
        scan_inter_channel_s: float = 5.0,
        # Active electrode experiment parameters
        active_stim_hz: float = 1.0,
        active_group_size: int = 10,
        active_group_pause_s: float = 5.0,
        active_total_repeats: int = 100,
        # STDP parameters
        stdp_testing_min: float = 20.0,
        stdp_learning_min: float = 50.0,
        stdp_validation_min: float = 20.0,
        stdp_amplitude_ua: float = 3.0,
        stdp_duration_us: float = 400.0,
        stdp_iti_s: float = 2.0,
        # Hebbian delay override (ms); 0 = use computed value
        hebbian_delay_ms: float = 0.0,
        # Primary STDP pair indices into KNOWN_PAIRS (0-based)
        primary_pair_index: int = 0,
    ):
        self.token = token
        self.booking_email = booking_email
        self.testing = testing
        self._output_dir = Path(output_dir)

        # Scan config
        self.scan_amplitudes = list(scan_amplitudes)
        self.scan_durations = list(scan_durations)
        self.scan_repeats = scan_repeats
        self.scan_hits_required = scan_hits_required
        self.scan_isi_s = scan_isi_s
        self.scan_inter_channel_s = scan_inter_channel_s

        # Active electrode config
        self.active_stim_hz = active_stim_hz
        self.active_group_size = active_group_size
        self.active_group_pause_s = active_group_pause_s
        self.active_total_repeats = active_total_repeats

        # STDP config
        self.stdp_testing_min = stdp_testing_min
        self.stdp_learning_min = stdp_learning_min
        self.stdp_validation_min = stdp_validation_min
        self.stdp_amplitude_ua = min(stdp_amplitude_ua, 4.0)
        self.stdp_duration_us = min(stdp_duration_us, 400.0)
        self.stdp_iti_s = stdp_iti_s
        self.hebbian_delay_ms = hebbian_delay_ms
        self.primary_pair_index = primary_pair_index

        # Hardware handles
        self.np_experiment = None
        self.trigger_controller = None
        self.intan = None
        self.database = None

        # Results storage
        self._stimulation_log: List[StimulationRecord] = []
        self._scan_results: List[ScanResult] = []
        self._responsive_pairs: List[Dict[str, Any]] = []
        self._correlogram_results: List[CrossCorrelogramResult] = []
        self._stdp_results: List[STDPTrialResult] = []
        self._active_stim_times: Dict[str, List[str]] = defaultdict(list)

        # Spike event cache per phase for cross-correlogram computation
        self._active_phase_spikes: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Execute the full experiment and return a results dict."""
        recording_start = None
        recording_stop = None
        try:
            logger.info("Initialising hardware connections")
            from neuroplatform import Experiment as NeuroPlatformExperiment

            self.np_experiment = NeuroPlatformExperiment(self.token)
            self.trigger_controller = TriggerController(email=self.booking_email)
            self.intan = IntanSofware()
            self.database = Database()

            logger.info("Experiment: %s", self.np_experiment.exp_name)
            logger.info("Electrodes: %s", self.np_experiment.electrodes)

            if not self.np_experiment.start():
                logger.error("Failed to start experiment")
                return {"status": "failed", "error": "Could not start experiment"}

            recording_start = datetime_now()

            # ---- Phase 1: Basic Excitability Scan ----
            logger.info("=== Phase 1: Basic Excitability Scan ===")
            self._phase_excitability_scan()

            # ---- Phase 2: Active Electrode Experiment ----
            logger.info("=== Phase 2: Active Electrode Experiment ===")
            self._phase_active_electrode()

            # ---- Phase 3: STDP Hebbian Learning ----
            logger.info("=== Phase 3: STDP Hebbian Learning ===")
            self._phase_stdp()

            recording_stop = datetime_now()

            results = self._compile_results(recording_start, recording_stop)

            # Persist all raw data
            self._save_all(recording_start, recording_stop)

            return results

        except Exception as exc:
            logger.error("Experiment error: %s", exc, exc_info=True)
            if recording_start is None:
                recording_start = datetime_now()
            if recording_stop is None:
                recording_stop = datetime_now()
            try:
                self._save_all(recording_start, recording_stop)
            except Exception as save_exc:
                logger.error("Save failed: %s", save_exc)
            return {"status": "failed", "error": str(exc)}
        finally:
            self._cleanup()

    # ------------------------------------------------------------------
    # Phase 1: Basic Excitability Scan
    # ------------------------------------------------------------------

    def _phase_excitability_scan(self) -> None:
        """Sweep electrodes, amplitudes, durations, polarities; identify responsive pairs."""
        electrodes = self.np_experiment.electrodes
        polarities = [StimPolarity.PositiveFirst, StimPolarity.NegativeFirst]
        polarity_names = {StimPolarity.PositiveFirst: "PositiveFirst", StimPolarity.NegativeFirst: "NegativeFirst"}

        # Use known pairs from prior scan as the primary source of truth,
        # but also run a lightweight scan on available electrodes.
        # To keep runtime manageable, scan only the electrodes involved in known pairs.
        scan_electrodes = list({p["electrode_from"] for p in self.KNOWN_PAIRS} |
                               {p["electrode_to"] for p in self.KNOWN_PAIRS})
        scan_electrodes = [e for e in scan_electrodes if e in electrodes]

        logger.info("Scanning %d electrodes", len(scan_electrodes))

        window_s = 0.05  # 50 ms response window

        for stim_elec in scan_electrodes:
            for amplitude in self.scan_amplitudes:
                for duration in self.scan_durations:
                    # Charge balance: A1*D1 = A2*D2, equal phases
                    for polarity in polarities:
                        hits_per_resp: Dict[int, int] = defaultdict(int)
                        latencies_per_resp: Dict[int, List[float]] = defaultdict(list)

                        for rep in range(self.scan_repeats):
                            t_stim = datetime_now()
                            self._send_stim(
                                electrode_idx=stim_elec,
                                amplitude_ua=amplitude,
                                duration_us=duration,
                                polarity=polarity,
                                trigger_key=0,
                                phase="scan",
                            )
                            self._wait(window_s + 0.01)
                            t_after = datetime_now()

                            # Query spikes in window
                            q_start = t_stim
                            q_stop = t_after
                            try:
                                spike_df = self.database.get_spike_event(
                                    q_start, q_stop, self.np_experiment.exp_name
                                )
                            except Exception as exc:
                                logger.warning("Spike query failed: %s", exc)
                                spike_df = pd.DataFrame()

                            if not spike_df.empty and "channel" in spike_df.columns:
                                for resp_elec in scan_electrodes:
                                    if resp_elec == stim_elec:
                                        continue
                                    resp_spikes = spike_df[spike_df["channel"] == resp_elec]
                                    if not resp_spikes.empty:
                                        hits_per_resp[resp_elec] += 1
                                        # Estimate latency from stim time
                                        if "Time" in resp_spikes.columns:
                                            try:
                                                t0 = pd.Timestamp(t_stim)
                                                lats = [(pd.Timestamp(ts) - t0).total_seconds() * 1000
                                                        for ts in resp_spikes["Time"]]
                                                lats = [l for l in lats if 0 < l < 50]
                                                if lats:
                                                    latencies_per_resp[resp_elec].extend(lats)
                                            except Exception:
                                                pass

                            if rep < self.scan_repeats - 1:
                                self._wait(self.scan_isi_s)

                        # Evaluate hits
                        for resp_elec, hits in hits_per_resp.items():
                            if hits >= self.scan_hits_required:
                                lats = latencies_per_resp.get(resp_elec, [])
                                med_lat = float(np.median(lats)) if lats else 0.0
                                result = ScanResult(
                                    electrode_from=stim_elec,
                                    electrode_to=resp_elec,
                                    amplitude=amplitude,
                                    duration=duration,
                                    polarity=polarity_names[polarity],
                                    hits=hits,
                                    repeats=self.scan_repeats,
                                    median_latency_ms=med_lat,
                                )
                                self._scan_results.append(result)
                                logger.info(
                                    "Responsive pair: %d->%d  amp=%.1f dur=%.0f pol=%s hits=%d/5 lat=%.1fms",
                                    stim_elec, resp_elec, amplitude, duration,
                                    polarity_names[polarity], hits, med_lat
                                )

                    self._wait(self.scan_inter_channel_s)

        # Build responsive pairs list: merge scan results with known pairs
        self._build_responsive_pairs()
        logger.info("Responsive pairs identified: %d", len(self._responsive_pairs))

    def _build_responsive_pairs(self) -> None:
        """Merge live scan results with known pairs from prior scan."""
        seen = set()
        # First add from live scan
        for r in self._scan_results:
            key = (r.electrode_from, r.electrode_to)
            if key not in seen:
                seen.add(key)
                self._responsive_pairs.append({
                    "electrode_from": r.electrode_from,
                    "electrode_to": r.electrode_to,
                    "amplitude": r.amplitude,
                    "duration": r.duration,
                    "polarity": r.polarity,
                    "median_latency_ms": r.median_latency_ms,
                    "source": "live_scan",
                })
        # Then add known pairs not already found
        for kp in self.KNOWN_PAIRS:
            key = (kp["electrode_from"], kp["electrode_to"])
            if key not in seen:
                seen.add(key)
                self._responsive_pairs.append({**kp, "source": "prior_scan"})

    # ------------------------------------------------------------------
    # Phase 2: Active Electrode Experiment
    # ------------------------------------------------------------------

    def _phase_active_electrode(self) -> None:
        """Stimulate each responsive pair at 1 Hz in groups of 10, compute cross-correlograms."""
        if not self._responsive_pairs:
            logger.warning("No responsive pairs; skipping active electrode phase")
            return

        isi_s = 1.0 / self.active_stim_hz  # 1 s between stimulations
        n_groups = self.active_total_repeats // self.active_group_size

        for pair in self._responsive_pairs:
            stim_elec = pair["electrode_from"]
            resp_elec = pair["electrode_to"]
            amplitude = min(pair["amplitude"], 4.0)
            duration = min(pair["duration"], 400.0)
            polarity = self._parse_polarity(pair["polarity"])
            pair_key = f"{stim_elec}->{resp_elec}"

            logger.info("Active electrode: pair %s  amp=%.1f dur=%.0f", pair_key, amplitude, duration)

            phase_start = datetime_now()
            stim_times: List[str] = []

            for group_idx in range(n_groups):
                for stim_idx in range(self.active_group_size):
                    t_stim = datetime_now()
                    self._send_stim(
                        electrode_idx=stim_elec,
                        amplitude_ua=amplitude,
                        duration_us=duration,
                        polarity=polarity,
                        trigger_key=1,
                        phase="active",
                    )
                    stim_times.append(t_stim.isoformat())
                    self._wait(isi_s)

                if group_idx < n_groups - 1:
                    self._wait(self.active_group_pause_s)

            phase_stop = datetime_now()
            self._active_stim_times[pair_key] = stim_times

            # Fetch spikes for this pair's active phase
            try:
                spike_df = self.database.get_spike_event(
                    phase_start, phase_stop, self.np_experiment.exp_name
                )
                self._active_phase_spikes[pair_key] = spike_df
            except Exception as exc:
                logger.warning("Failed to fetch spikes for pair %s: %s", pair_key, exc)
                self._active_phase_spikes[pair_key] = pd.DataFrame()

            # Compute cross-correlogram
            ccg = self._compute_cross_correlogram(
                stim_elec=stim_elec,
                resp_elec=resp_elec,
                stim_times=stim_times,
                spike_df=self._active_phase_spikes[pair_key],
                window_ms=50.0,
                bin_ms=1.0,
            )
            self._correlogram_results.append(ccg)
            logger.info(
                "CCG pair %s: peak_lag=%.1fms hebbian_delay=%.1fms",
                pair_key, ccg.peak_lag_ms, ccg.hebbian_delay_ms
            )

    def _compute_cross_correlogram(
        self,
        stim_elec: int,
        resp_elec: int,
        stim_times: List[str],
        spike_df: pd.DataFrame,
        window_ms: float = 50.0,
        bin_ms: float = 1.0,
    ) -> CrossCorrelogramResult:
        """Compute trigger-centred cross-correlogram for a pair."""
        n_bins = int(window_ms / bin_ms)
        bins_ms = [i * bin_ms for i in range(n_bins)]
        counts = [0] * n_bins

        if spike_df.empty or "channel" not in spike_df.columns or "Time" not in spike_df.columns:
            peak_lag = float(np.argmax(counts)) * bin_ms if any(c > 0 for c in counts) else 0.0
            return CrossCorrelogramResult(
                electrode_from=stim_elec,
                electrode_to=resp_elec,
                bins_ms=bins_ms,
                counts=counts,
                peak_lag_ms=peak_lag,
                hebbian_delay_ms=peak_lag,
            )

        resp_spikes = spike_df[spike_df["channel"] == resp_elec]
        if resp_spikes.empty:
            return CrossCorrelogramResult(
                electrode_from=stim_elec,
                electrode_to=resp_elec,
                bins_ms=bins_ms,
                counts=counts,
                peak_lag_ms=0.0,
                hebbian_delay_ms=0.0,
            )

        try:
            resp_times_s = np.array([
                pd.Timestamp(ts).timestamp() for ts in resp_spikes["Time"]
            ])
        except Exception:
            resp_times_s = np.array([])

        for t_str in stim_times:
            try:
                t0 = pd.Timestamp(t_str).timestamp()
            except Exception:
                continue
            for rt in resp_times_s:
                lag_ms = (rt - t0) * 1000.0
                if 0 <= lag_ms < window_ms:
                    bin_idx = int(lag_ms / bin_ms)
                    if 0 <= bin_idx < n_bins:
                        counts[bin_idx] += 1

        if any(c > 0 for c in counts):
            peak_bin = int(np.argmax(counts))
            peak_lag_ms = bins_ms[peak_bin]
        else:
            peak_lag_ms = 0.0

        # Hebbian delay: use peak lag as the pre->post delay for STDP
        hebbian_delay_ms = peak_lag_ms

        return CrossCorrelogramResult(
            electrode_from=stim_elec,
            electrode_to=resp_elec,
            bins_ms=bins_ms,
            counts=counts,
            peak_lag_ms=peak_lag_ms,
            hebbian_delay_ms=hebbian_delay_ms,
        )

    # ------------------------------------------------------------------
    # Phase 3: STDP Hebbian Learning
    # ------------------------------------------------------------------

    def _phase_stdp(self) -> None:
        """Three-phase STDP experiment: Testing / Learning / Validation."""
        # Select primary pair
        if not self._responsive_pairs:
            logger.warning("No responsive pairs for STDP; using first known pair")
            pairs_to_use = [self.KNOWN_PAIRS[0]]
        else:
            idx = min(self.primary_pair_index, len(self._responsive_pairs) - 1)
            pairs_to_use = [self._responsive_pairs[idx]]

        pair = pairs_to_use[0]
        stim_elec = pair["electrode_from"]
        resp_elec = pair["electrode_to"]
        amplitude = min(self.stdp_amplitude_ua, 4.0)
        duration = min(self.stdp_duration_us, 400.0)
        polarity = self._parse_polarity(pair.get("polarity", "PositiveFirst"))

        # Determine Hebbian delay
        if self.hebbian_delay_ms > 0:
            hebbian_delay_ms = self.hebbian_delay_ms
        else:
            # Look up from correlogram results
            pair_key = f"{stim_elec}->{resp_elec}"
            ccg_match = [c for c in self._correlogram_results
                         if c.electrode_from == stim_elec and c.electrode_to == resp_elec]
            if ccg_match:
                hebbian_delay_ms = ccg_match[0].hebbian_delay_ms
            else:
                # Fall back to known latency
                hebbian_delay_ms = pair.get("median_latency_ms", 15.0)

        # Clamp to STDP window
        hebbian_delay_ms = max(5.0, min(hebbian_delay_ms, 40.0))
        logger.info(
            "STDP pair: %d->%d  amp=%.1f dur=%.0f hebbian_delay=%.1fms",
            stim_elec, resp_elec, amplitude, duration, hebbian_delay_ms
        )

        # --- Testing Phase ---
        logger.info("STDP Testing Phase (%.0f min)", self.stdp_testing_min)
        self._stdp_phase(
            phase_name="testing",
            stim_elec=stim_elec,
            resp_elec=resp_elec,
            amplitude=amplitude,
            duration=duration,
            polarity=polarity,
            duration_min=self.stdp_testing_min,
            iti_s=self.stdp_iti_s,
            paired=False,
            hebbian_delay_ms=hebbian_delay_ms,
        )

        # --- Learning Phase ---
        logger.info("STDP Learning Phase (%.0f min)", self.stdp_learning_min)
        self._stdp_phase(
            phase_name="learning",
            stim_elec=stim_elec,
            resp_elec=resp_elec,
            amplitude=amplitude,
            duration=duration,
            polarity=polarity,
            duration_min=self.stdp_learning_min,
            iti_s=self.stdp_iti_s,
            paired=True,
            hebbian_delay_ms=hebbian_delay_ms,
        )

        # --- Validation Phase ---
        logger.info("STDP Validation Phase (%.0f min)", self.stdp_validation_min)
        self._stdp_phase(
            phase_name="validation",
            stim_elec=stim_elec,
            resp_elec=resp_elec,
            amplitude=amplitude,
            duration=duration,
            polarity=polarity,
            duration_min=self.stdp_validation_min,
            iti_s=self.stdp_iti_s,
            paired=False,
            hebbian_delay_ms=hebbian_delay_ms,
        )

    def _stdp_phase(
        self,
        phase_name: str,
        stim_elec: int,
        resp_elec: int,
        amplitude: float,
        duration: float,
        polarity: StimPolarity,
        duration_min: float,
        iti_s: float,
        paired: bool,
        hebbian_delay_ms: float,
    ) -> None:
        """Run one STDP phase for the specified duration."""
        phase_duration_s = duration_min * 60.0
        phase_start_wall = datetime_now()
        trial_idx = 0
        window_s = 0.05  # 50 ms response window

        while True:
            elapsed = (datetime_now() - phase_start_wall).total_seconds()
            if elapsed >= phase_duration_s:
                break

            t_stim = datetime_now()

            if paired:
                # Pre-electrode stimulation
                self._send_stim(
                    electrode_idx=stim_elec,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    polarity=polarity,
                    trigger_key=2,
                    phase=phase_name,
                )
                # Wait Hebbian delay then stimulate post-electrode
                self._wait(hebbian_delay_ms / 1000.0)
                self._send_stim(
                    electrode_idx=resp_elec,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    polarity=polarity,
                    trigger_key=3,
                    phase=phase_name + "_post",
                )
            else:
                # Single probe stimulation
                self._send_stim(
                    electrode_idx=stim_elec,
                    amplitude_ua=amplitude,
                    duration_us=duration,
                    polarity=polarity,
                    trigger_key=2,
                    phase=phase_name,
                )

            self._wait(window_s)
            t_after = datetime_now()

            # Query response
            spike_count = 0
            latency_ms = None
            try:
                spike_df = self.database.get_spike_event(
                    t_stim, t_after, self.np_experiment.exp_name
                )
                if not spike_df.empty and "channel" in spike_df.columns:
                    resp_spikes = spike_df[spike_df["channel"] == resp_elec]
                    spike_count = len(resp_spikes)
                    if spike_count > 0 and "Time" in resp_spikes.columns:
                        try:
                            t0 = pd.Timestamp(t_stim).timestamp()
                            lats = [
                                (pd.Timestamp(ts).timestamp() - t0) * 1000.0
                                for ts in resp_spikes["Time"]
                            ]
                            lats = [l for l in lats if 0 < l < 100]
                            if lats:
                                latency_ms = float(np.min(lats))
                        except Exception:
                            pass
            except Exception as exc:
                logger.warning("Spike query failed in STDP phase %s: %s", phase_name, exc)

            self._stdp_results.append(STDPTrialResult(
                phase=phase_name,
                trial_index=trial_idx,
                stim_electrode=stim_elec,
                resp_electrode=resp_elec,
                timestamp_utc=t_stim.isoformat(),
                spike_count=spike_count,
                latency_ms=latency_ms,
            ))

            trial_idx += 1

            # Wait remainder of ITI
            elapsed_trial = (datetime_now() - t_stim).total_seconds()
            remaining_iti = iti_s - elapsed_trial
            if remaining_iti > 0:
                self._wait(remaining_iti)

        logger.info("STDP phase '%s' complete: %d trials", phase_name, trial_idx)

    # ------------------------------------------------------------------
    # Stimulation helper
    # ------------------------------------------------------------------

    def _send_stim(
        self,
        electrode_idx: int,
        amplitude_ua: float,
        duration_us: float,
        polarity: StimPolarity,
        trigger_key: int = 0,
        phase: str = "",
    ) -> None:
        """Configure and fire a single charge-balanced biphasic pulse."""
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

        # Charge balance: A1*D1 = A2*D2 (equal amplitudes and durations)
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

        polarity_name = "PositiveFirst" if polarity == StimPolarity.PositiveFirst else "NegativeFirst"
        self._stimulation_log.append(StimulationRecord(
            electrode_idx=electrode_idx,
            amplitude_ua=amplitude_ua,
            duration_us=duration_us,
            polarity=polarity_name,
            timestamp_utc=datetime_now().isoformat(),
            trigger_key=trigger_key,
            phase=phase,
        ))

    # ------------------------------------------------------------------
    # Delay helper
    # ------------------------------------------------------------------

    def _wait(self, seconds: float) -> None:
        wait(seconds)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _parse_polarity(self, polarity_str: str) -> StimPolarity:
        if polarity_str == "PositiveFirst":
            return StimPolarity.PositiveFirst
        return StimPolarity.NegativeFirst

    # ------------------------------------------------------------------
    # Data persistence
    # ------------------------------------------------------------------

    def _save_all(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> None:
        """Persist all raw experiment data for downstream analysis."""
        fs_name = getattr(self.np_experiment, "exp_name", "unknown")
        saver = DataSaver(self._output_dir, fs_name)

        # Stimulation log
        saver.save_stimulation_log(self._stimulation_log)

        # All spike events
        try:
            spike_df = self.database.get_spike_event(
                recording_start, recording_stop, fs_name
            )
        except Exception as exc:
            logger.warning("Failed to fetch spike events: %s", exc)
            spike_df = pd.DataFrame()
        saver.save_spike_events(spike_df)

        # All triggers
        try:
            trigger_df = self.database.get_all_triggers(
                recording_start, recording_stop
            )
        except Exception as exc:
            logger.warning("Failed to fetch triggers: %s", exc)
            trigger_df = pd.DataFrame()
        saver.save_triggers(trigger_df)

        # Summary
        summary = {
            "fs_name": fs_name,
            "experiment_start_utc": recording_start.isoformat(),
            "experiment_stop_utc": recording_stop.isoformat(),
            "testing": self.testing,
            "total_stimulations": len(self._stimulation_log),
            "total_spike_events": len(spike_df),
            "total_triggers": len(trigger_df),
            "scan_results_count": len(self._scan_results),
            "responsive_pairs_count": len(self._responsive_pairs),
            "correlogram_results_count": len(self._correlogram_results),
            "stdp_trials_count": len(self._stdp_results),
        }
        saver.save_summary(summary)

        # Spike waveforms
        waveform_records = self._fetch_spike_waveforms(
            spike_df, recording_start, recording_stop
        )
        saver.save_spike_waveforms(waveform_records)

        # Extra: scan results
        saver.save_extra("scan_results", [asdict(r) for r in self._scan_results])

        # Extra: responsive pairs
        saver.save_extra("responsive_pairs", self._responsive_pairs)

        # Extra: correlogram results
        saver.save_extra("correlogram_results", [asdict(c) for c in self._correlogram_results])

        # Extra: STDP results
        saver.save_extra("stdp_results", [asdict(r) for r in self._stdp_results])

        # Extra: active stim times
        saver.save_extra("active_stim_times", dict(self._active_stim_times))

    def _fetch_spike_waveforms(
        self,
        spike_df: pd.DataFrame,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> list:
        """Fetch raw spike waveform data for each electrode that had spikes."""
        waveform_records = []
        if spike_df.empty:
            return waveform_records

        electrode_col = None
        for col in ["channel", "index", "electrode"]:
            if col in spike_df.columns:
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

    # ------------------------------------------------------------------
    # Results compilation
    # ------------------------------------------------------------------

    def _compile_results(
        self,
        recording_start: datetime,
        recording_stop: datetime,
    ) -> Dict[str, Any]:
        """Assemble a summary dict to be returned from run()."""
        logger.info("Compiling results")

        # STDP analysis: compare testing vs validation response rates
        testing_trials = [r for r in self._stdp_results if r.phase == "testing"]
        validation_trials = [r for r in self._stdp_results if r.phase == "validation"]

        def response_rate(trials: List[STDPTrialResult]) -> float:
            if not trials:
                return 0.0
            return sum(1 for t in trials if t.spike_count > 0) / len(trials)

        def mean_latency(trials: List[STDPTrialResult]) -> Optional[float]:
            lats = [t.latency_ms for t in trials if t.latency_ms is not None]
            return float(np.mean(lats)) if lats else None

        testing_rr = response_rate(testing_trials)
        validation_rr = response_rate(validation_trials)
        delta_r = validation_rr - testing_rr

        # Correlogram peak lags
        ccg_summary = [
            {
                "pair": f"{c.electrode_from}->{c.electrode_to}",
                "peak_lag_ms": c.peak_lag_ms,
                "hebbian_delay_ms": c.hebbian_delay_ms,
            }
            for c in self._correlogram_results
        ]

        summary: Dict[str, Any] = {
            "status": "completed",
            "experiment_name": self.np_experiment.exp_name,
            "recording_start": recording_start.isoformat(),
            "recording_stop": recording_stop.isoformat(),
            "duration_seconds": (recording_stop - recording_start).total_seconds(),
            "phase1_scan_results": len(self._scan_results),
            "phase1_responsive_pairs": len(self._responsive_pairs),
            "phase2_correlogram_results": ccg_summary,
            "phase3_testing_trials": len(testing_trials),
            "phase3_learning_trials": len([r for r in self._stdp_results if r.phase == "learning"]),
            "phase3_validation_trials": len(validation_trials),
            "phase3_testing_response_rate": testing_rr,
            "phase3_validation_response_rate": validation_rr,
            "phase3_delta_R": delta_r,
            "phase3_testing_mean_latency_ms": mean_latency(testing_trials),
            "phase3_validation_mean_latency_ms": mean_latency(validation_trials),
            "total_stimulations": len(self._stimulation_log),
        }

        return summary

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        """Release all hardware resources. Called from the finally block."""
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
