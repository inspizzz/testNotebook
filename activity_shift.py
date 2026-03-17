"""
Centre-of-Activity Shift Experiment
====================================
FinalSpark NeuroPlatform
 
Objective
---------
Measure whether focal electrical stimulation of one or more electrodes
causes a lasting shift in the spatial centre-of-activity (CoA) of the
organoid, computed as a weighted centroid of per-electrode spike counts
recorded before and after each stimulation epoch.
 
Experimental design
-------------------
1. Baseline recording  – passively record spike activity for
   `baseline_duration_s` seconds before any stimulation.
2. Stimulation epochs  – deliver `n_epochs` rounds of stimulation,
   each targeting `stim_electrode_indices`.  A configurable inter-epoch
   rest period allows the organoid to return to a resting state.
3. Post-epoch windows  – after every stimulation epoch, record spike
   activity for `response_window_s` seconds.
4. Analysis           – compute the weighted CoA for the baseline and
   each post-epoch window, then report the displacement vector.
 
Charge balance
--------------
The charge-balance constraint  A1*D1 = A2*D2  is enforced by the
`StimParamBuilder` helper class before any `StimParam` is accepted.
 
Amplitude limits (per documentation / safe defaults)
------  -----------------------------------------------
  phase_amplitude1 / phase_amplitude2 :  1 – 100 µA
  phase_duration1  / phase_duration2  :  10 – 1000 µs
 
Usage
-----
    from centre_of_activity_experiment import CoAExperiment
 
    # Real run
    exp = CoAExperiment(token="YOUR_TOKEN", booking_email="you@example.com")
    exp.run(testing=False)
 
    # Dry run (no hardware, no stimulations sent)
    exp = CoAExperiment(token="YOUR_TOKEN", booking_email="you@example.com")
    exp.run(testing=True)
"""
 
from __future__ import annotations
 
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
 
import numpy as np
 
# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("CoAExperiment")
 
 
# ---------------------------------------------------------------------------
# Constants / safe limits
# ---------------------------------------------------------------------------
AMPLITUDE_MIN_UA: float = 1.0      # µA – minimum safe phase amplitude
AMPLITUDE_MAX_UA: float = 100.0    # µA – maximum safe phase amplitude
DURATION_MIN_US: float = 10.0      # µs – minimum safe phase duration
DURATION_MAX_US: float = 1000.0    # µs – maximum safe phase duration
 
# Spike artifact exclusion window after each stimulation pulse (ms → s)
ARTIFACT_EXCLUSION_S: float = 0.010   # 10 ms
 
# MEA 4×8 electrode layout – 4 sites × 8 electrodes each, electrodes 0-31
# Positions are (row, col) on the 4×8 grid used for CoA spatial weighting.
# Index maps to (row, col) assuming row-major order: index = row*8 + col
MEA_ROWS: int = 4
MEA_COLS: int = 8
 
 
# ---------------------------------------------------------------------------
# Helper: electrode index → 2-D position on the MEA grid
# ---------------------------------------------------------------------------
def electrode_position(index: int) -> Tuple[float, float]:
    """Return the (row, col) grid position of an absolute electrode index."""
    row = (index % 32) // MEA_COLS   # within-site row
    col = (index % 32) % MEA_COLS    # within-site column
    return float(row), float(col)
 
 
# ---------------------------------------------------------------------------
# StimParamBuilder
# Validates limits and charge balance, then builds a configured StimParam.
# ---------------------------------------------------------------------------
@dataclass
class StimParamConfig:
    """
    All parameters needed to describe one biphasic stimulation waveform.
 
    Charge balance enforced: phase_amplitude1 * phase_duration1
                           = phase_amplitude2 * phase_duration2
    """
    electrode_index: int          # absolute electrode index [0-127]
    trigger_key: int              # trigger slot [0-15]
    phase_amplitude1: float       # A1 [µA]
    phase_duration1: float        # D1 [µs]
    phase_amplitude2: float       # A2 [µA]  – computed if not supplied
    phase_duration2: float        # D2 [µs]
    polarity: str = "NegativeFirst"
    nb_pulse: int = 1
    pulse_train_period: float = 10_000.0   # µs
 
    def __post_init__(self) -> None:
        self._validate()
 
    # ------------------------------------------------------------------
    def _validate(self) -> None:
        for amp, label in [
            (self.phase_amplitude1, "phase_amplitude1"),
            (self.phase_amplitude2, "phase_amplitude2"),
        ]:
            if not (AMPLITUDE_MIN_UA <= amp <= AMPLITUDE_MAX_UA):
                raise ValueError(
                    f"{label}={amp} µA is outside safe range "
                    f"[{AMPLITUDE_MIN_UA}, {AMPLITUDE_MAX_UA}] µA."
                )
 
        for dur, label in [
            (self.phase_duration1, "phase_duration1"),
            (self.phase_duration2, "phase_duration2"),
        ]:
            if not (DURATION_MIN_US <= dur <= DURATION_MAX_US):
                raise ValueError(
                    f"{label}={dur} µs is outside safe range "
                    f"[{DURATION_MIN_US}, {DURATION_MAX_US}] µs."
                )
 
        charge1 = self.phase_amplitude1 * self.phase_duration1
        charge2 = self.phase_amplitude2 * self.phase_duration2
        if not np.isclose(charge1, charge2, rtol=1e-4):
            raise ValueError(
                f"Charge balance violated: "
                f"A1*D1 = {charge1:.4f} ≠ A2*D2 = {charge2:.4f}. "
                "Please set parameters so that A1*D1 = A2*D2."
            )
 
        if not (0 <= self.trigger_key <= 15):
            raise ValueError(
                f"trigger_key={self.trigger_key} must be in [0, 15]."
            )
 
    # ------------------------------------------------------------------
    @classmethod
    def charge_balanced(
        cls,
        electrode_index: int,
        trigger_key: int,
        amplitude1: float,
        duration1: float,
        duration2: float,
        polarity: str = "NegativeFirst",
        nb_pulse: int = 1,
        pulse_train_period: float = 10_000.0,
    ) -> "StimParamConfig":
        """
        Convenience constructor: compute A2 automatically from
        A1*D1 = A2*D2  →  A2 = (A1*D1) / D2
        """
        amplitude2 = (amplitude1 * duration1) / duration2
        return cls(
            electrode_index=electrode_index,
            trigger_key=trigger_key,
            phase_amplitude1=amplitude1,
            phase_duration1=duration1,
            phase_amplitude2=amplitude2,
            phase_duration2=duration2,
            polarity=polarity,
            nb_pulse=nb_pulse,
            pulse_train_period=pulse_train_period,
        )
 
    # ------------------------------------------------------------------
    def summary(self) -> str:
        charge = self.phase_amplitude1 * self.phase_duration1
        return (
            f"Electrode={self.electrode_index} trigger={self.trigger_key} "
            f"A1={self.phase_amplitude1}µA D1={self.phase_duration1}µs "
            f"A2={self.phase_amplitude2:.4f}µA D2={self.phase_duration2}µs "
            f"charge={charge:.2f} µA·µs polarity={self.polarity} "
            f"nb_pulse={self.nb_pulse}"
        )
 
 
# ---------------------------------------------------------------------------
# StimulusLog – records stimulations sent during the experiment
# ---------------------------------------------------------------------------
@dataclass
class StimulusRecord:
    epoch: int
    timestamp_utc: datetime
    stim_configs: List[StimParamConfig]
    trigger_array: np.ndarray
    testing: bool = False
 
 
# ---------------------------------------------------------------------------
# CoAResult – stores a CoA measurement
# ---------------------------------------------------------------------------
@dataclass
class CoAResult:
    label: str
    window_start: datetime
    window_end: datetime
    spike_counts: Dict[int, int]          # {electrode_index: count}
    coa_row: float
    coa_col: float
 
    def displacement_from(self, baseline: "CoAResult") -> Tuple[float, float]:
        """Return (Δrow, Δcol) from a baseline CoA result."""
        return (self.coa_row - baseline.coa_row,
                self.coa_col - baseline.coa_col)
 
 
# ---------------------------------------------------------------------------
# CoAAnalyser – computes centre of activity from spike event DataFrames
# ---------------------------------------------------------------------------
class CoAAnalyser:
    """
    Computes the spatial centre of activity (weighted centroid) from
    spike-event data returned by `Database.get_spike_event`.
 
    The CoA is defined as:
        CoA_row = Σ(count_i * row_i) / Σ(count_i)
        CoA_col = Σ(count_i * col_i) / Σ(count_i)
 
    where row_i and col_i are the grid positions of electrode i.
    Artifacts within `artifact_exclusion_s` seconds after any stimulation
    timestamp are excluded.
    """
 
    def __init__(
        self,
        stim_timestamps: Optional[List[datetime]] = None,
        artifact_exclusion_s: float = ARTIFACT_EXCLUSION_S,
    ) -> None:
        self.stim_timestamps = stim_timestamps or []
        self.artifact_exclusion_s = artifact_exclusion_s
 
    # ------------------------------------------------------------------
    def compute(
        self,
        spike_df,
        label: str,
        window_start: datetime,
        window_end: datetime,
    ) -> CoAResult:
        """
        Parameters
        ----------
        spike_df : pd.DataFrame
            DataFrame from db.get_spike_event with columns:
            ['Time', 'channel', 'amplitude']
        label : str
            Human-readable label for this window (e.g. 'baseline').
        window_start, window_end : datetime
            UTC boundaries of the window.
 
        Returns
        -------
        CoAResult
        """
        df = spike_df.copy()
 
        # --- Filter to amplitude range that indicates real spikes ------
        # Per best-practices: <30 µV = noise, >200 µV = likely artifact
        df = df[(df["amplitude"] >= 30) & (df["amplitude"] <= 200)]
 
        # --- Exclude stimulation artifact window -----------------------
        excl = timedelta(seconds=self.artifact_exclusion_s)
        for ts in self.stim_timestamps:
            mask = (df["Time"] >= ts) & (df["Time"] < ts + excl)
            df = df[~mask]
 
        # --- Count spikes per channel ----------------------------------
        spike_counts: Dict[int, int] = {}
        if len(df) > 0:
            counts = df["channel"].value_counts()
            spike_counts = {int(ch): int(n) for ch, n in counts.items()}
 
        # --- Weighted centroid -----------------------------------------
        total = sum(spike_counts.values())
        if total == 0:
            coa_row, coa_col = 0.0, 0.0
        else:
            coa_row = sum(
                n * electrode_position(ch)[0]
                for ch, n in spike_counts.items()
            ) / total
            coa_col = sum(
                n * electrode_position(ch)[1]
                for ch, n in spike_counts.items()
            ) / total
 
        logger.info(
            "CoA [%s] rows=%.3f cols=%.3f  (total spikes=%d, channels=%d)",
            label, coa_row, coa_col, total, len(spike_counts),
        )
        return CoAResult(
            label=label,
            window_start=window_start,
            window_end=window_end,
            spike_counts=spike_counts,
            coa_row=coa_row,
            coa_col=coa_col,
        )
 
 
# ---------------------------------------------------------------------------
# Master experiment class
# ---------------------------------------------------------------------------
class CoAExperiment:
    """
    Centre-of-Activity Shift Experiment for FinalSpark NeuroPlatform.
 
    Parameters
    ----------
    token : str
        Experiment token provided by FinalSpark.
    booking_email : str
        Email address of the booking, passed to TriggerController.
    stim_configs : list[StimParamConfig], optional
        Stimulation parameter configurations.  A default charge-balanced
        example is used when not supplied.
    baseline_duration_s : float
        Seconds of passive recording before any stimulation.
    response_window_s : float
        Seconds of recording after each stimulation epoch.
    inter_epoch_rest_s : float
        Rest period between stimulation epochs (seconds).
    n_epochs : int
        Number of stimulation epochs.
    inter_stim_delay_s : float
        Sleep between consecutive trigger sends within one epoch.
    """
 
    def __init__(
        self,
        token: str,
        booking_email: str,
        stim_configs: Optional[List[StimParamConfig]] = None,
        baseline_duration_s: float = 120.0,
        response_window_s: float = 60.0,
        inter_epoch_rest_s: float = 30.0,
        n_epochs: int = 5,
        inter_stim_delay_s: float = 2.0,
    ) -> None:
        self.token = token
        self.booking_email = booking_email
        self.baseline_duration_s = baseline_duration_s
        self.response_window_s = response_window_s
        self.inter_epoch_rest_s = inter_epoch_rest_s
        self.n_epochs = n_epochs
        self.inter_stim_delay_s = inter_stim_delay_s
 
        # Build default stimulation configs if none provided.
        # Two charge-balanced biphasic pulses on electrodes 0 and 8,
        # each on its own trigger slot.
        if stim_configs is None:
            self.stim_configs: List[StimParamConfig] = [
                StimParamConfig.charge_balanced(
                    electrode_index=0,
                    trigger_key=0,
                    amplitude1=5.0,    # µA
                    duration1=200.0,   # µs  → charge = 1000 µA·µs
                    duration2=200.0,   # µs  → A2 = 5.0 µA  (symmetric)
                    polarity="NegativeFirst",
                    nb_pulse=1,
                ),
                StimParamConfig.charge_balanced(
                    electrode_index=8,
                    trigger_key=1,
                    amplitude1=5.0,
                    duration1=200.0,
                    duration2=200.0,
                    polarity="NegativeFirst",
                    nb_pulse=1,
                ),
            ]
        else:
            self.stim_configs = stim_configs
 
        # Internal state
        self._stimulus_log: List[StimulusRecord] = []
        self._coa_results: List[CoAResult] = []
        self._stim_timestamps: List[datetime] = []
 
    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(
        self,
        testing: bool = True,
    ) -> List[CoAResult]:
        """
        Execute the experiment.
 
        Parameters
        ----------
        testing : bool
            If True, no hardware is instantiated and no stimulations are
            sent to the organoid.  All other logic (timing, logging,
            analysis) runs as normal so the script can be validated
            before a live booking.
 
        Returns
        -------
        List[CoAResult]
            One baseline result followed by one result per epoch.
        """
        mode = "TEST (dry-run)" if testing else "LIVE"
        logger.info("=" * 60)
        logger.info("CoA Shift Experiment – %s mode", mode)
        logger.info("Token      : %s", self.token)
        logger.info("Email      : %s", self.booking_email)
        logger.info("Epochs     : %d", self.n_epochs)
        logger.info("Baseline   : %.1f s", self.baseline_duration_s)
        logger.info("Response   : %.1f s", self.response_window_s)
        logger.info("Rest       : %.1f s", self.inter_epoch_rest_s)
        logger.info("=" * 60)
 
        for cfg in self.stim_configs:
            logger.info("StimParam  : %s", cfg.summary())
 
        # ----------------------------------------------------------
        # Instantiate hardware (skipped in testing mode)
        # ----------------------------------------------------------
        experiment = None
        intan = None
        trigger_ctrl = None
        db = None
 
        if not testing:
            # Import neuroplatform here so the file is importable without
            # the package installed (e.g. during testing / CI).
            from neuroplatform import (  # type: ignore
                Database,
                Experiment,
                IntanSofware,
                StimParam,
                StimPolarity,
                TriggerController,
            )
 
            experiment = Experiment(self.token)
            db = Database()
            intan = IntanSofware()
            trigger_ctrl = TriggerController(booking_email=self.booking_email)
        else:
            logger.info("[TEST] Skipping hardware instantiation.")
 
        # ----------------------------------------------------------
        # Build StimParam objects from our validated configs
        # ----------------------------------------------------------
        stim_params = []
        if not testing:
            from neuroplatform import StimParam, StimPolarity  # type: ignore
 
            polarity_map = {
                "NegativeFirst": StimPolarity.NegativeFirst,
                "PositiveFirst": StimPolarity.PositiveFirst,
            }
            for cfg in self.stim_configs:
                sp = StimParam()
                sp.enable = True
                sp.index = cfg.electrode_index
                sp.trigger_key = cfg.trigger_key
                sp.polarity = polarity_map.get(cfg.polarity,
                                               StimPolarity.NegativeFirst)
                sp.phase_amplitude1 = cfg.phase_amplitude1
                sp.phase_duration1 = cfg.phase_duration1
                sp.phase_amplitude2 = cfg.phase_amplitude2
                sp.phase_duration2 = cfg.phase_duration2
                sp.nb_pulse = cfg.nb_pulse
                sp.pulse_train_period = cfg.pulse_train_period
                stim_params.append(sp)
        else:
            logger.info("[TEST] Skipping StimParam construction.")
 
        # ----------------------------------------------------------
        # Build trigger arrays for the stimulation epoch
        # ----------------------------------------------------------
        # All configured triggers fired together in each epoch.
        stim_trigger_array = np.zeros(16, dtype=np.uint8)
        for cfg in self.stim_configs:
            stim_trigger_array[cfg.trigger_key] = 1
 
        fs_name = None
 
        # ----------------------------------------------------------
        # Run experiment lifecycle
        # ----------------------------------------------------------
        try:
            if not testing:
                if experiment.start():
                    logger.info("Experiment started.")
                else:
                    raise RuntimeError("Failed to start experiment.")
 
                fs_name = experiment.exp_name
                logger.info("FS name: %s", fs_name)
 
                # Send stimulation parameters (takes ~10 s on shared access)
                logger.info("Sending StimParams to Intan (~10 s)…")
                intan.send_stimparam(stim_params)
                logger.info("StimParams sent.")
 
                # Disable variable threshold for consistent event capture
                # during stimulation windows.
                intan.var_threshold(False)
                logger.info("Variable threshold disabled.")
 
            # --- Baseline recording window ----------------------------
            baseline_start = datetime.utcnow()
            logger.info(
                "Baseline recording started at %s (%.1f s).",
                baseline_start.isoformat(), self.baseline_duration_s,
            )
            time.sleep(self.baseline_duration_s)
            baseline_end = datetime.utcnow()
            logger.info("Baseline recording ended at %s.", baseline_end.isoformat())
 
            # Fetch baseline spikes
            baseline_result = self._fetch_and_compute_coa(
                db=db,
                fs_name=fs_name,
                label="baseline",
                window_start=baseline_start,
                window_end=baseline_end,
                testing=testing,
            )
            self._coa_results.append(baseline_result)
 
            # --- Stimulation epochs -----------------------------------
            for epoch in range(1, self.n_epochs + 1):
                logger.info("-" * 50)
                logger.info("Epoch %d / %d", epoch, self.n_epochs)
 
                epoch_stim_start = datetime.utcnow()
 
                self._deliver_stimulation(
                    epoch=epoch,
                    trigger_array=stim_trigger_array,
                    trigger_ctrl=trigger_ctrl,
                    testing=testing,
                )
 
                # --- Response recording window ------------------------
                response_start = datetime.utcnow()
                logger.info(
                    "Response window started (%.1f s).", self.response_window_s
                )
                time.sleep(self.response_window_s)
                response_end = datetime.utcnow()
 
                epoch_result = self._fetch_and_compute_coa(
                    db=db,
                    fs_name=fs_name,
                    label=f"epoch_{epoch}",
                    window_start=response_start,
                    window_end=response_end,
                    testing=testing,
                )
                self._coa_results.append(epoch_result)
 
                # --- Log displacement vs baseline ---------------------
                drow, dcol = epoch_result.displacement_from(baseline_result)
                logger.info(
                    "Epoch %d CoA displacement: Δrow=%.3f  Δcol=%.3f",
                    epoch, drow, dcol,
                )
 
                # --- Rest between epochs ------------------------------
                if epoch < self.n_epochs:
                    logger.info(
                        "Resting for %.1f s before next epoch…",
                        self.inter_epoch_rest_s,
                    )
                    time.sleep(self.inter_epoch_rest_s)
 
        finally:
            # ----------------------------------------------------------
            # Clean-up hardware
            # ----------------------------------------------------------
            if not testing:
                # Re-enable variable threshold
                try:
                    intan.var_threshold(True)
                    logger.info("Variable threshold re-enabled.")
                except Exception as exc:
                    logger.warning("Could not re-enable variable threshold: %s", exc)
 
                # Disable all StimParams
                try:
                    for sp in stim_params:
                        sp.enable = False
                    intan.send_stimparam(stim_params)
                    logger.info("All StimParams disabled.")
                except Exception as exc:
                    logger.warning("Error disabling StimParams: %s", exc)
 
                # Close connections
                try:
                    trigger_ctrl.close()
                    logger.info("TriggerController closed.")
                except Exception as exc:
                    logger.warning("Error closing TriggerController: %s", exc)
 
                try:
                    intan.close()
                    logger.info("IntanSoftware connection closed.")
                except Exception as exc:
                    logger.warning("Error closing Intan: %s", exc)
 
                try:
                    experiment.stop()
                    logger.info("Experiment stopped.")
                except Exception as exc:
                    logger.warning("Error stopping experiment: %s", exc)
 
        # ----------------------------------------------------------
        # Summary report
        # ----------------------------------------------------------
        self._print_summary(baseline_result=self._coa_results[0])
        return self._coa_results
 
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
 
    def _deliver_stimulation(
        self,
        epoch: int,
        trigger_array: np.ndarray,
        trigger_ctrl,
        testing: bool,
    ) -> None:
        """
        Send a single trigger pulse.  In testing mode, the trigger is
        logged but not actually transmitted.
        """
        ts = datetime.utcnow()
        self._stim_timestamps.append(ts)
 
        record = StimulusRecord(
            epoch=epoch,
            timestamp_utc=ts,
            stim_configs=list(self.stim_configs),
            trigger_array=trigger_array.copy(),
            testing=testing,
        )
        self._stimulus_log.append(record)
 
        if testing:
            logger.info(
                "[TEST] Epoch %d – stimulation NOT sent (testing=True). "
                "Trigger array: %s  timestamp: %s",
                epoch,
                trigger_array.tolist(),
                ts.isoformat(),
            )
        else:
            logger.info(
                "Epoch %d – sending trigger array: %s",
                epoch, trigger_array.tolist(),
            )
            trigger_ctrl.send(trigger_array)
            logger.info("Trigger sent at %s.", ts.isoformat())
 
        # Brief delay after stimulation to allow the organoid to respond
        time.sleep(self.inter_stim_delay_s)
 
    def _fetch_and_compute_coa(
        self,
        db,
        fs_name: Optional[str],
        label: str,
        window_start: datetime,
        window_end: datetime,
        testing: bool,
    ) -> CoAResult:
        """
        Fetch spike events from the database for a given window and
        compute the centre of activity.
 
        In testing mode, synthetic data is generated instead.
        """
        analyser = CoAAnalyser(stim_timestamps=self._stim_timestamps)
 
        if testing:
            # Generate synthetic spike events so analysis code is exercised.
            logger.info("[TEST] Generating synthetic spike data for '%s'.", label)
            spike_df = _generate_synthetic_spikes(window_start, window_end)
        else:
            logger.info(
                "Fetching spike events for '%s' (%s → %s)…",
                label,
                window_start.isoformat(),
                window_end.isoformat(),
            )
            spike_df = db.get_spike_event(window_start, window_end, fs_name)
            logger.info("Fetched %d spike events.", len(spike_df))
 
        return analyser.compute(
            spike_df=spike_df,
            label=label,
            window_start=window_start,
            window_end=window_end,
        )
 
    def _print_summary(self, baseline_result: CoAResult) -> None:
        """Print a final summary of all CoA measurements and displacements."""
        logger.info("=" * 60)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("=" * 60)
        logger.info(
            "Stimulations logged : %d  (testing=%s)",
            len(self._stimulus_log),
            self._stimulus_log[0].testing if self._stimulus_log else "N/A",
        )
        logger.info(
            "Baseline CoA        : row=%.3f  col=%.3f",
            baseline_result.coa_row,
            baseline_result.coa_col,
        )
        for result in self._coa_results[1:]:
            drow, dcol = result.displacement_from(baseline_result)
            magnitude = (drow ** 2 + dcol ** 2) ** 0.5
            logger.info(
                "  [%s]  CoA row=%.3f col=%.3f  |  "
                "Δrow=%+.3f  Δcol=%+.3f  |dist|=%.3f",
                result.label,
                result.coa_row,
                result.coa_col,
                drow,
                dcol,
                magnitude,
            )
        logger.info("=" * 60)
 
    # ------------------------------------------------------------------
    # Accessors for post-hoc analysis
    # ------------------------------------------------------------------
 
    @property
    def stimulus_log(self) -> List[StimulusRecord]:
        """All stimulus records from the run."""
        return list(self._stimulus_log)
 
    @property
    def coa_results(self) -> List[CoAResult]:
        """All CoA results (baseline + epochs)."""
        return list(self._coa_results)
 
 
# ---------------------------------------------------------------------------
# Synthetic data generator – used only during testing
# ---------------------------------------------------------------------------
def _generate_synthetic_spikes(
    window_start: datetime,
    window_end: datetime,
    n_events: int = 200,
    seed: Optional[int] = None,
) -> "pd.DataFrame":
    """
    Return a synthetic spike-event DataFrame mimicking the output of
    ``Database.get_spike_event``.
 
    Columns: Time (datetime), channel (int), amplitude (float, µV)
    """
    import pandas as pd  # type: ignore
 
    rng = np.random.default_rng(seed)
    duration_s = (window_end - window_start).total_seconds()
    offsets = rng.uniform(0, duration_s, n_events)
    timestamps = [window_start + timedelta(seconds=float(o)) for o in offsets]
 
    # Channels drawn from a subset of 32 electrodes on one site
    channels = rng.integers(0, 32, size=n_events).tolist()
    # Amplitudes in the biologically plausible range 30–200 µV
    amplitudes = rng.uniform(30, 200, size=n_events).tolist()
 
    return pd.DataFrame({
        "Time": timestamps,
        "channel": channels,
        "amplitude": amplitudes,
    })
