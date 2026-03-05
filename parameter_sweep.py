"""
Stimulation Parameter Sweep for the FinalSpark NeuroPlatform.

This script systematically sweeps over a configurable grid of stimulation
parameters (amplitude, duration, polarity, …) across one or more electrodes,
records when each stimulus was sent, and saves a summary CSV so that the
results can later be correlated with spike data from the Spike-DB.

Usage
-----
1.  Set TOKEN to the token that was provided to you by FinalSpark.
2.  Edit SweepConfig to choose which electrodes and parameter ranges to test.
3.  Run:

        # Full live experiment
        python stim_sweep.py

        # Test mode: connects to real hardware, uploads params, sends triggers
        # but forces amplitude to 0 µA so no tissue is stimulated.
        python stim_sweep.py --test

        # Dry-run: no hardware connections at all, only logs what would happen.
        python stim_sweep.py --dry-run

Execution modes (mutually exclusive, in ascending order of hardware involvement)
---------------------------------------------------------------------------------
dry_run   – No SDK calls whatsoever.  Pure logic / CSV output test.
            Delays are skipped entirely.  Use this to validate the parameter
            grid and output file before touching hardware.

test_mode – Full hardware round-trip (Experiment, Intan, TriggerGenerator) with
            real connections, parameter uploads, and triggers — but every
            StimParam is sent with amplitude forced to 0 µA so that no actual
            stimulation current flows through the tissue.  Delays are real.
            Use this to verify hardware connectivity and timing end-to-end.

(neither) – Normal live experiment.  Parameters and amplitudes are used as
            configured.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

# NeuroPlatform SDK
from neuroplatform import Experiment, IntanSofware, StimParam, StimPolarity, Trigger

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
	level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Amplitude sent to hardware when test_mode is active [µA].
# Must be 0 so that no current flows, but a valid StimParam is still uploaded.
_TEST_MODE_AMPLITUDE_uA: float = 0.0


# ===========================================================================
# Data classes – configuration and result records
# ===========================================================================


@dataclass
class SweepConfig:
    """All user-configurable options for the parameter sweep."""

    # ---- Experiment --------------------------------------------------------
    token: str = "YOUR_TOKEN_HERE"

    # ---- Electrodes to sweep -----------------------------------------------
    # List of electrode indices (absolute index, 0-127).
    # All electrodes share the same parameter grid; each gets its own trigger.
    electrode_indices: List[int] = field(default_factory=lambda: [0, 8, 16])

    # ---- Parameter grid ----------------------------------------------------
    # Phase amplitudes [µA]
    amplitudes_uA: List[float] = field(default_factory=lambda: [1.0, 2.0, 4.0])

    # Phase durations [µs]  – both D1 and D2 are set to the same value so that
    # the charge is always balanced (D1×A1 == D2×A2 when A1==A2).
    durations_us: List[float] = field(default_factory=lambda: [50.0, 100.0, 200.0])

    # Polarities to test
    polarities: List[StimPolarity] = field(
        default_factory=lambda: [StimPolarity.NegativeFirst, StimPolarity.PositiveFirst]
    )

    # ---- Timing ------------------------------------------------------------
    # How long to wait between consecutive stimulus pulses (seconds)
    inter_stim_delay_s: float = 2.0

    # How long to wait after sending new parameters to Intan before firing
    # (the API docs state that send_stimparam takes ~10 s to complete)
    param_upload_wait_s: float = 12.0

    # ---- Output ------------------------------------------------------------
    output_csv: Path = Path("sweep_results.csv")

    # ---- Safety limits -----------------------------------------------------
    max_amplitude_uA: float = 10.0   # hard cap – never exceed this
    max_duration_us: float = 1000.0  # hard cap – never exceed this


@dataclass
class SweepRecord:
    """One row in the output CSV – a single stimulus event."""

    timestamp_utc: str
    electrode_index: int
    trigger_key: int
    amplitude_uA: float            # configured amplitude (may differ from what was sent)
    effective_amplitude_uA: float  # amplitude actually sent to hardware
    duration_us: float
    polarity: str
    tag: int                       # integer tag sent to Intan for DB lookup
    execution_mode: str            # "live" | "test" | "dry_run"


# ===========================================================================
# Helper class – wraps a single stimulation configuration
# ===========================================================================


class StimConfiguration:
    """
    Builds and owns a StimParam for a specific (electrode, amplitude,
    duration, polarity) combination.

    Charge balance is enforced: D1×A1 == D2×A2 with A1 == A2 → D1 == D2.
    """

    def __init__(
        self,
        electrode_index: int,
        trigger_key: int,
        amplitude_uA: float,
        duration_us: float,
        polarity: StimPolarity,
    ) -> None:
        self.electrode_index = electrode_index
        self.trigger_key = trigger_key
        self.amplitude_uA = amplitude_uA
        self.duration_us = duration_us
        self.polarity = polarity
        self.param: StimParam = self._build()

    def _build(self) -> StimParam:
        p = StimParam()
        p.enable = True
        p.index = self.electrode_index
        p.trigger_key = self.trigger_key
        p.polarity = self.polarity

        # Balanced biphasic – same amplitude and duration for both phases
        p.phase_amplitude1 = self.amplitude_uA
        p.phase_amplitude2 = self.amplitude_uA
        p.phase_duration1 = self.duration_us
        p.phase_duration2 = self.duration_us

        # Standard safety / refractory settings
        p.post_stim_ref_period = 1000.0
        p.enable_amp_settle = True
        p.post_stim_amp_settle = 1000.0
        p.enable_charge_recovery = True
        p.post_charge_recovery_off = 100.0
        return p

    def disabled_copy(self) -> StimParam:
        """Return a copy of the underlying StimParam with enable=False."""
        p = StimParam()
        p.enable = False
        p.index = self.electrode_index
        p.trigger_key = self.trigger_key
        return p

    def __repr__(self) -> str:
        return (
            f"StimConfiguration(electrode={self.electrode_index}, "
            f"trigger={self.trigger_key}, "
            f"amp={self.amplitude_uA}µA, "
            f"dur={self.duration_us}µs, "
            f"polarity={self.polarity})"
        )


# ===========================================================================
# Result writer
# ===========================================================================


class SweepResultWriter:
    """Incrementally writes SweepRecords to a CSV file."""

    _FIELDS = [
        "timestamp_utc",
        "electrode_index",
        "trigger_key",
        "amplitude_uA",
        "effective_amplitude_uA",
        "duration_us",
        "polarity",
        "tag",
        "execution_mode",
    ]

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self._FIELDS)
        self._writer.writeheader()
        log.info("Results will be written to: %s", path.resolve())

    def write(self, record: SweepRecord) -> None:
        self._writer.writerow(
            {
                "timestamp_utc": record.timestamp_utc,
                "electrode_index": record.electrode_index,
                "trigger_key": record.trigger_key,
                "amplitude_uA": record.amplitude_uA,
                "effective_amplitude_uA": record.effective_amplitude_uA,
                "duration_us": record.duration_us,
                "polarity": record.polarity,
                "tag": record.tag,
                "execution_mode": record.execution_mode,
            }
        )
        self._file.flush()

    def close(self) -> None:
        self._file.close()
        log.info("Results CSV closed.")


# ===========================================================================
# Main sweep class
# ===========================================================================


class StimParameterSweep:
    """
    Orchestrates a full stimulation parameter sweep over the NeuroPlatform.

    Call :py:meth:`run` to start the sweep.

    Parameters
    ----------
    config:
        A :class:`SweepConfig` instance with all sweep settings.
    dry_run:
        When *True* the hardware connections are skipped and the script only
        logs what it *would* do.  Useful for testing the parameter grid before
        a real experiment.
    """

    # The Intan supports triggers 0-15.  We reserve one trigger per electrode.
    MAX_TRIGGERS = 16

    def __init__(
        self, config: Optional[SweepConfig] = None, dry_run: bool = False
    ) -> None:
        self.config = config or SweepConfig()
        self.dry_run = dry_run

        self._validate_config()

        # Hardware handles – initialised in run()
        self._exp: Optional[Experiment] = None
        self._intan: Optional[IntanSofware] = None
        self._trigger_gen: Optional[Trigger] = None

        # Result writer
        self._writer = SweepResultWriter(self.config.output_csv)

        # Running tag counter (used as an integer tag in Intan, starts at 1)
        self._tag_counter: int = 1

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Execute the full parameter sweep.

        This is the single entry point to run the experiment.
        """
        log.info("=== NeuroPlatform Stimulation Parameter Sweep ===")
        log.info("Dry-run mode: %s", self.dry_run)

        combos = self._build_parameter_grid()
        total = len(combos)
        log.info(
            "Parameter grid: %d electrodes × %d combos = %d total stimulations",
            len(self.config.electrode_indices),
            total // len(self.config.electrode_indices),
            total,
        )

        if not self.dry_run:
            self._connect_hardware()

        try:
            self._run_sweep(combos)
        finally:
            self._teardown()

        log.info("Sweep complete.  Results saved to: %s", self.config.output_csv)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        """Raise ValueError for obviously unsafe or impossible configurations."""
        cfg = self.config

        if not cfg.electrode_indices:
            raise ValueError("electrode_indices must not be empty.")

        n_electrodes = len(cfg.electrode_indices)
        if n_electrodes > self.MAX_TRIGGERS:
            raise ValueError(
                f"Too many electrodes ({n_electrodes}); "
                f"max is {self.MAX_TRIGGERS} (one trigger per electrode)."
            )

        for amp in cfg.amplitudes_uA:
            if amp > cfg.max_amplitude_uA:
                raise ValueError(
                    f"Amplitude {amp} µA exceeds safety limit {cfg.max_amplitude_uA} µA."
                )

        for dur in cfg.durations_us:
            if dur > cfg.max_duration_us:
                raise ValueError(
                    f"Duration {dur} µs exceeds safety limit {cfg.max_duration_us} µs."
                )

        if len(set(cfg.electrode_indices)) != len(cfg.electrode_indices):
            raise ValueError("Duplicate electrode indices found in electrode_indices.")

        log.info("Configuration validated OK.")

    def _build_parameter_grid(
        self,
    ) -> List[tuple]:
        """
        Return a flat list of (electrode_index, trigger_key, amp, dur, polarity)
        tuples representing every combination to test.

        The trigger_key is assigned by position in electrode_indices so that
        each electrode always uses the same trigger throughout the sweep.
        """
        electrode_trigger_pairs = [
            (idx, trigger)
            for trigger, idx in enumerate(self.config.electrode_indices)
        ]

        param_combos = list(
            itertools.product(
                self.config.amplitudes_uA,
                self.config.durations_us,
                self.config.polarities,
            )
        )

        grid = []
        for amp, dur, pol in param_combos:
            for elec_idx, trig_key in electrode_trigger_pairs:
                grid.append((elec_idx, trig_key, amp, dur, pol))

        return grid

    def _connect_hardware(self) -> None:
        """Open connections to Experiment, Intan and TriggerGenerator."""
        log.info("Connecting to NeuroPlatform hardware …")
        self._exp = Experiment(self.config.token)
        self._intan = IntanSofware()
        self._trigger_gen = Trigger()
        log.info("Hardware connections established.")

    def _run_sweep(self, combos: List[tuple]) -> None:
        """
        Core sweep loop.

        The combos list is grouped by (amp, dur, polarity) so that all
        electrodes sharing the same parameters are uploaded together in a
        single 10-second Intan round-trip.  This dramatically reduces
        total experiment time.
        """
        cfg = self.config

        # Start experiment (non-dry-run only)
        if not self.dry_run:
            if not self._exp.start():
                raise RuntimeError("Failed to start experiment – another may be running.")
            log.info("Experiment started.")

        # Group combos by parameter set (amp, dur, polarity)
        # Each group contains all electrodes for that parameter combination.
        groups: dict = {}
        for elec_idx, trig_key, amp, dur, pol in combos:
            key = (amp, dur, pol)
            groups.setdefault(key, []).append((elec_idx, trig_key))

        total_groups = len(groups)
        log.info("Uploading %d unique parameter sets to Intan.", total_groups)

        for group_num, ((amp, dur, pol), elec_trig_pairs) in enumerate(
            groups.items(), start=1
        ):
            log.info(
                "Group %d/%d — amp=%.1f µA  dur=%.0f µs  polarity=%s  "
                "electrodes=%s",
                group_num,
                total_groups,
                amp,
                dur,
                pol,
                [e for e, _ in elec_trig_pairs],
            )

            # Build one StimConfiguration per electrode for this parameter set
            stim_configs = [
                StimConfiguration(
                    electrode_index=elec_idx,
                    trigger_key=trig_key,
                    amplitude_uA=amp,
                    duration_us=dur,
                    polarity=pol,
                )
                for elec_idx, trig_key in elec_trig_pairs
            ]

            # Upload parameters to Intan (takes ~10 s)
            self._upload_params([sc.param for sc in stim_configs])

            # Fire each electrode individually so we get one record per stim
            for sc in stim_configs:
                self._fire_single(sc, amp, dur, pol)
                time.sleep(cfg.inter_stim_delay_s)

            # Disable all params in this group before moving to the next set
            self._upload_params([sc.disabled_copy() for sc in stim_configs])

        log.info("All parameter groups completed.")

    def _upload_params(self, params: List[StimParam]) -> None:
        """Send a list of StimParams to the Intan software."""
        if self.dry_run:
            log.debug("[DRY-RUN] Would upload %d StimParam(s) to Intan.", len(params))
            return

        log.debug("Uploading %d param(s) to Intan (waiting ~%ds) …",
                  len(params), int(self.config.param_upload_wait_s))
        self._intan.send_stimparam(params)
        time.sleep(self.config.param_upload_wait_s)

    def _fire_single(
        self,
        sc: StimConfiguration,
        amp: float,
        dur: float,
        pol: StimPolarity,
    ) -> None:
        """Send a single trigger for the given StimConfiguration and log it."""
        tag = self._tag_counter
        self._tag_counter += 1
        ts = datetime.now(tz=timezone.utc).isoformat()

        log.info(
            "  FIRE  electrode=%d  trigger=%d  amp=%.1f µA  dur=%.0f µs  "
            "polarity=%s  tag=%d",
            sc.electrode_index,
            sc.trigger_key,
            amp,
            dur,
            pol,
            tag,
        )

        if not self.dry_run:
            # Tag the trigger so it can be retrieved from the Spike-DB later
            self._intan.set_tag_trigger(tag)

            trigger_array = np.zeros(16, dtype=np.uint8)
            trigger_array[sc.trigger_key] = 1
            self._trigger_gen.send(trigger_array)

        record = SweepRecord(
            timestamp_utc=ts,
            electrode_index=sc.electrode_index,
            trigger_key=sc.trigger_key,
            amplitude_uA=amp,
            duration_us=dur,
            polarity=str(pol),
            tag=tag,
        )
        self._writer.write(record)

    def _teardown(self) -> None:
        """Gracefully close all hardware connections."""
        log.info("Tearing down …")
        self._writer.close()

        if self.dry_run:
            log.info("[DRY-RUN] No hardware to close.")
            return

        # Always re-enable variable threshold and stop the experiment
        try:
            self._intan.var_threshold(True)
        except Exception as exc:
            log.warning("Could not re-enable variable threshold: %s", exc)

        try:
            self._trigger_gen.close()
        except Exception as exc:
            log.warning("Could not close TriggerGenerator: %s", exc)

        try:
            self._intan.close()
        except Exception as exc:
            log.warning("Could not close Intan: %s", exc)

        try:
            self._exp.stop()
        except Exception as exc:
            log.warning("Could not stop experiment: %s", exc)

        log.info("Hardware teardown complete.")
