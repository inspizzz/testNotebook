"""
Stimulation Parameter Sweep — NeuroPlatform / FinalSpark
=========================================================
Designed for remote execution via the Datalore API + GitHub integration.

HOW PARAMETERS ARE INJECTED
-----------------------------
The Datalore runner passes parameters in two ways, matching the API contract:

    CLASS_PARAMETERS  -> passed as keyword arguments to __init__()
    FUNCTION_PARAMETERS -> passed as keyword arguments to run()

Example Datalore API payload:

    CLASS_PARAMETERS = '[
        {"name": "token",            "value": "9T5KLS6T7X"},
        {"name": "dry_run",          "value": false}
    ]'

    FUNCTION_PARAMETERS = '[
        {"name": "electrode_indices", "value": [0, 8, 16]},
        {"name": "amplitudes_uA",     "value": [1.0, 2.0, 4.0]},
        {"name": "durations_us",      "value": [50.0, 100.0, 200.0]},
        {"name": "polarities",        "value": ["NegativeFirst", "PositiveFirst"]},
        {"name": "inter_stim_delay_s","value": 2.0},
        {"name": "param_upload_wait_s","value": 12.0},
        {"name": "output_csv",        "value": "sweep_results.csv"}
    ]'

POLARITY VALUES
---------------
Pass polarities as strings — the class resolves them automatically:
    "NegativeFirst"  ->  StimPolarity.NegativeFirst
    "PositiveFirst"  ->  StimPolarity.PositiveFirst

LOCAL / MANUAL TESTING
-----------------------
    sweep = StimParameterSweep(token="MY_TOKEN", dry_run=True)
    sweep.run(amplitudes_uA=[1.0, 2.0], electrode_indices=[0, 8])
"""

from __future__ import annotations

import csv
import itertools
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
from neuroplatform import Experiment, IntanSofware, StimParam, StimPolarity, Trigger

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Polarity helper — accepts StimPolarity instances or plain strings
# ---------------------------------------------------------------------------
_POLARITY_MAP = {
    "negativefirst": StimPolarity.NegativeFirst,
    "positivefirst": StimPolarity.PositiveFirst,
}


def _resolve_polarity(value) -> StimPolarity:
    if isinstance(value, StimPolarity):
        return value
    key = str(value).lower().replace(" ", "").replace("_", "")
    if key not in _POLARITY_MAP:
        raise ValueError(
            f"Unknown polarity '{value}'. "
            f"Valid strings: {list(_POLARITY_MAP.keys())}"
        )
    return _POLARITY_MAP[key]


# ===========================================================================
# Internal helpers
# ===========================================================================


class _StimConfiguration:
    """
    Builds a charge-balanced biphasic StimParam for one
    (electrode, amplitude, duration, polarity) combination.
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
        # Charge-balanced: D1*A1 == D2*A2  (A1==A2 => D1==D2)
        p.phase_amplitude1 = self.amplitude_uA
        p.phase_amplitude2 = self.amplitude_uA
        p.phase_duration1 = self.duration_us
        p.phase_duration2 = self.duration_us
        # Recommended refractory / safety defaults
        p.post_stim_ref_period = 1000.0
        p.enable_amp_settle = True
        p.post_stim_amp_settle = 1000.0
        p.enable_charge_recovery = True
        p.post_charge_recovery_off = 100.0
        return p

    def disabled_copy(self) -> StimParam:
        p = StimParam()
        p.enable = False
        p.index = self.electrode_index
        p.trigger_key = self.trigger_key
        return p


@dataclass
class _SweepRecord:
    timestamp_utc: str
    electrode_index: int
    trigger_key: int
    amplitude_uA: float
    duration_us: float
    polarity: str
    tag: int


class _SweepResultWriter:
    _FIELDS = [
        "timestamp_utc", "electrode_index", "trigger_key",
        "amplitude_uA", "duration_us", "polarity", "tag",
    ]

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self._FIELDS)
        self._writer.writeheader()
        log.info("Results CSV: %s", path.resolve())

    def write(self, record: _SweepRecord) -> None:
        self._writer.writerow(record.__dict__)
        self._file.flush()

    def close(self) -> None:
        self._file.close()
        log.info("Results CSV closed.")


# ===========================================================================
# Main class
# ===========================================================================


class StimParameterSweep:
    """
    Stimulation parameter sweep for the FinalSpark NeuroPlatform.

    Parameters
    ----------
    All __init__ arguments map directly to CLASS_PARAMETERS entries.
    All run() arguments map directly to FUNCTION_PARAMETERS entries.

    __init__ (CLASS_PARAMETERS)
    ----------------------------
    token : str
        Experiment token provided by FinalSpark.
    dry_run : bool
        True  -> log everything, never touch hardware (safe for testing).
        False -> live experiment.
    max_amplitude_uA : float
        Hard safety cap on amplitude. Raises ValueError if exceeded.
    max_duration_us : float
        Hard safety cap on duration. Raises ValueError if exceeded.
    param_upload_wait_s : float
        Seconds to wait after send_stimparam(). Must be >= 10.0 (hardware limit).

    run() (FUNCTION_PARAMETERS)
    ----------------------------
    electrode_indices : list[int]
        Absolute electrode indices to stimulate (0-127). Max 16 electrodes.
    amplitudes_uA : list[float]
        Amplitude values to sweep [uA].
    durations_us : list[float]
        Phase duration values to sweep [us] — applied to both D1 and D2
        to keep stimulations charge-balanced.
    polarities : list[str | StimPolarity]
        Polarity values to sweep. Accepted strings: "NegativeFirst",
        "PositiveFirst".
    inter_stim_delay_s : float
        Pause between consecutive trigger fires [seconds].
    output_csv : str
        File path for the results CSV.
    """

    _MAX_TRIGGERS: int = 16

    # ------------------------------------------------------------------
    # Constructor  <-  CLASS_PARAMETERS
    # ------------------------------------------------------------------

    def __init__(
        self,
        token: str = "YOUR_TOKEN_HERE",
        dry_run: bool = True,
        max_amplitude_uA: float = 10.0,
        max_duration_us: float = 1000.0,
        param_upload_wait_s: float = 12.0,
    ) -> None:
        self.token = token
        self.dry_run = dry_run
        self.max_amplitude_uA = max_amplitude_uA
        self.max_duration_us = max_duration_us
        self.param_upload_wait_s = param_upload_wait_s

        # Internal state
        self._exp: Optional[Experiment] = None
        self._intan: Optional[IntanSofware] = None
        self._trigger_gen: Optional[Trigger] = None
        self._writer: Optional[_SweepResultWriter] = None
        self._tag_counter: int = 1

    # ------------------------------------------------------------------
    # Entry point  <-  FUNCTION_PARAMETERS
    # ------------------------------------------------------------------

    def run(
        self,
        electrode_indices: List[int] = None,
        amplitudes_uA: List[float] = None,
        durations_us: List[float] = None,
        polarities: List = None,
        inter_stim_delay_s: float = 2.0,
        output_csv: str = "sweep_results.csv",
    ) -> None:
        """
        Execute the full stimulation parameter sweep.

        All arguments correspond 1-to-1 with FUNCTION_PARAMETERS entries.
        Defaults are used for any parameter not supplied by the caller.
        """
        # Apply defaults here (avoids mutable default argument pitfall)
        electrode_indices = electrode_indices if electrode_indices is not None else [0, 8, 16]
        amplitudes_uA     = amplitudes_uA     if amplitudes_uA     is not None else [1.0, 2.0, 4.0]
        durations_us      = durations_us      if durations_us      is not None else [50.0, 100.0, 200.0]
        polarities        = polarities        if polarities        is not None else ["NegativeFirst", "PositiveFirst"]

        # Resolve polarity strings -> StimPolarity enum values
        resolved_polarities: List[StimPolarity] = [
            _resolve_polarity(p) for p in polarities
        ]

        log.info("=" * 60)
        log.info("NeuroPlatform  —  Stimulation Parameter Sweep")
        log.info("Token:    %s", self.token)
        log.info("Dry-run:  %s", self.dry_run)
        log.info("=" * 60)

        self._validate(electrode_indices, amplitudes_uA, durations_us)

        grid = self._build_grid(electrode_indices, amplitudes_uA, durations_us, resolved_polarities)
        log.info(
            "Grid: %d electrodes x %d combos = %d total stimulations",
            len(electrode_indices),
            len(amplitudes_uA) * len(durations_us) * len(resolved_polarities),
            len(grid),
        )

        self._writer = _SweepResultWriter(Path(output_csv))
        self._tag_counter = 1  # reset between runs

        if not self.dry_run:
            self._connect()

        try:
            self._sweep(grid, inter_stim_delay_s)
        finally:
            self._teardown()

        log.info("Sweep complete. Results: %s", Path(output_csv).resolve())

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(
        self,
        electrode_indices: List[int],
        amplitudes_uA: List[float],
        durations_us: List[float],
    ) -> None:
        if not electrode_indices:
            raise ValueError("electrode_indices must not be empty.")
        if len(electrode_indices) > self._MAX_TRIGGERS:
            raise ValueError(
                f"Too many electrodes ({len(electrode_indices)}); "
                f"max is {self._MAX_TRIGGERS}."
            )
        if len(set(electrode_indices)) != len(electrode_indices):
            raise ValueError("Duplicate values in electrode_indices.")
        for amp in amplitudes_uA:
            if amp > self.max_amplitude_uA:
                raise ValueError(
                    f"Amplitude {amp} uA exceeds safety cap {self.max_amplitude_uA} uA."
                )
        for dur in durations_us:
            if dur > self.max_duration_us:
                raise ValueError(
                    f"Duration {dur} us exceeds safety cap {self.max_duration_us} us."
                )
        if self.param_upload_wait_s < 10.0:
            raise ValueError("param_upload_wait_s must be >= 10.0 s (hardware limit).")
        log.info("Configuration validated OK.")

    # ------------------------------------------------------------------
    # Parameter grid
    # ------------------------------------------------------------------

    def _build_grid(
        self,
        electrode_indices: List[int],
        amplitudes_uA: List[float],
        durations_us: List[float],
        polarities: List[StimPolarity],
    ) -> List[tuple]:
        electrode_trigger_map = {
            idx: trig for trig, idx in enumerate(electrode_indices)
        }
        grid = []
        for amp, dur, pol in itertools.product(amplitudes_uA, durations_us, polarities):
            for elec_idx in electrode_indices:
                grid.append(
                    (elec_idx, electrode_trigger_map[elec_idx], amp, dur, pol)
                )
        return grid

    # ------------------------------------------------------------------
    # Hardware
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        log.info("Connecting to NeuroPlatform hardware ...")
        self._exp = Experiment(self.token)
        self._intan = IntanSofware()
        self._trigger_gen = Trigger()
        log.info("Hardware connected.")

    # ------------------------------------------------------------------
    # Core sweep loop
    # ------------------------------------------------------------------

    def _sweep(self, grid: List[tuple], inter_stim_delay_s: float) -> None:
        if not self.dry_run:
            if not self._exp.start():
                raise RuntimeError(
                    "Could not start experiment — another may already be running."
                )
            log.info("Experiment started.")

        # Group by (amp, dur, pol) to minimise 10-second Intan round-trips
        groups: dict = {}
        for elec_idx, trig_key, amp, dur, pol in grid:
            groups.setdefault((amp, dur, pol), []).append((elec_idx, trig_key))

        total = len(groups)
        for group_num, ((amp, dur, pol), pairs) in enumerate(groups.items(), 1):
            log.info(
                "Group %d/%d  amp=%.1f uA  dur=%.0f us  polarity=%s  electrodes=%s",
                group_num, total, amp, dur, pol, [e for e, _ in pairs],
            )

            configs = [
                _StimConfiguration(elec_idx, trig_key, amp, dur, pol)
                for elec_idx, trig_key in pairs
            ]

            self._upload([sc.param for sc in configs])

            for sc in configs:
                self._fire(sc, amp, dur, pol)
                time.sleep(inter_stim_delay_s)

            self._upload([sc.disabled_copy() for sc in configs])

        log.info("All groups completed.")

    # ------------------------------------------------------------------
    # Intan helpers
    # ------------------------------------------------------------------

    def _upload(self, params: List[StimParam]) -> None:
        if self.dry_run:
            log.debug("[DRY-RUN] Would upload %d StimParam(s).", len(params))
            return
        log.debug("Uploading %d param(s) -> waiting %.0f s ...", len(params), self.param_upload_wait_s)
        self._intan.send_stimparam(params)
        time.sleep(self.param_upload_wait_s)

    def _fire(
        self,
        sc: _StimConfiguration,
        amp: float,
        dur: float,
        pol: StimPolarity,
    ) -> None:
        tag = self._tag_counter
        self._tag_counter += 1
        ts = datetime.now(tz=timezone.utc).isoformat()

        log.info(
            "  FIRE  electrode=%-3d  trigger=%-2d  amp=%5.1f uA  dur=%6.1f us  polarity=%s  tag=%d",
            sc.electrode_index, sc.trigger_key, amp, dur, pol, tag,
        )

        if not self.dry_run:
            self._intan.set_tag_trigger(tag)
            trigger_array = np.zeros(16, dtype=np.uint8)
            trigger_array[sc.trigger_key] = 1
            self._trigger_gen.send(trigger_array)

        self._writer.write(
            _SweepRecord(
                timestamp_utc=ts,
                electrode_index=sc.electrode_index,
                trigger_key=sc.trigger_key,
                amplitude_uA=amp,
                duration_us=dur,
                polarity=str(pol),
                tag=tag,
            )
        )

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def _teardown(self) -> None:
        log.info("Tearing down ...")

        if self._writer:
            self._writer.close()

        if self.dry_run:
            log.info("[DRY-RUN] No hardware to close.")
            return

        try:
            self._intan.var_threshold(True)
        except Exception as exc:
            log.warning("Could not re-enable variable threshold: %s", exc)
        try:
            self._trigger_gen.close()
        except Exception as exc:
            log.warning("TriggerGenerator close failed: %s", exc)
        try:
            self._intan.close()
        except Exception as exc:
            log.warning("Intan close failed: %s", exc)
        try:
            self._exp.stop()
        except Exception as exc:
            log.warning("Experiment stop failed: %s", exc)

        log.info("Hardware teardown complete.")


# ===========================================================================
# Datalore API usage reference
# ===========================================================================
#
# CLASS_PARAMETERS = '[
#     {"name": "token",               "value": "9T5KLS6T7X"},
#     {"name": "dry_run",             "value": false},
#     {"name": "max_amplitude_uA",    "value": 10.0},
#     {"name": "max_duration_us",     "value": 1000.0},
#     {"name": "param_upload_wait_s", "value": 12.0}
# ]'
#
# FUNCTION_PARAMETERS = '[
#     {"name": "electrode_indices",    "value": [0, 8, 16]},
#     {"name": "amplitudes_uA",        "value": [1.0, 2.0, 4.0]},
#     {"name": "durations_us",         "value": [50.0, 100.0, 200.0]},
#     {"name": "polarities",           "value": ["NegativeFirst", "PositiveFirst"]},
#     {"name": "inter_stim_delay_s",   "value": 2.0},
#     {"name": "output_csv",           "value": "sweep_results.csv"}
# ]'
#
# ===========================================================================
# Local / manual testing
# ===========================================================================
#
#   sweep = StimParameterSweep(token="9T5KLS6T7X", dry_run=True)
#   sweep.run(
#       electrode_indices=[0, 8],
#       amplitudes_uA=[1.0, 2.0],
#       durations_us=[100.0],
#       polarities=["NegativeFirst"],
#   )
#
# ===========================================================================
