"""
FinalSpark NeuroPlatform - Responsive Electrode Characterisation Experiment
===========================================================================
Objective:
    Stimulate each active electrode (identified via a prior parameter scan)
    100 times **per parameter set** and record spike activity, triggers, and
    stimulation metadata so that the response properties of the responsive
    electrode(s) can be characterised across all validated parameter combinations.

Usage:
    1. Replace the `token` argument in main() with your assigned experiment token.
    2. Replace the `booking_email` argument in main() with the email used for booking.
    3. Set `testing=True` to validate logic without sending any stimulations to the organoid.
    4. Adjust `inter_stim_delay_s` if required (minimum ~0.5 s recommended).
    5. Run: python responsive_electrode_experiment.py

How all 13 parameter sets are covered
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The scan results contain 13 (electrode, amplitude, duration) combinations across
3 electrode indices (0, 8, 16).  The experiment groups these into **stimulation
rounds** — each round holds one parameter set per electrode.  Within a round every
electrode shares the same trigger_key so all fire simultaneously (parallel trigger
optimisation from the NeuroPlatform docs).  Rounds are executed sequentially; each
one reloads the Intan with the new parameters (mandatory 10-second wait) then fires
100 simultaneous trigger pulses.

Example with this scan data:
    Electrode 0  has 6 parameter sets  -> round index 0-5
    Electrode 8  has 4 parameter sets  -> round index 0-3
    Electrode 16 has 3 parameter sets  -> round index 0-2
    Total rounds = max(6, 4, 3) = 6.
    In rounds 4-5 only electrode 0 fires (electrodes 8 and 16 exhausted their sets).
    In round 3 electrodes 0 and 8 fire together; electrode 16 is absent.

This yields 13 x 100 = 1 300 total pulses at up to 3x speedup vs sequential.

Author : (your name)
Date   : 2026-03-10
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from neuroplatform import (
    Database,
    Experiment,
    IntanSofware,
    StimParam,
    StimPolarity,
    TriggerController,
)

# ---------------------------------------------------------------------------
# Module-level logger -- the only permitted module-level object
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ReliableConnection:
    """Mirrors one entry from the parameter-scan output."""

    electrode_from: int
    electrode_to: int
    hits_k: int
    repeats_n: int
    amplitude: float   # uA
    duration: float    # us
    polarity: str      # "NegativeFirst" | "PositiveFirst"

    @classmethod
    def from_dict(cls, d: dict) -> "ReliableConnection":
        stim = d["stimulation"]
        return cls(
            electrode_from=d["electrode_from"],
            electrode_to=d["electrode_to"],
            hits_k=d["hits_k"],
            repeats_n=d["repeats_n"],
            amplitude=stim["amplitude"],
            duration=stim["duration"],
            polarity=stim["polarity"],
        )


@dataclass
class StimulationRecord:
    """One sent (or simulated, in testing mode) stimulation pulse."""

    round_index: int       # which parameter-set round this belongs to
    rep_index: int         # repetition number within the round (1-based)
    electrode: int
    amplitude: float
    duration: float
    polarity: str
    trigger_key: int
    trigger_tag: int
    timestamp_utc: str
    testing: bool


@dataclass
class ExperimentResults:
    """Container that holds everything collected during the experiment."""

    fs_name: str
    experiment_start_utc: str
    experiment_stop_utc: str
    testing: bool
    total_rounds: int
    stimulations: List[StimulationRecord] = field(default_factory=list)
    spike_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    triggers: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# StimulationRound
# ---------------------------------------------------------------------------

@dataclass
class StimulationRound:
    """
    One round of stimulation: a snapshot of one parameter set per electrode,
    all sharing the same trigger_key for simultaneous parallel firing.

    Attributes
    ----------
    round_index : int
        Zero-based index of this round within the full experiment.
    connections : dict[int, ReliableConnection]
        Maps electrode index -> connection params for electrodes active this round.
    trigger_key : int
        The single trigger key shared by all StimParams in this round.
    """

    round_index: int
    connections: Dict[int, ReliableConnection]
    trigger_key: int = 0  # all rounds reuse key 0 (params are reloaded each round)

    def build_stim_params(self) -> List[StimParam]:
        """Return one charge-balanced StimParam per active electrode this round."""
        stim_params: List[StimParam] = []
        for electrode, conn in self.connections.items():
            sp = StimParam()
            sp.enable = True
            sp.index = electrode
            sp.trigger_key = self.trigger_key
            sp.polarity = (
                StimPolarity.NegativeFirst
                if conn.polarity == "NegativeFirst"
                else StimPolarity.PositiveFirst
            )
            # Charge-balanced biphasic: phase 2 mirrors phase 1
            sp.phase_duration1 = conn.duration
            sp.phase_amplitude1 = conn.amplitude
            sp.phase_duration2 = conn.duration
            sp.phase_amplitude2 = conn.amplitude
            stim_params.append(sp)
            log.debug(
                "    Round %d  electrode=%d  trigger_key=%d  "
                "amp=%.2f uA  dur=%.1f us  pol=%s",
                self.round_index,
                electrode,
                self.trigger_key,
                conn.amplitude,
                conn.duration,
                conn.polarity,
            )
        return stim_params

    def build_trigger_array(self) -> np.ndarray:
        """Return the 16-element uint8 array that fires this round's trigger key."""
        arr = np.zeros(16, dtype=np.uint8)
        arr[self.trigger_key] = 1
        return arr


# ---------------------------------------------------------------------------
# StimulationPlan
# ---------------------------------------------------------------------------

class StimulationPlan:
    """
    Converts a flat list of ReliableConnections into an ordered sequence of
    StimulationRounds that covers every (electrode, amplitude, duration) pair.

    Grouping strategy
    ~~~~~~~~~~~~~~~~~
    Connections are grouped by electrode index, preserving their original order
    (which reflects ascending charge injection from the parameter scan).  The
    groups are then interleaved by position: round 0 takes the first entry from
    each electrode, round 1 takes the second, and so on.

    Within each round every active electrode shares ``trigger_key=0``, so a
    single trigger send fires all of them in parallel.  Because Intan params
    are reloaded between rounds anyway, reusing key 0 every round is safe.
    """

    SHARED_TRIGGER_KEY: int = 0

    def __init__(self, connections: List[ReliableConnection]) -> None:
        self._connections = connections
        self.rounds: List[StimulationRound] = []
        self._build()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """Build the ordered list of StimulationRounds from the connections."""
        # Group connections by electrode, preserving scan order within each group
        by_electrode: Dict[int, List[ReliableConnection]] = {}
        for conn in self._connections:
            by_electrode.setdefault(conn.electrode_from, []).append(conn)

        n_rounds = max(len(v) for v in by_electrode.values())
        self.rounds = []

        for round_idx in range(n_rounds):
            round_connections: Dict[int, ReliableConnection] = {}
            for electrode, conn_list in by_electrode.items():
                if round_idx < len(conn_list):
                    round_connections[electrode] = conn_list[round_idx]
            self.rounds.append(
                StimulationRound(
                    round_index=round_idx,
                    connections=round_connections,
                    trigger_key=self.SHARED_TRIGGER_KEY,
                )
            )

        total_pairs = sum(len(r.connections) for r in self.rounds)
        log.info(
            "Stimulation plan: %d round(s) covering %d (electrode, param) pair(s) "
            "across %d electrode(s)  [trigger_key=%d for all rounds]",
            len(self.rounds),
            total_pairs,
            len(by_electrode),
            self.SHARED_TRIGGER_KEY,
        )
        for r in self.rounds:
            electrodes_this_round = {
                e: f"{c.amplitude}uA/{c.duration}us"
                for e, c in r.connections.items()
            }
            log.info("  Round %d: %s", r.round_index, electrodes_this_round)


# ---------------------------------------------------------------------------
# DataSaver
# ---------------------------------------------------------------------------

class DataSaver:
    """Handles persistence of stimulation records, spike events, and triggers."""

    def __init__(self, output_dir: Path, fs_name: str) -> None:
        self._dir = output_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._prefix = self._dir / f"{fs_name}_{timestamp}"

    def save_stimulation_log(self, stimulations: List[StimulationRecord]) -> Path:
        """Persist the list of sent (or simulated) stimulations as JSON."""
        path = Path(f"{self._prefix}_stimulations.json")
        records = [asdict(s) for s in stimulations]
        path.write_text(json.dumps(records, indent=2))
        log.info("Stimulation log saved -> %s  (%d records)", path, len(records))
        return path

    def save_spike_events(self, df: pd.DataFrame) -> Path:
        """Persist spike events DataFrame as CSV."""
        path = Path(f"{self._prefix}_spike_events.csv")
        df.to_csv(path, index=False)
        log.info("Spike events saved -> %s  (%d rows)", path, len(df))
        return path

    def save_triggers(self, df: pd.DataFrame) -> Path:
        """Persist triggers DataFrame as CSV."""
        path = Path(f"{self._prefix}_triggers.csv")
        df.to_csv(path, index=False)
        log.info("Triggers saved -> %s  (%d rows)", path, len(df))
        return path

    def save_summary(self, results: ExperimentResults) -> Path:
        """Persist a high-level experiment summary as JSON."""
        path = Path(f"{self._prefix}_summary.json")
        summary = {
            "fs_name": results.fs_name,
            "experiment_start_utc": results.experiment_start_utc,
            "experiment_stop_utc": results.experiment_stop_utc,
            "testing": results.testing,
            "total_rounds": results.total_rounds,
            "total_stimulations": len(results.stimulations),
            "total_spike_events": len(results.spike_events),
            "total_triggers": len(results.triggers),
        }
        path.write_text(json.dumps(summary, indent=2))
        log.info("Summary saved -> %s", path)
        return path

    def save_timing_log(self, timing_records: list) -> Path:
        """
        Persist per-rep hardware call timings as CSV.

        Columns
        -------
        round_index         : stimulation round (0-based)
        rep_index           : repetition within the round (1-based)
        tag                 : trigger tag sent to Intan
        set_tag_trigger_ms  : wall-clock time for intan.set_tag_trigger() in ms
        trigger_send_ms     : wall-clock time for trigger_ctrl.send() in ms
        sleep_ms            : actual time spent in time.sleep() in ms
        total_rep_ms        : total wall-clock time for the full rep in ms
        """
        path = Path(f"{self._prefix}_timing.csv")
        df = pd.DataFrame(timing_records)
        df.to_csv(path, index=False)
        log.info("Timing log saved -> %s  (%d rows)", path, len(df))
        return path

    def save_spike_waveforms(self, waveform_records: list) -> Path:
        """
        Persist per-spike raw waveforms for every spike detected within the
        response window after each database trigger ``up`` event.

        File format
        -----------
        JSON array.  Each element describes one spike and contains:

        ``trigger_tag``
            Integer tag identifying which stimulation pulse opened this response
            window.  Taken directly from the DB trigger record so that hardware
            timing -- not Python code timing -- anchors all latency measurements.

        ``trigger_time_utc``
            ISO-8601 UTC timestamp of the trigger ``up`` event from the DB.

        ``spike_time_utc``
            ISO-8601 UTC timestamp of the spike event from the DB.

        ``time_post_trigger_ms``
            Latency from trigger ``up`` to spike in milliseconds (float, 4 d.p.).

        ``channel``
            Recording electrode that detected this spike.

        ``peak_amplitude_uv``
            Maximum absolute voltage across the 3 ms waveform window (uV).

        ``n_samples``
            Number of samples returned by ``db.get_raw_spike()`` -- normally 90
            (30 kHz x 3 ms); fewer if the DB returned a truncated record.

        ``waveform_samples``
            List of floats -- the raw amplitude samples (uV) spanning 1 ms before
            to 2 ms after the spike peak, as returned by ``db.get_raw_spike()``.

        ``waveform_time_ms``
            Corresponding time axis in ms (-1.0 to +2.0, length == n_samples).
        """
        path = Path(f"{self._prefix}_spike_waveforms.json")
        path.write_text(json.dumps(waveform_records, indent=2))
        log.info(
            "Spike waveforms saved -> %s  (%d spike(s))", path, len(waveform_records)
        )
        return path


# ---------------------------------------------------------------------------
# ResponsiveElectrodeExperiment  (main class)
# ---------------------------------------------------------------------------

class ResponsiveElectrodeExperiment:
    """
    Stimulates each active electrode 100 times **per parameter set** and collects
    the resulting neural activity for post-hoc characterisation.

    All 13 (electrode, amplitude, duration) pairs from the scan results are tested.
    The experiment is structured as sequential rounds; within each round all active
    electrodes fire simultaneously via the parallel trigger optimisation.

    Parameters
    ----------
    token : str
        NeuroPlatform experiment token.
    booking_email : str
        Email address used when booking the session. Stored as
        ``self.booking_email`` and passed to ``TriggerController``.
    scan_results : dict
        Raw output from the parameter scan (the ``reliable_connections`` dict).
    stimulations_per_param_set : int
        Number of trigger pulses per (electrode, parameter-set) pair (default 100).
    inter_stim_delay_s : float
        Seconds to wait between consecutive stimulation pulses (default 1.0).
    param_send_wait_s : float
        Mandatory pause after ``send_stimparam()`` for hardware propagation
        (default 10.0).
    spike_window_post_stim_s : float
        Extra seconds added to the DB query window after the experiment stops
        (default 5.0).
    response_window_ms : float
        Width of the per-trigger response window used when extracting spike
        waveforms (default 100.0 ms).  Only spikes whose DB timestamp falls
        within [trigger_up_time, trigger_up_time + response_window_ms) are
        included in the waveform file.
    output_dir : str
        Directory string where all result files are written (default ``./results``).
    testing : bool
        When ``True``, no hardware connections are made and no triggers are sent.
        All logic, logging, and file saving still runs normally so the full
        pipeline can be validated without a booking slot.
    """

    def __init__(
        self,
        token: str,
        booking_email: str,
        scan_results: dict,
        stimulations_per_param_set: int = 100,
        inter_stim_delay_s: float = 1.0,
        param_send_wait_s: float = 10.0,
        spike_window_post_stim_s: float = 5.0,
        response_window_ms: float = 100.0,
        output_dir: str = "./results",
        testing: bool = False,
    ) -> None:
        self._token = token
        self.booking_email = booking_email
        self._n_stims = stimulations_per_param_set
        self._delay = inter_stim_delay_s
        self._param_send_wait = param_send_wait_s
        self._spike_window = spike_window_post_stim_s
        self._response_window_ms = response_window_ms
        self._output_dir = Path(output_dir)
        self._testing = testing

        connections = [
            ReliableConnection.from_dict(c)
            for c in scan_results.get("reliable_connections", [])
        ]
        if not connections:
            raise ValueError("No reliable_connections found in scan_results.")

        self._plan = StimulationPlan(connections)
        self._stimulation_log: List[StimulationRecord] = []
        self._timing_log: List[dict] = []   # per-rep hardware call timings
        self._results: Optional[ExperimentResults] = None

        # Hardware handles -- assigned inside _connect(); None until then
        self._exp: Optional[Experiment] = None
        self._intan: Optional[IntanSofware] = None
        self._trigger_ctrl: Optional[TriggerController] = None
        self._db: Optional[Database] = None

        if self._testing:
            log.warning(
                "TESTING MODE ENABLED -- Intan and TriggerController will NOT be "
                "connected and no stimulations will be delivered to the organoid."
            )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> ExperimentResults:
        """
        Execute all stimulation rounds and return the collected results.

        Each round:
          1. Builds StimParams for the active electrodes in that round.
          2. Uploads params to Intan (10-second wait).
          3. Fires ``stimulations_per_param_set`` parallel trigger pulses.
          4. Disables params before moving to the next round.
        """
        self._connect()
        self._db = Database()

        experiment_start: Optional[datetime] = None
        experiment_stop: Optional[datetime] = None

        try:
            if not self._testing:
                if not self._exp.start():
                    raise RuntimeError(
                        "Could not start experiment -- is another session already running?"
                    )

            experiment_start = datetime.now(timezone.utc)
            log.info("Experiment start (UTC): %s", experiment_start.isoformat())

            self._run_all_rounds()

            experiment_stop = datetime.now(timezone.utc)
            log.info("Experiment stop  (UTC): %s", experiment_stop.isoformat())

        finally:
            self._safe_close()

        fs_name = self._exp.exp_name if self._exp is not None else "unknown"

        spike_df, trigger_df = self._fetch_database_results(
            fs_name=fs_name,
            start=experiment_start,
            stop=experiment_stop,
        )

        self._results = ExperimentResults(
            fs_name=fs_name,
            experiment_start_utc=experiment_start.isoformat() if experiment_start else "",
            experiment_stop_utc=experiment_stop.isoformat() if experiment_stop else "",
            testing=self._testing,
            total_rounds=len(self._plan.rounds),
            stimulations=self._stimulation_log,
            spike_events=spike_df,
            triggers=trigger_df,
        )

        self._save_all()
        return self._results

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """
        Instantiate all hardware connections.

        When ``testing=True`` the Intan and TriggerController are skipped so
        the code can run outside of a booking slot without errors.  The
        Experiment object is always created (it is read-only until ``.start()``
        is called) so ``fs_name`` and ``electrodes`` are always available.
        """
        if self._testing:
            log.info(
                "[TESTING] Skipping IntanSofware and TriggerController connections."
            )
            return

        log.info("Connecting to IntanSofware...")
        self._intan = IntanSofware()

        log.info(
            "Connecting to TriggerController (booking_email=%s)...",
            self.booking_email,
        )
        self._trigger_ctrl = TriggerController(self.booking_email)

        log.info("Creating Experiment (token=%s)...", self._token)
        self._exp = Experiment(self._token)
        log.info(
            "FS name: %s  |  booking_email: %s  |  testing: %s  |  electrodes: %s",
            self._exp.exp_name,
            self.booking_email,
            self._testing,
            self._exp.electrodes,
        )

    # ------------------------------------------------------------------
    # Round execution
    # ------------------------------------------------------------------

    def _run_all_rounds(self) -> None:
        """
        Iterate over every StimulationRound, uploading fresh parameters to the
        Intan before each one and firing all electrodes in parallel.
        """
        n_rounds = len(self._plan.rounds)
        total_pairs = sum(len(r.connections) for r in self._plan.rounds)
        log.info(
            "Running %d round(s)  |  %d total (electrode, param) pair(s)  "
            "|  %d reps each  ->  %d total pulses",
            n_rounds,
            total_pairs,
            self._n_stims,
            total_pairs * self._n_stims,
        )

        for stim_round in self._plan.rounds:
            self._run_single_round(stim_round)

    def _run_single_round(self, stim_round: StimulationRound) -> None:
        """
        Execute one stimulation round:
          1. Log round summary.
          2. Upload StimParams for this round's active electrodes.
          3. Fire ``_n_stims`` simultaneous trigger pulses.
          4. Disable all params for this round.
        """
        electrode_summary = {
            e: f"{c.amplitude}uA/{c.duration}us"
            for e, c in stim_round.connections.items()
        }
        log.info(
            "--- Round %d / %d  |  electrodes: %s ---",
            stim_round.round_index + 1,
            len(self._plan.rounds),
            electrode_summary,
        )

        stim_params = stim_round.build_stim_params()
        self._send_stim_params(stim_params)

        trigger_array = stim_round.build_trigger_array()
        self._fire_round(stim_round, trigger_array)

        self._disable_stim_params(stim_params, stim_round.round_index)

    def _fire_round(
        self, stim_round: StimulationRound, trigger_array: np.ndarray
    ) -> None:
        """
        Send ``_n_stims`` trigger pulses for a single round, logging one
        StimulationRecord per electrode per repetition.

        Timing instrumentation
        ----------------------
        Each rep measures the wall-clock cost of:
          - ``set_tag_trigger()``
          - ``trigger_ctrl.send()``
          - ``time.sleep()``
          - total rep duration
        A summary (mean ± std, min, max) is logged at INFO level at the end of
        the round so the breakdown is always visible in the experiment log without
        needing to set DEBUG level.
        """
        electrode_list = list(stim_round.connections.keys())
        global_tag_base = stim_round.round_index * self._n_stims

        # Per-rep timing accumulators (only populated in live mode)
        t_tag_ms:   List[float] = []
        t_send_ms:  List[float] = []
        t_sleep_ms: List[float] = []
        t_total_ms: List[float] = []

        for rep in range(self._n_stims):
            t_rep_start = time.monotonic()
            ts = datetime.now(timezone.utc)
            tag = global_tag_base + rep + 1

            if self._testing:
                log.debug(
                    "  [TESTING] round=%d  rep=%04d/%04d  electrodes=%s  ts=%s",
                    stim_round.round_index,
                    rep + 1,
                    self._n_stims,
                    electrode_list,
                    ts.isoformat(),
                )
            else:
                t0 = time.monotonic()
                self._intan.set_tag_trigger(tag)
                t_tag = (time.monotonic() - t0) * 1000.0

                t0 = time.monotonic()
                self._trigger_ctrl.send(trigger_array)
                t_send = (time.monotonic() - t0) * 1000.0

                t_tag_ms.append(t_tag)
                t_send_ms.append(t_send)
                self._timing_log.append({
                    "round_index": stim_round.round_index,
                    "rep_index":   rep + 1,
                    "tag":         tag,
                    "set_tag_trigger_ms": round(t_tag, 3),
                    "trigger_send_ms":    round(t_send, 3),
                })

                log.debug(
                    "  round=%d  rep=%04d/%04d  electrodes=%s  tag=%d  ts=%s  "
                    "set_tag=%.1fms  send=%.1fms",
                    stim_round.round_index,
                    rep + 1,
                    self._n_stims,
                    electrode_list,
                    tag,
                    ts.isoformat(),
                    t_tag,
                    t_send,
                )

            for electrode in electrode_list:
                conn = stim_round.connections[electrode]
                self._stimulation_log.append(
                    StimulationRecord(
                        round_index=stim_round.round_index,
                        rep_index=rep + 1,
                        electrode=electrode,
                        amplitude=conn.amplitude,
                        duration=conn.duration,
                        polarity=conn.polarity,
                        trigger_key=stim_round.trigger_key,
                        trigger_tag=tag,
                        timestamp_utc=ts.isoformat(),
                        testing=self._testing,
                    )
                )

            t0 = time.monotonic()
            time.sleep(self._delay)
            t_sleep_ms.append((time.monotonic() - t0) * 1000.0)
            t_total_ms.append((time.monotonic() - t_rep_start) * 1000.0)

            # Back-fill sleep and total into the timing row written above
            if not self._testing and self._timing_log:
                self._timing_log[-1]["sleep_ms"]       = round(t_sleep_ms[-1], 3)
                self._timing_log[-1]["total_rep_ms"]   = round(t_total_ms[-1], 3)

        # ------------------------------------------------------------------
        # Round timing summary — logged at INFO so it always appears
        # ------------------------------------------------------------------
        if not self._testing and t_tag_ms:
            def _fmt(vals: List[float]) -> str:
                mean = sum(vals) / len(vals)
                std = (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5
                return f"{mean:7.1f} ± {std:5.1f} ms  (min {min(vals):.1f}  max {max(vals):.1f})"

            log.info(
                "  Round %d timing summary (%d reps):\n"
                "    set_tag_trigger : %s\n"
                "    trigger.send    : %s\n"
                "    time.sleep      : %s\n"
                "    total per rep   : %s",
                stim_round.round_index,
                len(t_tag_ms),
                _fmt(t_tag_ms),
                _fmt(t_send_ms),
                _fmt(t_sleep_ms),
                _fmt(t_total_ms),
            )

        log.info(
            "  Round %d complete: %d reps x %d electrode(s)%s",
            stim_round.round_index,
            self._n_stims,
            len(electrode_list),
            " [TESTING]" if self._testing else "",
        )

    # ------------------------------------------------------------------
    # Stimulation helpers
    # ------------------------------------------------------------------

    def _send_stim_params(self, stim_params: List[StimParam]) -> None:
        """Upload StimParams to the Intan. Skipped in testing mode."""
        if self._testing:
            log.info(
                "[TESTING] Skipping send_stimparam() for %d param(s).",
                len(stim_params),
            )
            return

        log.info(
            "Sending %d StimParam(s) to Intan (wait %ds)...",
            len(stim_params),
            self._param_send_wait,
        )
        self._intan.send_stimparam(stim_params)
        time.sleep(self._param_send_wait)

    def _disable_stim_params(
        self, stim_params: List[StimParam], round_index: int
    ) -> None:
        """Disable all StimParams and re-upload to Intan. Skipped in testing mode."""
        if self._testing:
            log.info("[TESTING] Skipping disable_stim_params() for round %d.", round_index)
            return

        log.info(
            "Disabling StimParams for round %d (wait %ds)...",
            round_index,
            self._param_send_wait,
        )
        for sp in stim_params:
            sp.enable = False
        self._intan.send_stimparam(stim_params)
        time.sleep(self._param_send_wait)

    # ------------------------------------------------------------------
    # Database schema normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_trigger_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise the trigger DataFrame returned by ``db.get_all_triggers()``
        to a consistent internal schema regardless of which DB version is running.

        Two schemas have been observed in the wild:

        Documented schema   : Time (datetime), trigger (int), status (str "up"/"down"), tag (float)
        Observed real schema: _time (datetime), _value (float), trigger (int), up (int 1/0)

        Output always has: Time (UTC datetime64), trigger (int), status (str), tag (float)
        """
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]

        if "_time" in df.columns and "up" in df.columns:
            # Real DB schema: rename and convert the int up/down flag to strings
            df = df.rename(columns={"_time": "Time", "_value": "tag"})
            df["status"] = df["up"].map({1: "up", 0: "down"})
            log.debug("Trigger schema: real (_time/up) -> normalised.")
        elif "Time" in df.columns and "status" in df.columns:
            # Documented schema: already correct
            if "tag" not in df.columns:
                df["tag"] = float("nan")
            log.debug("Trigger schema: documented (Time/status) -> no change needed.")
        else:
            log.warning(
                "Unrecognised trigger DataFrame schema; columns present: %s",
                list(df.columns),
            )

        df["Time"] = pd.to_datetime(df["Time"], utc=True)
        return df

    @staticmethod
    def _normalise_spike_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise the spike event DataFrame returned by ``db.get_spike_event()``
        to ensure the time column is called ``Time`` and is UTC-aware.
        """
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        for candidate in ("_time", "time", "timestamp"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "Time"})
                break
        df["Time"] = pd.to_datetime(df["Time"], utc=True)
        return df

    # ------------------------------------------------------------------
    # Database retrieval
    # ------------------------------------------------------------------

    def _fetch_database_results(
        self,
        fs_name: str,
        start: Optional[datetime],
        stop: Optional[datetime],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Retrieve spike events and trigger records for the full experiment window."""
        if start is None or stop is None:
            log.warning("Experiment times not set; skipping DB fetch.")
            return (
                pd.DataFrame(columns=["Time", "channel", "Amplitude"]),
                pd.DataFrame(columns=["Time", "trigger", "status", "tag"]),
            )

        buffered_stop = stop + timedelta(seconds=self._spike_window)
        log.info(
            "Fetching DB data for %s  [%s -> %s]%s",
            fs_name,
            start.isoformat(),
            buffered_stop.isoformat(),
            "  [TESTING - no live stimulations were sent]" if self._testing else "",
        )

        # FIX 3: Use typed empty DataFrames as fallbacks so that column names
        # are always present even when the DB call fails or returns no data.
        # This prevents KeyError crashes in _fetch_spike_waveforms when it
        # tries to access e.g. trigger_df["status"] on a schema-less DataFrame.
        try:
            spike_df = self._db.get_spike_event(start, buffered_stop, fs_name)
            spike_df = self._normalise_spike_df(spike_df)
            log.info("  Spike events retrieved: %d rows", len(spike_df))
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not retrieve spike events: %s", exc)
            spike_df = pd.DataFrame(columns=["Time", "channel", "Amplitude"])

        try:
            trigger_df = self._db.get_all_triggers(start, buffered_stop)
            trigger_df = self._normalise_trigger_df(trigger_df)
            log.info("  Triggers retrieved: %d rows", len(trigger_df))
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not retrieve triggers: %s", exc)
            trigger_df = pd.DataFrame(columns=["Time", "trigger", "status", "tag"])

        return spike_df, trigger_df

    # ------------------------------------------------------------------
    # Spike waveform extraction
    # ------------------------------------------------------------------

    def _fetch_spike_waveforms(
        self,
        fs_name: str,
        spike_df: pd.DataFrame,
        trigger_df: pd.DataFrame,
    ) -> list:
        """
        For every DB trigger ``up`` event, fetch the raw 3 ms waveform for each
        spike that occurred within ``_response_window_ms`` milliseconds of that
        trigger.

        Timing is anchored entirely to the database trigger timestamps -- not to
        the code-side ``StimulationRecord.timestamp_utc`` values -- so that any
        Python / OS scheduling jitter does not affect latency measurements.

        Strategy
        --------
        1.  Filter ``trigger_df`` to ``status == "up"`` rows only; these mark
            the exact hardware fire times recorded by the NeuroPlatform DB.
        2.  For each trigger ``up`` event, select all spike events whose DB
            ``Time`` falls in the half-open interval
            ``[trigger_time, trigger_time + response_window_ms)``.
        3.  For each matching spike, call ``db.get_raw_spike(t-1ms, t+2ms, ch)``
            to retrieve the standard NeuroPlatform 90-sample / 3 ms waveform.
        4.  Build and return one record dict per spike.

        Returns
        -------
        list of dict
            Each dict contains the fields documented in
            ``DataSaver.save_spike_waveforms``.
        """
        # --- guard: both DataFrames must be non-empty --------------------
        if spike_df.empty or trigger_df.empty:
            log.warning(
                "Spike waveform extraction skipped: "
                "spike_df empty=%s  trigger_df empty=%s",
                spike_df.empty,
                trigger_df.empty,
            )
            return []

        # FIX 1 & 2: Schema is already normalised by _normalise_trigger_df() in
        # _fetch_database_results, so Time/status/tag are always present here.
        # This guard is a last-resort safety net only.
        trigger_df = trigger_df.copy()
        if "status" not in trigger_df.columns:
            log.warning(
                "trigger_df is missing the 'status' column after normalisation. "
                "Available columns: %s  -- skipping waveform extraction.",
                list(trigger_df.columns),
            )
            return []

        # Keep only trigger "up" events -- the moment each pulse fires.
        # Columns from db.get_all_triggers(): Time, trigger, status, tag
        up_triggers = trigger_df[trigger_df["status"] == "up"].copy()
        if up_triggers.empty:
            log.warning(
                "No trigger 'up' events found in trigger_df; "
                "skipping waveform extraction."
            )
            return []

        log.info(
            "Extracting spike waveforms: %d trigger up-event(s)  |  "
            "%d total spike event(s)  |  response window=%.1f ms",
            len(up_triggers),
            len(spike_df),
            self._response_window_ms,
        )

        response_window_td = timedelta(milliseconds=self._response_window_ms)

        # Standard NeuroPlatform time axis: -1 ms to +2 ms at 30 kHz (90 samples)
        _standard_time_axis_ms: List[float] = list(np.linspace(-1.0, 2.0, 90))

        waveform_records: list = []
        skipped_waveforms: int = 0

        for _, trig_row in up_triggers.iterrows():
            trig_time: datetime = trig_row["Time"]
            # Guard against missing tag (None / NaN) -- use -1 as sentinel
            trig_tag: int = (
                int(trig_row["tag"])
                if pd.notna(trig_row.get("tag"))
                else -1
            )
            window_end: datetime = trig_time + response_window_td

            # All spikes within the 100 ms response window for this trigger
            mask = (spike_df["Time"] >= trig_time) & (spike_df["Time"] < window_end)
            window_spikes = spike_df[mask]

            for _, spike_row in window_spikes.iterrows():
                spike_time: datetime = spike_row["Time"]
                channel: int = int(spike_row["channel"])
                latency_ms: float = (
                    (spike_time - trig_time).total_seconds() * 1_000.0
                )

                # Fetch raw waveform: 1 ms pre-spike to 2 ms post-spike.
                # This matches the NeuroPlatform 3 ms / 90-sample convention
                # documented in db.get_raw_spike().
                raw_t1 = spike_time - timedelta(milliseconds=1)
                raw_t2 = spike_time + timedelta(milliseconds=2)

                try:
                    raw_df = self._db.get_raw_spike(raw_t1, raw_t2, channel)
                    if raw_df is None or raw_df.empty:
                        log.debug(
                            "  Empty raw waveform for spike at %s ch=%d; skipping.",
                            spike_time.isoformat(),
                            channel,
                        )
                        skipped_waveforms += 1
                        continue

                    # raw_df has an "Amplitude" column containing uV values
                    samples: List[float] = raw_df["Amplitude"].tolist()
                    n_samples: int = len(samples)

                    # Rebuild time axis if the DB returned fewer than 90 samples
                    time_axis_ms: List[float] = (
                        _standard_time_axis_ms
                        if n_samples == 90
                        else list(np.linspace(-1.0, 2.0, n_samples))
                    )
                    peak_amplitude_uv: float = float(
                        raw_df["Amplitude"].abs().max()
                    )

                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "  Could not fetch raw waveform for spike at %s ch=%d: %s",
                        spike_time.isoformat(),
                        channel,
                        exc,
                    )
                    skipped_waveforms += 1
                    continue

                waveform_records.append(
                    {
                        "trigger_tag": trig_tag,
                        "trigger_time_utc": trig_time.isoformat(),
                        "spike_time_utc": spike_time.isoformat(),
                        "time_post_trigger_ms": round(latency_ms, 4),
                        "channel": channel,
                        "peak_amplitude_uv": round(peak_amplitude_uv, 4),
                        "n_samples": n_samples,
                        "waveform_samples": [round(v, 4) for v in samples],
                        "waveform_time_ms": [round(t, 6) for t in time_axis_ms],
                    }
                )

        log.info(
            "Waveform extraction complete: %d record(s) collected  |  %d skipped",
            len(waveform_records),
            skipped_waveforms,
        )
        return waveform_records

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_all(self) -> None:
        """
        Persist all collected data to disk via DataSaver.

        Files written (all share the same ``{fs_name}_{timestamp}`` prefix):

        ``_stimulations.json``
            One record per trigger pulse sent (or simulated in testing mode).

        ``_spike_events.csv``
            All spike events returned by the DB for the full experiment window.

        ``_triggers.csv``
            All trigger records (up + down) returned by the DB.

        ``_summary.json``
            High-level counts and experiment metadata.

        ``_spike_waveforms.json``
            Per-spike raw waveforms for every spike detected within
            ``response_window_ms`` of each DB trigger ``up`` event.  Latency
            is computed from the DB trigger timestamp so that hardware timing --
            not Python code timing -- anchors every measurement.
        """
        if self._results is None:
            return

        saver = DataSaver(self._output_dir, self._results.fs_name)
        saver.save_stimulation_log(self._results.stimulations)
        saver.save_spike_events(self._results.spike_events)
        saver.save_triggers(self._results.triggers)
        saver.save_summary(self._results)
        saver.save_timing_log(self._timing_log)

        # Spike waveform extraction -----------------------------------------
        # Uses the DB trigger "up" timestamps as the timing anchor so that
        # response latencies are not contaminated by Python scheduling jitter.
        # For every trigger up-event, each spike within the response window
        # has its full 3 ms / 90-sample raw waveform fetched from the DB and
        # written to a dedicated JSON file for artifact inspection and waveform
        # shape characterisation.
        waveform_records = self._fetch_spike_waveforms(
            fs_name=self._results.fs_name,
            spike_df=self._results.spike_events,
            trigger_df=self._results.triggers,
        )
        saver.save_spike_waveforms(waveform_records)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _safe_close(self) -> None:
        """Close only the connections that were actually opened."""
        if self._trigger_ctrl is not None:
            try:
                self._trigger_ctrl.close()
                log.info("TriggerController closed.")
            except Exception as exc:  # noqa: BLE001
                log.warning("Error closing TriggerController: %s", exc)

        if self._intan is not None:
            try:
                self._intan.close()
                log.info("IntanSoftware closed.")
            except Exception as exc:  # noqa: BLE001
                log.warning("Error closing IntanSoftware: %s", exc)

        if self._exp is not None:
            try:
                self._exp.stop()
                log.info("Experiment stopped.")
            except Exception as exc:  # noqa: BLE001
                log.warning("Error stopping experiment: %s", exc)
