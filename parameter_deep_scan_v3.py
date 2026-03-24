"""
FinalSpark NeuroPlatform – Responsive Electrode Characterisation Experiment
===========================================================================
Objective:
    Stimulate each active electrode (identified via a prior parameter scan)
    N times **per parameter set** and record spike activity, triggers, and
    stimulation metadata so that the response properties of the responsive
    electrode(s) can be characterised across all validated parameter combinations.

Usage:
    Instantiated remotely by the Datalore Claude Runner notebook.
    Accepts the same flat keyword-argument interface that the notebook runner
    passes via CLASS_PARAMETERS.

Author : Wiktor
Date   : 2026-03-24
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
    StimParam,
    StimPolarity,
    TriggerController,
)

try:
    from neuroplatform import IntanSofware as IntanSoftware
except ImportError:
    try:
        from neuroplatform import IntanSoftware as IntanSoftware
    except ImportError:
        raise ImportError("No IntanSoftware or IntanSofware found in neuroplatform.")


# ---------------------------------------------------------------------------
# Logging setup – writes all actions to logs.txt and also prints to console.
# ---------------------------------------------------------------------------
logger = logging.getLogger("DeepScan")
logger.setLevel(logging.DEBUG)

_file_handler = logging.FileHandler("logs.txt", mode="w")
_file_handler.setLevel(logging.DEBUG)

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)

_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
)
_file_handler.setFormatter(_formatter)
_console_handler.setFormatter(_formatter)

logger.addHandler(_file_handler)
logger.addHandler(_console_handler)


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

    round_index: int
    rep_index: int
    electrode: int
    amplitude: float
    duration: float
    polarity: str
    trigger_key: int
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
    """One round of stimulation: one parameter set per electrode, all sharing
    the same trigger_key for simultaneous parallel firing."""

    round_index: int
    connections: Dict[int, ReliableConnection]
    trigger_key: int = 0

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
            logger.debug(
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
    """Converts a flat list of ReliableConnections into an ordered sequence of
    StimulationRounds covering every (electrode, amplitude, duration) pair.

    Grouping: connections are grouped by electrode index, then interleaved by
    position so that round 0 takes the first entry from each electrode, round 1
    takes the second, etc.  Within each round every active electrode shares
    ``trigger_key=0`` for parallel firing.
    """

    SHARED_TRIGGER_KEY: int = 0

    def __init__(self, connections: List[ReliableConnection]) -> None:
        self._connections = connections
        self.rounds: List[StimulationRound] = []
        self._build()

    def _build(self) -> None:
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
        logger.info(
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
            logger.info("  Round %d: %s", r.round_index, electrodes_this_round)


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
        path = Path(f"{self._prefix}_stimulations.json")
        records = [asdict(s) for s in stimulations]
        path.write_text(json.dumps(records, indent=2))
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

    def save_summary(self, results: ExperimentResults) -> Path:
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
        logger.info("Saved summary -> %s", path)
        return path

    def save_spike_waveforms(self, waveform_records: list) -> Path:
        path = Path(f"{self._prefix}_spike_waveforms.json")
        path.write_text(json.dumps(waveform_records, indent=2))
        logger.info("Saved spike waveforms -> %s  (%d spike(s))", path, len(waveform_records))
        return path


# ---------------------------------------------------------------------------
# Lightweight mocks used when testing=True so that no hardware is contacted.
# ---------------------------------------------------------------------------

class _MockIntan:
    """Drop-in replacement for IntanSoftware in testing mode."""

    def send_stimparam(self, params):
        logger.debug("[TESTING] MockIntan.send_stimparam(%d params) — skipped.", len(params))

    def var_threshold(self, enabled: bool):
        logger.debug("[TESTING] MockIntan.var_threshold(%s) — skipped.", enabled)

    def close(self):
        logger.debug("[TESTING] MockIntan.close() — skipped.")


class _MockTriggerCtrl:
    """Drop-in replacement for TriggerController in testing mode."""

    def send(self, triggers):
        logger.debug("[TESTING] MockTriggerCtrl.send(%s) — skipped.", triggers.tolist())

    def close(self):
        logger.debug("[TESTING] MockTriggerCtrl.close() — skipped.")


class _MockExperiment:
    """Drop-in replacement for Experiment in testing mode."""

    def __init__(self, exp_name: str, electrodes: list):
        self.exp_name = exp_name
        self.electrodes = electrodes

    def start(self) -> bool:
        logger.info("[TESTING] MockExperiment.start() — returning True.")
        return True

    def stop(self):
        logger.info("[TESTING] MockExperiment.stop() — skipped.")


# ---------------------------------------------------------------------------
# ResponsiveElectrodeExperiment  (main class)
# ---------------------------------------------------------------------------

class ResponsiveElectrodeExperiment:
    """Stimulates each active electrode N times per parameter set and collects
    the resulting neural activity for post-hoc characterisation.

    Accepts flat keyword arguments from the Datalore notebook runner.
    """

    PARAM_SEND_WAIT_S = 10  # seconds Intan needs after send_stimparam()

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

        # ---- Build Experiment (or mock) from token ----
        if self._testing:
            logger.info("[TESTING] Mode enabled — hardware classes will NOT be instantiated.")
            self._exp = _MockExperiment(
                exp_name=f"test_{token}", electrodes=list(range(128))
            )
            self._intan = _MockIntan()
            self._trigger_ctrl = _MockTriggerCtrl()
        else:
            logger.info("Creating Experiment (token=%s)...", token)
            self._exp = Experiment(token)
            logger.info("Connecting to IntanSoftware...")
            self._intan = IntanSoftware()
            logger.info("Connecting to TriggerController (booking_email=%s)...", booking_email)
            self._trigger_ctrl = TriggerController(booking_email)

        self._db = Database()

        # ---- Parse reliable connections ----
        connections = [
            ReliableConnection.from_dict(c)
            for c in scan_results.get("reliable_connections", [])
        ]
        if not connections:
            raise ValueError("No reliable_connections found in scan_results.")

        self._plan = StimulationPlan(connections)
        self._stimulation_log: List[StimulationRecord] = []
        self._results: Optional[ExperimentResults] = None

        # ---- Initialisation summary ----
        total_pairs = sum(len(r.connections) for r in self._plan.rounds)
        unique_electrodes = set()
        for r in self._plan.rounds:
            unique_electrodes.update(r.connections.keys())

        logger.info("Initializing ResponsiveElectrodeExperiment (testing=%s)", testing)
        logger.info("  Experiment token: %s", self._exp.exp_name)
        logger.info("  Allowed electrodes: %s", self._exp.electrodes)
        logger.info("  Booking email: %s", booking_email)
        logger.info("  Stimulation rounds: %d", len(self._plan.rounds))
        logger.info("  Total (electrode, param) pairs: %d", total_pairs)
        logger.info("  Unique electrodes: %s", sorted(unique_electrodes))
        logger.info("  Stimulations per param set: %d", self._n_stims)
        logger.info("  Inter-stim delay: %.2f s", self._delay)
        logger.info("  Param send wait: %.1f s", self._param_send_wait)
        logger.info("  Spike window post-stim: %.1f s", self._spike_window)
        logger.info("  Response window: %.1f ms", self._response_window_ms)

        # ---- Expected execution duration (always based on real timing) ----
        n_rounds = len(self._plan.rounds)
        stim_time_per_round = self._n_stims * self._delay
        param_load_time = self._param_send_wait * 2 * n_rounds  # send + disable
        total_stim_time = stim_time_per_round * n_rounds
        expected_seconds = total_stim_time + param_load_time
        hours, remainder = divmod(int(expected_seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        logger.info(
            "  Expected execution duration (live): %dh %02dm %02ds  "
            "(%d rounds × %d reps × %.1fs delay + %d param loads × %.1fs)",
            hours, minutes, secs,
            n_rounds, self._n_stims, self._delay,
            n_rounds * 2, self._param_send_wait,
        )

        # ---- Validate inter_stim_delay ----
        if self._delay < 0.5:
            logger.warning(
                "inter_stim_delay_s=%.2f s is below the recommended minimum of 0.5 s. "
                "Stimulation fatigue may occur (see FinalSpark troubleshooting).",
                self._delay,
            )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> ExperimentResults:
        """Execute all stimulation rounds and return the collected results.

        Each round:
          1. Builds StimParams for the active electrodes in that round.
          2. Uploads params to Intan (wait for hardware propagation).
          3. Fires ``stimulations_per_param_set`` parallel trigger pulses.
          4. Disables params before moving to the next round.
        """
        logger.info("=" * 60)
        logger.info("STARTING DEEP SCAN%s", " [TESTING MODE]" if self._testing else "")
        logger.info("=" * 60)

        experiment_start: Optional[datetime] = None
        experiment_stop: Optional[datetime] = None

        try:
            if not self._exp.start():
                raise RuntimeError(
                    "Could not start experiment — is another session already running?"
                )

            experiment_start = datetime.now(timezone.utc)
            logger.info("Experiment started: %s", self._exp.exp_name)
            logger.info("Start time: %s", experiment_start.isoformat())

            # Disable variable threshold for stable event detection
            # (FinalSpark docs: variable threshold changes during bursts/stimulation)
            if not self._testing:
                logger.info(
                    "Disabling variable threshold for consistent spike detection "
                    "during the deep scan."
                )
            self._intan.var_threshold(False)

            self._run_all_rounds()

            experiment_stop = datetime.now(timezone.utc)
            logger.info("Experiment stop time: %s", experiment_stop.isoformat())

        finally:
            logger.info("Entering cleanup / finally block.")

            # Re-enable variable threshold (FinalSpark docs: ALWAYS re-enable)
            if not self._testing:
                logger.info("Re-enabling variable threshold.")
            try:
                self._intan.var_threshold(True)
            except Exception as exc:
                logger.error("Failed to re-enable variable threshold: %s", exc)

            # Close hardware connections
            logger.info("Closing TriggerController.")
            try:
                self._trigger_ctrl.close()
            except Exception as exc:
                logger.warning("Error closing TriggerController: %s", exc)

            logger.info("Closing Intan connection.")
            try:
                self._intan.close()
            except Exception as exc:
                logger.warning("Error closing IntanSoftware: %s", exc)

            logger.info("Stopping experiment.")
            try:
                self._exp.stop()
            except Exception as exc:
                logger.warning("Error stopping experiment: %s", exc)

            if experiment_start is not None and experiment_stop is not None:
                elapsed = experiment_stop - experiment_start
                logger.info("Total elapsed: %s", elapsed)

            total_stims = len(self._stimulation_log)
            logger.info("Total stimulations recorded: %d", total_stims)

        # ---- Fetch database results & save ----
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

        logger.info("=" * 60)
        logger.info("DEEP SCAN COMPLETE")
        logger.info("=" * 60)

        return self._results

    # ------------------------------------------------------------------
    # Round execution
    # ------------------------------------------------------------------

    def _run_all_rounds(self) -> None:
        n_rounds = len(self._plan.rounds)
        total_pairs = sum(len(r.connections) for r in self._plan.rounds)
        logger.info(
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
        electrode_summary = {
            e: f"{c.amplitude}uA/{c.duration}us"
            for e, c in stim_round.connections.items()
        }
        logger.info(
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
        electrode_list = list(stim_round.connections.keys())

        for rep in range(self._n_stims):
            ts = datetime.now(timezone.utc)

            if self._testing:
                logger.debug(
                    "  [TESTING] round=%d  rep=%04d/%04d  electrodes=%s  ts=%s",
                    stim_round.round_index,
                    rep + 1,
                    self._n_stims,
                    electrode_list,
                    ts.isoformat(),
                )
            else:
                self._trigger_ctrl.send(trigger_array)
                logger.debug(
                    "  round=%d  rep=%04d/%04d  electrodes=%s  ts=%s",
                    stim_round.round_index,
                    rep + 1,
                    self._n_stims,
                    electrode_list,
                    ts.isoformat(),
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
                        timestamp_utc=ts.isoformat(),
                        testing=self._testing,
                    )
                )

            if not self._testing:
                time.sleep(self._delay)

        logger.info(
            "  Round %d complete: %d reps × %d electrode(s)%s",
            stim_round.round_index,
            self._n_stims,
            len(electrode_list),
            " [TESTING]" if self._testing else "",
        )

    # ------------------------------------------------------------------
    # Stimulation helpers
    # ------------------------------------------------------------------

    def _send_stim_params(self, stim_params: List[StimParam]) -> None:
        if self._testing:
            logger.info(
                "[TESTING] Skipping send_stimparam() for %d param(s).",
                len(stim_params),
            )
            return

        logger.info(
            "Sending %d StimParam(s) to Intan (wait %ds)...",
            len(stim_params),
            self._param_send_wait,
        )
        self._intan.send_stimparam(stim_params)
        time.sleep(self._param_send_wait)

    def _disable_stim_params(
        self, stim_params: List[StimParam], round_index: int
    ) -> None:
        if self._testing:
            logger.info("[TESTING] Skipping disable_stim_params() for round %d.", round_index)
            return

        logger.info(
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
        """Normalise trigger DataFrame to consistent schema:
        Time (UTC datetime64), trigger (int), status (str), tag (float)."""
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]

        if "_time" in df.columns and "up" in df.columns:
            df = df.rename(columns={"_time": "Time", "_value": "tag"})
            try:
                up_int = df["up"].astype(float).astype(int)
            except (ValueError, TypeError):
                up_int = df["up"]
            df["status"] = up_int.map({1: "up", 0: "down"})
            n_unmapped = df["status"].isna().sum()
            if n_unmapped > 0:
                logger.warning(
                    "Trigger normalisation: %d row(s) had unexpected 'up' values. "
                    "Unique values seen: %s",
                    n_unmapped,
                    df["up"].unique().tolist(),
                )
            logger.info(
                "Trigger schema: real (_time/up) -> normalised.  "
                "up-events: %d  down-events: %d  unmapped: %d",
                (df["status"] == "up").sum(),
                (df["status"] == "down").sum(),
                n_unmapped,
            )
        elif "Time" in df.columns and "status" in df.columns:
            if "tag" not in df.columns:
                df["tag"] = float("nan")
            logger.info(
                "Trigger schema: documented (Time/status) -> no change needed.  "
                "up-events: %d  down-events: %d",
                (df["status"] == "up").sum(),
                (df["status"] == "down").sum(),
            )
        else:
            logger.warning(
                "Unrecognised trigger DataFrame schema; columns present: %s",
                list(df.columns),
            )

        df["Time"] = pd.to_datetime(df["Time"], utc=True)
        return df

    @staticmethod
    def _normalise_spike_df(df: pd.DataFrame) -> pd.DataFrame:
        """Normalise spike event DataFrame: ensure time column is ``Time`` and UTC-aware."""
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
            logger.warning("Experiment times not set; skipping DB fetch.")
            return (
                pd.DataFrame(columns=["Time", "channel", "Amplitude"]),
                pd.DataFrame(columns=["Time", "trigger", "status", "tag"]),
            )

        buffered_stop = stop + timedelta(seconds=self._spike_window)
        logger.info(
            "Fetching DB data for %s  [%s -> %s]%s",
            fs_name,
            start.isoformat(),
            buffered_stop.isoformat(),
            "  [TESTING — no live stimulations were sent]" if self._testing else "",
        )

        try:
            spike_df = self._db.get_spike_event(start, buffered_stop, fs_name)
            spike_df = self._normalise_spike_df(spike_df)
            logger.info("  Spike events retrieved: %d rows", len(spike_df))
        except Exception as exc:
            logger.warning("Could not retrieve spike events: %s", exc)
            spike_df = pd.DataFrame(columns=["Time", "channel", "Amplitude"])

        try:
            trigger_df = self._db.get_all_triggers(start, buffered_stop)
            trigger_df = self._normalise_trigger_df(trigger_df)
            logger.info("  Triggers retrieved: %d rows", len(trigger_df))
        except Exception as exc:
            logger.warning("Could not retrieve triggers: %s", exc)
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
        """For every DB trigger ``up`` event, fetch the raw 3 ms waveform for each
        spike that occurred within ``_response_window_ms`` milliseconds of that
        trigger.  Timing is anchored to DB trigger timestamps."""

        if spike_df.empty or trigger_df.empty:
            logger.warning(
                "Spike waveform extraction skipped: "
                "spike_df empty=%s  trigger_df empty=%s",
                spike_df.empty,
                trigger_df.empty,
            )
            return []

        trigger_df = trigger_df.copy()
        if "status" not in trigger_df.columns:
            logger.warning(
                "trigger_df is missing the 'status' column after normalisation. "
                "Available columns: %s  — skipping waveform extraction.",
                list(trigger_df.columns),
            )
            return []

        up_triggers = trigger_df[trigger_df["status"] == "up"].copy()
        if up_triggers.empty:
            logger.warning(
                "No trigger 'up' events found in trigger_df; "
                "skipping waveform extraction."
            )
            return []

        logger.info(
            "Extracting spike waveforms: %d trigger up-event(s)  |  "
            "%d total spike event(s)  |  response window=%.1f ms",
            len(up_triggers),
            len(spike_df),
            self._response_window_ms,
        )

        response_window_td = timedelta(milliseconds=self._response_window_ms)
        _standard_time_axis_ms: List[float] = list(np.linspace(-1.0, 2.0, 90))

        waveform_records: list = []
        skipped_waveforms: int = 0

        for _, trig_row in up_triggers.iterrows():
            trig_time: datetime = trig_row["Time"]
            trig_tag: int = (
                int(trig_row["tag"])
                if pd.notna(trig_row.get("tag"))
                else -1
            )
            window_end: datetime = trig_time + response_window_td

            mask = (spike_df["Time"] >= trig_time) & (spike_df["Time"] < window_end)
            window_spikes = spike_df[mask]

            for _, spike_row in window_spikes.iterrows():
                spike_time: datetime = spike_row["Time"]
                channel: int = int(spike_row["channel"])
                latency_ms: float = (
                    (spike_time - trig_time).total_seconds() * 1_000.0
                )

                raw_t1 = spike_time - timedelta(milliseconds=1)
                raw_t2 = spike_time + timedelta(milliseconds=2)

                try:
                    raw_df = self._db.get_raw_spike(raw_t1, raw_t2, channel)
                    if raw_df is None or raw_df.empty:
                        logger.debug(
                            "  Empty raw waveform for spike at %s ch=%d; skipping.",
                            spike_time.isoformat(),
                            channel,
                        )
                        skipped_waveforms += 1
                        continue

                    samples: List[float] = raw_df["Amplitude"].tolist()
                    n_samples: int = len(samples)
                    time_axis_ms: List[float] = (
                        _standard_time_axis_ms
                        if n_samples == 90
                        else list(np.linspace(-1.0, 2.0, n_samples))
                    )
                    peak_amplitude_uv: float = float(
                        raw_df["Amplitude"].abs().max()
                    )

                except Exception as exc:
                    logger.warning(
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

        logger.info(
            "Waveform extraction complete: %d record(s) collected  |  %d skipped",
            len(waveform_records),
            skipped_waveforms,
        )
        return waveform_records

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_all(self) -> None:
        if self._results is None:
            return

        saver = DataSaver(self._output_dir, self._results.fs_name)
        saver.save_stimulation_log(self._results.stimulations)
        saver.save_spike_events(self._results.spike_events)
        saver.save_triggers(self._results.triggers)
        saver.save_summary(self._results)

        waveform_records = self._fetch_spike_waveforms(
            fs_name=self._results.fs_name,
            spike_df=self._results.spike_events,
            trigger_df=self._results.triggers,
        )
        saver.save_spike_waveforms(waveform_records)
