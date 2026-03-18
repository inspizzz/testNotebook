from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

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

log = logging.getLogger(__name__)


@dataclass
class ReliableConnection:
    """Mirrors one entry from the parameter-scan output."""

    electrode_from: int
    electrode_to: int
    hits_k: int
    repeats_n: int
    amplitude: float
    duration: float
    polarity: str

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


@dataclass
class StimulationRound:
    """
    One round of stimulation: a snapshot of one parameter set per electrode,
    all sharing the same trigger_key for simultaneous parallel firing.
    """

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


class StimulationPlan:
    """
    Converts a flat list of ReliableConnections into an ordered sequence of
    StimulationRounds that covers every (electrode, amplitude, duration) pair.
    """

    SHARED_TRIGGER_KEY: int = 0

    def __init__(self, connections: List[ReliableConnection]) -> None:
        self._connections = connections
        self.rounds: List[StimulationRound] = []
        self._build()

    def _build(self) -> None:
        """Build the ordered list of StimulationRounds from the connections."""
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


class Experiment:
    """
    Main experiment class for initial validation runs on responsive electrodes.
    
    Stimulates each identified responsive electrode pair with optimal parameters
    from the prior parameter scan, collecting neural responses for characterization.
    """

    def __init__(
        self,
        token: str,
        booking_email: str,
        scan_results: dict,
        stimulations_per_param_set: int = 10,
        inter_stim_delay_s: float = 0.5,
        param_send_wait_s: float = 10.0,
        spike_window_post_stim_s: float = 5.0,
        output_dir: Path = Path("./results"),
        testing: bool = False,
    ) -> None:
        self._token = token
        self.booking_email = booking_email
        self._n_stims = min(stimulations_per_param_set, 50)
        self._delay = inter_stim_delay_s
        self._param_send_wait = param_send_wait_s
        self._spike_window = spike_window_post_stim_s
        self._output_dir = output_dir
        self._testing = testing

        connections = [
            ReliableConnection.from_dict(c)
            for c in scan_results.get("reliable_connections", [])
        ]
        if not connections:
            raise ValueError("No reliable_connections found in scan_results.")

        self._plan = StimulationPlan(connections)
        self._stimulation_log: List[StimulationRecord] = []
        self._results: Optional[ExperimentResults] = None

        self._exp: Optional[object] = None
        self._intan: Optional[IntanSofware] = None
        self._trigger_ctrl: Optional[TriggerController] = None
        self._db: Optional[Database] = None

        if self._testing:
            log.warning(
                "TESTING MODE ENABLED — Intan and TriggerController will NOT be "
                "connected and no stimulations will be delivered to the organoid."
            )

    def run(self) -> ExperimentResults:
        """
        Execute all stimulation rounds and return the collected results.
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )

        self._connect()
        self._db = Database()

        experiment_start: Optional[datetime] = None
        experiment_stop: Optional[datetime] = None

        try:
            if not self._testing:
                if not self._exp.start():
                    raise RuntimeError(
                        "Could not start experiment — is another session already running?"
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

    def _connect(self) -> None:
        """
        Instantiate all hardware connections.
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
        from neuroplatform import Experiment as NeuroPlatformExperiment
        self._exp = NeuroPlatformExperiment(self._token)
        log.info(
            "FS name: %s  |  booking_email: %s  |  testing: %s  |  electrodes: %s",
            self._exp.exp_name,
            self.booking_email,
            self._testing,
            self._exp.electrodes,
        )

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
          1. Build and upload StimParams to Intan.
          2. Fire all electrodes in parallel via trigger.
          3. Disable params before next round.
        """
        log.info(
            "=== Round %d/%d: %d electrode(s) ===",
            stim_round.round_index + 1,
            len(self._plan.rounds),
            len(stim_round.connections),
        )

        stim_params = stim_round.build_stim_params()

        if not self._testing:
            log.info("Uploading %d StimParam(s) to Intan...", len(stim_params))
            self._intan.send_stimparam(stim_params)
            log.info("Waiting %.1f s for Intan propagation...", self._param_send_wait)
            time.sleep(self._param_send_wait)

        trigger_array = stim_round.build_trigger_array()

        for rep in range(1, self._n_stims + 1):
            timestamp_utc = datetime.now(timezone.utc).isoformat()

            if not self._testing:
                self._trigger_ctrl.send(trigger_array)
            else:
                log.debug("[TESTING] Would send trigger %s", trigger_array)

            for electrode, conn in stim_round.connections.items():
                record = StimulationRecord(
                    round_index=stim_round.round_index,
                    rep_index=rep,
                    electrode=electrode,
                    amplitude=conn.amplitude,
                    duration=conn.duration,
                    polarity=conn.polarity,
                    trigger_key=stim_round.trigger_key,
                    trigger_tag=stim_round.round_index,
                    timestamp_utc=timestamp_utc,
                    testing=self._testing,
                )
                self._stimulation_log.append(record)

            if rep % 5 == 0:
                log.info(
                    "  Round %d: %d/%d stimulations sent",
                    stim_round.round_index,
                    rep,
                    self._n_stims,
                )

            if rep < self._n_stims:
                time.sleep(self._delay)

        if not self._testing:
            log.info("Disabling StimParam(s) for round %d...", stim_round.round_index)
            for sp in stim_params:
                sp.enable = False
            self._intan.send_stimparam(stim_params)
            time.sleep(self._param_send_wait)

    def _fetch_database_results(
        self,
        fs_name: str,
        start: Optional[datetime],
        stop: Optional[datetime],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch spike events and triggers from the database.
        """
        spike_df = pd.DataFrame()
        trigger_df = pd.DataFrame()

        if self._testing or self._db is None or start is None or stop is None:
            log.info("[TESTING/OFFLINE] Skipping database queries.")
            return spike_df, trigger_df

        try:
            stop_extended = stop + timedelta(seconds=self._spike_window)
            log.info(
                "Querying spike events from %s to %s...",
                start.isoformat(),
                stop_extended.isoformat(),
            )
            spike_df = self._db.get_spike_event(start, stop_extended, fs_name)
            log.info("Retrieved %d spike events.", len(spike_df))

            log.info(
                "Querying triggers from %s to %s...",
                start.isoformat(),
                stop.isoformat(),
            )
            trigger_df = self._db.get_all_triggers(start, stop)
            log.info("Retrieved %d trigger events.", len(trigger_df))

        except Exception as e:
            log.warning("Database query failed: %s", e)

        return spike_df, trigger_df

    def _safe_close(self) -> None:
        """
        Close all hardware connections safely.
        """
        if self._trigger_ctrl is not None:
            try:
                self._trigger_ctrl.close()
                log.info("TriggerController closed.")
            except Exception as e:
                log.warning("Error closing TriggerController: %s", e)

        if self._intan is not None:
            try:
                self._intan.close()
                log.info("IntanSofware closed.")
            except Exception as e:
                log.warning("Error closing IntanSofware: %s", e)

        if self._exp is not None and not self._testing:
            try:
                self._exp.stop()
                log.info("Experiment stopped.")
            except Exception as e:
                log.warning("Error stopping experiment: %s", e)

    def _save_all(self) -> None:
        """
        Persist all results to disk.
        """
        if self._results is None:
            log.warning("No results to save.")
            return

        saver = DataSaver(self._output_dir, self._results.fs_name)
        saver.save_stimulation_log(self._results.stimulations)
        if not self._results.spike_events.empty:
            saver.save_spike_events(self._results.spike_events)
        if not self._results.triggers.empty:
            saver.save_triggers(self._results.triggers)
        saver.save_summary(self._results)
        log.info("All results saved to %s", self._output_dir)
