from time import sleep
from datetime import datetime, timedelta, UTC
from typing import List
from collections import OrderedDict
from dataclasses import dataclass, asdict, field
from enum import Enum
from itertools import product
from pathlib import Path
import logging
import logging.handlers
from sys import stdout as STDOUT

import numpy as np
import pandas as pd
from tqdm import tqdm

from neuroplatform import (
    Database,
    TriggerController,
    StimParam,
    StimShape,
    Experiment,
    StimPolarity,
)

try:
    from neuroplatform import IntanSofware as IntanSoftware
except ImportError:
    try:
        from neuroplatform import IntanSoftware as IntanSoftware
    except ImportError:
        raise ImportError("No IntanSoftware or IntanSofware")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)
_log.propagate = False

if not _log.handlers:
    _handler = logging.StreamHandler(STDOUT)
    _handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    _log.addHandler(_handler)


# ---------------------------------------------------------------------------
# File logger factory
# ---------------------------------------------------------------------------

def _make_file_logger(log_path: Path) -> logging.Logger:
    """Create (or retrieve) a dedicated file logger that writes to *log_path*.

    The logger name is derived from the path so that multiple StimScan
    instances each get their own file without handler duplication on re-use.
    """
    logger_name = f"stimscan.file.{log_path.stem}"
    file_logger = logging.getLogger(logger_name)

    # Avoid adding duplicate handlers if the object is recreated in the same
    # Python session (e.g. in a notebook that re-runs a cell).
    if not file_logger.handlers:
        file_logger.setLevel(logging.DEBUG)
        file_logger.propagate = False
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        file_logger.addHandler(fh)

    return file_logger


# ---------------------------------------------------------------------------
# LogHistory
# ---------------------------------------------------------------------------

class LogHistory:
    """Manages log messages and exposes them as a DataFrame.

    Args:
        verbose (bool): If True, INFO-level messages are printed. Defaults to True.
        file_logger (logging.Logger | None): Optional file logger. When provided
            every message is mirrored to the log file in addition to stdout.
    """

    def __init__(self, verbose: bool = True, file_logger: logging.Logger = None):
        self._history: OrderedDict = OrderedDict()
        self.logger = _log
        self.verbose = verbose
        self._file_logger = file_logger

    # -- internal ------------------------------------------------------------

    def _write_file(self, level: str, message: str):
        if self._file_logger is None:
            return
        method = getattr(self._file_logger, level.lower(), self._file_logger.info)
        method(message)

    # -- public --------------------------------------------------------------

    def info(self, message: str):
        self._history[datetime.now(UTC)] = ("INFO", message)
        if self.verbose:
            self.logger.info(message)
        self._write_file("info", message)

    def warning(self, message: str):
        self._history[datetime.now(UTC)] = ("WARNING", message)
        self.logger.warning(message)
        self._write_file("warning", message)

    def error(self, message: str):
        self._history[datetime.now(UTC)] = ("ERROR", message)
        self.logger.error(message)
        self._write_file("error", message)

    def debug(self, message: str):
        self._history[datetime.now(UTC)] = ("DEBUG", message)
        self.logger.debug(message)
        self._write_file("debug", message)

    def get_history(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            self._history, orient="index", columns=["Level", "Message"]
        )


# ---------------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------------

class MEAType(Enum):
    """Layout of the MEA."""

    MEA4x8 = 1
    MEA32 = 2

    def get_sites(self) -> int:
        if self == MEAType.MEA4x8:
            return 4
        elif self == MEAType.MEA32:
            return 1
        raise ValueError("MEA type not recognized.")

    def get_electrodes_per_site(self) -> int:
        if self == MEAType.MEA4x8:
            return 8
        elif self == MEAType.MEA32:
            return 32
        raise ValueError("MEA type not recognized.")


class MEA(Enum):
    """MEA number (0-indexed)."""

    One = 0
    Two = 1
    Three = 2
    Four = 3

    @staticmethod
    def get_from_electrode(electrode: int) -> "MEA":
        return MEA(electrode // 32)

    @staticmethod
    def get_electrode_range(mea_number: "MEA") -> list[int]:
        return list(range(mea_number * 32, (mea_number + 1) * 32))


class Site(Enum):
    """Neurosphere site ID (0-indexed, 0-3)."""

    One = 0
    Two = 1
    Three = 2
    Four = 3

    @staticmethod
    def get_from_electrode(electrode_id: int) -> "Site":
        site = (electrode_id % 32) // 8
        return Site(site)


# ---------------------------------------------------------------------------
# StimParamLoader
# ---------------------------------------------------------------------------

class StimParamLoader:
    """Manages stimulation parameters and sends them to the Intan software.

    Args:
        stimparams (List[StimParam] | None): List of stimulation parameters.
        intan (IntanSoftware, optional): Intan software instance. Defaults to None.
        verbose (bool, optional): If True, display log messages. Defaults to True.
        must_connect (bool, optional): Raise if Intan is not connected. Defaults to False.
        file_logger (logging.Logger | None): Optional file logger to mirror all
            log messages.

    Methods:
        add_stimparam(stimparam): Append a StimParam to the list.
        enable_all(): Enable all parameters.
        disable_all(): Disable all parameters.
        send_parameters(): Send parameters to the Intan.
        disable_all_and_send(): Disable all parameters then send.
        show_all_stimparams(): Print all parameters.
        all_parameters_enabled() -> bool: Check if all parameters are enabled.
        reset(): Clear all parameters.
        get_log() -> DataFrame: Return full log history.
    """

    def __init__(
        self,
        stimparams: List[StimParam] | None,
        intan: IntanSoftware = None,
        verbose: bool = True,
        must_connect: bool = False,
        file_logger: logging.Logger = None,
    ):
        self._stimparams: List[StimParam] = []
        self._used_electrodes: list = []
        self._used_triggers: list = []
        self._sites: dict = {}
        self._meas: dict = {}
        self._electrode_param_mapping: dict = {}

        self.log = LogHistory(verbose, file_logger=file_logger)
        self.log.info("Please remember to book the system before connecting to the Intan.")

        self.intan = intan
        self.stimparams = stimparams  # uses the setter

        if self.intan is None:
            if must_connect:
                raise RuntimeError("Could not connect to Intan")
            else:
                self.log.warning(
                    "Could not connect to Intan. You may validate parameters but "
                    "sending them to the Intan will not be possible."
                )

    # -- property ------------------------------------------------------------

    @property
    def stimparams(self) -> List[StimParam]:
        return self._stimparams

    @stimparams.setter
    def stimparams(self, new_stimparams: List[StimParam] | None):
        if new_stimparams is None:
            self._stimparams = []
            self.log.info("No parameters set.")
            return
        for param in new_stimparams:
            if not isinstance(param, StimParam):
                raise ValueError(f"{param} is not a StimParam instance")
        self._stimparams = new_stimparams
        self._update_parameters()

    # -- internal ------------------------------------------------------------

    def _clear_records(self):
        self._used_electrodes = []
        self._used_triggers = []
        self._electrode_param_mapping = {}
        self._sites = {}
        self._meas = {}

    def _update_parameters(self):
        if not self._stimparams:
            return
        self._clear_records()
        for param in self.stimparams:
            if param.index in self._used_electrodes:
                raise ValueError(
                    f"Electrode {param.index} is already in use. "
                    "Only one parameter per electrode is allowed."
                )
            self._used_electrodes.append(param.index)
            if param.trigger_key in self._used_triggers:
                self._used_triggers.append(param.trigger_key)
            if not (0 <= param.index <= 127):
                raise ValueError(f"Invalid electrode number: {param.index}")
            if not (0 <= param.trigger_key <= 15):
                raise ValueError(f"Invalid trigger key: {param.trigger_key}")
            if param.phase_duration1 < 0 or param.phase_duration2 < 0:
                raise ValueError(
                    f"Invalid phase duration: {param.phase_duration1}, {param.phase_duration2}"
                )
            if param.phase_amplitude1 < 0 or param.phase_amplitude2 < 0:
                raise ValueError(
                    f"Invalid phase amplitude: {param.phase_amplitude1}, {param.phase_amplitude2}"
                )
            if param.phase_duration1 > 500 or param.phase_duration2 > 500:
                self.log.warning(
                    f"Phase duration exceeds 500 us: "
                    f"{param.phase_duration1}, {param.phase_duration2}"
                )
            if (
                param.phase_duration1 * param.phase_amplitude1
                != param.phase_duration2 * param.phase_amplitude2
                and param.stim_shape in (
                    StimShape.Biphasic,
                    StimShape.BiphasicWithInterphaseDelay,
                )
            ):
                self.log.warning(
                    f"Pulses are not charge-balanced for electrode {param.index}. "
                    "Ensure phase_duration * phase_amplitude is equal for both phases."
                )
            if (
                param.stim_shape == StimShape.Triphasic
                and param.phase_duration2 * param.phase_amplitude2 * 2
                != param.phase_duration1 * param.phase_amplitude1
            ):
                self.log.warning(
                    f"Pulses are not charge-balanced for electrode {param.index} "
                    "(Triphasic). Ensure phase2_dur * phase2_amp * 2 == phase1_dur * phase1_amp."
                )
            self._electrode_param_mapping[param.index] = param
            self._sites[param.index] = Site.get_from_electrode(param.index)
            self._meas[param.index] = MEA.get_from_electrode(param.index)

        if len(set(self._meas.values())) > 1:
            self.log.warning(
                "Parameters span multiple MEAs. Please confirm this is intentional."
            )

    # -- public --------------------------------------------------------------

    def get_log(self) -> pd.DataFrame:
        """Return a DataFrame of all log messages."""
        return self.log.get_history()

    def reset(self):
        """Clear all parameters."""
        self.stimparams = []

    def add_stimparam(self, stimparam: StimParam) -> bool:
        """Append a new StimParam to the list."""
        try:
            self.stimparams.append(stimparam)
            self._update_parameters()
            return True
        except Exception as exc:
            self.log.error(f"Error: {exc}")
            return False

    def show_all_stimparams(self):
        """Print all parameters."""
        if not self.stimparams:
            self.log.info("No parameters to display.")
            return
        for electrode, stimparam in self._electrode_param_mapping.items():
            self.log.info(f"Electrode {electrode}:\n{stimparam.display_attributes()}")
            self.log.info("*" * 50)

    def all_parameters_enabled(self) -> bool:
        """Return True if all parameters are enabled."""
        if not self._stimparams:
            return False
        return all(param.enable for param in self.stimparams)

    def enable_all(self):
        """Enable all parameters."""
        if not self._stimparams:
            self.log.warning("No parameters to enable.")
            return
        for param in self.stimparams:
            param.enable = True

    def disable_all(self):
        """Disable all parameters."""
        if not self._stimparams:
            self.log.warning("No parameters to disable.")
            return
        for param in self.stimparams:
            param.enable = False

    def _send_parameters(self):
        if self.intan is None:
            raise ValueError("Intan not connected")
        if not self._stimparams:
            self.log.warning("No parameters to send.")
            return
        self._update_parameters()
        self.intan.send_stimparam(self.stimparams)

    def send_parameters(self) -> bool:
        """Send parameters to the Intan.

        Returns:
            bool: True on success, False on failure.
        """
        try:
            if self.intan is None:
                raise ValueError("Intan not connected")
            if not self._stimparams:
                self.log.warning("No parameters to send.")
                return False
            if not self.all_parameters_enabled():
                self.log.warning(
                    "Some parameters are disabled. Enable all parameters you intend to use."
                )
            self.log.info("Sending... Please wait 10 seconds")
            self.intan.send_stimparam(self.stimparams)
            self.log.info("Done.")
            return True
        except Exception as exc:
            self.log.error(f"Error: {exc}")
            return False

    def disable_all_and_send(self) -> bool:
        """Disable all parameters and send the update to the Intan.

        Returns:
            bool: True on success, False on failure.
        """
        try:
            if not self._stimparams:
                self.log.warning("No parameters to disable.")
                return False
            self.disable_all()
            self._send_parameters()
            self.log.info("All parameters disabled and sent to Intan.")
            return True
        except Exception as exc:
            self.log.error(f"Error: {exc}")
            return False


# ---------------------------------------------------------------------------
# ExtendedTimedelta
# ---------------------------------------------------------------------------

class ExtendedTimedelta(timedelta):
    """Minimal extension of timedelta with convenience unit-conversion methods."""

    def as_minutes(self) -> float:
        return self.total_seconds() / 60

    def as_seconds(self) -> float:
        return self.total_seconds()

    def as_milliseconds(self) -> float:
        return self.total_seconds() * 1e3

    def as_microseconds(self) -> float:
        return self.total_seconds() * 1e6

    def to_hertz(self) -> float:
        if self.total_seconds() == 0:
            print("Period is zero. Returning np.inf Hz.")
            return np.inf
        return 1 / self.total_seconds()


# ---------------------------------------------------------------------------
# StimParamGrid
# ---------------------------------------------------------------------------

@dataclass
class StimParamGrid:
    """Contains lists of all parameters to scan.

    Attributes:
        amplitudes: list[float]
            Amplitudes to scan (uA). Recommended range: 0.1-5.
        durations: list[float]
            Durations to scan (us). Recommended range: 10-400.
        polarities: list[StimPolarity]
            Polarities to scan. Accepted: StimPolarity.NegativeFirst / PositiveFirst.
        interphase_delays: list[float]
            Inter-phase delays to scan (us).
        nb_pulses: list[int]
            Number of pulses per train.
        pulse_train_periods: list[float]
            Pulse train periods (us). No effect when nb_pulses == 1.
        post_stim_ref_periods: list[float]
            Post-stimulation refractory periods (us).
        stim_shapes: list[StimShape]
            Stimulation shapes. Accepted: StimShape.Biphasic / BiphasicWithInterphaseDelay.
        mea_type: MEAType
            MEA layout. Defaults to MEAType.MEA4x8.
    """

    amplitudes: list[float] = field(default_factory=list)
    durations: list[float] = field(default_factory=list)
    polarities: list[StimPolarity] = field(default_factory=list)
    interphase_delays: list[float] = field(default_factory=list)
    nb_pulses: list[int] = field(default_factory=list)
    pulse_train_periods: list[float] = field(default_factory=list)
    post_stim_ref_periods: list[float] = field(default_factory=list)
    stim_shapes: list[StimShape] = field(default_factory=list)
    mea_type: MEAType = MEAType.MEA4x8

    def __post_init__(self):
        # Coerce raw integer values back to StimPolarity enum members.
        # Necessary when parameters arrive via JSON (e.g. from Datalore),
        # where StimPolarity.NegativeFirst.value serialises to plain 0.
        self.polarities = [
            StimPolarity(p) if not isinstance(p, StimPolarity) else p
            for p in self.polarities
        ]

        type_checks = {
            "amplitudes": (int, float),
            "durations": (int, float),
            "interphase_delays": (int, float),
            "nb_pulses": int,
            "pulse_train_periods": (int, float),
            "post_stim_ref_periods": (int, float),
            "stim_shapes": StimShape,
        }
        for attr, types in type_checks.items():
            if not all(isinstance(item, types) for item in getattr(self, attr)):
                raise ValueError(f"All items in {attr} must be of type {types}.")

        if not isinstance(self.mea_type, MEAType):
            raise ValueError("mea_type must be a MEAType instance.")

        if any(s == StimShape.Triphasic for s in self.stim_shapes):
            raise NotImplementedError(
                "Triphasic stimulation is not currently supported."
            )

        default_param = StimParam()
        defaults = {
            "amplitudes": default_param.phase_amplitude1,
            "durations": default_param.phase_duration1,
            "polarities": default_param.polarity,
            "interphase_delays": default_param.interphase_delay,
            "nb_pulses": default_param.nb_pulse,
            "pulse_train_periods": default_param.pulse_train_period,
            "post_stim_ref_periods": default_param.post_stim_ref_period,
            "stim_shapes": default_param.stim_shape,
        }
        for attr, default in defaults.items():
            if not getattr(self, attr):
                setattr(self, attr, [default])

    def total_combinations(self) -> int:
        """Return the total number of parameter combinations."""
        total = 1
        for name, val in self.__dict__.items():
            if isinstance(val, list) and not name.startswith("_"):
                total *= len(val)
        return total

    def display_grid(self):
        """Print all parameter lists."""
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                print(f"{k}: {v}")


# ---------------------------------------------------------------------------
# StimParamFactory
# ---------------------------------------------------------------------------

@dataclass
class StimParamFactory:
    """Creates StimParam objects from a single parameter combination."""

    amplitude1: float
    amplitude2: float
    duration1: float
    duration2: float
    polarity: StimPolarity
    interphase_delay: float
    nb_pulse: int
    pulse_train_period: float
    post_stim_ref_period: float
    stim_shape: StimShape

    def create_from(self) -> StimParam:
        p = StimParam()
        p.phase_amplitude1 = self.amplitude1
        p.phase_amplitude2 = self.amplitude2
        p.phase_duration1 = self.duration1
        p.phase_duration2 = self.duration2
        p.polarity = self.polarity
        p.interphase_delay = self.interphase_delay
        p.nb_pulse = self.nb_pulse
        p.pulse_train_period = self.pulse_train_period
        p.post_stim_ref_period = self.post_stim_ref_period
        p.stim_shape = self.stim_shape
        return p

    def get_names(self) -> dict:
        return {
            "Amplitude": self.amplitude1,
            "Duration": self.duration1,
            "Polarity": self.polarity,
            "Interphase delay": self.interphase_delay,
            "Number of pulses": self.nb_pulse,
            "Pulse train period": self.pulse_train_period,
            "Post stim ref period": self.post_stim_ref_period,
            "Stim shape": self.stim_shape,
        }

    def display_params(self):
        for k, v in self.get_names().items():
            print(f"- {k}: {v}")


# ---------------------------------------------------------------------------
# StimScan
# ---------------------------------------------------------------------------

class StimScan:
    def __init__(
        self,
        token: str,
        booking_email: str,
        amplitudes: list[float],
        durations: list[float],
        polarities: list[StimPolarity],
        scan_channels: list[int],
        delay_btw_stim: float,       # seconds
        delay_btw_channels: float,   # seconds
        repeats_per_channel: int,
        testing: bool = False,
        output_dir: str = ".",
    ):
        """Creates a stimulation parameter scan utility.

        Args:
            token: str
                Experiment token provided by neuroplatform.
            booking_email: str
                Email used for booking the trigger controller.
            amplitudes: list[float]
                List of amplitudes to scan (uA).
            durations: list[float]
                List of durations to scan (us).
            polarities: list[StimPolarity]
                List of polarities to scan.
            scan_channels: list[int]
                Channels to scan. Must be within the experiment's allowed electrodes.
            delay_btw_stim: float
                Delay between each stimulation in seconds. Should exceed the full
                pulse-train duration when nb_pulses > 1.
            delay_btw_channels: float
                Delay between channel stimulations in seconds.
            repeats_per_channel: int
                Number of times each parameter combination is repeated per channel.
            testing: bool
                Dry-run mode. Parameters are built and logged normally but no
                stimulations are sent to hardware. Defaults to False.
            output_dir: str
                Directory where all output files (CSVs, log) are written.
                Created if it does not exist. Defaults to current directory.
        """
        self.testing = testing

        # ------------------------------------------------------------------
        # Output directory and file logger — set up first so everything that
        # follows is captured in the log file.
        # ------------------------------------------------------------------
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Log file name encodes the UTC wall-clock time of instantiation so
        # multiple runs never overwrite each other.
        _run_ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        self._log_path = self._output_dir / f"stimscan_{_run_ts}.log"
        self._file_logger = _make_file_logger(self._log_path)

        # ------------------------------------------------------------------
        # Main log (stdout + file)
        # ------------------------------------------------------------------
        self.log = LogHistory(verbose=True, file_logger=self._file_logger)

        self.log.info(f"StimScan initialised. Log file: {self._log_path}")
        self.log.info(
            f"Parameters — token={token}, email={booking_email}, "
            f"channels={scan_channels}, amplitudes={amplitudes}, "
            f"durations={durations}, polarities={polarities}, "
            f"delay_btw_stim={delay_btw_stim}s, "
            f"delay_btw_channels={delay_btw_channels}s, "
            f"repeats_per_channel={repeats_per_channel}, "
            f"testing={testing}"
        )

        self.token = token
        self.booking_email = booking_email

        if self.testing:
            self.log.info(
                "[TESTING MODE] No stimulations will be sent. "
                "Parameter generation and timing logic will still execute."
            )

        self.parameter_grid = StimParamGrid(
            amplitudes=amplitudes,
            durations=durations,
            polarities=polarities,
            mea_type=MEAType.MEA4x8,
        )

        self.scan_channels = scan_channels
        self.delay_btw_stim = ExtendedTimedelta(seconds=delay_btw_stim)
        self.delay_btw_channels = ExtendedTimedelta(seconds=delay_btw_channels)
        self.repeats_per_channel = repeats_per_channel

        self.params_per_electrode = self.parameter_grid.total_combinations()
        self.parameters = self._create_parameters_factory()
        self.mea_type = self.parameter_grid.mea_type
        self.mea = None
        self.loaders = None

        self.start_time = None
        self.stop_time = None

        self.fs_experiment = None
        self._trigger_gen = None
        self._intan = None

        if not self.testing:
            self._connect()

            if not np.all(np.isin(scan_channels, self.fs_experiment.electrodes)):
                raise ValueError(
                    "Some channels are not in the allowed electrodes list for your experiment token."
                )

        self._db = Database()

        self._channels_per_trigger: dict = {}
        self._current_factory_id = None
        self._stim_history: OrderedDict = OrderedDict()
        self._params_per_site: dict = {}

        for channel in scan_channels:
            site = Site.get_from_electrode(channel)
            if site not in self._params_per_site:
                self._params_per_site[site] = 0
            self._params_per_site[site] += 1
            if self._params_per_site[site] > self.mea_type.get_electrodes_per_site():
                raise ValueError(
                    f"Too many channels provided for site {site}. "
                    "Are all channels on the same MEA?"
                )

        mea = MEA.get_from_electrode(scan_channels[0]).value
        if not all(MEA.get_from_electrode(ch).value == mea for ch in scan_channels):
            raise ValueError("All channels must be on the same MEA.")
        self.mea = mea

        self.log.info(
            f"Initialisation complete. MEA={self.mea}, "
            f"total parameter combinations={self.parameter_grid.total_combinations()}"
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_stimulation_parameter_history(self) -> pd.DataFrame:
        """Return a DataFrame of all stimulation parameters sent (or simulated)."""
        return pd.DataFrame.from_dict(self._stim_history, orient="index")

    def get_scan_duration(self) -> timedelta:
        """Return the predicted duration of the scan."""
        nb_combinations = self.parameter_grid.total_combinations()
        nb_channels = len(self.scan_channels)

        stim_time = self.repeats_per_channel * self.delay_btw_stim
        time_per_combination = nb_channels * (stim_time + self.delay_btw_channels)

        if self.mea_type == MEAType.MEA4x8:
            intan_load_time = timedelta(seconds=20)
            nb_loaders = 1
        elif self.mea_type == MEAType.MEA32:
            intan_load_time = timedelta(seconds=40)
            nb_loaders = 2
        else:
            raise ValueError(
                f"Unsupported MEA type: {self.mea_type}. "
                "Choose MEAType.MEA4x8 or MEAType.MEA32."
            )

        total_time = (
            nb_combinations * (time_per_combination + intan_load_time)
            + timedelta(seconds=10 * nb_loaders)
        )
        total_s = int(total_time.total_seconds())
        h, remainder = divmod(total_s, 3600)
        m, s = divmod(remainder, 60)
        msg = f"Predicted scan duration: {h}h {m}m {s}s"
        print(msg)
        self.log.info(msg)
        return total_time

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self):
        """Run the stimulation scan.

        In testing mode the experiment is started/stopped normally so timing
        and parameter-building logic can be validated, but no triggers are fired
        and no parameters are uploaded to hardware.
        """
        self.log.info("run() called — starting scan.")
        try:
            if not self.testing:
                if not self.fs_experiment.start():
                    raise RuntimeError("Experiment did not start properly.")
                self.log.info("Experiment started on NeuroPlatform.")

            self.start_time = datetime.now(UTC)
            self.log.info(f"Scan start time (UTC): {self.start_time.isoformat()}")

            nb_combinations = len(self.parameters)
            for factory_id in tqdm(self.parameters):
                self._current_factory_id = factory_id
                self.log.debug(
                    f"Processing factory {factory_id + 1}/{nb_combinations}"
                )
                self._make_parameters()
                self._send_stim()

            self.log.info("All parameter combinations complete.")

        except Exception as exc:
            self.log.error(f"Exception during run(): {exc}")
            raise

        finally:
            self.stop_time = datetime.now(UTC)
            self.log.info(f"Scan stop time (UTC): {self.stop_time.isoformat()}")

            if not self.testing:
                if self.loaders is not None:
                    for _, loader in self.loaders:
                        loader.disable_all_and_send()
                        self.log.info("Loader disabled and sent.")
                self._trigger_gen.close()
                self._intan.close()
                self.fs_experiment.stop()
                self.log.info("Hardware connections closed. Experiment stopped.")

            self.save_results(output_dir=str(self._output_dir))
            self._flush_file_logger()

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    def save_results(self, output_dir: str = ".") -> dict[str, Path]:
        """Fetch results from the database and save them to CSV files.

        Three files are written:

        * ``stim_history.csv``   - stimulation parameter log recorded during
          the run (local, no DB query needed).
        * ``spike_activity.csv`` - spike events for the run window via
          ``db.get_spike_event``. Skipped in testing mode (no experiment name).
        * ``triggers.csv``       - trigger records for the run window via
          ``db.get_all_triggers``. Skipped in testing mode (no experiment name).

        The log file path is also included in the returned dict under the key
        ``"log"``.

        Args:
            output_dir: str
                Directory in which to write the CSV files. Created if absent.
                Defaults to the current working directory.

        Returns:
            dict[str, Path]: Keys are ``"stim_history"``, ``"spike_activity"``,
            ``"triggers"``, and ``"log"``.

        Raises:
            RuntimeError: If called before a completed run.
        """
        if self.start_time is None or self.stop_time is None:
            raise RuntimeError(
                "No completed run found. Call run() before save_results()."
            )

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved_paths: dict[str, Path] = {}

        # 1. Stimulation history (local — always available) ------------------
        stim_history_path = out / "stim_history.csv"
        stim_df = self.get_stimulation_parameter_history()
        stim_df.index.name = "timestamp_utc"
        stim_df.to_csv(stim_history_path)
        saved_paths["stim_history"] = stim_history_path
        msg = f"Saved stimulation history  -> {stim_history_path}"
        print(msg)
        self.log.info(msg)

        if self.testing:
            self.log.warning(
                "Testing mode: skipping DB queries for spike activity and triggers "
                "(no live experiment was run)."
            )
            # Log file is included even in testing mode.
            saved_paths["log"] = self._log_path
            self.log.info(f"Log file available at     -> {self._log_path}")
            return saved_paths

        exp_name = self.fs_experiment.exp_name

        # 2. Spike activity --------------------------------------------------
        activity_path = out / "spike_activity.csv"
        try:
            spike_df = self._db.get_spike_event(
                self.start_time, self.stop_time, exp_name
            )
            spike_df.to_csv(activity_path, index=False)
            saved_paths["spike_activity"] = activity_path
            msg = f"Saved spike activity       -> {activity_path}"
            print(msg)
            self.log.info(msg)
        except Exception as exc:
            warn = f"Warning: could not fetch spike activity from DB: {exc}"
            print(warn)
            self.log.warning(warn)

        # 3. Triggers --------------------------------------------------------
        triggers_path = out / "triggers.csv"
        try:
            triggers_df = self._db.get_all_triggers(self.start_time, self.stop_time)
            triggers_df.to_csv(triggers_path, index=False)
            saved_paths["triggers"] = triggers_path
            msg = f"Saved triggers             -> {triggers_path}"
            print(msg)
            self.log.info(msg)
        except Exception as exc:
            warn = f"Warning: could not fetch triggers from DB: {exc}"
            print(warn)
            self.log.warning(warn)

        # 4. Log file --------------------------------------------------------
        saved_paths["log"] = self._log_path
        self.log.info(f"Log file available at     -> {self._log_path}")

        return saved_paths

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _flush_file_logger(self):
        """Flush and close all handlers on the file logger."""
        for handler in self._file_logger.handlers:
            handler.flush()
            handler.close()
        self._file_logger.handlers.clear()

    def _connect(self) -> None:
        """Connect to NeuroPlatform hardware. Only called in live mode."""
        self.log.info("Connecting to NeuroPlatform hardware ...")
        self.fs_experiment = Experiment(token=self.token)
        self._trigger_gen = TriggerController(self.booking_email)
        self._intan = IntanSoftware()
        self.log.info(f"Hardware connected. FS name: {self.fs_experiment.exp_name}")

    def _get_param_indices_by_trigger(self, trigger_key: int, loader: StimParamLoader) -> list[int]:
        return [p.index for p in loader.stimparams if p.trigger_key == trigger_key]

    def _create_parameters_factory(self) -> dict[int, StimParamFactory]:
        factories = {}
        for i, (amp, dur, pol, ipd, nbp, ptp, psrp, ss) in enumerate(
            product(
                self.parameter_grid.amplitudes,
                self.parameter_grid.durations,
                self.parameter_grid.polarities,
                self.parameter_grid.interphase_delays,
                self.parameter_grid.nb_pulses,
                self.parameter_grid.pulse_train_periods,
                self.parameter_grid.post_stim_ref_periods,
                self.parameter_grid.stim_shapes,
            )
        ):
            factories[i] = StimParamFactory(
                amplitude1=amp,
                amplitude2=amp,
                duration1=dur,
                duration2=dur,
                polarity=pol,
                interphase_delay=ipd,
                nb_pulse=nbp,
                pulse_train_period=ptp,
                post_stim_ref_period=psrp,
                stim_shape=ss,
            )
        self.log.info(f"Created {len(factories)} parameter factory combinations.")
        return factories

    def _bind_parameter(self, params, factory, trigger_key, trigger_counter, index):
        p = factory.create_from()
        p.trigger_key = trigger_key
        p.index = index
        p.enable = True

        if p.index in self.scan_channels:
            trigger_counter[p.trigger_key] = 1
            params.append(p)
            if p.trigger_key not in self._channels_per_trigger:
                self._channels_per_trigger[p.trigger_key] = [p.index]
            else:
                self._channels_per_trigger[p.trigger_key].append(p.index)

    def _make_parameters(self):
        """Build loader objects for the current factory.

        In testing mode loaders are still created for validation purposes,
        but ``send_parameters`` is not called. self._intan is None in testing
        mode; StimParamLoader accepts this and issues a warning rather than
        raising, so parameter building still completes successfully.
        """
        factory = self.parameters[self._current_factory_id]
        triggers_counter = np.zeros(16)

        self.log.debug(
            f"Building parameters for factory {self._current_factory_id}: "
            f"amp={factory.amplitude1}uA, dur={factory.duration1}us, "
            f"polarity={factory.polarity}, shape={factory.stim_shape}"
        )

        if self.mea_type == MEAType.MEA4x8:
            params = []
            for site in range(self.mea_type.get_sites()):
                for trigger_key in range(self.mea_type.get_electrodes_per_site()):
                    self._bind_parameter(
                        params,
                        factory,
                        trigger_key,
                        triggers_counter,
                        self.mea * 32 + site * 8 + trigger_key,
                    )
            needed_triggers = np.where(triggers_counter > 0)[0]
            self.loaders = [
                (needed_triggers, StimParamLoader(
                    params, self._intan, verbose=False,
                    file_logger=self._file_logger,
                ))
            ]

        elif self.mea_type == MEAType.MEA32:
            params_16, params_32 = [], []
            for trigger_key in range(16):
                self._bind_parameter(
                    params_16, factory, trigger_key, triggers_counter,
                    self.mea * 32 + trigger_key,
                )
                self._bind_parameter(
                    params_32, factory, trigger_key, triggers_counter,
                    self.mea * 32 + 16 + trigger_key,
                )
            needed_triggers = np.where(triggers_counter > 0)[0]
            self.loaders = [
                (needed_triggers, StimParamLoader(
                    params_16, self._intan, verbose=False,
                    file_logger=self._file_logger,
                )),
                (needed_triggers, StimParamLoader(
                    params_32, self._intan, verbose=False,
                    file_logger=self._file_logger,
                )),
            ]

        self.log.debug(
            f"Factory {self._current_factory_id}: "
            f"{len(needed_triggers)} trigger(s) needed — {list(needed_triggers)}"
        )

    def _send_stim(self):
        """Send stimulation triggers for the current factory.

        When ``self.testing`` is True:
          - Parameters are not uploaded to hardware.
          - Triggers are not fired.
          - The stim history is still updated so post-run analysis works.
          - ``sleep`` calls still execute so timing estimates remain valid.
        """
        for needed_triggers, loader in self.loaders:
            if not self.testing:
                loader.send_parameters()

            for trigger in needed_triggers:
                triggers = np.zeros(16, dtype=np.uint8)
                triggers[trigger] = 1

                for repeat_idx in range(self.repeats_per_channel):
                    params_data_dict = asdict(self.parameters[self._current_factory_id])
                    params_data_dict["trigger_key"] = int(trigger)
                    params_data_dict["param_id"] = self._current_factory_id
                    params_data_dict["channel"] = self._get_param_indices_by_trigger(
                        trigger, loader
                    )
                    ts = datetime.now(UTC)
                    self._stim_history[ts] = params_data_dict

                    self.log.debug(
                        f"{'[DRY-RUN] ' if self.testing else ''}"
                        f"Trigger {int(trigger)} | factory {self._current_factory_id} | "
                        f"repeat {repeat_idx + 1}/{self.repeats_per_channel} | "
                        f"channels={params_data_dict['channel']}"
                    )

                    if not self.testing:
                        self._trigger_gen.send(triggers)

                    sleep(self.delay_btw_stim.as_seconds())

                sleep(self.delay_btw_channels.as_seconds())

            if not self.testing:
                loader.disable_all_and_send()
