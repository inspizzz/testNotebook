from time import sleep
from datetime import datetime, timedelta, UTC
import numpy as np
from collections import OrderedDict
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict, field
from enum import Enum
from itertools import product
from pathlib import Path
import logging

from neuroplatform import (
    Database,
    TriggerController,
    StimParam,
    Experiment,
    StimPolarity,
    StimShape,
)

try:
    from neuroplatform import IntanSofware as IntanSoftware
except ImportError:
    try:
        from neuroplatform import IntanSoftware as IntanSoftware
    except ImportError:
        raise ImportError("No IntanSoftware or IntanSofware")

from parameters_loader import StimParamLoader

# ---------------------------------------------------------------------------
# Logging setup – writes all actions to logs.txt and also prints to console.
# ---------------------------------------------------------------------------
logger = logging.getLogger("StimScan")
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
# Helper types
# ---------------------------------------------------------------------------

class ExtendedTimedelta(timedelta):
    """Minimal extension of the timedelta class to add simple time unit conversion methods."""

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
            logger.warning("Period is zero. Returning np.inf Hz.")
            return np.inf
        return 1 / self.total_seconds()


class MEAType(Enum):
    """Layout of the MEA."""

    MEA4x8 = 1
    MEA32 = 2

    def get_sites(self) -> int:
        if self == MEAType.MEA4x8:
            return 4
        elif self == MEAType.MEA32:
            return 1
        else:
            raise ValueError("MEA type not recognized.")

    def get_electrodes_per_site(self) -> int:
        if self == MEAType.MEA4x8:
            return 8
        elif self == MEAType.MEA32:
            return 32
        else:
            raise ValueError("MEA type not recognized.")


class MEA(Enum):
    """MEA Number"""

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
    """Neurosphere site ID, from 1 to 4"""

    One = 0
    Two = 1
    Three = 2
    Four = 3

    def get_from_electrode(electrode_id: int):
        site = (electrode_id % 32) // 8
        return Site(site)


# ---------------------------------------------------------------------------
# Stimulation parameter grid & factory
# ---------------------------------------------------------------------------

@dataclass
class StimParamGrid:
    """Contains lists of all parameters to scan.

    Attributes:
        amplitudes: list[float]
            List of amplitudes to scan. Recommended to stay between 0.1 and 5.
        durations: list[float]
            List of durations to scan. Recommended to stay between 10 and 400.
        polarities: list[StimPolarity]
            List of polarities to scan. Accepted values are StimPolarity.NegativeFirst and StimPolarity.PositiveFirst.
        interphase_delays: list[float]
            List of interphase delays to scan. How long to wait between the end of the first phase and the start of the second phase.
        nb_pulses: list[int]
            List of number of pulses to scan. Will create a spike train with the period specified in pulse_train_periods.
        pulse_train_periods: list[float]
            List of pulse train periods to scan. No effect if nb_pulses is 1.
        post_stim_ref_periods: list[float]
            List of post stimulation refractory periods to scan. Affects the time after a stimulation where no other stimulation can be sent.
        stim_shapes: list[StimShape]
            List of stimulation shapes to scan. Accepted values are StimShape.Biphasic and StimShape.BiphasicWithInterphaseDelay.
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
        attributes = {
            "amplitudes": (int, float),
            "durations": (int, float),
            "interphase_delays": (int, float),
            "nb_pulses": int,
            "pulse_train_periods": (int, float),
            "post_stim_ref_periods": (int, float),
            "stim_shapes": StimShape,
        }

        for attr, types in attributes.items():
            if not all(isinstance(item, types) for item in getattr(self, attr)):
                raise ValueError(f"All items in {attr} must be of type {types}.")

        if not isinstance(self.mea_type, MEAType):
            raise ValueError("MEA type must be a MEAType object.")

        if any(shape == StimShape.Triphasic for shape in self.stim_shapes):
            raise NotImplementedError(
                "Triphasic stimulation is not supported by this utility currently."
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

        logger.info(
            "StimParamGrid created — %d total combinations",
            self.total_combinations(),
        )
        self.display_grid()

    def total_combinations(self) -> int:
        total_combinations = 1
        for attr_name, attr in self.__dict__.items():
            if isinstance(attr, list) and not attr_name.startswith("_"):
                total_combinations *= len(attr)
        return total_combinations

    def display_grid(self):
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                logger.debug("  Grid %s: %s", k, v)


@dataclass
class StimParamFactory:
    """Factory class to create StimParam objects from the grid.

    Charge-balancing rule (from FinalSpark docs):
        phase_duration1 × phase_amplitude1 == phase_duration2 × phase_amplitude2
    This is enforced automatically: amplitude2 and duration2 mirror amplitude1 and
    duration1 so the charge is always balanced.
    """

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

    def __post_init__(self):
        # Enforce charge balancing: D1*A1 == D2*A2  (FinalSpark best-practice)
        charge1 = self.duration1 * self.amplitude1
        charge2 = self.duration2 * self.amplitude2
        if not np.isclose(charge1, charge2, rtol=1e-6):
            logger.warning(
                "Charge imbalance detected: D1*A1=%.4f != D2*A2=%.4f — "
                "correcting amplitude2 to maintain charge balance.",
                charge1,
                charge2,
            )
            if self.duration2 > 0:
                self.amplitude2 = charge1 / self.duration2
            else:
                self.duration2 = self.duration1
                self.amplitude2 = self.amplitude1
            logger.info(
                "Corrected: amplitude2=%.4f, duration2=%.4f",
                self.amplitude2,
                self.duration2,
            )

    def create_from(self):
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

    def get_names(self):
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
            logger.debug("  - %s: %s", k, v)


# ---------------------------------------------------------------------------
# Main scan class
# ---------------------------------------------------------------------------

ARTIFACT_EXCLUSION_MS = 10  # ms post-stimulation to exclude for artifact removal
PARAM_SEND_WAIT_S = 10     # seconds Intan needs to send parameters to headstage


# ---------------------------------------------------------------------------
# Lightweight mocks used when testing=True so that no hardware is contacted.
# ---------------------------------------------------------------------------

class _MockLoader:
    """Drop-in replacement for StimParamLoader in testing mode."""

    def __init__(self, stimparams):
        self.stimparams = stimparams

    def send_parameters(self):
        logger.debug("[TESTING] MockLoader.send_parameters() — skipped.")

    def disable_all_and_send(self):
        logger.debug("[TESTING] MockLoader.disable_all_and_send() — skipped.")


class _MockTriggerGen:
    """Drop-in replacement for TriggerController in testing mode."""

    def send(self, triggers):
        logger.debug("[TESTING] MockTriggerGen.send(%s) — skipped.", triggers.tolist())

    def close(self):
        logger.debug("[TESTING] MockTriggerGen.close() — skipped.")


class _MockIntan:
    """Drop-in replacement for IntanSoftware in testing mode."""

    def var_threshold(self, enabled: bool):
        logger.debug("[TESTING] MockIntan.var_threshold(%s) — skipped.", enabled)

    def close(self):
        logger.debug("[TESTING] MockIntan.close() — skipped.")


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


class StimScan:
    def __init__(
        self,
        token,
        booking_email,
        scan_channels,
        delay_btw_stim,  # in seconds
        delay_btw_channels,  # in seconds
        repeats_per_channel,
        testing=True,
        amplitudes=None,
        durations=None,
        polarities=None,
        interphase_delays=None,
        nb_pulses=None,
        pulse_train_periods=None,
        post_stim_ref_periods=None,
        stim_shapes=None,
        mea_type=None,
    ):
        """Creates a stimulation parameter scan utility.

        Accepts the same flat keyword arguments that the Datalore notebook
        runner passes via CLASS_PARAMETERS so the class can be instantiated
        directly from the remote execution environment.

        Args:
            token: str
                Experiment token (used to create an ``Experiment`` object, or
                a ``_MockExperiment`` when ``testing=True``).
            booking_email: str
                E-mail used for the FinalSpark booking / trigger controller.
            scan_channels: list[int]
                The channels to scan. Must all reside on the same MEA and exist
                in the experiment's allowed electrode list.
            delay_btw_stim: float
                Delay between each stimulation in seconds.  Must be ≥ 1 s to
                allow for fatigue recovery (FinalSpark recommendation: 1–10 s).
            delay_btw_channels: float
                Delay between each channel stimulation in seconds.  Use a
                higher value to mitigate cross-talk and fatigue.
            repeats_per_channel: int
                How many times each trigger is fired per parameter combination.
            testing: bool
                If True, no hardware is contacted (IntanSoftware,
                TriggerController, Database are replaced with mocks), no
                stimulations are sent, all inter-stimulation delays are
                skipped, but stim_history and logs are still produced.
            amplitudes: list[float] | None
                Amplitudes to scan (µA).  Falls back to StimParam default.
            durations: list[float] | None
                Phase durations to scan (µs).  Falls back to StimParam default.
            polarities: list[int|StimPolarity] | None
                Polarities to scan (0 = NegativeFirst, 1 = PositiveFirst).
            interphase_delays: list[float] | None
            nb_pulses: list[int] | None
            pulse_train_periods: list[float] | None
            post_stim_ref_periods: list[float] | None
            stim_shapes: list[int|StimShape] | None
            mea_type: int | MEAType | None
        """
        self.testing = testing

        # ---- Build Experiment ----
        # In testing mode the notebook's neuroplatform is the
        # SimulatedOrganoid-backed stub, so Experiment is safe to create.
        self.fs_experiment = Experiment(token)

        # ---- Coerce polarity ints → StimPolarity enums ----
        if polarities is not None:
            polarities = [
                StimPolarity(p) if isinstance(p, int) else p for p in polarities
            ]

        # ---- Coerce stim_shape ints → StimShape enums ----
        if stim_shapes is not None:
            stim_shapes = [
                StimShape(s) if isinstance(s, int) else s for s in stim_shapes
            ]

        # ---- Coerce mea_type int → MEAType enum ----
        if mea_type is not None and not isinstance(mea_type, MEAType):
            mea_type = MEAType(mea_type)

        # ---- Build StimParamGrid from flat lists ----
        grid_kwargs = {}
        if amplitudes is not None:
            grid_kwargs["amplitudes"] = amplitudes
        if durations is not None:
            grid_kwargs["durations"] = durations
        if polarities is not None:
            grid_kwargs["polarities"] = polarities
        if interphase_delays is not None:
            grid_kwargs["interphase_delays"] = interphase_delays
        if nb_pulses is not None:
            grid_kwargs["nb_pulses"] = nb_pulses
        if pulse_train_periods is not None:
            grid_kwargs["pulse_train_periods"] = pulse_train_periods
        if post_stim_ref_periods is not None:
            grid_kwargs["post_stim_ref_periods"] = post_stim_ref_periods
        if stim_shapes is not None:
            grid_kwargs["stim_shapes"] = stim_shapes
        if mea_type is not None:
            grid_kwargs["mea_type"] = mea_type

        self.parameter_grid = StimParamGrid(**grid_kwargs)

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

        self._trigger_gen = TriggerController(booking_email)
        self._intan = IntanSoftware()
        self._db = Database()

        self._channels_per_trigger = {}
        self._current_factory_id = None
        self._stim_history = OrderedDict()
        self._params_per_site = {}

        logger.info("Initializing StimScan (testing=%s)", testing)
        logger.info("  Experiment token: %s", self.fs_experiment.exp_name)
        logger.info("  Allowed electrodes: %s", self.fs_experiment.electrodes)
        logger.info("  Scan channels: %s", scan_channels)
        logger.info("  Delay between stims: %.2f s", delay_btw_stim)
        logger.info("  Delay between channels: %.2f s", delay_btw_channels)
        logger.info("  Repeats per channel: %d", repeats_per_channel)
        logger.info("  Total parameter combinations: %d", self.params_per_electrode)

        # ---- Expected execution duration (always based on real timing) ----
        stims_per_channel = self.params_per_electrode * repeats_per_channel
        num_channels = len(scan_channels)
        total_stim_time = stims_per_channel * delay_btw_stim * num_channels
        total_channel_time = delay_btw_channels * max(num_channels - 1, 0)
        expected_seconds = total_stim_time + total_channel_time
        expected_td = timedelta(seconds=expected_seconds)
        hours, remainder = divmod(int(expected_seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        logger.info(
            "  Expected execution duration (live): %dh %02dm %02ds  "
            "(%d stims × %.1fs delay + %d channel gaps × %.1fs)",
            hours, minutes, secs,
            stims_per_channel * num_channels, delay_btw_stim,
            max(num_channels - 1, 0), delay_btw_channels,
        )

        # ---- Validate delay_btw_stim (docs recommend 1–10 s for fatigue) ----
        if delay_btw_stim < 1.0:
            logger.warning(
                "delay_btw_stim=%.2f s is below the recommended minimum of 1 s. "
                "Stimulation fatigue may occur (see FinalSpark troubleshooting).",
                delay_btw_stim,
            )

        # ---- Validate channels are in the allowed electrode list ----
        if not np.all(np.isin(scan_channels, self.fs_experiment.electrodes)):
            bad = set(scan_channels) - set(self.fs_experiment.electrodes)
            raise ValueError(
                f"Channels {bad} are not in the allowed electrodes list for your experiment token."
            )

        # ---- Validate max electrodes per site (4x8: 8 per site) ----
        for channel in scan_channels:
            site = Site.get_from_electrode(channel)
            if site not in self._params_per_site:
                self._params_per_site[site] = 0
            self._params_per_site[site] += 1
            if self._params_per_site[site] > self.mea_type.get_electrodes_per_site():
                raise ValueError(
                    f"Too many provided channels for site {site}. "
                    f"Max is {self.mea_type.get_electrodes_per_site()} for {self.mea_type.name}."
                )

        # ---- Validate all channels on same MEA ----
        mea = MEA.get_from_electrode(scan_channels[0]).value
        if not all(
            MEA.get_from_electrode(channel).value == mea for channel in scan_channels
        ):
            raise ValueError("All channels must be on the same MEA.")
        self.mea = mea
        logger.info("  All channels on MEA %d — validation passed.", mea)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_stimulation_parameter_history(self):
        """Returns a DataFrame of all the stimulation parameters sent."""
        return pd.DataFrame.from_dict(self._stim_history, orient="index")

    # ------------------------------------------------------------------
    # Internal parameter creation
    # ------------------------------------------------------------------

    def _get_param_indices_by_trigger(self, trigger_key, loader):
        channels = []
        for param in loader.stimparams:
            if param.trigger_key == trigger_key:
                channels.append(param.index)
        return channels

    def _create_parameters_factory(self):
        parameters_factories = {}
        for i, combination in enumerate(
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
            amp, dur, pol, ipd, nbp, ptp, psrp, ss = combination
            # Charge-balanced: amplitude2 == amplitude1, duration2 == duration1
            factory = StimParamFactory(
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
            parameters_factories[i] = factory
            logger.debug(
                "Factory[%d]: amp=%.2f dur=%.1f pol=%s ipd=%.1f nbp=%d ptp=%.1f psrp=%.1f shape=%s",
                i, amp, dur, pol, ipd, nbp, ptp, psrp, ss,
            )
        logger.info("Created %d parameter factories.", len(parameters_factories))
        return parameters_factories

    # ------------------------------------------------------------------
    # Electrode index uniqueness check
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_no_duplicate_indices(params: list):
        """FinalSpark docs: do not use the same electrode index in two different
        StimParams — the later one silently overwrites the earlier one."""
        seen = {}
        for p in params:
            if p.index in seen:
                logger.warning(
                    "Duplicate electrode index %d detected across StimParams "
                    "(trigger_key %d vs %d). The later definition will overwrite "
                    "the earlier one on the Intan controller.",
                    p.index,
                    seen[p.index],
                    p.trigger_key,
                )
            seen[p.index] = p.trigger_key

    # ------------------------------------------------------------------
    # Bind parameters & build loaders
    # ------------------------------------------------------------------

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
        factory = self.parameters[self._current_factory_id]
        triggers_counter = np.zeros(16)

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
            self._validate_no_duplicate_indices(params)
            needed_triggers = np.where(triggers_counter > 0)[0]
            self.loaders = [
                (
                    needed_triggers,
                    StimParamLoader(params, self._intan, verbose=False),
                )
            ]
        elif self.mea_type == MEAType.MEA32:
            params_16 = []
            params_32 = []
            for trigger_key in range(16):
                self._bind_parameter(
                    params_16,
                    factory,
                    trigger_key,
                    triggers_counter,
                    self.mea * 32 + trigger_key,
                )
                self._bind_parameter(
                    params_32,
                    factory,
                    trigger_key,
                    triggers_counter,
                    self.mea * 32 + 16 + trigger_key,
                )
            self._validate_no_duplicate_indices(params_16 + params_32)
            needed_triggers = np.where(triggers_counter > 0)[0]
            self.loaders = [
                (
                    needed_triggers,
                    StimParamLoader(params_16, self._intan, verbose=False),
                ),
                (
                    needed_triggers,
                    StimParamLoader(params_32, self._intan, verbose=False),
                ),
            ]

        logger.debug(
            "Factory[%d]: built %d loader(s), needed triggers: %s",
            self._current_factory_id,
            len(self.loaders),
            needed_triggers.tolist(),
        )

    # ------------------------------------------------------------------
    # Stimulation sending
    # ------------------------------------------------------------------

    def _send_stim(self):
        for loader_idx, (needed_triggers, loader) in enumerate(self.loaders):
            logger.info(
                "Sending parameters to Intan (loader %d/%d)%s",
                loader_idx + 1,
                len(self.loaders),
                " — [TESTING] skipped." if self.testing else f" — this takes ~{PARAM_SEND_WAIT_S} s...",
            )
            loader.send_parameters()
            logger.info("Parameters sent successfully for loader %d.", loader_idx + 1)

            for trigger in needed_triggers:
                triggers = np.zeros(16, dtype=np.uint8)
                triggers[trigger] = 1
                channels = self._get_param_indices_by_trigger(trigger, loader)

                for rep in range(self.repeats_per_channel):
                    params_data_dict = asdict(self.parameters[self._current_factory_id])
                    params_data_dict["trigger_key"] = trigger
                    params_data_dict["param_id"] = self._current_factory_id
                    params_data_dict["channel"] = channels

                    stim_time = datetime.now(UTC)
                    self._stim_history[stim_time] = params_data_dict

                    self._trigger_gen.send(triggers)
                    logger.debug(
                        "Trigger %d fired (factory=%d, rep=%d/%d, channels=%s) at %s",
                        trigger,
                        self._current_factory_id,
                        rep + 1,
                        self.repeats_per_channel,
                        channels,
                        stim_time.isoformat(),
                    )
                    if not self.testing:
                        sleep(self.delay_btw_stim.as_seconds())

                if not self.testing:
                    logger.debug(
                        "Channel delay (%.2f s) after trigger %d",
                        self.delay_btw_channels.as_seconds(),
                        trigger,
                    )
                    sleep(self.delay_btw_channels.as_seconds())

            # Disable all parameters on this loader when done
            # (FinalSpark docs: always disable params after use)
            logger.info("Disabling all StimParams for loader %d.", loader_idx + 1)
            loader.disable_all_and_send()

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self):
        """Runs the stimulation scan.

        Follows the FinalSpark recommended pattern:
        1. Start experiment inside try/finally.
        2. Optionally disable variable threshold for consistent event detection.
        3. Iterate over all parameter combinations.
        4. Disable all StimParams and re-enable variable threshold in finally.
        5. Close Intan, trigger generator, and stop experiment.
        """
        logger.info("=" * 60)
        logger.info("STARTING STIMULATION SCAN%s", " [TESTING MODE]" if self.testing else "")
        logger.info("=" * 60)

        try:
            if self.fs_experiment.start():
                logger.info("Experiment started: %s", self.fs_experiment.exp_name)
                self.start_time = datetime.now(UTC)
                logger.info("Start time: %s", self.start_time.isoformat())

                # Disable variable threshold for stable event detection during scan
                # (FinalSpark docs: variable threshold changes during bursts/stimulation)
                total = len(self.parameters)
                for idx, factory_id in enumerate(
                    tqdm(self.parameters, desc="Parameter sweep")
                ):
                    self._current_factory_id = factory_id
                    factory = self.parameters[factory_id]
                    logger.info(
                        "--- Combination %d/%d (factory_id=%d) ---",
                        idx + 1,
                        total,
                        factory_id,
                    )
                    logger.info(
                        "  amp=%.2f dur=%.1f pol=%s shape=%s",
                        factory.amplitude1,
                        factory.duration1,
                        factory.polarity,
                        factory.stim_shape,
                    )
                    self._make_parameters()
                    self._send_stim()
                    logger.info("Combination %d/%d complete.", idx + 1, total)

                logger.info("All combinations processed.")
            else:
                logger.error(
                    "Experiment failed to start — "
                    "verify booking and that no other experiment is running."
                )
        finally:
            logger.info("Entering cleanup / finally block.")

            # Disable all StimParams (FinalSpark docs: always disable when finished)
            if self.loaders is not None:
                for loader_idx, (_, loader) in enumerate(self.loaders):
                    logger.info("Disabling loader %d StimParams.", loader_idx + 1)
                    loader.disable_all_and_send()

            # Close hardware connections
            logger.info("Closing trigger generator.")
            self._trigger_gen.close()

            logger.info("Closing Intan connection.")
            self._intan.close()

            # Stop experiment
            logger.info("Stopping experiment.")
            self.fs_experiment.stop()

            self.stop_time = datetime.now(UTC)
            logger.info("Stop time: %s", self.stop_time.isoformat())

            if self.start_time is not None:
                elapsed = self.stop_time - self.start_time
                logger.info("Total elapsed: %s", elapsed)

            total_stims = len(self._stim_history)
            logger.info("Total stimulations recorded: %d", total_stims)

            # Save output files (stim_history, spike_activity, triggers)
            self.save_results()

            logger.info("=" * 60)
            logger.info("SCAN COMPLETE")
            logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Result persistence
    # ------------------------------------------------------------------

    def save_results(self, output_dir: str = ".") -> dict[str, Path]:
        """Fetch results from the database and save them to CSV files.

        Produces the same artifacts as the remote parameter_sweep_v2:
            - stim_history.csv   — from local ``_stim_history``
            - spike_activity.csv — ``Database.get_spike_event``
            - triggers.csv       — ``Database.get_all_triggers``

        Args:
            output_dir: Directory to write the files into (created if missing).

        Returns:
            A dict mapping artifact name to its ``Path``.
        """
        if self.start_time is None or self.stop_time is None:
            logger.warning("No completed run found — skipping save_results().")
            return {}

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved: dict[str, Path] = {}

        # 1. stim_history.csv (always available)
        stim_history_path = out / "stim_history.csv"
        stim_df = self.get_stimulation_parameter_history()
        stim_df.index.name = "timestamp_utc"
        stim_df.to_csv(stim_history_path)
        saved["stim_history"] = stim_history_path
        logger.info("Saved stimulation history  -> %s (%d rows)", stim_history_path, len(stim_df))

        exp_name = self.fs_experiment.exp_name

        # 2. spike_activity.csv (Database.get_spike_event)
        activity_path = out / "spike_activity.csv"
        try:
            spike_df = self._db.get_spike_event(
                self.start_time, self.stop_time, exp_name
            )
            spike_df.to_csv(activity_path, index=False)
            saved["spike_activity"] = activity_path
            logger.info("Saved spike activity       -> %s (%d rows)", activity_path, len(spike_df))
        except Exception as exc:
            logger.error("Could not fetch spike activity from DB: %s", exc)

        # 3. triggers.csv (Database.get_all_triggers)
        triggers_path = out / "triggers.csv"
        try:
            triggers_df = self._db.get_all_triggers(
                self.start_time, self.stop_time
            )
            triggers_df.to_csv(triggers_path, index=False)
            saved["triggers"] = triggers_path
            logger.info("Saved triggers             -> %s (%d rows)", triggers_path, len(triggers_df))
        except Exception as exc:
            logger.error("Could not fetch triggers from DB: %s", exc)

        return saved

    # ------------------------------------------------------------------
    # Duration estimation
    # ------------------------------------------------------------------

    def get_scan_duration(self):
        """Returns the predicted time needed to run the scan."""
        nb_combinations = self.parameter_grid.total_combinations()
        nb_channels = len(self.scan_channels)

        stim_time = self.repeats_per_channel * self.delay_btw_stim
        inter_channel_time = self.delay_btw_channels
        time_per_combination = nb_channels * (stim_time + inter_channel_time)

        if self.mea_type == MEAType.MEA4x8:
            intan_load_time = timedelta(seconds=PARAM_SEND_WAIT_S)
            nb_loaders = 1
        elif self.mea_type == MEAType.MEA32:
            intan_load_time = timedelta(seconds=PARAM_SEND_WAIT_S * 2)
            nb_loaders = 2
        else:
            raise ValueError(
                f"Unsupported MEA type: {self.mea_type}, "
                f"choose MEAType.MEA4x8 or MEAType.MEA32"
            )

        cleanup_overhead = timedelta(seconds=PARAM_SEND_WAIT_S * nb_loaders)

        total_time = (
            nb_combinations * (time_per_combination + intan_load_time)
            + cleanup_overhead
        )
        total_seconds = int(total_time.total_seconds())
        h = total_seconds // 3600
        m = (total_seconds % 3600) // 60
        s = total_seconds % 60

        msg = f"Predicted duration of the scan: {h}h {m}m {s}s"
        logger.info(msg)
        print(msg)
        return total_time

    # ------------------------------------------------------------------
    # Plotting utilities
    # ------------------------------------------------------------------

    def _plot_raster_for_channels(
        self,
        channel_df,
        show_electrodes=None,
        s_before=60,
        s_after=5,
        param_dict=None,
        guideline_freq=None,
        exp_name=None,
    ):
        channel_df = channel_df.copy()
        if show_electrodes is None:
            show_electrodes = self.scan_channels

        channel_df.index.name = "Time"
        channel_df = channel_df.reset_index(drop=False)
        channel_df["Time"] = pd.to_datetime(channel_df["Time"])
        channel_df = channel_df.sort_values(by="Time")
        _, ax = plt.subplots(figsize=(20, 10))

        first_stim = channel_df["Time"].iloc[0]
        last_stim = channel_df["Time"].iloc[-1]
        y_axis_labs = MEA.get_electrode_range(self.mea)
        ax.set_yticks(ticks=range(len(y_axis_labs)), labels=y_axis_labs)
        offset = 0.5
        ax.set_ylim(-offset, len(y_axis_labs) - offset)
        ax.set_xlim(
            first_stim - timedelta(seconds=s_before),
            last_stim + timedelta(seconds=s_after),
        )

        events = self._db.get_spike_event(
            first_stim - timedelta(seconds=s_before),
            last_stim + timedelta(seconds=s_after),
            self.fs_experiment.exp_name if exp_name is None else exp_name,
        )
        if events.empty:
            raise ValueError("No events for the selected time in database")

        # ---- Artifact exclusion (FinalSpark docs: exclude 10 ms post-stim) ----
        stim_times = channel_df["Time"].values
        artifact_mask = pd.Series(False, index=events.index)
        for st in stim_times:
            st_ts = pd.Timestamp(st)
            artifact_mask |= (
                (events["Time"] >= st_ts)
                & (events["Time"] <= st_ts + pd.Timedelta(milliseconds=ARTIFACT_EXCLUSION_MS))
            )
        n_excluded = artifact_mask.sum()
        if n_excluded > 0:
            logger.info(
                "Raster plot: excluding %d spike events within %d ms of stimulation (artifact removal).",
                n_excluded,
                ARTIFACT_EXCLUSION_MS,
            )
        events = events[~artifact_mask]

        if guideline_freq is not None:
            try:
                for t, time_ in enumerate(
                    pd.date_range(
                        first_stim - timedelta(seconds=s_before),
                        last_stim + timedelta(seconds=s_after),
                        freq=guideline_freq,
                    )
                ):
                    ax.axvline(
                        time_,
                        color="yellow",
                        linestyle="--",
                        label=f"{guideline_freq} reference grid" if t == 0 else None,
                    )
            except ValueError:
                logger.warning(
                    "Invalid guideline_freq '%s'. "
                    "Format must be a string like '1s'. See pandas docs.",
                    guideline_freq,
                )

        for t, time_ in enumerate(channel_df["Time"]):
            ax.axvline(
                time_,
                color="r",
                linestyle="--",
                label="Stimulation" if t == 0 else None,
            )

        for channel in channel_df["channel"].values[0]:
            ax.fill_betweenx(
                [
                    channel - 0.25 - min(y_axis_labs),
                    channel + 0.25 - min(y_axis_labs),
                ],
                first_stim,
                last_stim,
                color="blue",
                alpha=0.3,
            )

        events_channels = events["channel"].unique()
        for i, electrode in enumerate(y_axis_labs):
            if electrode in events_channels:
                spikes = events[events["channel"] == electrode]["Time"]
                ax.eventplot(
                    spikes,
                    lineoffsets=i,
                    linelengths=0.5,
                    color="blue",
                )

        title = "Raster plot of spike events"
        if param_dict is not None:
            title += "\n"
            for key, value in param_dict.items():
                title += f"{key}: {value} "
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Electrode")
        plt.legend()
        plt.show()

    def plot_spike_count_per_stim(self):
        """Plots the spike count per minute for each channel."""
        spike_count_df = self._db.get_spike_count(
            self.start_time, self.stop_time, self.fs_experiment.exp_name
        )
        plt.figure(figsize=(80, 20))
        sns.lineplot(
            data=spike_count_df,
            x="Time",
            y="Spike per minutes",
            hue="channel",
            palette="colorblind",
        )
        plt.yscale("log")
        plt.title("Spike count per minute")
        plt.xlabel("Time")
        plt.ylabel("Spike count")

        for time_, stim in self._stim_history.items():
            plt.axvline(time_, color="black", linestyle="--", alpha=0.5)
            plt.text(time_, spike_count_df["Spike per minutes"].max() + 1, f"{stim}")
        plt.show()

    def plot_all_stims_for_channel(
        self,
        channel,
        s_before=1,
        s_after=2.5,
        show_electrodes=None,
        guideline_freq=None,
    ):
        """Plots the raster plot of all stimulations for a given channel."""
        stim_df = self.get_stimulation_parameter_history()
        stim_df = stim_df[stim_df["channel"].apply(lambda x: channel in x)]
        for param_id in stim_df["param_id"].unique():
            try:
                param_df = stim_df[stim_df["param_id"] == param_id]
                self._plot_raster_for_channels(
                    param_df,
                    show_electrodes=show_electrodes,
                    s_before=s_before,
                    s_after=s_after,
                    param_dict=self.parameters[param_id].get_names(),
                    guideline_freq=guideline_freq,
                )
            except ValueError:
                logger.warning("No events for parameter %d.", param_id)
                continue

    def plot_all_stims_for_param(
        self,
        param_id,
        s_before=1,
        s_after=2.5,
        show_electrodes=None,
        guideline_freq=None,
    ):
        """Plots the raster plot of all stimulations for a given parameter."""
        stim_df = self.get_stimulation_parameter_history()
        stim_df = stim_df[stim_df["param_id"] == param_id].copy()
        for stim_channel in self.scan_channels:
            param_df = stim_df[stim_df["channel"].apply(lambda x: stim_channel in x)]
            try:
                self._plot_raster_for_channels(
                    param_df,
                    show_electrodes=show_electrodes,
                    s_before=s_before,
                    s_after=s_after,
                    param_dict=self.parameters[param_id].get_names(),
                    guideline_freq=guideline_freq,
                )
            except ValueError:
                logger.warning(
                    "No events for channel %d for parameter %d.",
                    stim_channel,
                    param_id,
                )
                continue

    def plot_all_stims(
        self,
        s_before=1,
        s_after=2.5,
        show_electrodes=None,
        guideline_freq=None,
    ):
        """Plots the raster plot of all stimulations.

        NOTE: May output a very large number of plots if the scan is large.
        """
        for channel in self.scan_channels:
            self.plot_all_stims_for_channel(
                channel,
                s_before=s_before,
                s_after=s_after,
                show_electrodes=show_electrodes,
                guideline_freq=guideline_freq,
            )