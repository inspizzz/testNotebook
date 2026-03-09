from time import sleep
from datetime import datetime, timedelta, UTC
import numpy as np
from collections import OrderedDict
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass, asdict, field
from enum import Enum
from itertools import product
from pathlib import Path

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

from .parameters_loader import StimParamLoader


class ExtendedTimedelta(timedelta):
    """Minimal extension of the timedelta class to add simple time unit conversion methods."""

    def as_minutes(self) -> float:
        """Returns the total number of minutes."""
        return self.total_seconds() / 60

    def as_seconds(self) -> float:
        """Returns the total number of seconds."""
        return self.total_seconds()

    def as_milliseconds(self) -> float:
        """Returns the total number of milliseconds."""
        return self.total_seconds() * 1e3

    def as_microseconds(self) -> float:
        """Returns the total number of microseconds."""
        return self.total_seconds() * 1e6

    def to_hertz(self) -> float:
        """Returns the frequency in hertz."""
        if self.total_seconds() == 0:
            print("Period is zero. Returning np.inf Hz.")
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
            List of number of pulses to scan. Will creat a spike train with the period specified in pulse_train_periods.
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

    def total_combinations(self) -> int:
        """Returns the total number of combinations."""
        total_combinations = 1
        for attr_name, attr in self.__dict__.items():
            if isinstance(attr, list) and not attr_name.startswith("_"):
                total_combinations *= len(attr)
        return total_combinations

    def display_grid(self):
        """Prints all the parameters in the grid."""
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                print(f"{k}: {v}")


@dataclass
class StimParamFactory:
    """Factory class to create StimParam objects from the grid."""

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
            print(f"- {k}: {v}")


class StimScan:
    def __init__(
        self,
        token: str,
        booking_email: str,
        amplitudes: list[float],
        durations: list[float],
        polarities: list[StimPolarity],
        scan_channels: list[int],
        delay_btw_stim: float,       # in seconds
        delay_btw_channels: float,   # in seconds
        repeats_per_channel: int,
        testing: bool = False,
    ):
        """Creates a stimulation parameter scan utility.

        Args:
            token: str
                Experiment token provided by neuroplatform.
            booking_email: str
                Email used for booking the trigger controller.
            amplitudes: list[float]
                List of amplitudes to scan.
            durations: list[float]
                List of durations to scan.
            polarities: list[StimPolarity]
                List of polarities to scan.
            scan_channels: list[int]
                The list of channels to scan. Must be part of electrodes listed in
                the experiment token.
            delay_btw_stim: float
                The delay between each stimulation in seconds. Note that if you are
                using pulse trains, this should be larger than the duration of the
                full pulse train.
            delay_btw_channels: float
                The delay between each channel stimulation in seconds. Use a higher
                value if you are concerned about fatigue or cross-talk.
            repeats_per_channel: int
                Number of times each parameter combination is repeated per channel.
            testing: bool
                If True, runs the scan in dry-run mode: parameters are built and
                logged as normal, but no stimulations are sent to the hardware.
                Useful for verifying scan configuration without using experiment time.
                Defaults to False.
        """
        self.testing = testing
        if self.testing:
            print(
                "[TESTING MODE] No stimulations will be sent. "
                "Parameter generation and timing logic will still execute."
            )

        self.fs_experiment = Experiment(token=token)
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

        self._trigger_gen = TriggerController(booking_email)
        self._intan = IntanSoftware()
        self._db = Database()

        self._channels_per_trigger = {}
        self._current_factory_id = None
        self._stim_history = OrderedDict()
        self._params_per_site = {}

        if not np.all(np.isin(scan_channels, self.fs_experiment.electrodes)):
            raise ValueError(
                "Some channels are not in the allowed electrodes list for your experiment token."
            )

        for channel in scan_channels:
            site = Site.get_from_electrode(channel)
            if site not in self._params_per_site:
                self._params_per_site[site] = 0
            self._params_per_site[site] += 1
            if self._params_per_site[site] > self.mea_type.get_electrodes_per_site():
                raise ValueError(
                    f"Too many provided channels for site {site}. Are all channels on the same MEA?"
                )

        mea = MEA.get_from_electrode(scan_channels[0]).value
        if not all(
            MEA.get_from_electrode(channel).value == mea for channel in scan_channels
        ):
            raise ValueError("All channels must be on the same MEA.")
        self.mea = mea

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_stimulation_parameter_history(self) -> pd.DataFrame:
        """Returns a DataFrame of all the stimulation parameters sent (or
        simulated in testing mode)."""
        return pd.DataFrame.from_dict(self._stim_history, orient="index")

    def get_scan_duration(self) -> timedelta:
        """Returns the predicted time needed to run the scan with the chosen parameters."""
        nb_combinations = self.parameter_grid.total_combinations()
        nb_channels = len(self.scan_channels)

        stim_time = self.repeats_per_channel * self.delay_btw_stim
        inter_channel_time = self.delay_btw_channels
        time_per_combination = nb_channels * (stim_time + inter_channel_time)

        if self.mea_type == MEAType.MEA4x8:
            intan_load_time = timedelta(seconds=20)
            nb_loaders = 1
        elif self.mea_type == MEAType.MEA32:
            intan_load_time = timedelta(seconds=40)
            nb_loaders = 2
        else:
            raise ValueError(
                f"Unsupported MEA type: {self.mea_type}, "
                "choose MEAType.MEA4x8 or MEAType.MEA32"
            )

        cleanup_overhead = timedelta(seconds=10 * nb_loaders)
        total_time = (
            nb_combinations * (time_per_combination + intan_load_time)
            + cleanup_overhead
        )

        total_seconds = int(total_time.total_seconds())
        h = total_seconds // 3600
        m = (total_seconds % 3600) // 60
        s = total_seconds % 60
        print(f"Predicted duration of the scan : {h}h {m}m {s}s")
        return total_time

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self):
        """Runs the stimulation scan.

        In testing mode the experiment is still started/stopped so that timing
        and parameter-building logic can be validated, but no triggers are fired
        and no parameters are loaded onto the hardware.
        """
        try:
            if self.fs_experiment.start():
                self.start_time = datetime.now(UTC)
                for factory_id in tqdm(self.parameters):
                    self._current_factory_id = factory_id
                    self._make_parameters()
                    self._send_stim()
        finally:
            if self.loaders is not None and not self.testing:
                for _, loader in self.loaders:
                    loader.disable_all_and_send()
            self._trigger_gen.close()
            self._intan.close()
            self.fs_experiment.stop()
            self.stop_time = datetime.now(UTC)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    def save_results(self, output_dir: str = ".") -> dict[str, Path]:
        """Fetches results from the database and saves them to CSV files.

        Three files are written:

        * ``stim_history.csv``    — the stimulation parameter log recorded
          during the run (generated locally, no DB query needed).
        * ``spike_activity.csv``  — spike events fetched from the database for
          the duration of the run, via ``db.get_spike_event``.
        * ``triggers.csv``        — trigger records fetched from the database
          for the duration of the run, via ``db.get_all_triggers``.

        Args:
            output_dir: str
                Directory where the CSV files will be written. Defaults to the
                current working directory. The directory is created if it does
                not exist.

        Returns:
            dict[str, Path]: Mapping of file role to the saved Path, with keys
            ``"stim_history"``, ``"spike_activity"``, and ``"triggers"``.

        Raises:
            RuntimeError: If the scan has not been run yet (``start_time`` or
            ``stop_time`` are None).
        """
        if self.start_time is None or self.stop_time is None:
            raise RuntimeError(
                "No completed run found. Call run() before save_results()."
            )

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        saved_paths: dict[str, Path] = {}
        exp_name = self.fs_experiment.exp_name

        # ---- 1. Stimulation parameter history (local, no DB needed) --------
        stim_history_path = out / "stim_history.csv"
        stim_df = self.get_stimulation_parameter_history()
        stim_df.index.name = "timestamp_utc"
        stim_df.to_csv(stim_history_path)
        saved_paths["stim_history"] = stim_history_path
        print(f"Saved stimulation history  -> {stim_history_path}")

        # ---- 2. Spike activity (db.get_spike_event) ------------------------
        # Per best-practices guidance: keep query windows manageable.
        # We query the full run window here; callers may wish to split this
        # for very long runs.
        activity_path = out / "spike_activity.csv"
        try:
            spike_df = self._db.get_spike_event(
                self.start_time,
                self.stop_time,
                exp_name,
            )
            spike_df.to_csv(activity_path, index=False)
            saved_paths["spike_activity"] = activity_path
            print(f"Saved spike activity       -> {activity_path}")
        except Exception as exc:
            print(f"Warning: could not fetch spike activity from DB: {exc}")

        # ---- 3. Triggers (db.get_all_triggers) -----------------------------
        triggers_path = out / "triggers.csv"
        try:
            triggers_df = self._db.get_all_triggers(
                self.start_time,
                self.stop_time,
            )
            triggers_df.to_csv(triggers_path, index=False)
            saved_paths["triggers"] = triggers_path
            print(f"Saved triggers             -> {triggers_path}")
        except Exception as exc:
            print(f"Warning: could not fetch triggers from DB: {exc}")

        return saved_paths

    # ------------------------------------------------------------------
    # Private helpers
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
            parameters_factories[i] = StimParamFactory(
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
        return parameters_factories

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
        """Builds loader objects for the current factory.

        In testing mode the loaders are still created so that parameter
        validation logic is exercised, but ``send_parameters`` is not called.
        """
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

    def _send_stim(self):
        """Sends stimulation triggers for the current factory.

        When ``self.testing`` is True:
          - Parameters are *not* loaded onto the hardware.
          - Triggers are *not* fired.
          - The stim history is still updated so that post-run analysis methods
            work correctly in testing mode.
          - ``sleep`` calls are still executed so that timing estimates remain
            valid.
        """
        for needed_triggers, loader in self.loaders:
            if not self.testing:
                loader.send_parameters()

            for trigger in needed_triggers:
                triggers = np.zeros(16, dtype=np.uint8)
                triggers[trigger] = 1

                for _ in range(self.repeats_per_channel):
                    params_data_dict = asdict(self.parameters[self._current_factory_id])
                    params_data_dict["trigger_key"] = trigger
                    params_data_dict["param_id"] = self._current_factory_id
                    params_data_dict["channel"] = self._get_param_indices_by_trigger(
                        trigger, loader
                    )
                    self._stim_history[datetime.now(UTC)] = params_data_dict

                    if not self.testing:
                        self._trigger_gen.send(triggers)

                    sleep(self.delay_btw_stim.as_seconds())

                sleep(self.delay_btw_channels.as_seconds())

            if not self.testing:
                loader.disable_all_and_send()
