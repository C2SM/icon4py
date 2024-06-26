# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import abc
import enum
import logging
import pathlib
import uuid
from abc import ABC
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Sequence, TypedDict

from typing_extensions import Required

import icon4py.model.common.exceptions as exceptions
from icon4py.model.common.components import monitor
from icon4py.model.common.grid import horizontal as h_grid, vertical as v_grid
from icon4py.model.common.io import cf_utils, ugrid, writers


log = logging.getLogger(__name__)


class OutputInterval(str, enum.Enum):
    SECOND = "SECOND"
    MINUTE = "MINUTE"
    HOUR = "HOUR"
    DAY = "DAY"


def to_delta(value: str) -> timedelta:
    vals = value.split(" ")
    num = 1 if not vals[0].isnumeric() else int(vals[0])
    assert num >= 0, f"Delta value must be positive: {num}"

    value = vals[0].upper() if len(vals) < 2 else vals[1].upper()
    value = value[:-1] if value.endswith("S") else value

    if value == OutputInterval.HOUR:
        return timedelta(hours=num)
    elif value == OutputInterval.DAY:
        return timedelta(days=num)
    elif value == OutputInterval.MINUTE:
        return timedelta(minutes=num)
    elif value == OutputInterval.SECOND:
        return timedelta(seconds=num)
    else:
        raise NotImplementedError(f" Delta '{value}' is not supported.")


class Config(ABC):
    """
    Base class for all config classes.

    # TODO (halungge) Need to visit this, when we address configuration
    """

    def __str__(self):
        return "instance of {}(Config)".format(self.__class__)

    @abc.abstractmethod
    def validate(self) -> None:
        """
        Validate the config.

        Raises:
            InvalidConfigError: if the config is invalid
        """

        pass


@dataclass(frozen=True)
class FieldGroupIOConfig(Config):
    """
    Structured config for IO of a field group.

    Field group is a number of fields that are output at the same time intervals on the same grid
    (can be any horizontal dimension) and vertical levels.

    """

    output_interval: str
    start_time: Optional[str]
    filename: str
    variables: list[str]
    timesteps_per_file: int = 10
    nc_title: str = "ICON4Py Simulation"
    nc_comment: str = "ICON inspired code in Python and GT4Py"

    def __post_init__(self):
        self.validate()

    def _validate_filename(self) -> None:
        if not self.filename:
            raise exceptions.InvalidConfigError("Output filename is missing.")
        if self.filename.startswith("/"):
            raise exceptions.InvalidConfigError(
                f"Filename may not be an absolute path: {self.filename}."
            )

    def validate(self) -> None:
        if not self.output_interval:
            raise exceptions.InvalidConfigError("No output interval provided.")
        if not self.variables:
            raise exceptions.InvalidConfigError("No variables provided for output.")
        self._validate_filename()


@dataclass(frozen=True)
class IOConfig(Config):
    """
    Structured and hierarchical config for IO.

    Holds some general configuration and a collection of configurations for each field group.

    """

    output_path: str = "./output/"
    field_groups: Sequence[FieldGroupIOConfig] = ()

    time_units = cf_utils.DEFAULT_TIME_UNIT
    calendar = cf_utils.DEFAULT_CALENDAR

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        if not self.field_groups:
            log.warning("No field configurations provided for output")
        else:
            for field_config in self.field_groups:
                field_config.validate()


class IOMonitor(monitor.Monitor):
    """
    Composite Monitor for all IO groups.
    """

    def __init__(
        self,
        config: IOConfig,
        vertical_size: v_grid.VerticalModelParams,
        horizontal_size: h_grid.HorizontalGridSize,
        grid_file_name: str,
        grid_id: uuid.UUID,
    ):
        self.config = config
        self._grid_file = grid_file_name
        self._initialize_output()
        self._group_monitors = [
            FieldGroupMonitor(
                conf,
                vertical=vertical_size,
                horizontal=horizontal_size,
                grid_id=grid_id,
                output_path=self._output_path,
            )
            for conf in config.field_groups
        ]

    def _read_grid_attrs(self) -> dict:
        with ugrid.load_data_file(self._grid_file) as ds:
            return ds.attrs

    def _initialize_output(self) -> None:
        self._create_output_dir()
        self._write_ugrid()

    def _create_output_dir(self) -> None:
        path = pathlib.Path(self.config.output_path)
        try:
            path.mkdir(parents=True, exist_ok=False, mode=0o777)
            self._output_path = path
        except OSError as error:
            log.error(f"Output directory at {path} exists: {error}.")

    def _write_ugrid(self) -> None:
        writer = ugrid.IconUGridWriter(self._grid_file, self._output_path)
        writer(validate=True)

    @property
    def path(self):
        return self._output_path

    def store(self, state: dict, model_time: datetime, **kwargs) -> None:
        for m in self._group_monitors:
            m.store(state, model_time, **kwargs)

    def close(self):
        for m in self._group_monitors:
            m.close()


class GlobalFileAttributes(TypedDict, total=False):
    """
    Global file attributes of  a ICON generated netCDF file.

    Attribute map what ICON produces, (including the upper, lower case pattern).
    Omissions (possibly incomplete):
    - 'CDI' used for the supported CDI version (http://mpimet.mpg.de/cdi) since we do not support it

    Additions:
    - 'external_variables': variable used by CF conventions if cell_measure variables are used from an external file'
    """

    #: version of the supported CF conventions
    Conventions: Required[str]  # TODO (halungge) check changelog? latest version is 1.11

    #: unique id of the horizontal grid used in the simulation (from grid file)
    uuidOfHGrid: Required[uuid.UUID]

    #: institution name
    institution: Required[str]

    #: title of the file or simulation
    title: Required[str]

    #: source code repository
    source: Required[str]

    #: path of the binary and generation time stamp of the file
    history: Required[str]

    #: references for publication # TODO (halungge) check if this is the right reference
    references: str
    comment: str
    external_variables: str


class FieldGroupMonitor(monitor.Monitor):
    """
    Monitor for a group of fields.

    This monitor is responsible for storing a group of fields that are output at the same time intervals.
    """

    @property
    def next_output_time(self):
        return self._next_output_time

    @property
    def time_delta(self):
        return self._time_delta

    def __init__(
        self,
        config: FieldGroupIOConfig,
        vertical: v_grid.VerticalModelParams,
        horizontal: h_grid.HorizontalGridSize,
        grid_id: uuid.UUID,
        time_units: str = cf_utils.DEFAULT_TIME_UNIT,
        calendar: str = cf_utils.DEFAULT_CALENDAR,
        output_path: pathlib.Path = pathlib.Path(__file__).parent,
    ):
        self._global_attrs: GlobalFileAttributes = {
            "Conventions": "CF-1.7",  # TODO (halungge) check changelog? latest version is 1.11
            "title": config.nc_title,
            "comment": config.nc_comment,
            "institution": "ETH Zurich and MeteoSwiss",
            "source": "https://icon4py.github.io",
            "history": output_path.absolute().as_posix()
            + " "
            + datetime.now().isoformat(),  # TODO (halungge) this is actually the path to the binary in ICON not the output path
            "references": "https://icon4py.github.io",
            "uuidOfHGrid": grid_id,
        }
        self.config = config
        self._time_properties = writers.TimeProperties(time_units, calendar)
        self._vertical_size = vertical
        self._horizontal_size = horizontal
        self._field_names = config.variables
        self._handle_output_path(output_path, config.filename)
        self._next_output_time = datetime.fromisoformat(config.start_time)
        self._time_delta = to_delta(config.output_interval)
        self._file_counter = 0
        self._current_timesteps_in_file = 0
        self._dataset = None

    @property
    def output_path(self) -> pathlib.Path:
        return self._output_path

    def _handle_output_path(self, output_path: pathlib.Path, filename: str):
        file = output_path.joinpath(filename).absolute()
        path = file.parent
        path.mkdir(parents=True, exist_ok=True, mode=0o777)
        self._output_path = path
        self._file_name_pattern = file.name

    def _init_dataset(
        self,
        vertical_params: v_grid.VerticalModelParams,
        horizontal_size: h_grid.HorizontalGridSize,
    ) -> None:
        """Initialise the dataset with global attributes and dimensions.

        TODO (magdalena): as long as we have no terrain it is probably ok to take vct_a as vertical
                          coordinate once there is terrain k-heights become [horizontal, vertical ] field

        """
        if self._dataset is not None:
            self._dataset.close()
        self._file_counter += 1
        filename = generate_name(self._file_name_pattern, self._file_counter)
        filename = self._output_path.joinpath(filename)
        df = writers.NETCDFWriter(
            filename,
            vertical_params,
            horizontal_size,
            self._time_properties,
            self._global_attrs,
        )
        df.initialize_dataset()
        self._dataset = df

    def _update_fetch_times(self) -> None:
        self._next_output_time = self._next_output_time + self._time_delta

    def store(self, state: dict, model_time: datetime, **kwargs) -> None:
        """Pick fields from the state dictionary to be written to disk.

        Args:
            state: dict  model state dictionary
            model_time: the current time step of the simulation
        """
        # TODO (halungge) how to handle non time matches? That is if the model time jumps over the output time
        if self._at_capture_time(model_time):
            # TODO (halungge) this should do a deep copy of the data
            try:
                state_to_store = {field: state[field] for field in self._field_names}
            except KeyError as e:
                log.error(f"Field '{e.args[0]}' is missing in state.")
                self.close()
                raise exceptions.IncompleteStateError(e.args[0]) from e

            log.info(f"Storing fields {state_to_store.keys()} at {model_time}")
            self._update_fetch_times()

            if self._do_initialize_new_file():
                self._init_dataset(self._vertical_size, self._horizontal_size)
            self._append_data(state_to_store, model_time)

            self._update_current_file_count()
            if self._is_file_limit_reached():
                self.close()

    def _update_current_file_count(self) -> None:
        self._current_timesteps_in_file = self._current_timesteps_in_file + 1

    def _do_initialize_new_file(self) -> bool:
        return self._current_timesteps_in_file == 0

    def _is_file_limit_reached(self) -> bool:
        return 0 < self.config.timesteps_per_file == self._current_timesteps_in_file

    def _append_data(self, state_to_store: dict, model_time: datetime) -> None:
        self._dataset.append(state_to_store, model_time)

    def _at_capture_time(self, model_time) -> bool:
        return self._next_output_time == model_time

    def close(self) -> None:
        if self._dataset is not None:
            self._dataset.close()
            self._current_timesteps_in_file = 0


def generate_name(fname: str, counter: int) -> str:
    stem = fname.split(".")[0]
    return f"{stem}_{counter:0>4}.nc"
