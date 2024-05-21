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
import logging
from abc import ABC
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence

from icon4py.model.common.components.exceptions import InvalidConfigError
from icon4py.model.common.components.monitor import Monitor
from icon4py.model.common.grid.horizontal import HorizontalGridSize
from icon4py.model.common.grid.vertical import VerticalGridSize
from icon4py.model.driver.io.cf_utils import (
    DEFAULT_CALENDAR,
    DEFAULT_TIME_UNIT,
)
from icon4py.model.driver.io.ugrid import IconUGridWriter, load_data_file
from icon4py.model.driver.io.writers import NetcdfWriter


log = logging.getLogger(__name__)


class OutputInterval(str, Enum):
    SECOND = "SECOND"
    MINUTE = "MINUTE"
    HOUR = "HOUR"
    DAY = "DAY"


def to_delta(value: str) -> timedelta:
    vals = value.split(" ")
    num = 1 if not vals[0].isnumeric() else int(vals[0])

    value = vals[0].upper() if len(vals) < 2 else vals[1].upper()
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

    #TODO (halungge) Need to visit this, when we address configuration
    """

    def __str__(self):
        return "instance of {}(Config)".format(self.__class__)

    @abc.abstractmethod
    def validate(self):
        """
        Validate the config.

        Raises:
            InvalidConfigError: if the config is invalid
        """

        pass


@dataclass(frozen=True)
class FieldGroupIoConfig(Config):
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

    def _validate_filename(self):
        assert self.filename, "No filename provided for output."
        if self.filename.startswith("/"):
            raise InvalidConfigError(f"Filename may not be an absolute path: {self.filename}.")

    def validate(self):
        if not self.output_interval:
            raise InvalidConfigError("No output interval provided.")
        if not self.variables:
            raise InvalidConfigError("No variables provided for output.")
        self._validate_filename()


@dataclass(frozen=True)
class IoConfig(Config):
    """
    Structured and hierarchical config for IO.

    Holds some general configuration and a collection of configuraitions for each field group

    """

    output_path: str = "./output/"
    field_configs: Sequence[FieldGroupIoConfig] = ()

    time_units = DEFAULT_TIME_UNIT
    calendar = DEFAULT_CALENDAR

    def validate(self):
        if not self.field_configs:
            log.warning("No field configurations provided for output")
        else:
            for field_config in self.field_configs:
                field_config.validate()


class IoMonitor(Monitor):
    """
    Composite Monitor for all IO Groups
    """

    def __init__(
        self,
        config: IoConfig,
        vertical_size: VerticalGridSize,
        horizontal_size: HorizontalGridSize,
        grid_file_name: str,
        grid_id: str,
    ):
        config.validate()
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
            for conf in config.field_configs
        ]

    def _read_grid_attrs(self) -> dict:
        with load_data_file(self._grid_file) as ds:
            return ds.attrs

    def _initialize_output(self):
        self._create_output_dir()
        self._write_ugrid()

    def _create_output_dir(self):
        path = Path(self.config.output_path)
        try:
            path.mkdir(parents=True, exist_ok=False, mode=0o777)
            self._output_path = path
        except OSError as error:
            log.error(f"Output directory at {path} exists: {error}.")

    def _write_ugrid(self):
        writer = IconUGridWriter(self._grid_file, self._output_path)
        writer(validate=True)

    @property
    def path(self):
        return self._output_path

    def store(self, state, model_time: datetime, **kwargs):
        for monitor in self._group_monitors:
            monitor.store(state, model_time, **kwargs)

    def close(self):
        for monitor in self._group_monitors:
            monitor.close()


class FieldGroupMonitor(Monitor):
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

    @property
    def output_path(self) -> Path:
        return self._output_path

    def __init__(
        self,
        config: FieldGroupIoConfig,
        vertical: VerticalGridSize,
        horizontal: HorizontalGridSize,
        grid_id: str,
        output_path: Path = Path(__file__).parent,
    ):
        self._global_attrs = dict(
            Conventions="CF-1.7",  # TODO (halungge) check changelog? latest version is 1.11
            title=config.nc_title,
            comment=config.nc_comment,
            institution="ETH Zurich and MeteoSwiss",
            source="ICON4Py",
            history="Created by ICON4Py",
            references="https://icon4py.github.io",
            external_variables="",  # TODO (halungge) needed if cell_measure (cell area) variables are in external file
            uuidOfHGrid=grid_id,
        )
        self.config = config
        self._vertical_size = vertical
        self._horizontal_size = horizontal
        self._field_names = config.variables
        self._handle_output_path(output_path, config.filename)
        self._next_output_time = datetime.fromisoformat(config.start_time)
        self._time_delta = to_delta(config.output_interval)
        self._file_counter = 0
        self._current_timesteps_in_file = 0
        self._dataset = None

    def _handle_output_path(self, output_path: Path, filename: str):
        file = output_path.joinpath(filename).absolute()
        path = file.parent
        path.mkdir(parents=True, exist_ok=True, mode=0o777)
        self._output_path = path
        self._file_name_pattern = file.name

    def _init_dataset(self, horizontal_size: HorizontalGridSize, vertical_grid: VerticalGridSize):
        """Initialise the dataset with global attributes and dimensions.

        TODO (magdalena): as long as we have no terrain it is probably ok to take vct_a as vertical
                          coordinate once there is terrain k-heights become [horizontal, vertical ] field

        """
        if self._dataset is not None:
            self._dataset.close()
        self._file_counter += 1
        filename = generate_name(self._file_name_pattern, self._file_counter)
        filename = self._output_path.joinpath(filename)
        df = NetcdfWriter(filename, vertical_grid, horizontal_size, self._global_attrs)
        df.initialize_dataset()
        self._dataset = df

    def _update_fetch_times(self):
        self._next_output_time = self._next_output_time + self._time_delta

    def store(self, state: dict, model_time: datetime, **kwargs):
        """Pick fields from the state dictionary to be written to disk.

        Args:
            state: dict  model state dictionary
            time: float  model time
        """
        # TODO (halungge) how to handle non time matches? That is if the model time jumps over the output time
        if self._at_capture_time(model_time):
            # TODO this should do a deep copy of the data
            state_to_store = {field: state[field] for field in self._field_names}
            logging.info(f"Storing fields {state_to_store.keys()} at {model_time}")
            self._update_fetch_times()

            if self._do_initialize_new_file():
                self._init_dataset(self._horizontal_size, self._vertical_size)
            self._append_data(state_to_store, model_time)

            self._update_current_file_count()
            if self._is_file_limit_reached():
                self.close()

    def _update_current_file_count(self):
        self._current_timesteps_in_file = self._current_timesteps_in_file + 1

    def _do_initialize_new_file(self):
        return self._current_timesteps_in_file == 0

    def _is_file_limit_reached(self):
        return (
            self.config.timesteps_per_file
            > 0  # since _current_timesteps_in_file >=0 this is not even necessary
            and self._current_timesteps_in_file == self.config.timesteps_per_file
        )

    def _append_data(self, state_to_store: dict, model_time: datetime):
        self._dataset.append(state_to_store, model_time)

    def _at_capture_time(self, model_time):
        return self._next_output_time == model_time

    def close(self):
        if self._dataset is not None:
            self._dataset.close()
            self._current_timesteps_in_file = 0


def generate_name(fname: str, counter: int) -> str:
    stem = fname.split(".")[0]
    return f"{stem}_{counter:0>4}.nc"
