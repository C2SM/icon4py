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
from pathlib import Path
from typing import Optional, Sequence

import netCDF4 as nc
import numpy as np
import xarray
import xarray as xr
from cftime import date2num
from dask.delayed import Delayed

from icon4py.model.common.decomposition.definitions import (
    ProcessProperties,
    SingleNodeProcessProperties,
)
from icon4py.model.common.grid.horizontal import HorizontalGridSize
from icon4py.model.common.grid.vertical import VerticalGridSize
from icon4py.model.driver.io.cf_utils import (
    COARDS_T_POS,
    DEFAULT_CALENDAR,
    DEFAULT_TIME_UNIT,
    INTERFACE_LEVEL_NAME,
    LEVEL_NAME,
)
from icon4py.model.driver.io.xgrid import IconUGridWriter, load_data_file


log = logging.getLogger(__name__)
processor_properties = SingleNodeProcessProperties()


def to_delta(value: str) -> timedelta:
    vals = value.split(" ")
    num = 1 if not vals[0].isnumeric() else int(vals[0])

    value = vals[0].upper() if len(vals) < 2 else vals[1].upper()
    if value == "HOUR":
        return timedelta(hours=num)
    elif value == "DAY":
        return timedelta(days=num)
    elif value == "MINUTE":
        return timedelta(minutes=num)
    elif value == "SECOND":
        return timedelta(seconds=num)
    else:
        raise NotImplementedError(f" delta {value} is not supported")


class Config(ABC):
    """
    Base class for all config classes.
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


class InvalidConfigError(Exception):
    pass


class Monitor(ABC):
    """
    Monitor component of the model.

    Monitor is a base class for components that store or freeze state for later usage but don't modify it or return any new state objects.

    Named after Sympl Monitor component: https://sympl.readthedocs.io/en/latest/monitors.html
    """

    def __str__(self):
        return "instance of {}(Monitor)".format(self.__class__)

    @abc.abstractmethod
    def store(self, state: dict, model_time: datetime, *args, **kwargs):
        """Store state and perform class specific actions on it.


        Args:
            state: dict  model state dictionary
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

        TODO (magdalena): as long as we have no terrain it is probably ok to take vct_a as vertical coordinate once there is
        terrain k-heights become [horizontal, vertical ] field

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

    # TODO (halungge) rename?
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

            if self._current_timesteps_in_file == 0:
                self._init_dataset(self._horizontal_size, self._vertical_size)
            # xarray to_netcdf does not support appending to an existing dimension: see https://github.com/pydata/xarray/issues/1672
            # we use netcdf4-python to write the file
            self._append_data(state_to_store, model_time)

            self._current_timesteps_in_file = self._current_timesteps_in_file + 1
            if self._current_timesteps_in_file == self.config.timesteps_per_file:
                self.close()

    def _append_data(self, state_to_store: dict, model_time: datetime):
        self._dataset.append(state_to_store, model_time)

    def _at_capture_time(self, model_time):
        return self._next_output_time == model_time

    def close(self):
        if self._dataset is not None:
            self._dataset.close()
            self._current_timesteps_in_file = 0


class NetcdfWriter:
    """
    Writer for netcdf files.

    Writes a netcdf file using netcdf4-python directly. Currently, this seems to be the only way that we can
      - get support for parallel (MPI available) writing
      - the possibility to append time slices to a variable already present in the file. (Xarray.to_netcdf does not support this https://github.com/pydata/xarray/issues/1672)
    """

    def __getitem__(self, item):
        return self.dataset.getncattr(item)

    def __init__(
        self,
        file_name: Path,
        vertical: VerticalGridSize,
        horizontal: HorizontalGridSize,
        global_attrs: dict,
        process_properties: ProcessProperties = processor_properties,
    ):
        self._file_name = str(file_name)
        self._process_properties = process_properties
        self.num_levels = vertical.num_lev
        self.horizontal_size = horizontal
        self.attrs = global_attrs
        self.dataset = None

    def add_dimension(self, name: str, values: xr.DataArray):
        self.dataset.createDimension(name, values.shape[0])
        dim = self.dataset.createVariable(name, values.dtype, (name,))
        dim.units = values.units
        if hasattr(values.attrs, "calendar"):
            dim.calendar = values.calendar

        dim.long_name = values.long_name
        dim.standard_name = values.standard_name

    def initialize_dataset(self):
        # TODO (magdalena) (what mode do we need `a` or `w`?
        self.dataset = nc.Dataset(
            self._file_name,
            "w",
            format="NETCDF4",
            persist=True,
            parallel=self._process_properties.comm_size > 1,
            comm=self._process_properties.comm,
        )
        log.info(f"Creating file {self._file_name} at {self.dataset.filepath()}")
        self.dataset.setncatts(self.attrs)
        ## create dimensions all except time are fixed
        self.dataset.createDimension("time", None)
        self.dataset.createDimension("level", self.num_levels)
        self.dataset.createDimension("interface_level", self.num_levels + 1)
        self.dataset.createDimension("cell", self.horizontal_size.num_cells)
        self.dataset.createDimension("vertex", self.horizontal_size.num_vertices)
        self.dataset.createDimension("edge", self.horizontal_size.num_edges)
        log.debug(f"Creating dimensions {self.dataset.dimensions} in {self._file_name}")
        # create time variables
        times = self.dataset.createVariable("times", "f8", ("time",))
        times.units = DEFAULT_TIME_UNIT
        times.calendar = DEFAULT_CALENDAR
        times.standard_name = "time"
        times.long_name = "time"
        # create vertical coordinates:
        levels = self.dataset.createVariable("levels", np.int32, ("level",))
        levels.units = "1"
        levels.long_name = "model full levels"
        levels.standard_name = LEVEL_NAME
        levels[:] = np.arange(self.num_levels, dtype=np.int32)

        interface_levels = self.dataset.createVariable(
            "interface_levels", np.int32, ("interface_level",)
        )
        interface_levels.units = "1"
        interface_levels.long_name = "model interface levels"
        interface_levels.standard_name = INTERFACE_LEVEL_NAME
        interface_levels[:] = np.arange(self.num_levels + 1, dtype=np.int32)

        # TODO (magdalena) add vct_a as vertical coordinate?

    def append(self, state_to_append: dict[str, xarray.DataArray], model_time: datetime):
        """
        Append the fields to the dataset.

        Appends a time slice of the fields in the state_to_append dictionary to the dataset for the `model_time` expanding the time coordinate by the `model_time`.
        Args:
            state_to_append: fields to append
            model_time: time of the model state

        Returns:

        """
        time = self.dataset["times"]
        time_pos = len(time)
        time[time_pos] = date2num(model_time, units=time.units, calendar=time.calendar)
        for k, new_slice in state_to_append.items():
            standard_name = new_slice.standard_name
            new_slice = to_canonical_dim_order(new_slice)
            assert standard_name is not None, f"No short_name provided for {standard_name}."
            ds_var = filter_by_standard_name(self.dataset.variables, standard_name)
            if not ds_var:
                dimensions = ("time", *new_slice.dims)
                new_var = self.dataset.createVariable(k, new_slice.dtype, dimensions)
                new_var[0, :] = new_slice.data
                new_var.units = new_slice.units
                new_var.standard_name = new_slice.standard_name
                new_var.long_name = new_slice.long_name
                new_var.coordinates = new_slice.coordinates
                new_var.mesh = new_slice.mesh
                new_var.location = new_slice.location

            else:
                var_name = ds_var.get(k).name
                dims = ds_var.get(k).dimensions
                shape = ds_var.get(k).shape
                assert (
                    len(new_slice.dims) == len(dims) - 1
                ), f"Data variable dimensions do not match for {standard_name}."

                # TODO (magdalena) change for parallel/distributed case: where we write at `global_index` field on the node for the horizontal dim.
                # we can acutally assume fixed index ordering here, input arrays are  re-shaped to canonical order (see above)

                right = (slice(None),) * (len(dims) - 1)
                expand_slice = (slice(shape[COARDS_T_POS] - 1, shape[COARDS_T_POS]),)
                slices = expand_slice + right
                self.dataset.variables[var_name][slices] = new_slice.data

    def close(self):
        if self.dataset.isopen():
            self.dataset.close()

    @property
    def dims(self) -> dict:
        return self.dataset.dimensions

    @property
    def variables(self) -> dict:
        return self.dataset.variables


# TODO (magdalena) this should be moved to a separate file
class XArrayNetCDFWriter:
    from xarray import Dataset

    def __init__(self, filename, mode="a"):
        self.filename = filename
        self.mode = mode

    def write(self, dataset: Dataset, immediate=True) -> [Delayed | None]:
        delayed = dataset.to_netcdf(
            self.filename,
            mode=self.mode,
            engine="netcdf4",
            format="NETCDF4",
            unlimited_dims=["time"],
            compute=immediate,
        )
        return delayed

    def close(self):
        self.dataset.close()
        self.dataset = None
        return self.dataset

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def generate_name(fname: str, counter: int) -> str:
    stem = fname.split(".")[0]
    return f"{stem}_{counter:0>4}.nc"


def filter_by_standard_name(model_state: dict, value: str):
    return {k: v for k, v in model_state.items() if value == v.standard_name}


def to_canonical_dim_order(data: xarray.DataArray) -> xarray.DataArray:
    """Check for dimension being in canoncial order ('T', 'Z', 'Y', 'X') and return them in this order."""
    dims = data.dims
    if len(dims) >= 2:
        if dims[0] in ("cell", "edge", "vertex") and dims[1] in (
            "height",
            "level",
            "interface_level",
        ):
            return data.transpose(dims[1], dims[0], *dims[2:], transpose_coords=True)
        else:
            return data
