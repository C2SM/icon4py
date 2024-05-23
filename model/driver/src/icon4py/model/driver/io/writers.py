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

import logging
from datetime import datetime
from pathlib import Path

import netCDF4 as nc
import numpy as np
import xarray as xr

from icon4py.model.common.decomposition.definitions import (
    ProcessProperties,
    SingleNodeProcessProperties,
)
from icon4py.model.common.grid.base import HorizontalGridSize
from icon4py.model.common.grid.vertical import VerticalGridSize
from icon4py.model.driver.io.cf_utils import (
    COARDS_T_POS,
    DEFAULT_CALENDAR,
    DEFAULT_TIME_UNIT,
    INTERFACE_LEVEL_NAME,
    LEVEL_NAME,
    date2num,
    to_canonical_dim_order,
)


log = logging.getLogger(__name__)
processor_properties = SingleNodeProcessProperties()


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

    def append(self, state_to_append: dict[str, xr.DataArray], model_time: datetime):
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


def filter_by_standard_name(model_state: dict, value: str):
    return {k: v for k, v in model_state.items() if value == v.standard_name}
