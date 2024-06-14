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
import dataclasses
import functools
import logging
import pathlib
from datetime import datetime

import netCDF4 as nc
import numpy as np
import xarray as xr

from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.grid import horizontal as h_grid, vertical as v_grid
from icon4py.model.common.io import cf_utils


EDGE = "edge"
VERTEX = "vertex"
CELL = "cell"
MODEL_INTERFACE_LEVEL = "interface_level"
MODEL_LEVEL = "level"
TIME = "time"

log = logging.getLogger(__name__)
processor_properties = decomp_defs.SingleNodeProcessProperties()


@dataclasses.dataclass
class TimeProperties:
    units: str
    calendar: str


class NETCDFWriter:
    """
    Writer for netcdf files.

    Writes a netcdf file using netcdf4-python directly. Currently, this seems to be the only way that we can
      - get support for parallel (MPI available) writing
      - the possibility to append time slices to a variable already present in the file. (Xarray.to_netcdf does not support this https://github.com/pydata/xarray/issues/1672)
    """

    def __init__(
        self,
        file_name: pathlib.Path,
        vertical: v_grid.VerticalModelParams,
        horizontal: h_grid.HorizontalGridSize,
        time_properties: TimeProperties,
        global_attrs: dict,
        process_properties: decomp_defs.ProcessProperties = processor_properties,
    ):
        self._file_name = str(file_name)
        self._process_properties = process_properties
        self._time_properties = time_properties
        self._vertical_params = vertical
        self._horizontal_size = horizontal
        self.attrs = global_attrs
        self.dataset = None

    def __getitem__(self, item):
        return self.dataset.getncattr(item)

    @functools.cached_property
    def num_levels(self) -> int:
        # TODO (@halungge) fix once PR 470 (https://github.com/C2SM/icon4py/pull/470) is merged
        return self._vertical_params.vct_a.ndarray.shape[0] - 1

    @functools.cached_property
    def num_interfaces(self) -> int:
        # TODO (@halungge) fix once PR 470 (https://github.com/C2SM/icon4py/pull/470) is merged
        return self._vertical_params.vct_a.ndarray.shape[0]

    def initialize_dataset(self) -> None:
        self.dataset = nc.Dataset(
            self._file_name,
            "w",
            format="NETCDF4",
            persist=True,
            parallel=self._process_properties.comm_size > 1,
            comm=self._process_properties.comm,
        )
        log.info(f"Creating file {self._file_name} at {self.dataset.filepath()}")
        self.dataset.setncatts({k: str(v) for (k, v) in self.attrs.items()})
        ## create dimensions all except time are fixed
        self.dataset.createDimension(TIME, None)
        self.dataset.createDimension(MODEL_LEVEL, self.num_levels)
        self.dataset.createDimension(MODEL_INTERFACE_LEVEL, self.num_interfaces)
        self.dataset.createDimension(CELL, self._horizontal_size.num_cells)
        self.dataset.createDimension(VERTEX, self._horizontal_size.num_vertices)
        self.dataset.createDimension(EDGE, self._horizontal_size.num_edges)
        log.debug(f"Creating dimensions {self.dataset.dimensions} in {self._file_name}")
        # create time variables
        times = self.dataset.createVariable(TIME, "f8", (TIME,))
        times.units = self._time_properties.units
        times.axis = cf_utils.COARDS_TIME_COORDINATE_NAME
        times.calendar = self._time_properties.calendar
        times.standard_name = TIME
        times.long_name = TIME
        # create vertical coordinates:
        levels = self.dataset.createVariable(MODEL_LEVEL, np.int32, (MODEL_LEVEL,))
        levels.units = "1"
        levels.positive = "down"
        levels.long_name = "model full level index"
        levels.standard_name = cf_utils.LEVEL_STANDARD_NAME
        levels[:] = np.arange(self.num_levels, dtype=np.int32)

        interface_levels = self.dataset.createVariable(
            MODEL_INTERFACE_LEVEL, np.int32, (MODEL_INTERFACE_LEVEL,)
        )
        interface_levels.units = "1"
        interface_levels.positive = "down"
        interface_levels.long_name = "model interface level index"
        interface_levels.standard_name = cf_utils.INTERFACE_LEVEL_STANDARD_NAME
        interface_levels[:] = np.arange(self.num_levels + 1, dtype=np.int32)

        heights = self.dataset.createVariable("height", np.float64, (MODEL_INTERFACE_LEVEL,))
        heights.units = "m"
        heights.positive = "up"
        heights.axis = cf_utils.COARDS_VERTICAL_COORDINATE_NAME
        heights.long_name = "height value of half levels for flat topography"
        heights.standard_name = cf_utils.INTERFACE_LEVEL_HEIGHT_STANDARD_NAME
        heights[:] = self._vertical_params.vct_a.ndarray

    def append(self, state_to_append: dict[str, xr.DataArray], model_time: datetime) -> None:
        """
        Append the fields to the dataset.

        Appends a time slice of the fields in the state_to_append dictionary to the dataset for the `model_time` expanding the time coordinate by the `model_time`.
        Args:
            state_to_append: fields to append
            model_time: time of the model state

        Returns:

        """
        time = self.dataset[TIME]
        time_pos = len(time)
        time[time_pos] = cf_utils.date2num(model_time, units=time.units, calendar=time.calendar)
        for var_name, new_slice in state_to_append.items():
            standard_name = new_slice.standard_name
            new_slice = cf_utils.to_canonical_dim_order(new_slice)
            assert standard_name is not None, f"No short_name provided for {standard_name}."
            ds_var = filter_by_standard_name(self.dataset.variables, standard_name)
            if not ds_var:
                dimensions = ("time", *new_slice.dims)
                new_var = self.dataset.createVariable(var_name, new_slice.dtype, dimensions)
                new_var[0, :] = new_slice.data
                new_var.units = new_slice.units
                new_var.standard_name = new_slice.standard_name
                new_var.long_name = new_slice.long_name
                new_var.coordinates = new_slice.coordinates
                new_var.mesh = new_slice.mesh
                new_var.location = new_slice.location

            else:
                var_name = ds_var.get(var_name).name
                dims = ds_var.get(var_name).dimensions
                shape = ds_var.get(var_name).shape
                assert (
                    len(new_slice.dims) == len(dims) - 1
                ), f"Data variable dimensions do not match for {standard_name}."

                # TODO (magdalena) change for parallel/distributed case: where we write at `global_index` field on the node for the horizontal dim.
                # we can acutally assume fixed index ordering here, input arrays are  re-shaped to canonical order (see above)

                right = (slice(None),) * (len(dims) - 1)
                expand_slice = (
                    slice(shape[cf_utils.COARDS_T_POS] - 1, shape[cf_utils.COARDS_T_POS]),
                )
                slices = expand_slice + right
                self.dataset.variables[var_name][slices] = new_slice.data

    def close(self) -> None:
        if self.dataset.isopen():
            self.dataset.close()

    @property
    def dims(self) -> dict:
        return self.dataset.dimensions

    @property
    def variables(self) -> dict:
        return self.dataset.variables


def filter_by_standard_name(model_state: dict, value: str) -> dict:
    return {k: v for k, v in model_state.items() if value == v.standard_name}
