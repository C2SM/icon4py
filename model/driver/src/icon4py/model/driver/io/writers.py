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
import logging
import pathlib
from datetime import datetime

import netCDF4 as nc
import numpy as np
import xarray as xr

from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.grid import horizontal as h_grid, vertical as v_grid
from icon4py.model.driver.io import cf_utils


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
        vertical: v_grid.VerticalGridSize,
        horizontal: h_grid.HorizontalGridSize,
        time_properties: TimeProperties,
        global_attrs: dict,
        process_properties: decomp_defs.ProcessProperties = processor_properties,
    ):
        self._file_name = str(file_name)
        self._process_properties = process_properties
        self._time_properties = time_properties
        self.num_levels = vertical.num_lev
        self.horizontal_size = horizontal
        self.attrs = global_attrs
        self.dataset = None

    def __getitem__(self, item):
        return self.dataset.getncattr(item)

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
        self.dataset.createDimension("time", None)
        self.dataset.createDimension("level", self.num_levels)
        self.dataset.createDimension("interface_level", self.num_levels + 1)
        self.dataset.createDimension("cell", self.horizontal_size.num_cells)
        self.dataset.createDimension("vertex", self.horizontal_size.num_vertices)
        self.dataset.createDimension("edge", self.horizontal_size.num_edges)
        log.debug(f"Creating dimensions {self.dataset.dimensions} in {self._file_name}")
        # create time variables
        times = self.dataset.createVariable("times", "f8", ("time",))
        times.units = self._time_properties.units
        times.calendar = self._time_properties.calendar
        times.standard_name = "time"
        times.long_name = "time"
        # create vertical coordinates:
        levels = self.dataset.createVariable("levels", np.int32, ("level",))
        levels.units = "1"
        levels.long_name = "model full levels"
        levels.standard_name = cf_utils.LEVEL_NAME
        levels[:] = np.arange(self.num_levels, dtype=np.int32)

        interface_levels = self.dataset.createVariable(
            "interface_levels", np.int32, ("interface_level",)
        )
        interface_levels.units = "1"
        interface_levels.long_name = "model interface levels"
        interface_levels.standard_name = cf_utils.INTERFACE_LEVEL_NAME
        interface_levels[:] = np.arange(self.num_levels + 1, dtype=np.int32)

        # TODO (magdalena) add vct_a as vertical coordinate?

    def append(self, state_to_append: dict[str, xr.DataArray], model_time: datetime) -> None:
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
        time[time_pos] = cf_utils.date2num(model_time, units=time.units, calendar=time.calendar)
        for item_key, item_data in state_to_append.items():
            standard_name = item_data.standard_name
            item_data = cf_utils.to_canonical_dim_order(item_data)
            assert standard_name is not None, f"No short_name provided for {standard_name}."
            ds_var = filter_by_standard_name(self.dataset.variables, standard_name)
            if not ds_var:
                dimensions = ("time", *item_data.dims)
                new_var = self.dataset.createVariable(item_key, item_data.dtype, dimensions)
                new_var[0, :] = item_data.data
                new_var.units = item_data.units
                new_var.standard_name = item_data.standard_name
                new_var.long_name = item_data.long_name
                new_var.coordinates = item_data.coordinates
                new_var.mesh = item_data.mesh
                new_var.location = item_data.location

            else:
                var_name = ds_var.get(item_key).name
                dims = ds_var.get(item_key).dimensions
                shape = ds_var.get(item_key).shape
                assert (
                    len(item_data.dims) == len(dims) - 1
                ), f"Data variable dimensions do not match for {standard_name}."

                # TODO (magdalena) change for parallel/distributed case: where we write at `global_index` field on the node for the horizontal dim.
                # we can acutally assume fixed index ordering here, input arrays are  re-shaped to canonical order (see above)

                right = (slice(None),) * (len(dims) - 1)
                expand_slice = (
                    slice(shape[cf_utils.COARDS_T_POS] - 1, shape[cf_utils.COARDS_T_POS]),
                )
                slices = expand_slice + right
                self.dataset.variables[var_name][slices] = item_data.data

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
