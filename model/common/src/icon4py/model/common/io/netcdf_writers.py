# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import datetime as dt
import functools
import logging
import pathlib
import types
from typing import Self

import netCDF4 as nc
import numpy as np
import xarray as xr

from icon4py.model.common.grid import base, vertical as v_grid
from icon4py.model.common.io import cf_utils, writers
from icon4py.model.common.utils import data_allocation as data_alloc


log = logging.getLogger(__name__)


class NETCDFWriter:
    """
    Writer for netcdf files.

    Writes a netcdf file using netcdf4-python directly. Currently, this seems to be the only way that we can
      - append time slices to a variable already present in the file. (Xarray.to_netcdf does not support this https://github.com/pydata/xarray/issues/1672)

    This is a serial writer: in a distributed run it is used on the root rank only, on
    fields already gathered to global size (`distributed.GatherDistribution`).
    TODO (kotsaloscv): add a parallel netCDF writer once an MPI-enabled netCDF4 build is
    available (pip wheels ship without parallel support).
    """

    def __init__(
        self,
        *,
        file_name: pathlib.Path | str,
        vertical: v_grid.VerticalGrid,
        horizontal: base.HorizontalGridSize,
        time_properties: writers.TimeProperties,
        global_attrs: writers.GlobalFileAttributes,
    ):
        self._file_name = str(file_name)
        self._time_properties = time_properties
        self._vertical_params = vertical
        self._horizontal_size = horizontal
        self.attrs = global_attrs
        self.dataset = None

    def __getitem__(self, item: str) -> str:
        assert self.dataset is not None
        return self.dataset.getncattr(item)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self.close()

    @functools.cached_property
    def num_levels(self) -> int:
        return self._vertical_params.interface_physical_height.ndarray.shape[0] - 1

    @functools.cached_property
    def num_interfaces(self) -> int:
        return self._vertical_params.interface_physical_height.ndarray.shape[0]

    def initialize_dataset(self) -> None:
        self.dataset = nc.Dataset(  # type: ignore [assignment] # dataset is reassigned here
            self._file_name,
            "w",
            format="NETCDF4",
            persist=True,
        )
        assert self.dataset is not None
        log.info(f"Creating file {self._file_name} at {self.dataset.filepath()}")
        self.dataset.setncatts({k: str(v) for (k, v) in self.attrs.items()})
        ## create dimensions all except time are fixed
        self.dataset.createDimension(writers.TIME, None)
        self.dataset.createDimension(writers.MODEL_LEVEL, self.num_levels)
        self.dataset.createDimension(writers.MODEL_HALF_LEVEL, self.num_interfaces)
        self.dataset.createDimension(writers.CELL, self._horizontal_size.num_cells)
        self.dataset.createDimension(writers.VERTEX, self._horizontal_size.num_vertices)
        self.dataset.createDimension(writers.EDGE, self._horizontal_size.num_edges)
        log.debug(f"Creating dimensions {self.dataset.dimensions} in {self._file_name}")
        # create time variables
        times = self.dataset.createVariable(writers.TIME, "f8", (writers.TIME,))
        times.setncatts(writers.time_attributes(self._time_properties))
        # create vertical coordinates:
        levels = self.dataset.createVariable(writers.MODEL_LEVEL, np.int32, (writers.MODEL_LEVEL,))
        levels.setncatts(writers.LEVEL_ATTRIBUTES)
        levels[:] = np.arange(self.num_levels, dtype=np.int32)

        half_levels = self.dataset.createVariable(
            writers.MODEL_HALF_LEVEL, np.int32, (writers.MODEL_HALF_LEVEL,)
        )
        half_levels.setncatts(writers.HALF_LEVEL_ATTRIBUTES)
        half_levels[:] = np.arange(self.num_levels + 1, dtype=np.int32)

        heights = self.dataset.createVariable("height", np.float64, (writers.MODEL_HALF_LEVEL,))
        heights.setncatts(writers.HEIGHT_ATTRIBUTES)
        heights[:] = data_alloc.as_numpy(self._vertical_params.interface_physical_height)

    def append(self, state_to_append: dict[str, xr.DataArray], model_time: dt.datetime) -> None:
        """
        Append the fields to the dataset.

        Appends a time slice of the fields in the state_to_append dictionary to the dataset for the `model_time` expanding the time coordinate by the `model_time`.
        Args:
            state_to_append: fields to append
            model_time: time of the model state

        Returns:

        """
        assert self.dataset is not None
        time = self.dataset[writers.TIME]
        time_pos = len(time)
        time[time_pos] = cf_utils.date2num(model_time, units=time.units, calendar=time.calendar)
        for var_name, new_slice in state_to_append.items():
            standard_name = new_slice.standard_name
            canonical_new_slice = cf_utils.to_canonical_dim_order(new_slice)
            assert standard_name is not None, f"No short_name provided for {standard_name}."
            ds_var = writers.filter_by_standard_name(self.dataset.variables, standard_name)
            if not ds_var:
                dimensions = ("time", *canonical_new_slice.dims)
                new_var = self.dataset.createVariable(
                    var_name, canonical_new_slice.dtype, dimensions
                )
                new_var[0, :] = data_alloc.as_numpy(canonical_new_slice.data)
                new_var.setncatts(writers.data_variable_attributes(canonical_new_slice))

            else:
                assert ds_var is not None
                actual_var_name = ds_var.get(var_name).name
                dims = ds_var.get(actual_var_name).dimensions
                shape = ds_var.get(actual_var_name).shape
                assert len(canonical_new_slice.dims) == len(dims) - 1, (
                    f"Data variable dimensions do not match for {standard_name}."
                )

                # Fields arriving here span the full file dimensions (single-rank state or
                # gathered global fields); rank-local writes are the ZarrWriter's business.
                right = (slice(None),) * (len(dims) - 1)
                expand_slice = (
                    slice(shape[cf_utils.COARDS_T_POS] - 1, shape[cf_utils.COARDS_T_POS]),
                )
                slices = expand_slice + right
                self.dataset.variables[actual_var_name][slices] = data_alloc.as_numpy(
                    canonical_new_slice.data
                )

    def close(self) -> None:
        assert self.dataset is not None
        if self.dataset.isopen():
            self.dataset.close()

    @property
    def dims(self) -> dict:
        assert self.dataset is not None
        return self.dataset.dimensions

    @property
    def variables(self) -> dict:
        assert self.dataset is not None
        return self.dataset.variables
