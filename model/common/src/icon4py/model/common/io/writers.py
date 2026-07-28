# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import datetime as dt
import functools
import logging
import pathlib
import uuid
import warnings
from typing import Final, Protocol, Required, TypedDict

import netCDF4 as nc
import numpy as np
import xarray as xr
import zarr

import icon4py.model.common.states.metadata
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import base, vertical as v_grid
from icon4py.model.common.io import cf_utils, distributed
from icon4py.model.common.utils import data_allocation as data_alloc


EDGE: Final[str] = "edge"
VERTEX: Final[str] = "vertex"
CELL: Final[str] = "cell"
MODEL_HALF_LEVEL: Final[str] = "half_level"
MODEL_LEVEL: Final[str] = "level"
TIME: Final[str] = "time"

#: Prefix of the global-index coordinates of rank-block distributed zarr stores.
GLOBAL_INDEX_PREFIX: Final[str] = "global_index"

log = logging.getLogger(__name__)


class GlobalFileAttributes(TypedDict, total=False):
    """
    Global file attributes of an ICON generated netCDF file.

    Attribute map what ICON produces, (including the upper, lower case pattern).
    Omissions (possibly incomplete):
    - 'CDI' used for the supported CDI version (http://mpimet.mpg.de/cdi) since we do not support it

    Additions:
    - 'external_variables': variable used by CF conventions if cell_measure variables are used from an external file'
    """

    #: version of the supported CF conventions
    Conventions: Required[str]  # TODO(halungge): check changelog? latest version is 1.11

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

    #: references for publication # TODO(halungge): check if this is the right reference
    references: str
    comment: str
    external_variables: str


@dataclasses.dataclass
class TimeProperties:
    units: str
    calendar: str


class FieldWriter(Protocol):
    """Writer for one output file: create it, append time slices to it, close it."""

    def initialize_dataset(self) -> None: ...

    def append(self, state_to_append: dict[str, xr.DataArray], model_time: dt.datetime) -> None: ...

    def close(self) -> None: ...


def _horizontal_axis_sizes(horizontal: base.HorizontalGridSize) -> dict[str, int]:
    return {
        CELL: horizontal.num_cells,
        EDGE: horizontal.num_edges,
        VERTEX: horizontal.num_vertices,
    }


# ------------------------------------------------------------------------------------
# Coordinate attributes, shared between the writers so the file formats stay identical
# ------------------------------------------------------------------------------------


def _time_attributes(time_properties: TimeProperties) -> dict[str, str]:
    return {
        "units": time_properties.units,
        "axis": cf_utils.COARDS_TIME_COORDINATE_NAME,
        "calendar": time_properties.calendar,
        "standard_name": TIME,
        "long_name": TIME,
    }


LEVEL_ATTRIBUTES: Final[dict[str, str]] = {
    "units": "1",
    "positive": "down",
    "long_name": "model full level index",
    "standard_name": cf_utils.LEVEL_STANDARD_NAME,
}

HALF_LEVEL_ATTRIBUTES: Final[dict[str, str]] = {
    "units": "1",
    "positive": "down",
    "long_name": "model half level index",
    "standard_name": icon4py.model.common.states.metadata.INTERFACE_LEVEL_STANDARD_NAME,
}

HEIGHT_ATTRIBUTES: Final[dict[str, str]] = {
    "units": "m",
    "positive": "up",
    "axis": cf_utils.COARDS_VERTICAL_COORDINATE_NAME,
    "long_name": "height value of half levels without topography",
    "standard_name": icon4py.model.common.states.metadata.INTERFACE_LEVEL_HEIGHT_STANDARD_NAME,
}

#: CF/UGRID attributes carried from a field's DataArray onto its file variable.
DATA_VARIABLE_ATTRIBUTES: Final[tuple[str, ...]] = (
    "units",
    "standard_name",
    "long_name",
    "coordinates",
    "mesh",
    "location",
)


def _data_variable_attributes(canonical_slice: xr.DataArray) -> dict[str, str]:
    return {name: getattr(canonical_slice, name) for name in DATA_VARIABLE_ATTRIBUTES}


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
        time_properties: TimeProperties,
        global_attrs: GlobalFileAttributes,
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
        self.dataset.createDimension(TIME, None)
        self.dataset.createDimension(MODEL_LEVEL, self.num_levels)
        self.dataset.createDimension(MODEL_HALF_LEVEL, self.num_interfaces)
        self.dataset.createDimension(CELL, self._horizontal_size.num_cells)
        self.dataset.createDimension(VERTEX, self._horizontal_size.num_vertices)
        self.dataset.createDimension(EDGE, self._horizontal_size.num_edges)
        log.debug(f"Creating dimensions {self.dataset.dimensions} in {self._file_name}")
        # create time variables
        times = self.dataset.createVariable(TIME, "f8", (TIME,))
        times.setncatts(_time_attributes(self._time_properties))
        # create vertical coordinates:
        levels = self.dataset.createVariable(MODEL_LEVEL, np.int32, (MODEL_LEVEL,))
        levels.setncatts(LEVEL_ATTRIBUTES)
        levels[:] = np.arange(self.num_levels, dtype=np.int32)

        half_levels = self.dataset.createVariable(MODEL_HALF_LEVEL, np.int32, (MODEL_HALF_LEVEL,))
        half_levels.setncatts(HALF_LEVEL_ATTRIBUTES)
        half_levels[:] = np.arange(self.num_levels + 1, dtype=np.int32)

        heights = self.dataset.createVariable("height", np.float64, (MODEL_HALF_LEVEL,))
        heights.setncatts(HEIGHT_ATTRIBUTES)
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
        time = self.dataset[TIME]
        time_pos = len(time)
        time[time_pos] = cf_utils.date2num(model_time, units=time.units, calendar=time.calendar)
        for var_name, new_slice in state_to_append.items():
            standard_name = new_slice.standard_name
            canonical_new_slice = cf_utils.to_canonical_dim_order(new_slice)
            assert standard_name is not None, f"No short_name provided for {standard_name}."
            ds_var = filter_by_standard_name(self.dataset.variables, standard_name)
            if not ds_var:
                dimensions = ("time", *canonical_new_slice.dims)
                new_var = self.dataset.createVariable(
                    var_name, canonical_new_slice.dtype, dimensions
                )
                new_var[0, :] = data_alloc.as_numpy(canonical_new_slice.data)
                new_var.setncatts(_data_variable_attributes(canonical_new_slice))

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


class ZarrWriter:
    """
    Writer for zarr stores, mirroring the layout of the netCDF files.

    The store carries the same dimensions (time, level, half_level, cell, edge, vertex),
    coordinate variables and CF attributes as the netCDF files; arrays carry their
    dimension names natively (zarr format 3), so ``xarray.open_zarr`` reads the store
    like the corresponding netCDF file.

    Serial mode (``rank_blocks is None``): whole time slices are appended, exactly like
    the NETCDFWriter -- used for single-rank runs and on the root rank of gathered output.

    Rank-block mode (``rank_blocks`` given): every rank writes only its rank-contiguous
    block of the horizontal axes (see ``distributed.RankBlock``). The horizontal axes are
    chunked with exactly one chunk per rank, so concurrent writes of different ranks never
    touch the same chunk file. The root rank performs all store-metadata operations
    (array creation, time-axis resizing); a single barrier per append orders them before
    the data writes of all ranks. ``global_index_<dim>`` coordinates map store positions
    to the undecomposed global grid; padding positions carry the fill value -1 (data
    padding reads as NaN for floating dtypes and as zero otherwise).
    """

    def __init__(
        self,
        *,
        file_name: pathlib.Path,
        vertical: v_grid.VerticalGrid,
        horizontal: base.HorizontalGridSize,
        time_properties: TimeProperties,
        global_attrs: GlobalFileAttributes,
        rank_blocks: dict[str, distributed.RankBlock] | None = None,
        process_props: decomposition.ProcessProperties | None = None,
    ):
        self._file_name = str(file_name)
        self._time_properties = time_properties
        self._vertical_params = vertical
        self._horizontal_sizes = _horizontal_axis_sizes(horizontal)
        self.attrs = global_attrs
        self._rank_blocks = rank_blocks
        self._process_props = (
            process_props
            if process_props is not None
            else decomposition.SingleNodeProcessProperties()
        )
        self._group: zarr.Group | None = None
        # The append count doubles as the time index of the next slice. It is kept
        # locally (identical on all ranks: in rank-block mode append is called once per
        # capture step on every rank) instead of being derived from store metadata,
        # which a lagging rank could re-read only after the root already resized for a
        # later step.
        self._append_count = 0

    @functools.cached_property
    def num_levels(self) -> int:
        return self._vertical_params.interface_physical_height.ndarray.shape[0] - 1

    def _is_collective(self) -> bool:
        """Whether store operations are shared between the ranks (rank-block mode).

        In serial mode the writer never synchronizes, whatever communicator it holds:
        it is used by a single process (a single-rank run, or the root rank of a
        gathered write while the other ranks never touch the writer).
        """
        return self._rank_blocks is not None

    def _is_root(self) -> bool:
        return not self._is_collective() or self._process_props.rank == 0

    def _barrier(self) -> None:
        if self._is_collective() and not self._process_props.is_single_rank():
            self._process_props.comm.Barrier()

    def _horizontal_write_range(self, dim_name: str) -> slice:
        if self._rank_blocks is None:
            return slice(0, self._horizontal_sizes[dim_name])
        block = self._rank_blocks[dim_name]
        return slice(block.start, block.start + block.count)

    def _horizontal_chunk(self, dim_name: str) -> int:
        if self._rank_blocks is None:
            return self._horizontal_sizes[dim_name]
        return self._rank_blocks[dim_name].chunk

    def _data_array(self, name: str) -> zarr.Array:
        """Typed access to a store array (``Group.__getitem__`` may also yield groups)."""
        assert self._group is not None
        array = self._group[name]
        assert isinstance(array, zarr.Array)
        return array

    def initialize_dataset(self) -> None:
        if self._is_root():
            # mode "w-" refuses to overwrite an existing store; format 3 carries the
            # dimension names of every array natively (``dimension_names``), which
            # xarray reads like netCDF dimensions.
            group = zarr.open_group(self._file_name, mode="w-", zarr_format=3)
            log.info(f"Creating store {self._file_name}")
            group.attrs.update({k: str(v) for (k, v) in self.attrs.items()})

            # format 3 requires a fill value on every array (defaulting to the dtype's
            # zero); unlike a format-2 fill value it is not decoded as a missing-value
            # marker by xarray, so real coordinate values (e.g. level index 0) are safe
            times = group.create_array(
                TIME, shape=(0,), chunks=(1,), dtype="f8", dimension_names=[TIME]
            )
            times.attrs.update(_time_attributes(self._time_properties))

            levels = group.create_array(
                MODEL_LEVEL,
                shape=(self.num_levels,),
                dtype=np.int32,
                dimension_names=[MODEL_LEVEL],
            )
            levels[:] = np.arange(self.num_levels, dtype=np.int32)
            levels.attrs.update(LEVEL_ATTRIBUTES)

            half_levels = group.create_array(
                MODEL_HALF_LEVEL,
                shape=(self.num_levels + 1,),
                dtype=np.int32,
                dimension_names=[MODEL_HALF_LEVEL],
            )
            half_levels[:] = np.arange(self.num_levels + 1, dtype=np.int32)
            half_levels.attrs.update(HALF_LEVEL_ATTRIBUTES)

            heights = group.create_array(
                "height",
                shape=(self.num_levels + 1,),
                dtype=np.float64,
                dimension_names=[MODEL_HALF_LEVEL],
            )
            heights[:] = data_alloc.as_numpy(self._vertical_params.interface_physical_height)
            heights.attrs.update(HEIGHT_ATTRIBUTES)

            if self._rank_blocks is not None:
                for dim_name, block in self._rank_blocks.items():
                    global_index = group.create_array(
                        f"{GLOBAL_INDEX_PREFIX}_{dim_name}",
                        shape=(block.padded_size,),
                        chunks=(block.chunk,),
                        dtype=np.int64,
                        fill_value=-1,
                        dimension_names=[dim_name],
                    )
                    global_index.attrs.update(
                        units="1",
                        long_name=(
                            f"position of each {dim_name} entry in the undecomposed global "
                            f"grid of {block.global_size} entries (-1 marks padding)"
                        ),
                    )
            self._group = group

        self._barrier()
        if self._is_collective():
            if not self._is_root():
                self._group = zarr.open_group(self._file_name, mode="r+")
            assert self._rank_blocks is not None
            for dim_name, block in self._rank_blocks.items():
                index_range = self._horizontal_write_range(dim_name)
                self._data_array(f"{GLOBAL_INDEX_PREFIX}_{dim_name}")[index_range] = (
                    block.global_index
                )

    def append(self, state_to_append: dict[str, xr.DataArray], model_time: dt.datetime) -> None:
        """
        Append the fields to the store.

        Appends a time slice of the fields in the state_to_append dictionary to the store
        for the `model_time`, expanding the time coordinate by the `model_time`. In
        rank-block mode only this rank's horizontal block is written.

        Args:
            state_to_append: fields to append
            model_time: time of the model state
        """
        assert self._group is not None
        canonical_slices: dict[str, xr.DataArray] = {}
        host_data: dict[str, np.ndarray] = {}
        # canonicalize and transfer to host up front: a failure here (unsupported
        # dimensions, device buffer that cannot be converted) must precede any store
        # mutation, or the store would be left with a phantom time slice
        for var_name, new_slice in state_to_append.items():
            canonical_slice = cf_utils.to_canonical_dim_order(new_slice)
            if canonical_slice is None:
                raise ValueError(
                    f"Cannot write field '{var_name}': only fields with a horizontal and a "
                    f"vertical dimension are supported."
                )
            canonical_slices[var_name] = canonical_slice
            host_data[var_name] = data_alloc.as_numpy(canonical_slice.data)
        time_pos = self._append_count
        if self._is_root():
            times = self._data_array(TIME)
            times.resize((time_pos + 1,))
            times[time_pos] = cf_utils.date2num(
                model_time,
                units=self._time_properties.units,
                calendar=self._time_properties.calendar,
            )
            for var_name, canonical_slice in canonical_slices.items():
                if var_name not in self._group:
                    self._create_variable(var_name, canonical_slice)
                variable = self._data_array(var_name)
                variable.resize((time_pos + 1, *variable.shape[1:]))
        self._barrier()
        for var_name, canonical_slice in canonical_slices.items():
            variable = self._data_array(var_name)
            horizontal_range = self._horizontal_write_range(str(canonical_slice.dims[-1]))
            middle = (slice(None),) * (len(canonical_slice.dims) - 1)
            index: tuple[int | slice, ...] = (time_pos, *middle, horizontal_range)
            variable[index] = host_data[var_name]
        self._append_count += 1

    def _create_variable(self, var_name: str, canonical_slice: xr.DataArray) -> None:
        """Create the (empty along time) array of a data variable on the root rank."""
        assert self._group is not None
        horizontal_name = str(canonical_slice.dims[-1])
        shape = (0, *canonical_slice.shape[:-1], self._horizontal_sizes[horizontal_name])
        chunks = (1, *canonical_slice.shape[:-1], self._horizontal_chunk(horizontal_name))
        # NaN marks never-written positions (rank-block padding) but exists only for
        # floating dtypes; other dtypes fall back to the dtype's default fill value
        # (padding then reads as zero)
        fill_value = float("nan") if np.issubdtype(canonical_slice.dtype, np.floating) else None
        variable = self._group.create_array(
            var_name,
            shape=shape,
            chunks=chunks,
            dtype=canonical_slice.dtype,
            fill_value=fill_value,
            dimension_names=[TIME, *(str(d) for d in canonical_slice.dims)],
        )
        variable.attrs.update(_data_variable_attributes(canonical_slice))

    def close(self) -> None:
        if self._group is None:
            return
        if self._is_root():
            with warnings.catch_warnings():
                # consolidated metadata speeds up opening the store (single metadata
                # read for all arrays) but is a zarr-python extension of the format 3
                # specification; silence the spec-stability warning it emits on every
                # close (other readers simply ignore the extra metadata)
                warnings.filterwarnings(
                    "ignore", message="Consolidated metadata", category=UserWarning
                )
                zarr.consolidate_metadata(self._group.store)
        self._group = None


def filter_by_standard_name(model_state: dict, value: str) -> dict:
    return {k: v for k, v in model_state.items() if value == v.standard_name}
