# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import importlib.util
import pathlib
from datetime import datetime, timedelta

import gt4py.next as gtx
import netCDF4 as nc
import numpy as np
import pytest
import xarray as xr
import zarr

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base as grid_def, vertical as v_grid
from icon4py.model.common.io import (
    cf_utils,
    distributed,
    netcdf_writers,
    utils,
    writers,
    zarr_writers,
)
from icon4py.model.common.states import data, metadata
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import test_utils

from ...fixtures import random_name, test_path
from .. import utils as test_io_utils


@pytest.mark.parametrize("value", ["air_density", "upward_air_velocity"])
def test_filter_by_standard_name(value):
    state = test_io_utils.model_state(test_io_utils.simple_grid)
    assert writers.filter_by_standard_name(state, value) == {value: state[value]}


def test_filter_by_standard_name_key_differs_from_name():
    state = test_io_utils.model_state(test_io_utils.simple_grid)
    assert writers.filter_by_standard_name(state, "virtual_potential_temperature") == {
        "theta_v": state["theta_v"]
    }


def test_filter_by_standard_name_non_existing_name():
    state = test_io_utils.model_state(test_io_utils.simple_grid)
    assert writers.filter_by_standard_name(state, "does_not_exist") == {}


def _vertical_params(grid: grid_def.Grid) -> v_grid.VerticalGrid:
    num_levels = grid.config.vertical_size
    heights = np.linspace(start=12000.0, stop=0.0, num=num_levels + 1)
    vertical_config = v_grid.VerticalGridConfig(num_levels=num_levels)
    return v_grid.VerticalGrid(
        vertical_config,
        vct_a=gtx.as_field((dims.KDim,), heights),
        vct_b=None,
    )


def initialized_writer(
    test_path: pathlib.Path, random_name: str, grid: grid_def.Grid = test_io_utils.simple_grid
) -> tuple[netcdf_writers.NETCDFWriter, grid_def.Grid]:
    horizontal = grid.config.horizontal_config
    fname = str(test_path.absolute()) + "/" + random_name + ".nc"
    writer = netcdf_writers.NETCDFWriter(
        file_name=fname,
        vertical=_vertical_params(grid),
        horizontal=horizontal,
        time_properties=writers.TimeProperties(
            cf_utils.DEFAULT_TIME_UNIT, cf_utils.DEFAULT_CALENDAR
        ),
        global_attrs={"title": "test", "institution": "EXCLAIM - ETH Zurich"},
    )
    writer.initialize_dataset()
    return writer, grid


def test_initialize_writer_time_var(test_path, random_name):
    dataset, _ = initialized_writer(test_path, random_name)
    time_var = dataset.variables[writers.TIME]
    assert time_var.dimensions == ("time",)
    assert time_var.units == "seconds since 1970-01-01 00:00:00"
    assert time_var.calendar == "proleptic_gregorian"
    assert time_var.long_name == "time"
    assert time_var.standard_name == "time"
    assert len(time_var) == 0


def test_initialize_writer_vertical_model_levels(test_path, random_name):
    dataset, grid = initialized_writer(test_path, random_name)
    vertical = dataset.variables[writers.MODEL_LEVEL]
    assert vertical.units == "1"
    assert vertical.dimensions == (writers.MODEL_LEVEL,)
    assert vertical.long_name == "model full level index"
    assert vertical.standard_name == cf_utils.LEVEL_STANDARD_NAME
    assert vertical.datatype == np.int32
    assert len(vertical) == grid.num_levels
    assert np.all(vertical == np.arange(grid.num_levels))


def test_initialize_writer_half_levels(test_path, random_name):
    dataset, grid = initialized_writer(test_path, random_name)
    half_levels = dataset.variables[writers.MODEL_HALF_LEVEL]
    assert half_levels.units == "1"
    assert half_levels.datatype == np.int32
    assert half_levels.long_name == "model half level index"
    assert half_levels.standard_name == metadata.INTERFACE_LEVEL_STANDARD_NAME
    assert len(half_levels) == grid.num_levels + 1
    assert np.all(half_levels == np.arange(grid.num_levels + 1))


def test_initialize_writer_heights(test_path, random_name):
    dataset, grid = initialized_writer(test_path, random_name)
    heights = dataset.variables["height"]
    assert heights.units == "m"
    assert heights.datatype == np.float64
    assert heights.long_name == "height value of half levels without topography"
    assert heights.standard_name == metadata.INTERFACE_LEVEL_HEIGHT_STANDARD_NAME
    assert len(heights) == grid.num_levels + 1
    assert heights[0] == 12000.0
    assert heights[-1] == 0.0


def test_writer_append_timeslice(test_path, random_name):
    writer, _ = initialized_writer(test_path, random_name)
    time = datetime.now()
    assert len(writer.variables[writers.TIME]) == 0
    slice1 = {}
    writer.append(slice1, time)
    assert len(writer.variables[writers.TIME]) == 1
    time1 = time + timedelta(hours=1)
    writer.append(slice1, time1)
    assert len(writer.variables[writers.TIME]) == 2
    time2 = time1 + timedelta(hours=1)
    writer.append(slice1, time2)
    assert len(writer.variables[writers.TIME]) == 3
    time_units = writer.variables[writers.TIME].units
    cal = writer.variables[writers.TIME].calendar
    assert np.all(
        writer.variables[writers.TIME][:]
        == np.array(cf_utils.date2num((time, time1, time2), units=time_units, calendar=cal))
    )


def test_writer_append_timeslice_create_new_var(test_path, random_name):
    dataset, grid = initialized_writer(test_path, random_name)
    time = datetime.now()
    assert len(dataset.variables[writers.TIME]) == 0
    assert "air_density" not in dataset.variables

    state = dict(air_density=test_io_utils.model_state(grid)["air_density"])
    dataset.append(state, time)
    assert len(dataset.variables[writers.TIME]) == 1
    assert "air_density" in dataset.variables
    assert dataset.variables["air_density"].dimensions == (
        writers.TIME,
        writers.MODEL_LEVEL,
        writers.CELL,
    )
    assert dataset.variables["air_density"].shape == (
        1,
        grid.num_levels,
        grid.num_cells,
    )
    test_utils.assert_dallclose(dataset.variables["air_density"][0], state["air_density"].data.T)


def test_writer_append_timeslice_to_existing_var(test_path, random_name):
    dataset, grid = initialized_writer(test_path, random_name)
    time = datetime.now()
    state = dict(air_density=test_io_utils.model_state(grid)["air_density"])
    dataset.append(state, time)
    assert len(dataset.variables[writers.TIME]) == 1
    assert "air_density" in dataset.variables

    new_rho = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=np.float32)
    state["air_density"] = utils.to_data_array(
        new_rho, data.PROGNOSTIC_CF_ATTRIBUTES["air_density"]
    )

    new_time = time + timedelta(hours=1)
    dataset.append(state, new_time)

    assert len(dataset.variables[writers.TIME]) == 2
    assert dataset.variables["air_density"].shape == (
        2,
        grid.num_levels,
        grid.num_cells,
    )
    test_utils.assert_dallclose(dataset.variables["air_density"][1], new_rho.ndarray.T)


def initialized_zarr_writer(
    test_path: pathlib.Path,
    random_name: str,
    grid: grid_def.Grid = test_io_utils.simple_grid,
    rank_blocks: dict[str, distributed.RankBlock] | None = None,
    horizontal: grid_def.HorizontalGridSize | None = None,
) -> tuple[zarr_writers.ZarrWriter, pathlib.Path]:
    store_path = test_path.absolute() / f"{random_name}.zarr"
    writer = zarr_writers.ZarrWriter(
        file_name=store_path,
        vertical=_vertical_params(grid),
        horizontal=horizontal if horizontal is not None else grid.config.horizontal_config,
        time_properties=writers.TimeProperties(
            cf_utils.DEFAULT_TIME_UNIT, cf_utils.DEFAULT_CALENDAR
        ),
        global_attrs={"title": "test", "institution": "EXCLAIM - ETH Zurich"},
        rank_blocks=rank_blocks,
    )
    writer.initialize_dataset()
    return writer, store_path


def test_zarr_writer_initialize_coordinates(test_path: pathlib.Path, random_name: str) -> None:
    writer, store_path = initialized_zarr_writer(test_path, random_name)
    writer.close()
    grid = test_io_utils.simple_grid
    assert zarr.open_group(store_path, mode="r").metadata.zarr_format == 3
    with xr.open_zarr(store_path) as ds:
        assert ds.attrs["title"] == "test"
        assert ds.sizes[writers.TIME] == 0
        assert np.all(ds[writers.MODEL_LEVEL].values == np.arange(grid.num_levels))
        assert np.all(ds[writers.MODEL_HALF_LEVEL].values == np.arange(grid.num_levels + 1))
        assert ds["height"].values[0] == 12000.0
        assert ds["height"].values[-1] == 0.0
        assert ds["height"].attrs["units"] == "m"


def test_zarr_writer_append_timeslice_create_new_var(
    test_path: pathlib.Path, random_name: str
) -> None:
    writer, store_path = initialized_zarr_writer(test_path, random_name)
    grid = test_io_utils.simple_grid
    state = dict(air_density=test_io_utils.model_state(grid)["air_density"])
    writer.append(state, datetime.now())
    writer.close()
    with xr.open_zarr(store_path) as ds:
        assert ds["air_density"].dims == (writers.TIME, writers.MODEL_LEVEL, writers.CELL)
        assert ds["air_density"].shape == (1, grid.num_levels, grid.num_cells)
        test_utils.assert_dallclose(ds["air_density"].values[0], state["air_density"].data.T)
        assert ds["air_density"].attrs["standard_name"] == "air_density"


def test_zarr_writer_append_timeslice_to_existing_var(
    test_path: pathlib.Path, random_name: str
) -> None:
    writer, store_path = initialized_zarr_writer(test_path, random_name)
    grid = test_io_utils.simple_grid
    state = dict(air_density=test_io_utils.model_state(grid)["air_density"])
    first_time = datetime.now()
    writer.append(state, first_time)

    new_rho = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=np.float32)
    state["air_density"] = utils.to_data_array(
        new_rho, data.PROGNOSTIC_CF_ATTRIBUTES["air_density"]
    )
    writer.append(state, first_time + timedelta(hours=1))
    writer.close()
    with xr.open_zarr(store_path) as ds:
        assert ds["air_density"].shape == (2, grid.num_levels, grid.num_cells)
        test_utils.assert_dallclose(ds["air_density"].values[1], new_rho.ndarray.T)
        assert ds.sizes[writers.TIME] == 2
    # the raw time values must be the CF-encoded model times, in append order
    with xr.open_zarr(store_path, decode_times=False) as ds:
        expected_times = [
            cf_utils.date2num(t) for t in (first_time, first_time + timedelta(hours=1))
        ]
        test_utils.assert_dallclose(ds[writers.TIME].values, expected_times)


def test_zarr_writer_refuses_to_overwrite(test_path: pathlib.Path, random_name: str) -> None:
    writer, store_path = initialized_zarr_writer(test_path, random_name)
    writer.close()
    duplicate = zarr_writers.ZarrWriter(
        file_name=store_path,
        vertical=_vertical_params(test_io_utils.simple_grid),
        horizontal=test_io_utils.simple_grid.config.horizontal_config,
        time_properties=writers.TimeProperties(
            cf_utils.DEFAULT_TIME_UNIT, cf_utils.DEFAULT_CALENDAR
        ),
        global_attrs={"title": "test", "institution": "EXCLAIM - ETH Zurich"},
    )
    with pytest.raises(FileExistsError):
        duplicate.initialize_dataset()


def test_zarr_writer_rank_block_writes_padded_block(
    test_path: pathlib.Path, random_name: str
) -> None:
    # single-rank rank-block layout with padding: the store's horizontal axes are the
    # padded sizes, data lands in the rank's block, padding stays NaN and the
    # global-index coordinates mark it with -1
    grid = test_io_utils.simple_grid
    padding = 3
    rank_blocks = {
        dim_name: distributed.RankBlock(
            start=0,
            count=size,
            chunk=size + padding,
            padded_size=size + padding,
            global_size=size,
            global_index=np.arange(size, dtype=np.int64),
        )
        for dim_name, size in (
            (writers.CELL, grid.num_cells),
            (writers.EDGE, grid.num_edges),
            (writers.VERTEX, grid.num_vertices),
        )
    }
    padded_horizontal = grid_def.HorizontalGridSize(
        num_cells=grid.num_cells + padding,
        num_edges=grid.num_edges + padding,
        num_vertices=grid.num_vertices + padding,
    )
    writer, store_path = initialized_zarr_writer(
        test_path, random_name, grid, rank_blocks=rank_blocks, horizontal=padded_horizontal
    )
    state = dict(air_density=test_io_utils.model_state(grid)["air_density"])
    writer.append(state, datetime.now())
    writer.close()
    with xr.open_zarr(store_path) as ds:
        assert ds["air_density"].shape == (1, grid.num_levels, grid.num_cells + padding)
        values = ds["air_density"].values[0]
        test_utils.assert_dallclose(values[:, : grid.num_cells], state["air_density"].data.T)
        assert np.all(np.isnan(values[:, grid.num_cells :]))
    # the -1 padding marker doubles as the store's fill value (which xarray does not
    # decode as a missing value for format 3); read undecoded to pin the on-disk
    # contract independent of the reader's decoding defaults
    with xr.open_zarr(store_path, mask_and_scale=False) as ds:
        global_index = ds[f"{writers.GLOBAL_INDEX_PREFIX}_{writers.CELL}"].values
        assert np.all(global_index[: grid.num_cells] == np.arange(grid.num_cells))
        assert np.all(global_index[grid.num_cells :] == -1)
    # exactly one horizontal chunk per rank block: the invariant that keeps concurrent
    # writes of different ranks out of each other's chunk files
    group = zarr.open_group(store_path, mode="r")
    cell_chunk = rank_blocks[writers.CELL].chunk
    air_density = group["air_density"]
    global_index_cell = group[f"{writers.GLOBAL_INDEX_PREFIX}_{writers.CELL}"]
    assert isinstance(air_density, zarr.Array)
    assert isinstance(global_index_cell, zarr.Array)
    assert air_density.chunks == (1, grid.num_levels, cell_chunk)
    assert global_index_cell.chunks == (cell_chunk,)


def test_zarr_writer_rank_block_writes_at_nonzero_start(
    test_path: pathlib.Path, random_name: str
) -> None:
    # store view of a non-root rank: its block starts at rank * chunk; data and global
    # indices must land inside the block only, everything before and after stays padding
    grid = test_io_utils.simple_grid
    rank = 1
    rank_blocks = {}
    for dim_name, size in (
        (writers.CELL, grid.num_cells),
        (writers.EDGE, grid.num_edges),
        (writers.VERTEX, grid.num_vertices),
    ):
        chunk = size + 1  # uneven layout: one padding entry per block
        rank_blocks[dim_name] = distributed.RankBlock(
            start=rank * chunk,
            count=size,
            chunk=chunk,
            padded_size=2 * chunk,
            global_size=2 * size,
            global_index=np.arange(size, 2 * size, dtype=np.int64),
        )
    padded_horizontal = grid_def.HorizontalGridSize(
        num_cells=rank_blocks[writers.CELL].padded_size,
        num_edges=rank_blocks[writers.EDGE].padded_size,
        num_vertices=rank_blocks[writers.VERTEX].padded_size,
    )
    writer, store_path = initialized_zarr_writer(
        test_path, random_name, grid, rank_blocks=rank_blocks, horizontal=padded_horizontal
    )
    state = dict(air_density=test_io_utils.model_state(grid)["air_density"])
    writer.append(state, datetime.now())
    writer.close()
    cell_block = rank_blocks[writers.CELL]
    block = slice(cell_block.start, cell_block.start + cell_block.count)
    with xr.open_zarr(store_path, mask_and_scale=False) as ds:
        values = ds["air_density"].values[0]
        test_utils.assert_dallclose(values[:, block], state["air_density"].data.T)
        global_index = ds[f"{writers.GLOBAL_INDEX_PREFIX}_{writers.CELL}"].values
        assert np.all(global_index[block] == cell_block.global_index)
        assert np.all(global_index[: cell_block.start] == -1)
    with xr.open_zarr(store_path) as ds:
        values = ds["air_density"].values[0]
        assert np.all(np.isnan(values[:, : cell_block.start]))
        assert np.all(np.isnan(values[:, cell_block.start + cell_block.count :]))


def test_initialize_writer_create_dimensions(
    test_path,
    random_name,
):
    writer, grid = initialized_writer(test_path, random_name)

    assert writer["title"] == "test"
    assert writer["institution"] == "EXCLAIM - ETH Zurich"
    assert len(writer.dims) == 6
    assert writer.dims[writers.MODEL_LEVEL].size == grid.num_levels
    assert writer.dims[writers.MODEL_HALF_LEVEL].size == grid.num_levels + 1
    assert writer.dims[writers.CELL].size == grid.num_cells
    assert writer.dims[writers.VERTEX].size == grid.num_vertices
    assert writer.dims[writers.EDGE].size == grid.num_edges
    assert writer.dims[writers.TIME].size == 0
    assert writer.dims[writers.TIME].isunlimited

    assert writer.variables[writers.TIME].units == cf_utils.DEFAULT_TIME_UNIT
    assert writer.variables[writers.TIME].calendar == cf_utils.DEFAULT_CALENDAR


def initialized_netcdf_rank_block_writer(
    test_path: pathlib.Path,
    random_name: str,
    grid: grid_def.Grid,
    rank_blocks: dict[str, distributed.RankBlock],
    horizontal: grid_def.HorizontalGridSize,
) -> tuple[netcdf_writers.NETCDFWriter, pathlib.Path]:
    """Rank-block netCDF writer on a single-rank communicator (serial file handle)."""
    file_path = test_path.absolute() / f"{random_name}.nc"
    writer = netcdf_writers.NETCDFWriter(
        file_name=file_path,
        vertical=_vertical_params(grid),
        horizontal=horizontal,
        time_properties=writers.TimeProperties(
            cf_utils.DEFAULT_TIME_UNIT, cf_utils.DEFAULT_CALENDAR
        ),
        global_attrs={"title": "test", "institution": "EXCLAIM - ETH Zurich"},
        rank_blocks=rank_blocks,
    )
    writer.initialize_dataset()
    return writer, file_path


def test_netcdf_writer_rank_block_writes_padded_block(
    test_path: pathlib.Path, random_name: str
) -> None:
    # the netCDF twin of the zarr rank-block test: padded horizontal axes, data in the
    # rank's block, NaN data padding, -1 global-index padding. A single-rank
    # communicator uses a serial file handle, so the layout is exercised without an
    # MPI-parallel netCDF4 installation.
    grid = test_io_utils.simple_grid
    padding = 3
    rank_blocks = {
        dim_name: distributed.RankBlock(
            start=0,
            count=size,
            chunk=size + padding,
            padded_size=size + padding,
            global_size=size,
            global_index=np.arange(size, dtype=np.int64),
        )
        for dim_name, size in (
            (writers.CELL, grid.num_cells),
            (writers.EDGE, grid.num_edges),
            (writers.VERTEX, grid.num_vertices),
        )
    }
    padded_horizontal = grid_def.HorizontalGridSize(
        num_cells=grid.num_cells + padding,
        num_edges=grid.num_edges + padding,
        num_vertices=grid.num_vertices + padding,
    )
    writer, file_path = initialized_netcdf_rank_block_writer(
        test_path, random_name, grid, rank_blocks, padded_horizontal
    )
    state = dict(air_density=test_io_utils.model_state(grid)["air_density"])
    writer.append(state, datetime.now())
    writer.close()
    with xr.open_dataset(file_path) as ds:
        assert ds["air_density"].shape == (1, grid.num_levels, grid.num_cells + padding)
        values = ds["air_density"].values[0]
        test_utils.assert_dallclose(values[:, : grid.num_cells], state["air_density"].data.T)
        # padding reads as NaN: written explicitly by the writer, matching the
        # variable's fill value
        assert np.all(np.isnan(values[:, grid.num_cells :]))
        # the -1 padding is written explicitly, not encoded as a _FillValue attribute
        # (xarray would decode that to NaN, turning the integer coordinate into floats)
        global_index = ds[f"{writers.GLOBAL_INDEX_PREFIX}_{writers.CELL}"].values
        assert global_index.dtype == np.int64
        assert np.all(global_index[: grid.num_cells] == np.arange(grid.num_cells))
        assert np.all(global_index[grid.num_cells :] == -1)
    # exactly one horizontal chunk per rank block: the same on-disk layout as the
    # rank-block zarr store, keeping concurrent writes of different ranks in
    # disjoint chunks
    cell_chunk = rank_blocks[writers.CELL].chunk
    with nc.Dataset(file_path) as raw:
        assert raw["air_density"].chunking() == [1, grid.num_levels, cell_chunk]
        assert raw[f"{writers.GLOBAL_INDEX_PREFIX}_{writers.CELL}"].chunking() == [cell_chunk]


def test_netcdf_writer_rank_block_writes_at_nonzero_start(
    test_path: pathlib.Path, random_name: str
) -> None:
    # file view of a non-root rank: its block starts at rank * chunk; data and global
    # indices must land inside the block only. Unlike the zarr store (whose fill value
    # covers the whole array), regions of other ranks are simply not written by this
    # rank -- in a real run every rank covers its own block.
    grid = test_io_utils.simple_grid
    rank = 1
    rank_blocks = {}
    for dim_name, size in (
        (writers.CELL, grid.num_cells),
        (writers.EDGE, grid.num_edges),
        (writers.VERTEX, grid.num_vertices),
    ):
        chunk = size + 1  # uneven layout: one padding entry per block
        rank_blocks[dim_name] = distributed.RankBlock(
            start=rank * chunk,
            count=size,
            chunk=chunk,
            padded_size=2 * chunk,
            global_size=2 * size,
            global_index=np.arange(size, 2 * size, dtype=np.int64),
        )
    padded_horizontal = grid_def.HorizontalGridSize(
        num_cells=rank_blocks[writers.CELL].padded_size,
        num_edges=rank_blocks[writers.EDGE].padded_size,
        num_vertices=rank_blocks[writers.VERTEX].padded_size,
    )
    writer, file_path = initialized_netcdf_rank_block_writer(
        test_path, random_name, grid, rank_blocks, padded_horizontal
    )
    state = dict(air_density=test_io_utils.model_state(grid)["air_density"])
    writer.append(state, datetime.now())
    writer.close()
    cell_block = rank_blocks[writers.CELL]
    block = slice(cell_block.start, cell_block.start + cell_block.count)
    with xr.open_dataset(file_path) as ds:
        values = ds["air_density"].values[0]
        test_utils.assert_dallclose(values[:, block], state["air_density"].data.T)
        # everything this rank did not write reads as the NaN fill value: the other
        # rank's block and the padding entry of this rank's block
        assert np.all(np.isnan(values[:, : cell_block.start]))
        assert np.all(np.isnan(values[:, cell_block.start + cell_block.count :]))
        global_index = ds[f"{writers.GLOBAL_INDEX_PREFIX}_{writers.CELL}"].values
        assert np.all(global_index[block] == cell_block.global_index)
        # the padding inside this rank's block is written as -1 (the other rank's
        # block region belongs to the other rank and is not asserted here)
        assert np.all(
            global_index[cell_block.start + cell_block.count : cell_block.start + cell_block.chunk]
            == -1
        )


class _TwoRankProcessProperties:
    """Multi-rank ProcessProperties stand-in; the guard raises before any communication."""

    comm = None
    rank = 0
    comm_name = ""
    comm_size = 2

    def is_single_rank(self) -> bool:
        return False


def test_netcdf_writer_rejects_multi_rank_blocks_without_parallel_support(
    test_path: pathlib.Path, random_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # patched instead of relying on the local installation: PyPI wheels are always
    # serial, but the test must also pass on a machine with a parallel build
    monkeypatch.setattr(netcdf_writers, "missing_parallel_support", lambda: "<serial build>")
    with pytest.raises(RuntimeError) as err:
        netcdf_writers.NETCDFWriter(
            file_name=test_path / f"{random_name}.nc",
            vertical=_vertical_params(test_io_utils.simple_grid),
            horizontal=test_io_utils.simple_grid.config.horizontal_config,
            time_properties=writers.TimeProperties(
                cf_utils.DEFAULT_TIME_UNIT, cf_utils.DEFAULT_CALENDAR
            ),
            global_attrs={"title": "test", "institution": "EXCLAIM - ETH Zurich"},
            rank_blocks={},  # the guard fires on the mode alone, before any block is used
            process_props=_TwoRankProcessProperties(),
        )
    message = str(err.value)
    assert "<serial build>" in message
    assert "pip install --no-binary netcdf4" in message
    assert "__has_parallel4_support__" in message


class _FakeNetCDF4Module:
    """netCDF4 module stand-in with a controlled parallel-support flag.

    The real predicate is bypassed (monkeypatched) everywhere else, so its three
    branches are pinned here against a stub instead of the local installation.
    """

    __version__ = "0.0.0-test"
    __netcdf4libversion__ = "0.0.0"
    __hdf5libversion__ = "0.0.0"

    def __init__(self, has_parallel4_support: bool) -> None:
        self.__has_parallel4_support__ = has_parallel4_support


def _find_spec_pretending_mpi4py(present: bool):
    real_find_spec = importlib.util.find_spec

    def find_spec(name, *args):
        if name == "mpi4py":
            return object() if present else None
        return real_find_spec(name, *args)

    return find_spec


def test_missing_parallel_support_reports_serial_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(netcdf_writers, "nc", _FakeNetCDF4Module(False))
    reason = netcdf_writers.missing_parallel_support()
    assert reason is not None
    assert "serial build" in reason
    assert "__has_parallel4_support__" in reason
    assert "PyPI wheels" in reason


def test_missing_parallel_support_requires_mpi4py(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(netcdf_writers, "nc", _FakeNetCDF4Module(True))
    monkeypatch.setattr(importlib.util, "find_spec", _find_spec_pretending_mpi4py(present=False))
    assert netcdf_writers.missing_parallel_support() == "the 'mpi4py' package is not installed"


def test_missing_parallel_support_accepts_parallel_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(netcdf_writers, "nc", _FakeNetCDF4Module(True))
    monkeypatch.setattr(importlib.util, "find_spec", _find_spec_pretending_mpi4py(present=True))
    assert netcdf_writers.missing_parallel_support() is None


def test_bounded_middle_chunks_keeps_small_shapes_whole() -> None:
    assert netcdf_writers._bounded_middle_chunks((80,), 10_000, 8) == (80,)


def test_bounded_middle_chunks_shrinks_to_the_hdf5_limit() -> None:
    # 80 levels x 10.5M cells x 8 B = 6.7 GB exceeds the 4 GiB chunk limit: the
    # vertical chunk must shrink while the horizontal axis stays one chunk per block
    horizontal_chunk = 10_500_000
    (vertical_chunk,) = netcdf_writers._bounded_middle_chunks((80,), horizontal_chunk, 8)
    assert 1 <= vertical_chunk < 80
    assert vertical_chunk * horizontal_chunk * 8 <= netcdf_writers._MAX_CHUNK_BYTES


def test_bounded_middle_chunks_rejects_oversized_rank_block() -> None:
    # a single horizontal row of the block already exceeds the limit
    with pytest.raises(RuntimeError, match="chunk size limit"):
        netcdf_writers._bounded_middle_chunks((80,), 2**30, 8)
