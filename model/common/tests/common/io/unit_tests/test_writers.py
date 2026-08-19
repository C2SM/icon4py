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
from icon4py.model.common.decomposition import definitions as decomposition_defs
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


TEST_GLOBAL_ATTRS: writers.GlobalFileAttributes = {  # type: ignore[typeddict-item]  # Tests use only a subset of required attributes
    "title": "test",
    "institution": "EXCLAIM - ETH Zurich",
}


def _vertical_params(grid: grid_def.Grid) -> v_grid.VerticalGrid:
    num_levels = grid.config.vertical_size
    heights = np.linspace(start=12000.0, stop=0.0, num=num_levels + 1)
    vertical_config = v_grid.VerticalGridConfig(num_levels=num_levels)
    return v_grid.VerticalGrid(
        vertical_config,
        vct_a=gtx.as_field((dims.KDim,), heights),  # type: ignore[arg-type]  # NDArrayObject Protocol mismatch
        vct_b=None,
    )


def initialized_writer(
    test_path: pathlib.Path,
    random_name: str,
    grid: grid_def.Grid = test_io_utils.simple_grid,
    horizontal_chunk_size: int | None = None,
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
        global_attrs=TEST_GLOBAL_ATTRS,
        horizontal_chunk_size=horizontal_chunk_size,
        rank_blocks=None,
        process_props=decomposition_defs.SingleNodeProcessProperties(),
    )
    writer.initialize_dataset()
    return writer, grid


def test_initialize_writer_time_var(test_path: pathlib.Path, random_name: str) -> None:
    dataset, _ = initialized_writer(test_path, random_name)
    time_var = dataset.variables[writers.TIME]
    assert time_var.dimensions == ("time",)
    assert time_var.units == "seconds since 1970-01-01 00:00:00"
    assert time_var.calendar == "proleptic_gregorian"
    assert time_var.long_name == "time"
    assert time_var.standard_name == "time"
    assert len(time_var) == 0


def test_initialize_writer_vertical_model_levels(test_path: pathlib.Path, random_name: str) -> None:
    dataset, grid = initialized_writer(test_path, random_name)
    vertical = dataset.variables[writers.MODEL_LEVEL]
    assert vertical.units == "1"
    assert vertical.dimensions == (writers.MODEL_LEVEL,)
    assert vertical.long_name == "model full level index"
    assert vertical.standard_name == metadata.LEVEL_STANDARD_NAME
    assert vertical.datatype == np.int32
    assert len(vertical) == grid.num_levels
    assert np.all(vertical == np.arange(grid.num_levels))


def test_initialize_writer_half_levels(test_path: pathlib.Path, random_name: str) -> None:
    dataset, grid = initialized_writer(test_path, random_name)
    half_levels = dataset.variables[writers.MODEL_HALF_LEVEL]
    assert half_levels.units == "1"
    assert half_levels.datatype == np.int32
    assert half_levels.long_name == "model half level index"
    assert half_levels.standard_name == metadata.INTERFACE_LEVEL_STANDARD_NAME
    assert len(half_levels) == grid.num_levels + 1
    assert np.all(half_levels == np.arange(grid.num_levels + 1))


def test_initialize_writer_heights(test_path: pathlib.Path, random_name: str) -> None:
    dataset, grid = initialized_writer(test_path, random_name)
    heights = dataset.variables["height"]
    assert heights.units == "m"
    assert heights.datatype == np.float64
    assert heights.long_name == "height value of half levels without topography"
    assert heights.standard_name == metadata.INTERFACE_LEVEL_HEIGHT_STANDARD_NAME
    assert len(heights) == grid.num_levels + 1
    assert heights[0] == 12000.0
    assert heights[-1] == 0.0


def test_writer_append_timeslice(test_path: pathlib.Path, random_name: str) -> None:
    writer, _ = initialized_writer(test_path, random_name)
    time = datetime.now()
    assert len(writer.variables[writers.TIME]) == 0
    slice1: dict[str, xr.DataArray] = {}
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


def test_writer_append_timeslice_create_new_var(test_path: pathlib.Path, random_name: str) -> None:
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


def test_writer_append_timeslice_to_existing_var(test_path: pathlib.Path, random_name: str) -> None:
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
    test_utils.assert_dallclose(dataset.variables["air_density"][1], new_rho.ndarray.T)  # type: ignore[attr-defined]  # NDArrayObject Protocol lacks .T attribute


def initialized_zarr_writer(
    test_path: pathlib.Path,
    random_name: str,
    grid: grid_def.Grid = test_io_utils.simple_grid,
    *,
    rank_blocks: dict[str, distributed.RankBlock] | None = None,
    horizontal: grid_def.HorizontalGridSize | None = None,
    horizontal_chunk_size: int | None = None,
    horizontal_shard_size: int | None = None,
) -> tuple[zarr_writers.ZarrWriter, pathlib.Path]:
    store_path = test_path.absolute() / f"{random_name}.zarr"
    writer = zarr_writers.ZarrWriter(
        file_name=store_path,
        vertical=_vertical_params(grid),
        horizontal=horizontal if horizontal is not None else grid.config.horizontal_config,
        time_properties=writers.TimeProperties(
            cf_utils.DEFAULT_TIME_UNIT, cf_utils.DEFAULT_CALENDAR
        ),
        global_attrs=TEST_GLOBAL_ATTRS,
        rank_blocks=rank_blocks,
        horizontal_chunk_size=horizontal_chunk_size,
        horizontal_shard_size=horizontal_shard_size,
        process_props=decomposition_defs.SingleNodeProcessProperties(),
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
        # serial store: the horizontal axis is in global order, the UGRID association holds
        assert ds["air_density"].attrs["mesh"] == "mesh"
        assert ds["air_density"].attrs["location"] == "face"
        assert writers.LAYOUT_ATTRIBUTE not in ds["air_density"].attrs


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
        test_utils.assert_dallclose(ds["air_density"].values[1], new_rho.ndarray.T)  # type: ignore[attr-defined]  # NDArrayObject Protocol lacks .T attribute
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
        global_attrs=TEST_GLOBAL_ATTRS,
        rank_blocks=None,
        process_props=decomposition_defs.SingleNodeProcessProperties(),
    )
    with pytest.raises(FileExistsError):
        duplicate.initialize_dataset()


class _ReplayingComm:
    """Communicator stand-in replaying the root rank's broadcast to a non-root rank."""

    def __init__(self, root_message: str | None) -> None:
        self._root_message = root_message

    def bcast(self, _obj: object, root: int = 0) -> str | None:
        return self._root_message


class _NonRootProcessProperties:
    """Rank 1 of a two-rank run; ``comm`` replays what rank 0 broadcast."""

    def __init__(self, comm: _ReplayingComm) -> None:
        self.comm = comm

    rank = 1
    comm_name = ""
    comm_size = 2

    def is_single_rank(self) -> bool:
        return False


def test_zarr_writer_root_failure_reaches_non_root_ranks(
    test_path: pathlib.Path, random_name: str
) -> None:
    """A store-creation failure on the root rank raises on the other ranks too.

    Rank-block mode: the root creates the store while the other ranks wait for its
    verdict. If the root fails (existing store, full disk), the verdict is a message
    and every rank raises -- instead of the non-root ranks waiting forever for a
    store that never appears (a hang, not an error). Here rank 1 receives the
    broadcast the root would have sent after its ``zarr.open_group`` failed.
    """
    non_root = zarr_writers.ZarrWriter(
        file_name=test_path / f"{random_name}.zarr",
        vertical=_vertical_params(test_io_utils.simple_grid),
        horizontal=test_io_utils.simple_grid.config.horizontal_config,
        time_properties=writers.TimeProperties(
            cf_utils.DEFAULT_TIME_UNIT, cf_utils.DEFAULT_CALENDAR
        ),
        global_attrs=TEST_GLOBAL_ATTRS,
        rank_blocks=_single_rank_block(10),
        process_props=_NonRootProcessProperties(
            _ReplayingComm("FileExistsError: store already exists")
        ),
    )
    with pytest.raises(RuntimeError, match=r"failed on the root rank.*store already exists"):
        non_root.initialize_dataset()


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
            size=size + padding,
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
        # the rank-ordered, padded axis invalidates the UGRID association: the layout
        # marker replaces it, so no reader places the values on the mesh unreordered
        assert ds["air_density"].attrs[writers.LAYOUT_ATTRIBUTE] == writers.RANK_BLOCK_LAYOUT
        assert "mesh" not in ds["air_density"].attrs
        assert "location" not in ds["air_density"].attrs
    # the -1 padding marker doubles as the store's fill value (which xarray does not
    # decode as a missing value for format 3); read undecoded to pin the on-disk
    # contract independent of the reader's decoding defaults
    with xr.open_zarr(store_path, mask_and_scale=False) as ds:
        global_index = ds[f"{writers.GLOBAL_INDEX_PREFIX}_{writers.CELL}"].values
        assert np.all(global_index[: grid.num_cells] == np.arange(grid.num_cells))
        assert np.all(global_index[grid.num_cells :] == -1)
    # default layout: one chunk per rank block
    group = zarr.open_group(store_path, mode="r")
    cell_chunk = rank_blocks[writers.CELL].size
    air_density = group["air_density"]
    global_index_cell = group[f"{writers.GLOBAL_INDEX_PREFIX}_{writers.CELL}"]
    assert isinstance(air_density, zarr.Array)
    assert isinstance(global_index_cell, zarr.Array)
    assert air_density.chunks == (1, grid.num_levels, cell_chunk)
    assert global_index_cell.chunks == (cell_chunk,)


def test_zarr_writer_rank_block_writes_at_nonzero_start(
    test_path: pathlib.Path, random_name: str
) -> None:
    # store view of a non-root rank: its block starts at rank * block size; data and
    # global indices must land inside the block only, everything before and after
    # stays padding
    grid = test_io_utils.simple_grid
    rank = 1
    rank_blocks = {}
    for dim_name, size in (
        (writers.CELL, grid.num_cells),
        (writers.EDGE, grid.num_edges),
        (writers.VERTEX, grid.num_vertices),
    ):
        block_size = size + 1  # uneven layout: one padding entry per block
        rank_blocks[dim_name] = distributed.RankBlock(
            start=rank * block_size,
            count=size,
            size=block_size,
            padded_size=2 * block_size,
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


def test_zarr_writer_horizontal_chunk_and_shard_sizes(
    test_path: pathlib.Path, random_name: str
) -> None:
    # a serial store with configured chunking/sharding: the horizontal axes split into
    # chunks of the configured size, grouped into shards of whole chunks; the data
    # content is independent of the layout
    grid = test_io_utils.simple_grid
    writer, store_path = initialized_zarr_writer(
        test_path, random_name, horizontal_chunk_size=4, horizontal_shard_size=8
    )
    state = dict(air_density=test_io_utils.model_state(grid)["air_density"])
    writer.append(state, datetime.now())
    writer.close()
    air_density = zarr.open_group(store_path, mode="r")["air_density"]
    assert isinstance(air_density, zarr.Array)
    assert air_density.chunks == (1, grid.num_levels, 4)
    assert air_density.shards == (1, grid.num_levels, 8)
    with xr.open_zarr(store_path) as ds:
        test_utils.assert_dallclose(ds["air_density"].values[0], state["air_density"].data.T)


def test_zarr_writer_rank_block_chunk_and_shard_sizes(
    test_path: pathlib.Path, random_name: str
) -> None:
    # rank-block store with sub-chunked, sharded blocks (block size a multiple of
    # the shard size); data, padding and global indices are unchanged
    grid = test_io_utils.simple_grid
    chunk_size, shard_size = 3, 6
    rank_blocks = {}
    for dim_name, size in (
        (writers.CELL, grid.num_cells),
        (writers.EDGE, grid.num_edges),
        (writers.VERTEX, grid.num_vertices),
    ):
        block_size = (size + shard_size - 1) // shard_size * shard_size
        rank_blocks[dim_name] = distributed.RankBlock(
            start=0,
            count=size,
            size=block_size,
            padded_size=block_size,
            global_size=size,
            global_index=np.arange(size, dtype=np.int64),
        )
    padded_horizontal = grid_def.HorizontalGridSize(
        num_cells=rank_blocks[writers.CELL].padded_size,
        num_edges=rank_blocks[writers.EDGE].padded_size,
        num_vertices=rank_blocks[writers.VERTEX].padded_size,
    )
    writer, store_path = initialized_zarr_writer(
        test_path,
        random_name,
        grid,
        rank_blocks=rank_blocks,
        horizontal=padded_horizontal,
        horizontal_chunk_size=chunk_size,
        horizontal_shard_size=shard_size,
    )
    state = dict(air_density=test_io_utils.model_state(grid)["air_density"])
    writer.append(state, datetime.now())
    writer.close()
    group = zarr.open_group(store_path, mode="r")
    air_density = group["air_density"]
    global_index_cell = group[f"{writers.GLOBAL_INDEX_PREFIX}_{writers.CELL}"]
    assert isinstance(air_density, zarr.Array)
    assert isinstance(global_index_cell, zarr.Array)
    assert air_density.chunks == (1, grid.num_levels, chunk_size)
    assert air_density.shards == (1, grid.num_levels, shard_size)
    assert global_index_cell.chunks == (chunk_size,)
    assert global_index_cell.shards == (shard_size,)
    with xr.open_zarr(store_path) as ds:
        values = ds["air_density"].values[0]
        test_utils.assert_dallclose(values[:, : grid.num_cells], state["air_density"].data.T)
        assert np.all(np.isnan(values[:, grid.num_cells :]))
    with xr.open_zarr(store_path, mask_and_scale=False) as ds:
        global_index = ds[f"{writers.GLOBAL_INDEX_PREFIX}_{writers.CELL}"].values
        assert np.all(global_index[: grid.num_cells] == np.arange(grid.num_cells))
        assert np.all(global_index[grid.num_cells :] == -1)


def test_zarr_writer_rejects_chunks_crossing_rank_blocks(
    test_path: pathlib.Path, random_name: str
) -> None:
    # a block size of 10 is no multiple of a chunk size of 4
    with pytest.raises(ValueError, match="not a multiple"):
        zarr_writers.ZarrWriter(
            file_name=test_path / f"{random_name}.zarr",
            vertical=_vertical_params(test_io_utils.simple_grid),
            horizontal=test_io_utils.simple_grid.config.horizontal_config,
            time_properties=writers.TimeProperties(
                cf_utils.DEFAULT_TIME_UNIT, cf_utils.DEFAULT_CALENDAR
            ),
            global_attrs=TEST_GLOBAL_ATTRS,
            rank_blocks=_single_rank_block(10),
            horizontal_chunk_size=4,
            process_props=decomposition_defs.SingleNodeProcessProperties(),
        )


def test_initialize_writer_create_dimensions(
    test_path: pathlib.Path,
    random_name: str,
) -> None:
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
    *,
    horizontal_chunk_size: int | None = None,
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
        global_attrs=TEST_GLOBAL_ATTRS,
        rank_blocks=rank_blocks,
        horizontal_chunk_size=horizontal_chunk_size,
        process_props=decomposition_defs.SingleNodeProcessProperties(),
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
            size=size + padding,
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
        # the rank-ordered, padded axis invalidates the UGRID association: the layout
        # marker replaces it, so no reader places the values on the mesh unreordered
        assert ds["air_density"].attrs[writers.LAYOUT_ATTRIBUTE] == writers.RANK_BLOCK_LAYOUT
        assert "mesh" not in ds["air_density"].attrs
        assert "location" not in ds["air_density"].attrs
        # the -1 padding is written explicitly, not encoded as a _FillValue attribute
        # (xarray would decode that to NaN, turning the integer coordinate into floats)
        global_index = ds[f"{writers.GLOBAL_INDEX_PREFIX}_{writers.CELL}"].values
        assert global_index.dtype == np.int64
        assert np.all(global_index[: grid.num_cells] == np.arange(grid.num_cells))
        assert np.all(global_index[grid.num_cells :] == -1)
    # default layout: one chunk per rank block, same as the rank-block zarr store
    cell_chunk = rank_blocks[writers.CELL].size
    with nc.Dataset(file_path) as raw:
        assert raw["air_density"].chunking() == [1, grid.num_levels, cell_chunk]
        assert raw[f"{writers.GLOBAL_INDEX_PREFIX}_{writers.CELL}"].chunking() == [cell_chunk]


def test_netcdf_writer_rank_block_writes_at_nonzero_start(
    test_path: pathlib.Path, random_name: str
) -> None:
    # file view of a non-root rank: its block starts at rank * block size; data and
    # global indices must land inside the block only. Unlike the zarr store (whose fill value
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
        block_size = size + 1  # uneven layout: one padding entry per block
        rank_blocks[dim_name] = distributed.RankBlock(
            start=rank * block_size,
            count=size,
            size=block_size,
            padded_size=2 * block_size,
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
            global_index[cell_block.start + cell_block.count : cell_block.start + cell_block.size]
            == -1
        )


def test_netcdf_writer_horizontal_chunk_size(test_path: pathlib.Path, random_name: str) -> None:
    # a serial file is chunked only when configured
    writer, grid = initialized_writer(test_path, random_name, horizontal_chunk_size=4)
    state = dict(air_density=test_io_utils.model_state(grid)["air_density"])
    writer.append(state, datetime.now())
    assert writer.variables["air_density"].chunking() == [1, grid.num_levels, 4]
    writer.close()


def test_netcdf_writer_horizontal_chunk_size_clamped_to_axis(
    test_path: pathlib.Path, random_name: str
) -> None:
    # netCDF rejects chunks larger than a fixed dimension: an oversized configured
    # chunk is clamped to the axis size
    writer, grid = initialized_writer(test_path, random_name, horizontal_chunk_size=10_000)
    state = dict(air_density=test_io_utils.model_state(grid)["air_density"])
    writer.append(state, datetime.now())
    assert writer.variables["air_density"].chunking() == [1, grid.num_levels, grid.num_cells]
    writer.close()


def test_netcdf_writer_rank_block_horizontal_chunk_size(
    test_path: pathlib.Path, random_name: str
) -> None:
    # rank-block file with sub-chunked blocks: the chunk size divides every block size
    # (padded sizes 21/30/12), so chunks never cross block boundaries; data and
    # padding are unchanged
    grid = test_io_utils.simple_grid
    padding = 3
    chunk_size = 3
    rank_blocks = {
        dim_name: distributed.RankBlock(
            start=0,
            count=size,
            size=size + padding,
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
        test_path,
        random_name,
        grid,
        rank_blocks,
        padded_horizontal,
        horizontal_chunk_size=chunk_size,
    )
    state = dict(air_density=test_io_utils.model_state(grid)["air_density"])
    writer.append(state, datetime.now())
    writer.close()
    with nc.Dataset(file_path) as raw:
        assert raw["air_density"].chunking() == [1, grid.num_levels, chunk_size]
        assert raw[f"{writers.GLOBAL_INDEX_PREFIX}_{writers.CELL}"].chunking() == [chunk_size]
    with xr.open_dataset(file_path) as ds:
        values = ds["air_density"].values[0]
        test_utils.assert_dallclose(values[:, : grid.num_cells], state["air_density"].data.T)
        assert np.all(np.isnan(values[:, grid.num_cells :]))


def test_netcdf_writer_rejects_chunks_crossing_rank_blocks(
    test_path: pathlib.Path, random_name: str
) -> None:
    # a block size of 10 is no multiple of a chunk size of 4
    with pytest.raises(ValueError, match="not a multiple"):
        netcdf_writers.NETCDFWriter(
            file_name=test_path / f"{random_name}.nc",
            vertical=_vertical_params(test_io_utils.simple_grid),
            horizontal=test_io_utils.simple_grid.config.horizontal_config,
            time_properties=writers.TimeProperties(
                cf_utils.DEFAULT_TIME_UNIT, cf_utils.DEFAULT_CALENDAR
            ),
            global_attrs=TEST_GLOBAL_ATTRS,
            rank_blocks=_single_rank_block(10),
            horizontal_chunk_size=4,
            process_props=decomposition_defs.SingleNodeProcessProperties(),
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
            global_attrs=TEST_GLOBAL_ATTRS,
            rank_blocks={},  # the guard fires on the mode alone, before any block is used
            process_props=_TwoRankProcessProperties(),
        )
    message = str(err.value)
    assert "<serial build>" in message
    assert "pip install --no-binary netcdf4" in message
    assert "__has_parallel4_support__" in message
    assert "'zarr' backend" in message and "'gather' mode" in message


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


def _find_spec_pretending_mpi4py(present: bool):  # type: ignore[no-untyped-def]  # Returns a dynamically-constructed finder function
    real_find_spec = importlib.util.find_spec

    def find_spec(name, *args):  # type: ignore[no-untyped-def]  # Dynamically-constructed finder function
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


def _single_rank_block(size: int) -> dict[str, distributed.RankBlock]:
    return {
        writers.CELL: distributed.RankBlock(
            start=0,
            count=size,
            size=size,
            padded_size=size,
            global_size=size,
            global_index=np.arange(size, dtype=np.int64),
        )
    }


def test_zarr_writer_rejects_shards_crossing_rank_blocks(
    test_path: pathlib.Path, random_name: str
) -> None:
    # a block size of 24 holds whole chunks of 8 but not whole shards of 16
    with pytest.raises(ValueError, match="shard"):
        zarr_writers.ZarrWriter(
            file_name=test_path / f"{random_name}.zarr",
            vertical=_vertical_params(test_io_utils.simple_grid),
            horizontal=test_io_utils.simple_grid.config.horizontal_config,
            time_properties=writers.TimeProperties(
                cf_utils.DEFAULT_TIME_UNIT, cf_utils.DEFAULT_CALENDAR
            ),
            global_attrs=TEST_GLOBAL_ATTRS,
            rank_blocks=_single_rank_block(24),
            horizontal_chunk_size=8,
            horizontal_shard_size=16,
            process_props=decomposition_defs.SingleNodeProcessProperties(),
        )


def test_zarr_writer_rejects_shard_without_dividing_chunk(
    test_path: pathlib.Path, random_name: str
) -> None:
    # every rank must reject the layout at construction; zarr itself would raise on
    # the root rank only, inside a pre-barrier store operation
    with pytest.raises(ValueError, match="shard size"):
        zarr_writers.ZarrWriter(
            file_name=test_path / f"{random_name}.zarr",
            vertical=_vertical_params(test_io_utils.simple_grid),
            horizontal=test_io_utils.simple_grid.config.horizontal_config,
            time_properties=writers.TimeProperties(
                cf_utils.DEFAULT_TIME_UNIT, cf_utils.DEFAULT_CALENDAR
            ),
            global_attrs=TEST_GLOBAL_ATTRS,
            horizontal_chunk_size=3,
            horizontal_shard_size=8,
            rank_blocks=None,
            process_props=decomposition_defs.SingleNodeProcessProperties(),
        )


def test_zarr_writer_append_invalid_field_leaves_store_unchanged(
    test_path: pathlib.Path, random_name: str
) -> None:
    # a bad field must raise before any store mutation (identically on every rank in
    # rank-block mode), or the store would keep a phantom time slice
    writer, store_path = initialized_zarr_writer(test_path, random_name)
    grid = test_io_utils.simple_grid
    no_attrs = xr.DataArray(np.zeros((grid.num_cells, grid.num_levels)), dims=("cell", "level"))
    with pytest.raises(ValueError, match="missing the CF attributes"):
        writer.append({"air_density": no_attrs}, datetime.now())
    bogus_dims = xr.DataArray(np.zeros((4, 5)), dims=("foo", "bar"))
    with pytest.raises(ValueError, match="unknown horizontal dimension"):
        writer.append({"junk": bogus_dims}, datetime.now())
    writer.close()
    with xr.open_zarr(store_path) as ds:
        assert ds.sizes[writers.TIME] == 0


def test_netcdf_writer_append_invalid_field_leaves_file_unchanged(
    test_path: pathlib.Path, random_name: str
) -> None:
    writer, grid = initialized_writer(test_path, random_name)
    no_attrs = xr.DataArray(np.zeros((grid.num_cells, grid.num_levels)), dims=("cell", "level"))
    with pytest.raises(ValueError, match="missing the CF attributes"):
        writer.append({"air_density": no_attrs}, datetime.now())
    assert len(writer.variables[writers.TIME]) == 0
    writer.close()


def test_netcdf_writer_append_two_fields_sharing_standard_name(
    test_path: pathlib.Path, random_name: str
) -> None:
    # variables resolve by name (like the zarr writer): a shared standard_name must
    # yield two variables, not silently overwrite the first
    writer, grid = initialized_writer(test_path, random_name)
    field = test_io_utils.model_state(grid)["air_density"]
    writer.append({"air_density": field, "air_density_copy": field.copy(deep=True)}, datetime.now())
    assert "air_density" in writer.variables
    assert "air_density_copy" in writer.variables
    writer.close()
