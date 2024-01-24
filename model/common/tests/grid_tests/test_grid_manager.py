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
from __future__ import annotations

import functools
import logging
import typing
from uuid import uuid4

import numpy as np
import pytest

from icon4py.model.common.test_utils.datatest_utils import GLOBAL_EXPERIMENT, REGIONAL_EXPERIMENT


if typing.TYPE_CHECKING:
    import netCDF4

try:
    import netCDF4  # noqa: F811
except ImportError:
    pytest.skip("optional netcdf dependency not installed", allow_module_level=True)

from icon4py.model.common.dimension import (
    C2E2CDim,
    C2EDim,
    C2VDim,
    CellDim,
    E2C2EDim,
    E2CDim,
    E2VDim,
    EdgeDim,
    V2CDim,
    V2EDim,
    VertexDim,
)
from icon4py.model.common.grid.grid_manager import (
    GridFile,
    GridFileName,
    GridManager,
    IndexTransformation,
    ToGt4PyTransformation,
)
from icon4py.model.common.grid.horizontal import HorizontalMarkerIndex
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.grid.vertical import VerticalGridSize

from .utils import MCH_GRID_FILE, R02B04_GLOBAL, resolve_file_from_gridfile_name


SIMPLE_GRID_NC = "simple_grid.nc"
R02B04_GLOBLA_NUM_VERTICES = 10242

R02B04_GLOBAL_NUM_EDGES = 30720

R02B04_GLOBAL_NUM_CELLS = 20480


@pytest.fixture
def simple_grid_gridfile(tmp_path):
    path = tmp_path.joinpath(SIMPLE_GRID_NC).absolute()
    grid = SimpleGrid()
    dataset = netCDF4.Dataset(path, "w", format="NETCDF4")
    dataset.setncattr(GridFile.PropertyName.GRID_ID, str(uuid4()))
    dataset.createDimension(GridFile.DimensionName.VERTEX_NAME, size=grid.num_vertices)

    dataset.createDimension(GridFile.DimensionName.EDGE_NAME, size=grid.num_edges)
    dataset.createDimension(GridFile.DimensionName.CELL_NAME, size=grid.num_cells)
    dataset.createDimension(GridFile.DimensionName.NEIGHBORS_TO_EDGE_SIZE, size=grid.size[E2VDim])
    dataset.createDimension(GridFile.DimensionName.DIAMOND_EDGE_SIZE, size=grid.size[E2C2EDim])
    dataset.createDimension(GridFile.DimensionName.MAX_CHILD_DOMAINS, size=1)
    # add dummy values for the grf dimensions
    dataset.createDimension(GridFile.DimensionName.CELL_GRF, size=14)
    dataset.createDimension(GridFile.DimensionName.EDGE_GRF, size=24)
    dataset.createDimension(GridFile.DimensionName.VERTEX_GRF, size=13)
    _add_to_dataset(
        dataset,
        np.zeros(grid.num_edges),
        GridFile.GridRefinementName.CONTROL_EDGES,
        (GridFile.DimensionName.EDGE_NAME,),
    )

    _add_to_dataset(
        dataset,
        np.zeros(grid.num_cells),
        GridFile.GridRefinementName.CONTROL_CELLS,
        (GridFile.DimensionName.CELL_NAME,),
    )
    _add_to_dataset(
        dataset,
        np.zeros(grid.num_vertices),
        GridFile.GridRefinementName.CONTROL_VERTICES,
        (GridFile.DimensionName.VERTEX_NAME,),
    )

    dataset.createDimension(GridFile.DimensionName.NEIGHBORS_TO_CELL_SIZE, size=grid.size[C2EDim])
    dataset.createDimension(GridFile.DimensionName.NEIGHBORS_TO_VERTEX_SIZE, size=grid.size[V2CDim])

    _add_to_dataset(
        dataset,
        grid.connectivities[C2EDim],
        GridFile.OffsetName.C2E,
        (
            GridFile.DimensionName.NEIGHBORS_TO_CELL_SIZE,
            GridFile.DimensionName.CELL_NAME,
        ),
    )

    _add_to_dataset(
        dataset,
        grid.connectivities[E2CDim],
        GridFile.OffsetName.E2C,
        (
            GridFile.DimensionName.NEIGHBORS_TO_EDGE_SIZE,
            GridFile.DimensionName.EDGE_NAME,
        ),
    )
    _add_to_dataset(
        dataset,
        grid.connectivities[E2VDim],
        GridFile.OffsetName.E2V,
        (
            GridFile.DimensionName.NEIGHBORS_TO_EDGE_SIZE,
            GridFile.DimensionName.EDGE_NAME,
        ),
    )

    _add_to_dataset(
        dataset,
        grid.connectivities[V2CDim],
        GridFile.OffsetName.V2C,
        (
            GridFile.DimensionName.NEIGHBORS_TO_VERTEX_SIZE,
            GridFile.DimensionName.VERTEX_NAME,
        ),
    )

    _add_to_dataset(
        dataset,
        grid.connectivities[C2VDim],
        GridFile.OffsetName.C2V,
        (
            GridFile.DimensionName.NEIGHBORS_TO_CELL_SIZE,
            GridFile.DimensionName.CELL_NAME,
        ),
    )
    _add_to_dataset(
        dataset,
        np.zeros((grid.num_vertices, 4), dtype=np.int32),
        GridFile.OffsetName.V2E2V,
        (GridFile.DimensionName.DIAMOND_EDGE_SIZE, GridFile.DimensionName.VERTEX_NAME),
    )
    _add_to_dataset(
        dataset,
        grid.connectivities[V2EDim],
        GridFile.OffsetName.V2E,
        (
            GridFile.DimensionName.NEIGHBORS_TO_VERTEX_SIZE,
            GridFile.DimensionName.VERTEX_NAME,
        ),
    )
    _add_to_dataset(
        dataset,
        grid.connectivities[C2E2CDim],
        GridFile.OffsetName.C2E2C,
        (
            GridFile.DimensionName.NEIGHBORS_TO_CELL_SIZE,
            GridFile.DimensionName.CELL_NAME,
        ),
    )

    _add_to_dataset(
        dataset,
        np.ones((1, 24), dtype=np.int32),
        GridFile.GridRefinementName.START_INDEX_EDGES,
        (GridFile.DimensionName.MAX_CHILD_DOMAINS, GridFile.DimensionName.EDGE_GRF),
    )
    _add_to_dataset(
        dataset,
        np.ones((1, 14), dtype=np.int32),
        GridFile.GridRefinementName.START_INDEX_CELLS,
        (GridFile.DimensionName.MAX_CHILD_DOMAINS, GridFile.DimensionName.CELL_GRF),
    )
    _add_to_dataset(
        dataset,
        np.ones((1, 13), dtype=np.int32),
        GridFile.GridRefinementName.START_INDEX_VERTICES,
        (GridFile.DimensionName.MAX_CHILD_DOMAINS, GridFile.DimensionName.VERTEX_GRF),
    )
    _add_to_dataset(
        dataset,
        np.ones((1, 24), dtype=np.int32),
        GridFile.GridRefinementName.END_INDEX_EDGES,
        (GridFile.DimensionName.MAX_CHILD_DOMAINS, GridFile.DimensionName.EDGE_GRF),
    )
    _add_to_dataset(
        dataset,
        np.ones((1, 14), dtype=np.int32),
        GridFile.GridRefinementName.END_INDEX_CELLS,
        (GridFile.DimensionName.MAX_CHILD_DOMAINS, GridFile.DimensionName.CELL_GRF),
    )
    _add_to_dataset(
        dataset,
        np.ones((1, 13), dtype=np.int32),
        GridFile.GridRefinementName.END_INDEX_VERTICES,
        (GridFile.DimensionName.MAX_CHILD_DOMAINS, GridFile.DimensionName.VERTEX_GRF),
    )
    dataset.close()
    yield path
    path.unlink()


def _add_to_dataset(
    dataset: netCDF4.Dataset,
    data: np.ndarray,
    var_name: str,
    dims: tuple[GridFileName, GridFileName],
):
    var = dataset.createVariable(var_name, np.int32, dims)
    var[:] = np.transpose(data)[:]


@pytest.mark.with_netcdf
def test_gridparser_dimension(simple_grid_gridfile):
    data = netCDF4.Dataset(simple_grid_gridfile, "r")
    grid_parser = GridFile(data)
    grid = SimpleGrid()
    assert grid_parser.dimension(GridFile.DimensionName.CELL_NAME) == grid.num_cells
    assert grid_parser.dimension(GridFile.DimensionName.VERTEX_NAME) == grid.num_vertices
    assert grid_parser.dimension(GridFile.DimensionName.EDGE_NAME) == grid.num_edges


@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [(MCH_GRID_FILE, REGIONAL_EXPERIMENT), (R02B04_GLOBAL, GLOBAL_EXPERIMENT)],
)
def test_gridfile_vertex_cell_edge_dimensions(grid_savepoint, grid_file):
    file = resolve_file_from_gridfile_name(grid_file)
    dataset = netCDF4.Dataset(file, "r")
    grid_file = GridFile(dataset)

    assert grid_file.dimension(GridFile.DimensionName.CELL_NAME) == grid_savepoint.num(CellDim)
    assert grid_file.dimension(GridFile.DimensionName.EDGE_NAME) == grid_savepoint.num(EdgeDim)
    assert grid_file.dimension(GridFile.DimensionName.VERTEX_NAME) == grid_savepoint.num(VertexDim)


@pytest.mark.with_netcdf
def test_grid_parser_index_fields(simple_grid_gridfile, caplog):
    caplog.set_level(logging.DEBUG)
    data = netCDF4.Dataset(simple_grid_gridfile, "r")
    grid = SimpleGrid()
    grid_parser = GridFile(data)

    assert np.allclose(grid_parser.int_field(GridFile.OffsetName.C2E), grid.connectivities[C2EDim])
    assert np.allclose(grid_parser.int_field(GridFile.OffsetName.E2C), grid.connectivities[E2CDim])
    assert np.allclose(grid_parser.int_field(GridFile.OffsetName.V2E), grid.connectivities[V2EDim])
    assert np.allclose(grid_parser.int_field(GridFile.OffsetName.V2C), grid.connectivities[V2CDim])


# TODO @magdalena add test cases for hexagon vertices v2e2v
# v2e2v: grid,???


# v2e: exists in serial, simple, grid
@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [(MCH_GRID_FILE, REGIONAL_EXPERIMENT), (R02B04_GLOBAL, GLOBAL_EXPERIMENT)],
)
def test_gridmanager_eval_v2e(caplog, grid_savepoint, grid_file):
    caplog.set_level(logging.DEBUG)
    file = resolve_file_from_gridfile_name(grid_file)
    grid = init_grid_manager(file).get_grid()
    seralized_v2e = grid_savepoint.v2e()[0 : grid.num_vertices, :]
    # there are vertices at the boundary of a local domain or at a pentagon point that have less than
    # 6 neighbors hence there are "Missing values" in the grid file
    # they get substituted by the "last valid index" in preprocessing step in icon.
    assert not has_invalid_index(seralized_v2e)
    v2e_table = grid.get_offset_provider("V2E").table
    assert has_invalid_index(v2e_table)
    reset_invalid_index(seralized_v2e)
    assert np.allclose(v2e_table, seralized_v2e)


# v2c: exists in serial, simple, grid
@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [(MCH_GRID_FILE, REGIONAL_EXPERIMENT), (R02B04_GLOBAL, GLOBAL_EXPERIMENT)],
)
def test_gridmanager_eval_v2c(caplog, grid_savepoint, grid_file):
    caplog.set_level(logging.DEBUG)
    file = resolve_file_from_gridfile_name(grid_file)
    grid = init_grid_manager(file).get_grid()
    serialized_v2c = grid_savepoint.v2c()[0 : grid.num_vertices, :]
    # there are vertices that have less than 6 neighboring cells: either pentagon points or
    # vertices at the boundary of the domain for a limited area mode
    # hence in the grid file there are "missing values"
    # they get substituted by the "last valid index" in preprocessing step in icon.
    assert not has_invalid_index(serialized_v2c)
    assert has_invalid_index(grid.get_offset_provider("V2C").table)
    reset_invalid_index(serialized_v2c)

    assert np.allclose(grid.get_offset_provider("V2C").table, serialized_v2c)


def reset_invalid_index(index_array: np.ndarray):
    """
    Revert changes from mo_model_domimp_patches.

    Helper function to revert mo_model_domimp_patches.f90: move_dummies_to_end_idxblk.
    see:
    # ! Checks for the pentagon case and moves dummy cells to end.
    #  ! The dummy entry is either set to 0 or duplicated from the last one
    #  SUBROUTINE move_dummies_to_end(array, array_size, max_connectivity, duplicate)

    After reading the grid file ICON moves all invalid indices (neighbors not existing in the
    grid file) to the end of the neighbor list and replaces them with the "last valid neighbor index"
    it is up to the user then to ensure that any coefficients in neighbor some multiplied with
    these values are zero in order to "remove" them again from the sum.

    For testing we resubstitute those to the GridFile.INVALID_INDEX
    Args:
        index_array: the array where values the invalid values have to be reset

    Returns: an array where the spurious "last valid index" are replaced by GridFile.INVALID_INDEX

    """
    for i in range(0, index_array.shape[0]):
        uq, index = np.unique(index_array[i, :], return_index=True)
        index_array[i, max(index) + 1 :] = GridFile.INVALID_INDEX


# e2v: exists in serial, simple, grid
@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [(MCH_GRID_FILE, REGIONAL_EXPERIMENT), (R02B04_GLOBAL, GLOBAL_EXPERIMENT)],
)
def test_gridmanager_eval_e2v(caplog, grid_savepoint, grid_file):
    caplog.set_level(logging.DEBUG)
    file = resolve_file_from_gridfile_name(grid_file)
    grid = init_grid_manager(file).get_grid()

    serialized_e2v = grid_savepoint.e2v()[0 : grid.num_edges, :]
    # all vertices in the system have to neighboring edges, there no edges that point nowhere
    # hence this connectivity has no "missing values" in the grid file
    assert not has_invalid_index(serialized_e2v)
    assert not has_invalid_index(grid.get_offset_provider("E2V").table)
    assert np.allclose(grid.get_offset_provider("E2V").table, serialized_e2v)


def has_invalid_index(ar: np.ndarray):
    return np.any(invalid_index(ar))


def invalid_index(ar):
    return np.where(ar == GridFile.INVALID_INDEX)


def _is_local(grid_file: str):
    return grid_file == MCH_GRID_FILE


def assert_invalid_indices(e2c_table: np.ndarray, grid_file: str):
    """
    Assert invalid indices for E2C connectivity.

    Local grids: there are edges at the boundary that have only one
    neighboring cell, there are "missing values" in the grid file
    and for E2C they do not get substituted in the ICON preprocessing.

    Global grids have no "missing values" indices since all edges always have 2 neighboring cells.

    Args:
        e2c_table: E2C connectivity
        grid_file: name of grid file used

    """
    if _is_local(grid_file):
        assert has_invalid_index(e2c_table)
    else:
        assert not has_invalid_index(e2c_table)


# e2c : exists in serial, simple, grid
@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [(MCH_GRID_FILE, REGIONAL_EXPERIMENT), (R02B04_GLOBAL, GLOBAL_EXPERIMENT)],
)
def test_gridmanager_eval_e2c(caplog, grid_savepoint, grid_file):
    caplog.set_level(logging.DEBUG)
    file = resolve_file_from_gridfile_name(grid_file)
    grid = init_grid_manager(file).get_grid()
    serialized_e2c = grid_savepoint.e2c()[0 : grid.num_edges, :]
    e2c_table = grid.get_offset_provider("E2C").table
    assert_invalid_indices(serialized_e2c, grid_file)
    assert_invalid_indices(e2c_table, grid_file)
    assert np.allclose(e2c_table, serialized_e2c)


# c2e: serial, simple, grid
@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [(MCH_GRID_FILE, REGIONAL_EXPERIMENT), (R02B04_GLOBAL, GLOBAL_EXPERIMENT)],
)
def test_gridmanager_eval_c2e(caplog, grid_savepoint, grid_file):
    caplog.set_level(logging.DEBUG)
    file = resolve_file_from_gridfile_name(grid_file)
    grid = init_grid_manager(file).get_grid()

    serialized_c2e = grid_savepoint.c2e()[0 : grid.num_cells, :]
    # no cells with less than 3 neighboring edges exist, otherwise the cell is not there in the
    # first place
    # hence there are no "missing values" in the grid file
    assert not has_invalid_index(serialized_c2e)
    assert not has_invalid_index(grid.get_offset_provider("C2E").table)
    assert np.allclose(grid.get_offset_provider("C2E").table, serialized_c2e)


# c2e2c: exists in  serial, simple_mesh, grid
@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [(MCH_GRID_FILE, REGIONAL_EXPERIMENT), (R02B04_GLOBAL, GLOBAL_EXPERIMENT)],
)
def test_gridmanager_eval_c2e2c(caplog, grid_savepoint, grid_file):
    caplog.set_level(logging.DEBUG)
    file = resolve_file_from_gridfile_name(grid_file)
    grid = init_grid_manager(file).get_grid()
    assert np.allclose(
        grid.get_offset_provider("C2E2C").table,
        grid_savepoint.c2e2c()[0 : grid.num_cells, :],
    )


# e2c2e (e2c2eo) - diamond: exists in serial, simple_mesh
@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [(MCH_GRID_FILE, REGIONAL_EXPERIMENT), (R02B04_GLOBAL, GLOBAL_EXPERIMENT)],
)
def test_gridmanager_eval_e2c2e(caplog, grid_savepoint, grid_file):
    caplog.set_level(logging.DEBUG)
    file = resolve_file_from_gridfile_name(grid_file)
    grid = init_grid_manager(file).get_grid()
    serialized_e2c2e = grid_savepoint.e2c2e()[0 : grid.num_cells, :]
    assert_invalid_indices(serialized_e2c2e, grid_file)

    e2c2e_table = grid.get_offset_provider("E2C2E").table

    assert_invalid_indices(e2c2e_table, grid_file)
    # ICON calculates diamond edges only from rl_start = 2 (lateral_boundary(EdgeDim) + 1 for
    # boundaries all values are INVALID even though the half diamond exists (see mo_model_domimp_setup.f90 ll 163ff.)
    assert_unless_invalid(e2c2e_table, serialized_e2c2e)


def assert_unless_invalid(table, serialized_ref):
    invalid_positions = invalid_index(table)
    np.allclose(table[invalid_positions], serialized_ref[invalid_positions])


@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [(MCH_GRID_FILE, REGIONAL_EXPERIMENT), (R02B04_GLOBAL, GLOBAL_EXPERIMENT)],
)
def test_gridmanager_eval_e2c2v(caplog, grid_savepoint, grid_file):
    caplog.set_level(logging.DEBUG)
    file = resolve_file_from_gridfile_name(grid_file)
    grid = init_grid_manager(file).get_grid()
    # the "far" (adjacent to edge normal ) is not always there, because ICON only calculates those starting from
    #   (lateral_boundary(EdgeDim) + 1) to end(EdgeDim)  (see mo_intp_coeffs.f90) and only for owned cells
    serialized_ref = grid_savepoint.e2c2v()[: grid.num_edges, :]
    table = grid.get_offset_provider("E2C2V").table
    assert_unless_invalid(table, serialized_ref)


@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [(MCH_GRID_FILE, REGIONAL_EXPERIMENT), (R02B04_GLOBAL, GLOBAL_EXPERIMENT)],
)
def test_gridmanager_eval_c2v(caplog, grid_savepoint, grid_file):
    caplog.set_level(logging.DEBUG)
    file = resolve_file_from_gridfile_name(grid_file)
    grid = init_grid_manager(file).get_grid()
    c2v = grid.get_offset_provider("C2V").table
    assert np.allclose(c2v, grid_savepoint.c2v()[0 : grid.num_cells, :])


@functools.cache
def init_grid_manager(fname, num_levels=65, transformation=ToGt4PyTransformation()):
    grid_manager = GridManager(transformation, fname, VerticalGridSize(num_levels))
    grid_manager()
    return grid_manager


@pytest.mark.parametrize("dim, size", [(CellDim, 18), (EdgeDim, 27), (VertexDim, 9)])
@pytest.mark.with_netcdf
def test_grid_manager_getsize(simple_grid_gridfile, dim, size, caplog):
    caplog.set_level(logging.DEBUG)
    gm = init_grid_manager(
        simple_grid_gridfile, num_levels=10, transformation=IndexTransformation()
    )
    gm()
    assert size == gm.get_size(dim)


def assert_up_to_order(table, diamond_table):
    assert table.shape == diamond_table.shape
    for n in range(table.shape[0]):
        assert np.all(np.in1d(table[n, :], diamond_table[n, :]))


@pytest.mark.with_netcdf
def test_grid_manager_diamond_offset(simple_grid_gridfile):
    simple_grid = SimpleGrid()
    gm = init_grid_manager(
        simple_grid_gridfile,
        num_levels=simple_grid.num_levels,
        transformation=IndexTransformation(),
    )
    gm()
    icon_grid = gm.get_grid()
    table = icon_grid.get_offset_provider("E2C2V").table
    assert_up_to_order(table, simple_grid.diamond_table)


@pytest.mark.with_netcdf
def test_gridmanager_given_file_not_found_then_abort():
    fname = "./unknown_grid.nc"
    with pytest.raises(SystemExit) as error:
        gm = GridManager(IndexTransformation(), fname, VerticalGridSize(num_lev=80))
        gm()
        assert error.type == SystemExit
        assert error.value == 1


@pytest.mark.parametrize("size", [100, 1500, 20000])
@pytest.mark.with_netcdf
def test_gt4py_transform_offset_by_1_where_valid(size):
    trafo = ToGt4PyTransformation()
    input_field = np.random.randint(-1, size, (size,))
    offset = trafo.get_offset_for_index_field(input_field)
    expected = np.where(input_field >= 0, -1, 0)
    assert np.allclose(expected, offset)


@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "dim, marker, start_index",
    [
        (CellDim, HorizontalMarkerIndex.interior(CellDim), 4104),
        (CellDim, HorizontalMarkerIndex.interior(CellDim) + 1, 0),
        (CellDim, HorizontalMarkerIndex.local(CellDim) - 1, 20896),
        (CellDim, HorizontalMarkerIndex.halo(CellDim), 20896),
        (CellDim, HorizontalMarkerIndex.nudging(CellDim), 3316),
        (CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 3, 2511),
        (CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 2, 1688),
        (CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 1, 850),
        (CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 0, 0),
        (EdgeDim, HorizontalMarkerIndex.interior(EdgeDim), 6176),
        (EdgeDim, HorizontalMarkerIndex.local(EdgeDim) - 2, 31558),
        (EdgeDim, HorizontalMarkerIndex.local(EdgeDim) - 1, 31558),
        (EdgeDim, HorizontalMarkerIndex.nudging(EdgeDim) + 1, 5387),
        (EdgeDim, HorizontalMarkerIndex.nudging(EdgeDim), 4989),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 7, 4184),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 6, 3777),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 5, 2954),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 4, 2538),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 3, 1700),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 2, 1278),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1, 428),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 0, 0),
        (VertexDim, HorizontalMarkerIndex.interior(VertexDim), 2071),
        (VertexDim, HorizontalMarkerIndex.local(VertexDim) - 1, 10663),
        (VertexDim, HorizontalMarkerIndex.nudging(VertexDim) + 1, 10663),
        (VertexDim, HorizontalMarkerIndex.nudging(VertexDim), 10663),
        (VertexDim, HorizontalMarkerIndex.end(VertexDim), 10663),
        (VertexDim, HorizontalMarkerIndex.lateral_boundary(VertexDim) + 4, 1673),
        (VertexDim, HorizontalMarkerIndex.lateral_boundary(VertexDim) + 3, 1266),
        (VertexDim, HorizontalMarkerIndex.lateral_boundary(VertexDim) + 2, 850),
        (VertexDim, HorizontalMarkerIndex.lateral_boundary(VertexDim) + 1, 428),
        (VertexDim, HorizontalMarkerIndex.lateral_boundary(VertexDim) + 0, 0),
    ],
)
def test_get_start_index_for_local_grid(grid_savepoint, dim, marker, start_index):
    file = resolve_file_from_gridfile_name(MCH_GRID_FILE)
    from_savepoint = grid_savepoint.construct_icon_grid()
    from_grid_file = init_grid_manager(file).get_grid()
    assert from_grid_file.get_start_index(dim, marker) == start_index
    assert from_grid_file.get_start_index(dim, marker) == from_savepoint.get_start_index(
        dim, marker
    )


@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "dim, marker, index",
    [
        (CellDim, HorizontalMarkerIndex.interior(CellDim), 20896),
        (CellDim, HorizontalMarkerIndex.interior(CellDim) + 1, 850),
        (CellDim, HorizontalMarkerIndex.local(CellDim) - 2, 20896),
        (CellDim, HorizontalMarkerIndex.local(CellDim) - 1, 20896),
        (CellDim, HorizontalMarkerIndex.local(CellDim), 20896),
        (CellDim, HorizontalMarkerIndex.nudging(CellDim), 4104),
        (CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 3, 3316),
        (CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 2, 2511),
        (CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 1, 1688),
        (CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 0, 850),
        (EdgeDim, HorizontalMarkerIndex.interior(EdgeDim), 31558),
        (EdgeDim, HorizontalMarkerIndex.local(EdgeDim) - 2, 31558),
        (EdgeDim, HorizontalMarkerIndex.local(EdgeDim) - 1, 31558),
        (EdgeDim, HorizontalMarkerIndex.local(EdgeDim), 31558),
        (EdgeDim, HorizontalMarkerIndex.nudging(EdgeDim) + 1, 6176),
        (EdgeDim, HorizontalMarkerIndex.nudging(EdgeDim), 5387),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 7, 4989),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 6, 4184),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 5, 3777),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 4, 2954),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 3, 2538),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 2, 1700),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1, 1278),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 0, 428),
        (VertexDim, HorizontalMarkerIndex.interior(VertexDim), 10663),
        (VertexDim, HorizontalMarkerIndex.local(VertexDim) - 2, 10663),
        (VertexDim, HorizontalMarkerIndex.local(VertexDim) - 1, 10663),
        (VertexDim, HorizontalMarkerIndex.local(VertexDim), 10663),
        (VertexDim, HorizontalMarkerIndex.nudging(VertexDim) + 1, 10663),
        (VertexDim, HorizontalMarkerIndex.nudging(VertexDim), 10663),
        (VertexDim, HorizontalMarkerIndex.end(VertexDim), 10663),
        (VertexDim, HorizontalMarkerIndex.lateral_boundary(VertexDim) + 4, 2071),
        (VertexDim, HorizontalMarkerIndex.lateral_boundary(VertexDim) + 3, 1673),
        (VertexDim, HorizontalMarkerIndex.lateral_boundary(VertexDim) + 2, 1266),
        (VertexDim, HorizontalMarkerIndex.lateral_boundary(VertexDim) + 1, 850),
        (VertexDim, HorizontalMarkerIndex.lateral_boundary(VertexDim) + 0, 428),
    ],
)
@pytest.mark.parametrize(
    "grid_file, experiment, num_levels", [(MCH_GRID_FILE, REGIONAL_EXPERIMENT, 65)]
)
def test_get_end_index_for_local_grid(grid_file, num_levels, grid_savepoint, dim, marker, index):
    file = resolve_file_from_gridfile_name(grid_file)
    from_savepoint = grid_savepoint.construct_icon_grid()
    from_grid_file = init_grid_manager(file, num_levels=num_levels).get_grid()
    assert from_grid_file.get_end_index(dim, marker) == index
    assert from_grid_file.get_end_index(dim, marker) == from_savepoint.get_end_index(dim, marker)


@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "dim, marker,start_index,end_index",
    [
        (CellDim, HorizontalMarkerIndex.interior(CellDim), 0, 0),
        (CellDim, HorizontalMarkerIndex.local(CellDim), 0, R02B04_GLOBAL_NUM_CELLS),
        (CellDim, HorizontalMarkerIndex.nudging(CellDim), 0, 0),
        (CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim), 0, 0),
        (
            CellDim,
            HorizontalMarkerIndex.end(CellDim),
            R02B04_GLOBAL_NUM_CELLS,
            R02B04_GLOBAL_NUM_CELLS,
        ),
        (
            CellDim,
            HorizontalMarkerIndex.halo(CellDim),
            R02B04_GLOBAL_NUM_CELLS,
            R02B04_GLOBAL_NUM_CELLS,
        ),
        (EdgeDim, HorizontalMarkerIndex.interior(EdgeDim), 0, 0),
        (EdgeDim, HorizontalMarkerIndex.local(EdgeDim), 0, R02B04_GLOBAL_NUM_EDGES),
        (EdgeDim, HorizontalMarkerIndex.nudging(EdgeDim), 0, 0),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim), 0, 0),
        (
            EdgeDim,
            HorizontalMarkerIndex.end(EdgeDim),
            R02B04_GLOBAL_NUM_EDGES,
            R02B04_GLOBAL_NUM_EDGES,
        ),
        (
            EdgeDim,
            HorizontalMarkerIndex.halo(EdgeDim),
            R02B04_GLOBAL_NUM_EDGES,
            R02B04_GLOBAL_NUM_EDGES,
        ),
        (VertexDim, HorizontalMarkerIndex.interior(VertexDim), 0, 0),
        (VertexDim, HorizontalMarkerIndex.local(VertexDim), 0, R02B04_GLOBLA_NUM_VERTICES),
        (VertexDim, HorizontalMarkerIndex.lateral_boundary(VertexDim), 0, 0),
        (
            VertexDim,
            HorizontalMarkerIndex.end(VertexDim),
            R02B04_GLOBLA_NUM_VERTICES,
            R02B04_GLOBLA_NUM_VERTICES,
        ),
        (
            VertexDim,
            HorizontalMarkerIndex.halo(VertexDim),
            R02B04_GLOBLA_NUM_VERTICES,
            R02B04_GLOBLA_NUM_VERTICES,
        ),
    ],
)
@pytest.mark.parametrize("grid_file, num_levels", [(R02B04_GLOBAL, 80)])
def test_get_start_end_index_for_global_grid(
    grid_file, num_levels, dim, marker, start_index, end_index
):
    file = resolve_file_from_gridfile_name(grid_file)
    from_grid_file = init_grid_manager(file, num_levels=num_levels).get_grid()
    assert from_grid_file.get_start_index(dim, marker) == start_index
    assert from_grid_file.get_end_index(dim, marker) == end_index
