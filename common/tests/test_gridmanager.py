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
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from netCDF4 import Dataset

from icon4py.common.dimension import CellDim, EdgeDim, VertexDim
from icon4py.grid.grid_manager import (
    GridFile,
    GridFileName,
    GridManager,
    IndexTransformation,
    ToGt4PyTransformation,
)
from icon4py.grid.horizontal import HorizontalMarkerIndex
from icon4py.grid.vertical import VerticalGridConfig
from icon4py.testutils.simple_mesh import SimpleMesh


SIMPLE_MESH_NC = "./simple_mesh_grid.nc"


@pytest.fixture
def simple_mesh_path(simple_mesh_data):
    return Path(SIMPLE_MESH_NC).absolute()


@pytest.fixture(scope="session")
def simple_mesh_data():
    mesh = SimpleMesh()
    dataset = Dataset(SIMPLE_MESH_NC, "w", format="NETCDF4")
    dataset.setncattr(GridFile.PropertyName.GRID_ID, str(uuid4()))
    dataset.createDimension(GridFile.DimensionName.VERTEX_NAME, size=mesh.n_vertices)

    dataset.createDimension(GridFile.DimensionName.EDGE_NAME, size=mesh.n_edges)
    dataset.createDimension(GridFile.DimensionName.CELL_NAME, size=mesh.n_cells)
    dataset.createDimension(
        GridFile.DimensionName.NEIGHBORS_TO_EDGE_SIZE, size=mesh.n_e2v
    )
    dataset.createDimension(GridFile.DimensionName.DIAMOND_EDGE_SIZE, size=mesh.n_e2c2e)
    dataset.createDimension(GridFile.DimensionName.MAX_CHILD_DOMAINS, size=1)
    # add dummy values for the grf dimensions
    # TODO @magdalena fix to something more useful?
    dataset.createDimension(GridFile.DimensionName.CELL_GRF, size=14)
    dataset.createDimension(GridFile.DimensionName.EDGE_GRF, size=24)
    dataset.createDimension(GridFile.DimensionName.VERTEX_GRF, size=13)
    _add_to_dataset(
        dataset,
        np.zeros(mesh.n_edges),
        GridFile.GridRefinementName.CONTROL_EDGES,
        (GridFile.DimensionName.EDGE_NAME,),
    )

    _add_to_dataset(
        dataset,
        np.zeros(mesh.n_cells),
        GridFile.GridRefinementName.CONTROL_CELLS,
        (GridFile.DimensionName.CELL_NAME,),
    )
    _add_to_dataset(
        dataset,
        np.zeros(mesh.n_vertices),
        GridFile.GridRefinementName.CONTROL_VERTICES,
        (GridFile.DimensionName.VERTEX_NAME,),
    )

    dataset.createDimension(
        GridFile.DimensionName.NEIGHBORS_TO_CELL_SIZE, size=mesh.n_c2e
    )
    dataset.createDimension(
        GridFile.DimensionName.NEIGHBORS_TO_VERTEX_SIZE, size=mesh.n_v2c
    )

    _add_to_dataset(
        dataset,
        mesh.c2e,
        GridFile.OffsetName.C2E,
        (
            GridFile.DimensionName.NEIGHBORS_TO_CELL_SIZE,
            GridFile.DimensionName.CELL_NAME,
        ),
    )

    _add_to_dataset(
        dataset,
        mesh.e2c,
        GridFile.OffsetName.E2C,
        (
            GridFile.DimensionName.NEIGHBORS_TO_EDGE_SIZE,
            GridFile.DimensionName.EDGE_NAME,
        ),
    )
    _add_to_dataset(
        dataset,
        mesh.e2v,
        GridFile.OffsetName.E2V,
        (
            GridFile.DimensionName.NEIGHBORS_TO_EDGE_SIZE,
            GridFile.DimensionName.EDGE_NAME,
        ),
    )

    _add_to_dataset(
        dataset,
        mesh.v2c,
        GridFile.OffsetName.V2C,
        (
            GridFile.DimensionName.NEIGHBORS_TO_VERTEX_SIZE,
            GridFile.DimensionName.VERTEX_NAME,
        ),
    )

    _add_to_dataset(
        dataset,
        mesh.c2v,
        GridFile.OffsetName.C2V,
        (
            GridFile.DimensionName.NEIGHBORS_TO_CELL_SIZE,
            GridFile.DimensionName.CELL_NAME,
        ),
    )
    _add_to_dataset(
        dataset,
        np.zeros((mesh.n_vertices, 4), dtype=np.int32),
        GridFile.OffsetName.V2E2V,
        (GridFile.DimensionName.DIAMOND_EDGE_SIZE, GridFile.DimensionName.VERTEX_NAME),
    )
    _add_to_dataset(
        dataset,
        mesh.v2e,
        GridFile.OffsetName.V2E,
        (
            GridFile.DimensionName.NEIGHBORS_TO_VERTEX_SIZE,
            GridFile.DimensionName.VERTEX_NAME,
        ),
    )
    # add_to_dataset(data, mesh.v2e2v, GridFile.Offsets.V2E2V, (GridFile.Dimension.V2E_SIZE, GridFile.Dimension.VERTEX_NAME))
    _add_to_dataset(
        dataset,
        mesh.c2e2c,
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


def _add_to_dataset(
    dataset: Dataset,
    data: np.ndarray,
    var_name: str,
    dims: tuple[GridFileName, GridFileName],
):
    var = dataset.createVariable(var_name, np.int32, dims)
    var[:] = np.transpose(data)[:]




def test_gridparser_dimension(simple_mesh_data):

    data = Dataset(SIMPLE_MESH_NC, "r")
    grid_parser = GridFile(data)
    mesh = SimpleMesh()
    assert grid_parser.dimension(GridFile.DimensionName.CELL_NAME) == mesh.n_cells
    assert grid_parser.dimension(GridFile.DimensionName.VERTEX_NAME) == mesh.n_vertices
    assert grid_parser.dimension(GridFile.DimensionName.EDGE_NAME) == mesh.n_edges


def test_gridfile_vertex_cell_edge_dimensions(grid_savepoint, r04b09_dsl_gridfile):
    data = Dataset(r04b09_dsl_gridfile, "r")
    grid_file = GridFile(data)

    assert grid_file.dimension(GridFile.DimensionName.CELL_NAME) == grid_savepoint.num(
        CellDim
    )
    assert grid_file.dimension(GridFile.DimensionName.EDGE_NAME) == grid_savepoint.num(
        EdgeDim
    )
    # TODO: @magdalena fix in serialized data. it returns the num_cells
    assert grid_file.dimension(
        GridFile.DimensionName.VERTEX_NAME
    ) == grid_savepoint.num(VertexDim)


def test_grid_parser_index_fields(simple_mesh_data, caplog):
    caplog.set_level(logging.DEBUG)
    data = Dataset(SIMPLE_MESH_NC, "r")
    mesh = SimpleMesh()
    grid_parser = GridFile(data)

    assert np.allclose(grid_parser.int_field(GridFile.OffsetName.C2E), mesh.c2e)
    assert np.allclose(grid_parser.int_field(GridFile.OffsetName.E2C), mesh.e2c)
    assert np.allclose(grid_parser.int_field(GridFile.OffsetName.V2E), mesh.v2e)
    assert np.allclose(grid_parser.int_field(GridFile.OffsetName.V2C), mesh.v2c)


# TODO @magdalena add test cases for
# v2e2v: grid,???

# v2e: serial, simple, grid
@pytest.mark.datatest
def test_gridmanager_eval_v2e(caplog, grid_savepoint, r04b09_dsl_gridfile):
    caplog.set_level(logging.DEBUG)
    gm = init_grid_manager(r04b09_dsl_gridfile)
    num_vertex = gm.get_size(VertexDim)
    seralized_v2e = grid_savepoint.v2e()[0:num_vertex, :]
    # there are vertices at the boundary of a local domain or at a pentagon point that have less than
    # 6 neighbors hence there are "Missing values" in the grid file
    # they get substituted by the "last valid index" in preprocessing step in icon.
    assert not has_invalid_index(seralized_v2e)
    assert has_invalid_index(gm.get_v2e_connectivity().table)
    reset_invalid_index(seralized_v2e)
    assert np.allclose(gm.get_v2e_connectivity().table, seralized_v2e)



# v2c: serial, simple, grid
@pytest.mark.datatest
def test_gridmanager_eval_v2c(caplog, grid_savepoint, r04b09_dsl_gridfile):
    caplog.set_level(logging.DEBUG)
    gm = init_grid_manager(r04b09_dsl_gridfile)
    num_vertex = gm.get_size(VertexDim)
    serialized_v2c = grid_savepoint.v2c()[0:num_vertex, :]
    # there are vertices that have less than 6 neighboring cells: either pentagon points or
    # vertices at the boundary of the domain for a limited area mode
    # hence in the grid file there are "missing values"
    # they get substituted by the "last valid index" in preprocessing step in icon.
    assert not has_invalid_index(serialized_v2c)
    assert has_invalid_index(gm.get_v2c_connectivity().table)
    reset_invalid_index(serialized_v2c)

    assert np.allclose(gm.get_v2c_connectivity().table, serialized_v2c)


def reset_invalid_index(index_array: np.ndarray):
    """
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


# e2v: serial, simple, grid
@pytest.mark.datatest
def test_gridmanager_eval_e2v(caplog, grid_savepoint, r04b09_dsl_gridfile):
    caplog.set_level(logging.DEBUG)
    gm = init_grid_manager(r04b09_dsl_gridfile)
    num_edges = gm.get_size(EdgeDim)

    serialized_e2v = grid_savepoint.e2v()[0:num_edges, :]
    # all vertices in the system have to neighboring edges, there no edges that point nowhere
    # hence this connectivity has no "missing values" in the grid file
    assert not has_invalid_index(serialized_e2v)
    assert not has_invalid_index(gm.get_e2v_connectivity().table)
    assert np.allclose(gm.get_e2v_connectivity().table, serialized_e2v)



def has_invalid_index(ar: np.ndarray):
    return np.any(np.where(ar == GridFile.INVALID_INDEX))




# e2c :serial, simple, grid
@pytest.mark.datatest
def test_gridmanager_eval_e2c(caplog, grid_savepoint, r04b09_dsl_gridfile):
    caplog.set_level(logging.DEBUG)
    gm = init_grid_manager(r04b09_dsl_gridfile)
    num_edges = gm.get_size(EdgeDim)
    serialized_c2e = grid_savepoint.e2c()[0:num_edges, :]
    # there are edges at the boundary that have only one
    # neighboring cell, there are "missing values" in the grid file
    # and here they do not get substituted in the ICON preprocessing
    assert has_invalid_index(serialized_c2e)
    assert has_invalid_index(gm.get_e2c_connectivity().table)
    assert np.allclose(gm.get_e2c_connectivity().table, serialized_c2e)


# c2e: serial, simple, grid
@pytest.mark.datatest
def test_gridmanager_eval_c2e(caplog, grid_savepoint, r04b09_dsl_gridfile):
    caplog.set_level(logging.DEBUG)
    gm = init_grid_manager(r04b09_dsl_gridfile)
    num_cells = gm.get_size(CellDim)

    serialized_c2e = grid_savepoint.c2e()[0:num_cells, :]
    # no cells with less than 3 neighboring edges exist, otherwise the cell is not there in the
    # first place
    # hence there are no "missing values" in the grid file
    assert not has_invalid_index(serialized_c2e)
    assert not has_invalid_index(gm.get_c2e_connectivity().table)
    assert np.allclose(gm.get_c2e_connectivity().table, serialized_c2e)


# e2c2e (e2c2eo) - diamond: serial, simple_mesh, grid file????
@pytest.mark.skip("does this array exist in grid file?")
@pytest.mark.datatest
def test_gridmanager_eval_e2c2e(caplog, grid_savepoint, r04b09_dsl_gridfile):
    caplog.set_level(logging.DEBUG)
    gm, num_cells, num_edges, num_vertex = init_grid_manager(r04b09_dsl_gridfile)
    serialized_e2c2e = grid_savepoint.e2c2e()[0:num_cells, :]
    assert has_invalid_index(serialized_e2c2e)
    assert has_invalid_index(gm.get_e2c2e_connectivity().table)
    assert np.allclose(gm.get_e2c2e_connectivity().table, serialized_e2c2e)




# c2e2c: serial, simple_mesh, grid
@pytest.mark.datatest
def test_gridmanager_eval_c2e2c(caplog, grid_savepoint, r04b09_dsl_gridfile):
    caplog.set_level(logging.DEBUG)
    gm = init_grid_manager(r04b09_dsl_gridfile)
    num_cells = gm.get_size(CellDim)
    assert np.allclose(
        gm.get_c2e2c_connectivity().table, grid_savepoint.c2e2c()[0:num_cells, :]
    )

def test_gridmanager_eval_e2c2v(caplog, grid_savepoint, r04b09_dsl_gridfile):
    caplog.set_level(logging.DEBUG)
    gm = init_grid_manager(r04b09_dsl_gridfile)
    num_edges = gm.get_size(EdgeDim)
    c2v = gm.get_c2v_connectivity().table
    assert np.allclose(
        gm.get_e2c2v_connectivity().table, grid_savepoint.e2c2v()[0:num_edges, :]
    )
@pytest.mark.skip("add to savepoint")
def test_gridmanager_eval_c2v(caplog, grid_savepoint, r04b09_dsl_gridfile):
    caplog.set_level(logging.DEBUG)
    gm = init_grid_manager(r04b09_dsl_gridfile)
    num_cells = gm.get_size(CellDim)
    c2v = gm.get_c2v_connectivity().table
    assert np.allclose(
        c2v, grid_savepoint.c2v()[0:num_cells, :]
    )
def init_grid_manager(fname):
    gm = GridManager(ToGt4PyTransformation(), fname, VerticalGridConfig(65))
    gm.init()
    return gm


@pytest.mark.parametrize("dim, size", [(CellDim, 18), (EdgeDim, 27), (VertexDim, 9)])
def test_grid_manager_getsize(simple_mesh_data, simple_mesh_path, dim, size, caplog):
    caplog.set_level(logging.DEBUG)
    gm = GridManager(IndexTransformation(), simple_mesh_path)
    gm.init()
    assert size == gm.get_size(dim)

def test_grid_manager_diamond_offset(simple_mesh_path):
    mesh = SimpleMesh()
    gm = GridManager(IndexTransformation(), simple_mesh_path, VerticalGridConfig(num_lev=mesh.k_level))
    gm.init()
    assert np.allclose(np.sort(gm.get_e2c2v_connectivity().table, 1), np.sort(mesh.diamond_arr, 1))


def test_gridmanager_given_file_not_found_then_abort():
    fname = "./unknown_grid.nc"
    with pytest.raises(SystemExit) as error:
        gm = GridManager(IndexTransformation(), fname)
        gm.init()
        assert error.type == SystemExit
        assert error.value == 1


@pytest.mark.parametrize("size", [100, 1500, 20000])
def test_gt4py_transform_offset_by_1_where_valid(size):
    trafo = ToGt4PyTransformation()
    input_field = np.random.randint(-1, size, (size,))
    offset = trafo.get_offset_for_index_field(input_field)
    expected = np.where(input_field >= 0, -1, 0)
    assert np.allclose(expected, offset)


@pytest.mark.parametrize(
    "dim, marker, index",
    [
        (VertexDim, HorizontalMarkerIndex.lateral_boundary(VertexDim), 0),
        (VertexDim, HorizontalMarkerIndex.local(VertexDim), 10663),
        (VertexDim, HorizontalMarkerIndex.interior(VertexDim), 2071),
        (VertexDim, HorizontalMarkerIndex.nudging(VertexDim), 10663),
        (CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim), 0),
        (CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 1, 850),
        (CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 2, 1688),
        (CellDim, HorizontalMarkerIndex.local(CellDim), 20896),
        (CellDim, HorizontalMarkerIndex.interior(CellDim), 4104),
        (CellDim, HorizontalMarkerIndex.nudging(CellDim), 3316),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim), 0),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 4, 2538),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 5, 2954),
        (EdgeDim, HorizontalMarkerIndex.local(EdgeDim), 31558),
        (EdgeDim, HorizontalMarkerIndex.interior(EdgeDim), 6176),
        (EdgeDim, HorizontalMarkerIndex.nudging(EdgeDim), 4989),
    ],
)
def test_get_start_indices(r04b09_dsl_gridfile, dim, marker, index):
    gm = init_grid_manager(r04b09_dsl_gridfile)
    assert gm.get_start_index(dim, marker) == index


@pytest.mark.parametrize(
    "dim, marker, index",
    [
        (VertexDim, HorizontalMarkerIndex.lateral_boundary(VertexDim), 428),
        (VertexDim, HorizontalMarkerIndex.local(VertexDim), 10663),
        (VertexDim, HorizontalMarkerIndex.interior(VertexDim), 10663),
        (VertexDim, HorizontalMarkerIndex.nudging(VertexDim), 10663),
        (CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim), 850),
        (CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 1, 1688),
        (CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 2, 2511),
        (CellDim, HorizontalMarkerIndex.local(CellDim), 20896),
        (CellDim, HorizontalMarkerIndex.local(CellDim) + 1, 20896),
        (CellDim, HorizontalMarkerIndex.interior(CellDim), 20896),
        (CellDim, HorizontalMarkerIndex.nudging(CellDim), 4104),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim), 428),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 4, 2954),
        (EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 5, 3777),
        (EdgeDim, HorizontalMarkerIndex.local(EdgeDim), 31558),
        (EdgeDim, HorizontalMarkerIndex.interior(EdgeDim), 31558),
        (EdgeDim, HorizontalMarkerIndex.nudging(EdgeDim), 5387),
    ],
)
def test_get_start_indices(r04b09_dsl_gridfile, dim, marker, index):
    gm = init_grid_manager(r04b09_dsl_gridfile)
    assert gm.get_end_index(dim, marker) == index
