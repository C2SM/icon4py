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
from icon4py.grid.icon_grid import (
    GridFile,
    GridFileName,
    GridManager,
    GridTransformation,
    ToGt4PyTransformation,
)
from icon4py.testutils.data_handling import download_and_extract
from icon4py.testutils.simple_mesh import SimpleMesh


SIMPLE_MESH_NC = "./simple_mesh_grid.nc"

grid_uri = "https://polybox.ethz.ch/index.php/s/hD232znfEPBh4Oh/download"
grids_path = Path(__file__).parent.joinpath("grids")
r04b09_dsl_grid_path = grids_path.joinpath("mch_ch_r04b09_dsl")
data_file = r04b09_dsl_grid_path.joinpath("mch_ch_r04b09_dsl_grids_v1.tar.gz").name


@pytest.fixture(scope="session")
def get_grid_files():
    """
    Get the grid files used for testing.

    Session scoped fixture which is a prerequisite of all the other fixtures in this file.
    """
    download_and_extract(grid_uri, r04b09_dsl_grid_path, data_file)


@pytest.fixture
def simple_mesh_path():
    return Path(SIMPLE_MESH_NC).absolute()


@pytest.fixture(scope="session")
def simple_mesh_data():
    mesh = SimpleMesh()
    dataset = Dataset(SIMPLE_MESH_NC, "w", format="NETCDF4")
    dataset.setncattr(GridFile.Property.GRID_ID, str(uuid4()))
    dataset.createDimension(GridFile.Dimension.VERTEX_NAME, size=mesh.n_vertices)
    dataset.createDimension(GridFile.Dimension.EDGE_NAME, size=mesh.n_edges)
    dataset.createDimension(GridFile.Dimension.CELL_NAME, size=mesh.n_cells)
    dataset.createDimension(GridFile.Dimension.E2V_SIZE, size=mesh.n_e2v)
    dataset.createDimension(GridFile.Dimension.DIAMOND_EDGE_SIZE, size=mesh.n_e2c2e)
    dataset.createDimension(
        GridFile.Dimension.NEIGHBORING_EDGES_TO_CELL_SIZE, size=mesh.n_c2e
    )
    dataset.createDimension(GridFile.Dimension.V2E_SIZE, size=mesh.n_v2c)

    add_to_dataset(
        dataset,
        mesh.c2e,
        GridFile.Offsets.C2E,
        (
            GridFile.Dimension.NEIGHBORING_EDGES_TO_CELL_SIZE,
            GridFile.Dimension.CELL_NAME,
        ),
    )
    # add_to_dataset(data, mesh.c2v, GridFile.Offsets.C2V, (GridFile.Dimension.C2E_SIZE, GridFile.Dimension.CELL_NAME))
    add_to_dataset(
        dataset,
        mesh.e2c,
        GridFile.Offsets.E2C,
        (GridFile.Dimension.E2V_SIZE, GridFile.Dimension.EDGE_NAME),
    )
    add_to_dataset(
        dataset,
        mesh.e2v,
        GridFile.Offsets.E2V,
        (GridFile.Dimension.E2V_SIZE, GridFile.Dimension.EDGE_NAME),
    )

    add_to_dataset(
        dataset,
        mesh.v2c,
        GridFile.Offsets.V2C,
        (GridFile.Dimension.V2E_SIZE, GridFile.Dimension.VERTEX_NAME),
    )
    # TODO fix me there is no v2c in the simple mesh
    add_to_dataset(
        dataset,
        np.zeros((mesh.n_cells, 3), dtype=np.int32),
        GridFile.Offsets.C2V,
        (
            GridFile.Dimension.NEIGHBORING_EDGES_TO_CELL_SIZE,
            GridFile.Dimension.CELL_NAME,
        ),
    )
    add_to_dataset(
        dataset,
        np.zeros((mesh.n_vertices, 4), dtype=np.int32),
        GridFile.Offsets.V2E2V,
        (GridFile.Dimension.DIAMOND_EDGE_SIZE, GridFile.Dimension.VERTEX_NAME),
    )
    add_to_dataset(
        dataset,
        mesh.v2e,
        GridFile.Offsets.V2E,
        (GridFile.Dimension.V2E_SIZE, GridFile.Dimension.VERTEX_NAME),
    )
    # add_to_dataset(data, mesh.v2e2v, GridFile.Offsets.V2E2V, (GridFile.Dimension.V2E_SIZE, GridFile.Dimension.VERTEX_NAME))
    add_to_dataset(
        dataset,
        mesh.c2e2c,
        GridFile.Offsets.C2E2C,
        (
            GridFile.Dimension.NEIGHBORING_EDGES_TO_CELL_SIZE,
            GridFile.Dimension.CELL_NAME,
        ),
    )
    dataset.close()


def add_to_dataset(
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
    assert grid_parser.dimension(GridFile.Dimension.CELL_NAME) == mesh.n_cells
    assert grid_parser.dimension(GridFile.Dimension.VERTEX_NAME) == mesh.n_vertices
    assert grid_parser.dimension(GridFile.Dimension.EDGE_NAME) == mesh.n_edges


def test_grid_parser_index_fields(simple_mesh_data, caplog):
    caplog.set_level(logging.DEBUG)
    data = Dataset(SIMPLE_MESH_NC, "r")
    mesh = SimpleMesh()
    grid_parser = GridFile(data)

    assert np.allclose(grid_parser.int_field(GridFile.Offsets.C2E), mesh.c2e)
    assert np.allclose(grid_parser.int_field(GridFile.Offsets.E2C), mesh.e2c)
    assert np.allclose(grid_parser.int_field(GridFile.Offsets.V2E), mesh.v2e)
    assert np.allclose(grid_parser.int_field(GridFile.Offsets.V2C), mesh.v2c)




#e2c2v - diamond: serial, simple

#c2v: grid, ???
#v2e2v: grid,???

# v2e: serial, simple, grid
@pytest.mark.skip("TODO: handling of boundary values")
@pytest.mark.datatest
def test_gridmanager_eval_v2e(caplog, grid_savepoint, get_grid_files):
    caplog.set_level(logging.DEBUG)
    fname = r04b09_dsl_grid_path.joinpath("grid.nc")
    gm, num_cells, num_edges, num_vertex = _init_grid_manager(fname)
    assert np.allclose(
        gm.get_v2e_connectivity().table, grid_savepoint.v2e()[0:num_vertex, :]
    )

#v2c: serial, simple, grid
#@pytest.mark.skip("TODO: handling of boundary values")

#mo_model_domimp_patches.f90 lines 2183ff
# ! Checks for the pentagon case and moves dummy cells to end.
#  ! The dummy entry is either set to 0 or duplicated from the last one
#  SUBROUTINE move_dummies_to_end(array, array_size, max_connectivity, duplicate)


@pytest.mark.datatest
def test_gridmanager_eval_v2c(caplog, grid_savepoint, get_grid_files):
    caplog.set_level(logging.DEBUG)
    fname = r04b09_dsl_grid_path.joinpath("grid.nc")
    gm, num_cells, num_edges, num_vertex = _init_grid_manager(fname)
    assert np.allclose(
        gm.get_v2c_connectivity().table, grid_savepoint.v2c()[0:num_vertex, :]
    )

#e2v: serial, simple, grid
@pytest.mark.datatest
def test_gridmanager_eval_e2v(caplog, grid_savepoint, get_grid_files):
    caplog.set_level(logging.DEBUG)
    fname = r04b09_dsl_grid_path.joinpath("grid.nc")
    gm, num_cells, num_edges, num_vertex = _init_grid_manager(fname)
    assert np.allclose(
        gm.get_e2v_connectivity().table, grid_savepoint.e2v()[0:num_edges, :]
    )


# e2c :serial, simple, grid
@pytest.mark.datatest
def test_gridmanager_eval_e2c(caplog, grid_savepoint, get_grid_files):
    caplog.set_level(logging.DEBUG)
    fname = r04b09_dsl_grid_path.joinpath("grid.nc")
    gm, num_cells, num_edges, num_vertex = _init_grid_manager(fname)

    assert np.allclose(
        gm.get_e2c_connectivity().table, grid_savepoint.e2c()[0:num_edges, :]
    )


#c2e: serial, simple, grid
@pytest.mark.datatest
def test_gridmanager_eval_c2e(caplog, grid_savepoint, get_grid_files):
    caplog.set_level(logging.DEBUG)
    fname = r04b09_dsl_grid_path.joinpath("grid.nc")
    gm, num_cells, num_edges, num_vertex = _init_grid_manager(fname)
    assert np.allclose(
        gm.get_c2e_connectivity().table, grid_savepoint.c2e()[0:num_cells, :]
    )

# e2c2e (e2c2eo) - diamond: serial, simple
# @pytest.mark.datatest
# def test_gridmanager_eval_e2c2e(caplog, grid_savepoint, get_grid_files):
#     caplog.set_level(logging.DEBUG)
#     fname = r04b09_dsl_grid_path.joinpath("grid.nc")
#     gm, num_cells, num_edges, num_vertex = _init_grid_manager(fname)
#     assert np.allclose(
#         gm.get_e2c2e_connectivity().table, grid_savepoint.e2c2e()[0:num_cells, :]
#     )

#c2e2c: serial, simple, grid
@pytest.mark.datatest
def test_gridmanager_eval_c2e2c(caplog, grid_savepoint, get_grid_files):
    caplog.set_level(logging.DEBUG)
    fname = r04b09_dsl_grid_path.joinpath("grid.nc")
    gm, num_cells, num_edges, num_vertex = _init_grid_manager(fname)
    assert np.allclose(
        gm.get_c2e2c_connectivity().table, grid_savepoint.c2e2c()[0:num_cells, :]
    )




def _init_grid_manager(fname):
    gm = GridManager(ToGt4PyTransformation(), fname)
    gm.init()
    num_vertex = gm.get_size(VertexDim)
    num_edges = gm.get_size(EdgeDim)
    num_cells = gm.get_size(CellDim)
    return gm, num_cells, num_edges, num_vertex


@pytest.mark.parametrize("dim, size", [(CellDim, 18), (EdgeDim, 27), (VertexDim, 9)])
def test_grid_manager_getsize(simple_mesh_data, simple_mesh_path, dim, size, caplog):
    caplog.set_level(logging.DEBUG)
    gm = GridManager(GridTransformation(), simple_mesh_path)
    gm.init()
    assert size == gm.get_size(dim)


def test_gridmanager_given_file_not_found_then_abort():
    fname = "./unknown_grid.nc"
    with pytest.raises(SystemExit) as error:
        gm = GridManager(GridTransformation(), fname)
        gm.init()
        assert error.type == SystemExit
        assert error.value == 1


@pytest.mark.parametrize("size", [100, 1500, 20000])
def test_gt4py_transform_offset_by_1_where_valid(size):
    trafo = ToGt4PyTransformation()
    input_field = np.random.randint(-1, size, (size,))
    offset = trafo.get_offset_for_field(input_field)
    expected = np.where(input_field >= 0, -1, 0)
    assert np.allclose(expected, offset)
