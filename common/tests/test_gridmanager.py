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
import pathlib
from uuid import uuid4

import numpy as np
import pytest
from netCDF4 import Dataset

from icon4py.common.dimension import CellDim, EdgeDim, VertexDim
from icon4py.grid.icon_grid import GridFile, GridFileName, GridManager
from icon4py.testutils.simple_mesh import SimpleMesh


SIMPLE_MESH_NC = "./simple_mesh_grid.nc"


@pytest.fixture
def simple_mesh_path():
    return pathlib.Path(SIMPLE_MESH_NC).absolute()


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
    dataset.createDimension(GridFile.Dimension.C2E_SIZE, size=mesh.n_c2e)
    dataset.createDimension(GridFile.Dimension.V2E_SIZE, size=mesh.n_v2c)

    add_to_dataset(
        dataset,
        mesh.c2e,
        GridFile.Offsets.C2E,
        (GridFile.Dimension.C2E_SIZE, GridFile.Dimension.CELL_NAME),
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
        (GridFile.Dimension.C2E_SIZE, GridFile.Dimension.CELL_NAME),
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
        (GridFile.Dimension.C2E_SIZE, GridFile.Dimension.CELL_NAME),
    )
    dataset.close()


def add_to_dataset(
    dataset: Dataset,
    data: np.ndarray,
    var_name: str,
    dims: tuple[GridFileName, GridFileName],
):
    var = dataset.createVariable(var_name, np.int32, dims)
    var[:] = np.transpose(data)[:] + 1


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


@pytest.mark.skip("TODO: how are -1 values handled?")
@pytest.mark.datatest
def test_gridmanager_read_gridfile(caplog, grid_savepoint):
    caplog.set_level(logging.DEBUG)
    fname = "/home/magdalena/data/exclaim/grids/mch_ch_r04b09_dsl/grid.nc"
    gm = GridManager(fname)
    gm.init()
    num_vertex = gm.get_size(VertexDim)
    num_edges = gm.get_size(EdgeDim)
    num_cells = gm.get_size(CellDim)
    assert np.allclose(
        gm.get_v2e_connectivity().table, grid_savepoint.v2e()[0:num_vertex, :]
    )
    assert np.allclose(
        gm.get_v2c_connectivity().table, grid_savepoint.v2c()[0:num_vertex, :]
    )
    assert np.allclose(
        gm.get_e2c_connectivity().table, grid_savepoint.e2c()[0:num_edges, :]
    )
    assert np.allclose(
        gm.get_c2e_connectivity().table, grid_savepoint.c2e()[0:num_cells, :]
    )
    assert np.allclose(
        gm.get_e2v_connectivity().table, grid_savepoint.v2e()[0:num_vertex, :]
    )


@pytest.mark.parametrize("dim, size", [(CellDim, 18), (EdgeDim, 27), (VertexDim, 9)])
def test_grid_manager_getsize(simple_mesh_data, simple_mesh_path, dim, size):
    gm = GridManager(simple_mesh_path)
    gm.init()
    assert size == gm.get_size(dim)


def test_gridmanager_given_file_not_found_then_abort():
    fname = "./unknown_grid.nc"
    with pytest.raises(SystemExit) as error:
        gm = GridManager(fname)
        gm.init()
        assert error.type == SystemExit
        assert error.value == 1
