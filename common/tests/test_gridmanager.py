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
    ToGt4PyTransformation, IconDomainZone,
)
from icon4py.testutils.data_handling import download_and_extract
from icon4py.testutils.simple_mesh import SimpleMesh


SIMPLE_MESH_NC = "./simple_mesh_grid.nc"

mch_ch_r04b09_dsl_grid_uri = "https://polybox.ethz.ch/index.php/s/hD232znfEPBh4Oh/download"
r02b04_global_grid_uri = "https://polybox.ethz.ch/index.php/s/0EM8O8U53GKGsst/download"
grids_path = Path(__file__).parent.joinpath("grids")
r04b09_dsl_grid_path = grids_path.joinpath("mch_ch_r04b09_dsl")
r04b09_dsl_data_file = r04b09_dsl_grid_path.joinpath("mch_ch_r04b09_dsl_grids_v1.tar.gz").name
r02b04_global_grid_path = grids_path.joinpath("icon_r02b04_global")
r02b04_global_data_file = r02b04_global_grid_path.joinpath("icon_grid_0013_R02B04_G.tar.gz").name

@pytest.fixture(scope="session")
def get_grid_files():
    """
    Get the grid files used for testing.

    Session scoped fixture which is a prerequisite of all the other fixtures in this file.
    """
    download_and_extract(mch_ch_r04b09_dsl_grid_uri, r04b09_dsl_grid_path, r04b09_dsl_data_file)
    download_and_extract(r02b04_global_grid_uri, r02b04_global_grid_path, r02b04_global_data_file)


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
    dataset.createDimension(GridFile.Dimension.MAX_CHILD_DOMAINS, size=1)
    # add dummy values for the grf dimensions
    # TODO @magdalena fix to something more useful?
    dataset.createDimension(GridFile.Dimension.CELL_GRF, size=14)
    dataset.createDimension(GridFile.Dimension.EDGE_GRF, size=24)
    dataset.createDimension(GridFile.Dimension.VERTEX_GRF, size=13)
    add_to_dataset(dataset, np.zeros(mesh.n_edges), GridFile.GridRefinement.CONTROL_EDGES, (GridFile.Dimension.EDGE_NAME,))
    add_to_dataset(dataset, np.zeros(mesh.n_cells), GridFile.GridRefinement.CONTROL_CELLS, (GridFile.Dimension.CELL_NAME,))
    add_to_dataset(dataset,  np.zeros(mesh.n_vertices), GridFile.GridRefinement.CONTROL_VERTICES,(GridFile.Dimension.VERTEX_NAME,))

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
    # TODO @magdalena: there is no v2c in the simple mesh
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

    add_to_dataset(dataset, np.ones((1, 24), dtype=np.int32), GridFile.GridRefinement.START_INDEX_EDGES, (GridFile.Dimension.MAX_CHILD_DOMAINS, GridFile.Dimension.EDGE_GRF))
    add_to_dataset(dataset, np.ones((1, 14), dtype=np.int32), GridFile.GridRefinement.START_INDEX_CELLS, (GridFile.Dimension.MAX_CHILD_DOMAINS, GridFile.Dimension.CELL_GRF))
    add_to_dataset(dataset, np.ones((1, 13), dtype=np.int32), GridFile.GridRefinement.START_INDEX_VERTICES, (GridFile.Dimension.MAX_CHILD_DOMAINS, GridFile.Dimension.VERTEX_GRF))
    add_to_dataset(dataset, np.ones((1, 24), dtype=np.int32), GridFile.GridRefinement.END_INDEX_EDGES, (GridFile.Dimension.MAX_CHILD_DOMAINS, GridFile.Dimension.EDGE_GRF))
    add_to_dataset(dataset, np.ones((1, 14), dtype=np.int32), GridFile.GridRefinement.END_INDEX_CELLS, (GridFile.Dimension.MAX_CHILD_DOMAINS, GridFile.Dimension.CELL_GRF))
    add_to_dataset(dataset, np.ones((1, 13), dtype=np.int32), GridFile.GridRefinement.END_INDEX_VERTICES,(GridFile.Dimension.MAX_CHILD_DOMAINS, GridFile.Dimension.VERTEX_GRF))
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

def test_gridfile_vertex_cell_edge_dimensions(grid_savepoint):
    fname = r04b09_dsl_grid_path.joinpath("grid.nc")
    data = Dataset(fname, "r")
    grid_file = GridFile(data)

    assert grid_file.dimension(GridFile.Dimension.CELL_NAME) == grid_savepoint.num(CellDim)
    assert grid_file.dimension(GridFile.Dimension.EDGE_NAME) == grid_savepoint.num(EdgeDim)
    # TODO: @magdalena fix in serialized data. it returns the num_cells
    assert grid_file.dimension(GridFile.Dimension.VERTEX_NAME) == grid_savepoint.num(VertexDim)




def test_grid_parser_index_fields(simple_mesh_data, caplog):
    caplog.set_level(logging.DEBUG)
    data = Dataset(SIMPLE_MESH_NC, "r")
    mesh = SimpleMesh()
    grid_parser = GridFile(data)

    assert np.allclose(grid_parser.int_field(GridFile.Offsets.C2E), mesh.c2e)
    assert np.allclose(grid_parser.int_field(GridFile.Offsets.E2C), mesh.e2c)
    assert np.allclose(grid_parser.int_field(GridFile.Offsets.V2E), mesh.v2e)
    assert np.allclose(grid_parser.int_field(GridFile.Offsets.V2C), mesh.v2c)



# TODO @magdalena add test cases for
#e2c2v - diamond: serial, simple
#c2v: grid, ???
#v2e2v: grid,???

# v2e: serial, simple, grid
@pytest.mark.datatest
def test_gridmanager_eval_v2e(caplog, grid_savepoint, get_grid_files):
    caplog.set_level(logging.DEBUG)
    fname = r04b09_dsl_grid_path.joinpath("grid.nc")
    gm = _init_grid_manager(fname)
    num_vertex = gm.get_size(VertexDim)
    seralized_v2e = grid_savepoint.v2e()[0:num_vertex, :]
    # there are vertices at the boundary of a local domain or at a pentagon point that have less than
    # 6 neighbors hence there are "Missing values" in the grid file
    # they get substituted by the "last valid index" in preprocessing step in icon.
    assert not has_invalid_index(seralized_v2e)
    assert has_invalid_index(gm.get_v2e_connectivity().table)
    reset_invalid_index(seralized_v2e)
    assert np.allclose(
        gm.get_v2e_connectivity().table, seralized_v2e
    )

#v2c: serial, simple, grid
@pytest.mark.datatest
def test_gridmanager_eval_v2c(caplog, grid_savepoint, get_grid_files):
    caplog.set_level(logging.DEBUG)
    fname = r04b09_dsl_grid_path.joinpath("grid.nc")
    gm = _init_grid_manager(fname)
    num_vertex = gm.get_size(VertexDim)
    serialized_v2c = grid_savepoint.v2c()[0:num_vertex, :]
    # there are vertices that have less than 6 neighboring cells: either pentagon points or
    # vertices at the boundary of the domain for a limited area mode
    # hence in the grid file there are "missing values"
    # they get substituted by the "last valid index" in preprocessing step in icon.
    assert not has_invalid_index(serialized_v2c)
    assert has_invalid_index(gm.get_v2c_connectivity().table)
    reset_invalid_index(serialized_v2c)

    assert np.allclose(
        gm.get_v2c_connectivity().table, serialized_v2c
    )


def reset_invalid_index(index_array: np.ndarray):
    """
    helper function to revert mo_model_domimp_patches.f90: move_dummies_to_end_idxblk.
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
        index_array[i, max(index) + 1:] = GridFile.INVALID_INDEX


#e2v: serial, simple, grid
@pytest.mark.datatest
def test_gridmanager_eval_e2v(caplog, grid_savepoint, get_grid_files):
    caplog.set_level(logging.DEBUG)
    fname = r04b09_dsl_grid_path.joinpath("grid.nc")
    gm = _init_grid_manager(fname)
    num_edges = gm.get_size(EdgeDim)

    serialized_e2v = grid_savepoint.e2v()[0:num_edges, :]
    # all vertices in the system have to neighboring edges, there no edges that point nowhere
    # hence this connectivity has no "missing values" in the grid file
    assert not has_invalid_index(serialized_e2v)
    assert not has_invalid_index(gm.get_e2v_connectivity().table)
    assert np.allclose(
        gm.get_e2v_connectivity().table, serialized_e2v
    )


def has_invalid_index(ar :np.ndarray):
    return np.any(np.where(ar == GridFile.INVALID_INDEX))


# e2c :serial, simple, grid
@pytest.mark.datatest
def test_gridmanager_eval_e2c(caplog, grid_savepoint, get_grid_files):
    caplog.set_level(logging.DEBUG)
    fname = r04b09_dsl_grid_path.joinpath("grid.nc")
    gm = _init_grid_manager(fname)
    num_edges = gm.get_size(EdgeDim)
    serialized_c2e = grid_savepoint.e2c()[0:num_edges, :]
    # there are edges at the boundary that have only one
    # neighboring cell, there are "missing values" in the grid file
    # and here they do not get substituted in the ICON preprocessing
    assert has_invalid_index(serialized_c2e)
    assert has_invalid_index(gm.get_e2c_connectivity().table)
    assert np.allclose(
        gm.get_e2c_connectivity().table, serialized_c2e
    )


#c2e: serial, simple, grid
@pytest.mark.datatest
def test_gridmanager_eval_c2e(caplog, grid_savepoint, get_grid_files):
    caplog.set_level(logging.DEBUG)
    fname = r04b09_dsl_grid_path.joinpath("grid.nc")
    gm = _init_grid_manager(fname)
    num_cells = gm.get_size(CellDim)

    serialized_c2e = grid_savepoint.c2e()[0:num_cells, :]
    # no cells with less than 3 neighboring edges exist, otherwise the cell is not there in the
    # first place
    # hence there are no "missing values" in the grid file
    assert not has_invalid_index(serialized_c2e)
    assert not has_invalid_index(gm.get_c2e_connectivity().table)
    assert np.allclose(
        gm.get_c2e_connectivity().table, serialized_c2e
    )

#e2c2e (e2c2eo) - diamond: serial, simple_mesh
@pytest.mark.skip("does this array exist in grid file?")
@pytest.mark.datatest
def test_gridmanager_eval_e2c2e(caplog, grid_savepoint, get_grid_files):
    caplog.set_level(logging.DEBUG)
    fname = r04b09_dsl_grid_path.joinpath("grid.nc")
    gm, num_cells, num_edges, num_vertex = _init_grid_manager(fname)
    serialized_e2c2e = grid_savepoint.e2c2e()[0:num_cells, :]
    assert has_invalid_index(serialized_e2c2e)
    assert has_invalid_index(gm.get_e2c2e_connectivity().table)
    assert np.allclose(
        gm.get_e2c2e_connectivity().table, serialized_e2c2e
    )

#c2e2c: serial, simple_mesh, grid
@pytest.mark.datatest
def test_gridmanager_eval_c2e2c(caplog, grid_savepoint, get_grid_files):
    caplog.set_level(logging.DEBUG)
    fname = r04b09_dsl_grid_path.joinpath("grid.nc")
    gm = _init_grid_manager(fname)
    num_cells = gm.get_size(CellDim)
    assert np.allclose(
        gm.get_c2e2c_connectivity().table, grid_savepoint.c2e2c()[0:num_cells, :]
    )



def _init_grid_manager(fname):
    gm = GridManager(ToGt4PyTransformation(), fname)
    gm.init()
    return gm


@pytest.mark.parametrize("dim, size", [(CellDim, 18), (EdgeDim, 27), (VertexDim, 9)])
def test_grid_manager_getsize(simple_mesh_data, simple_mesh_path, dim, size, caplog):
    caplog.set_level(logging.DEBUG)
    gm = GridManager(IndexTransformation(), simple_mesh_path)
    gm.init()
    assert size == gm.get_size(dim)


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

@pytest.mark.parametrize("zone, bounds", [
    (IconDomainZone.NUDGING,(3316, 4104)),
    (IconDomainZone.HALO, (20896, 20896)),
    (IconDomainZone.LATERAL_BOUNDARY, (0, 850)),
    (IconDomainZone.INTERIOR, (4104, 20896))
] )
def test_start_end_indices_cells(get_grid_files, zone, bounds):
    fname = r04b09_dsl_grid_path.joinpath("grid.nc")
    gm = _init_grid_manager(fname)
    assert gm.get_domain_boundaries(CellDim, zone) == bounds


@pytest.mark.parametrize("zone, bounds", [
    (IconDomainZone.NUDGING,(4989, 5387)),
    (IconDomainZone.HALO, (31558, 31558)),
    (IconDomainZone.LATERAL_BOUNDARY, (0, 428)),
    (IconDomainZone.INTERIOR, (6176, 31558))])
def test_start_end_indices_edges(get_grid_files, zone, bounds):
    fname = r04b09_dsl_grid_path.joinpath("grid.nc")
    gm = _init_grid_manager(fname)
    assert gm.get_domain_boundaries(EdgeDim, zone) == bounds


@pytest.mark.parametrize("zone, bounds", [
    (IconDomainZone.NUDGING,(10663, 10663)),
    (IconDomainZone.HALO, (10663, 10663)),
    (IconDomainZone.LATERAL_BOUNDARY, (0, 428)),
    (IconDomainZone.INTERIOR, (2071, 10663)),
    (IconDomainZone.END, (10663, 10663))
])
def test_start_end_indices_vertices(get_grid_files, zone, bounds):
    fname = r04b09_dsl_grid_path.joinpath("grid.nc")
    gm = _init_grid_manager(fname)
    assert gm.get_domain_boundaries(VertexDim, zone) == bounds

