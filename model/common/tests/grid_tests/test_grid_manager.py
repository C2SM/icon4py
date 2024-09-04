# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
import logging
import typing
import uuid

import numpy as np
import pytest

import icon4py.model.common.test_utils.datatest_utils as dt_utils
from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions as defs, halo
from icon4py.model.common.grid import (
    grid_manager as gm,
    horizontal as h_grid,
    refinement as refin,
    simple,
    vertical as v_grid,
)
from icon4py.model.common.settings import xp


if typing.TYPE_CHECKING:
    import netCDF4

try:
    import netCDF4
except ImportError:
    pytest.skip("optional netcdf dependency not installed", allow_module_level=True)


from . import utils


SIMPLE_GRID_NC = "simple_grid.nc"

R02B04_GLOBAL_NUM_VERTICES = 10242
R02B04_GLOBAL_NUM_EDGES = 30720
R02B04_GLOBAL_NUM_CELLS = 20480

MCH_CH_04B09_NUM_VERTICES = 10663
MCH_CH_R04B09_LOCAL_NUM_EDGES = 31558
MCH_CH_RO4B09_LOCAL_NUM_CELLS = 20896
MCH_CH_RO4B09_GLOBAL_NUM_CELLS = 83886080


MCH_CH_R04B09_CELL_DOMAINS = {
    "2ND_BOUNDARY_LINE": 850,
    "3D_BOUNDARY_LINE": 1688,
    "4TH_BOUNDARY_LINE": 2511,
    "NUDGING": 3316,
    "INTERIOR": 4104,
    "HALO": 20896,
    "LOCAL": 0,
}

MCH_CH_R04B09_VERTEX_DOMAINS = {
    "2ND_BOUNDARY_LINE": 428,
    "3D_BOUNDARY_LINE": 850,
    "4TH_BOUNDARY_LINE": 1266,
    "5TH_BOUNDARY_LINE": 1673,
    "INTERIOR": 2071,
    "HALO": 10663,
    "LOCAL": 0,
}

MCH_CH_R04B09_EDGE_DOMAINS = {
    "2ND_BOUNDARY_LINE": 428,
    "3D_BOUNDARY_LINE": 1278,
    "4TH_BOUNDARY_LINE": 1700,
    "5TH_BOUNDARY_LINE": 2538,
    "6TH_BOUNDARY_LINE": 2954,
    "7TH_BOUNDARY_LINE": 3777,
    "8TH_BOUNDARY_LINE": 4184,
    "NUDGING": 4989,
    "2ND_NUDGING": 5387,
    "INTERIOR": 6176,
    "HALO": 31558,
    "LOCAL": 0,
    "END": 31558,
}


@pytest.fixture
def simple_grid_gridfile(tmp_path):
    path = tmp_path.joinpath(SIMPLE_GRID_NC).absolute()
    grid = simple.SimpleGrid()

    dataset = netCDF4.Dataset(path, "w", format="NETCDF4")
    dataset.setncattr(gm.PropertyName.GRID_ID, str(uuid.uuid4()))
    dataset.setncattr(gm.PropertyName.LEVEL, 0)
    dataset.setncattr(gm.PropertyName.ROOT, 0)
    dataset.createDimension(gm.DimensionName.VERTEX_NAME, size=grid.num_vertices)

    dataset.createDimension(gm.DimensionName.EDGE_NAME, size=grid.num_edges)
    dataset.createDimension(gm.DimensionName.CELL_NAME, size=grid.num_cells)
    dataset.createDimension(gm.DimensionName.NEIGHBORS_TO_EDGE_SIZE, size=grid.size[dims.E2VDim])
    dataset.createDimension(gm.DimensionName.DIAMOND_EDGE_SIZE, size=grid.size[dims.E2C2EDim])
    dataset.createDimension(gm.DimensionName.MAX_CHILD_DOMAINS, size=1)
    # add dummy values for the grf dimensions
    dataset.createDimension(gm.DimensionName.CELL_GRF, size=14)
    dataset.createDimension(gm.DimensionName.EDGE_GRF, size=24)
    dataset.createDimension(gm.DimensionName.VERTEX_GRF, size=13)
    _add_to_dataset(
        dataset,
        np.zeros(grid.num_edges),
        gm.GridRefinementName.CONTROL_EDGES,
        (gm.DimensionName.EDGE_NAME,),
    )

    _add_to_dataset(
        dataset,
        np.zeros(grid.num_cells),
        gm.GridRefinementName.CONTROL_CELLS,
        (gm.DimensionName.CELL_NAME,),
    )
    _add_to_dataset(
        dataset,
        np.zeros(grid.num_vertices),
        gm.GridRefinementName.CONTROL_VERTICES,
        (gm.DimensionName.VERTEX_NAME,),
    )

    dataset.createDimension(gm.DimensionName.NEIGHBORS_TO_CELL_SIZE, size=grid.size[dims.C2EDim])
    dataset.createDimension(gm.DimensionName.NEIGHBORS_TO_VERTEX_SIZE, size=grid.size[dims.V2CDim])

    _add_to_dataset(
        dataset,
        grid.connectivities[dims.C2EDim],
        gm.ConnectivityName.C2E,
        (
            gm.DimensionName.NEIGHBORS_TO_CELL_SIZE,
            gm.DimensionName.CELL_NAME,
        ),
    )

    _add_to_dataset(
        dataset,
        grid.connectivities[dims.E2CDim],
        gm.ConnectivityName.E2C,
        (
            gm.DimensionName.NEIGHBORS_TO_EDGE_SIZE,
            gm.DimensionName.EDGE_NAME,
        ),
    )
    _add_to_dataset(
        dataset,
        grid.connectivities[dims.E2VDim],
        gm.ConnectivityName.E2V,
        (
            gm.DimensionName.NEIGHBORS_TO_EDGE_SIZE,
            gm.DimensionName.EDGE_NAME,
        ),
    )

    _add_to_dataset(
        dataset,
        grid.connectivities[dims.V2CDim],
        gm.ConnectivityName.V2C,
        (
            gm.DimensionName.NEIGHBORS_TO_VERTEX_SIZE,
            gm.DimensionName.VERTEX_NAME,
        ),
    )

    _add_to_dataset(
        dataset,
        grid.connectivities[dims.C2VDim],
        gm.ConnectivityName.C2V,
        (
            gm.DimensionName.NEIGHBORS_TO_CELL_SIZE,
            gm.DimensionName.CELL_NAME,
        ),
    )
    _add_to_dataset(
        dataset,
        np.zeros((grid.num_vertices, 4), dtype=np.int32),
        gm.ConnectivityName.V2E2V,
        (gm.DimensionName.DIAMOND_EDGE_SIZE, gm.DimensionName.VERTEX_NAME),
    )
    _add_to_dataset(
        dataset,
        grid.connectivities[dims.V2EDim],
        gm.ConnectivityName.V2E,
        (
            gm.DimensionName.NEIGHBORS_TO_VERTEX_SIZE,
            gm.DimensionName.VERTEX_NAME,
        ),
    )
    _add_to_dataset(
        dataset,
        grid.connectivities[dims.C2E2CDim],
        gm.ConnectivityName.C2E2C,
        (
            gm.DimensionName.NEIGHBORS_TO_CELL_SIZE,
            gm.DimensionName.CELL_NAME,
        ),
    )

    _add_to_dataset(
        dataset,
        np.ones((1, 24), dtype=np.int32),
        gm.GridRefinementName.START_INDEX_EDGES,
        (gm.DimensionName.MAX_CHILD_DOMAINS, gm.DimensionName.EDGE_GRF),
    )
    _add_to_dataset(
        dataset,
        np.ones((1, 14), dtype=np.int32),
        gm.GridRefinementName.START_INDEX_CELLS,
        (gm.DimensionName.MAX_CHILD_DOMAINS, gm.DimensionName.CELL_GRF),
    )
    _add_to_dataset(
        dataset,
        np.ones((1, 13), dtype=np.int32),
        gm.GridRefinementName.START_INDEX_VERTICES,
        (gm.DimensionName.MAX_CHILD_DOMAINS, gm.DimensionName.VERTEX_GRF),
    )
    _add_to_dataset(
        dataset,
        np.ones((1, 24), dtype=np.int32),
        gm.GridRefinementName.END_INDEX_EDGES,
        (gm.DimensionName.MAX_CHILD_DOMAINS, gm.DimensionName.EDGE_GRF),
    )
    _add_to_dataset(
        dataset,
        np.ones((1, 14), dtype=np.int32),
        gm.GridRefinementName.END_INDEX_CELLS,
        (gm.DimensionName.MAX_CHILD_DOMAINS, gm.DimensionName.CELL_GRF),
    )
    _add_to_dataset(
        dataset,
        np.ones((1, 13), dtype=np.int32),
        gm.GridRefinementName.END_INDEX_VERTICES,
        (gm.DimensionName.MAX_CHILD_DOMAINS, gm.DimensionName.VERTEX_GRF),
    )
    dataset.close()
    yield path
    path.unlink()


def _add_to_dataset(
    dataset: netCDF4.Dataset,
    data: np.ndarray,
    var_name: str,
    dims: tuple[gm.GridFileName, gm.GridFileName],
):
    var = dataset.createVariable(var_name, np.int32, dims)
    var[:] = np.transpose(data)[:]


@functools.cache
def grid_manager(fname, num_levels=65, transformation=None) -> gm.GridManager:
    if transformation is None:
        transformation = gm.ToZeroBasedIndexTransformation()
    grid_manager = gm.GridManager(transformation, fname, v_grid.VerticalGridConfig(num_levels))
    grid_manager()
    return grid_manager


@pytest.mark.with_netcdf
def test_gridparser_dimension(simple_grid_gridfile):
    data = netCDF4.Dataset(simple_grid_gridfile, "r")
    parser = gm.GridFile(data)
    grid = simple.SimpleGrid()
    assert parser.dimension(gm.DimensionName.CELL_NAME) == grid.num_cells
    assert parser.dimension(gm.DimensionName.VERTEX_NAME) == grid.num_vertices
    assert parser.dimension(gm.DimensionName.EDGE_NAME) == grid.num_edges


@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "parser, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_gridfile_vertex_cell_edge_dimensions(grid_savepoint, parser):
    file = utils.resolve_file_from_gridfile_name(parser)
    parser = gm.GridFile(netCDF4.Dataset(file, "r"))

    assert parser.dimension(gm.DimensionName.CELL_NAME) == grid_savepoint.num(dims.CellDim)
    assert parser.dimension(gm.DimensionName.EDGE_NAME) == grid_savepoint.num(dims.EdgeDim)
    assert parser.dimension(gm.DimensionName.VERTEX_NAME) == grid_savepoint.num(dims.VertexDim)


@pytest.mark.with_netcdf
def test_grid_parser_index_fields(simple_grid_gridfile, caplog):
    caplog.set_level(logging.DEBUG)
    data = netCDF4.Dataset(simple_grid_gridfile, "r")
    simple_grid = simple.SimpleGrid()
    parser = gm.GridFile(data)

    assert np.allclose(
        parser.int_field(gm.ConnectivityName.C2E), simple_grid.connectivities[dims.C2EDim]
    )
    assert np.allclose(
        parser.int_field(gm.ConnectivityName.E2C), simple_grid.connectivities[dims.E2CDim]
    )
    assert np.allclose(
        parser.int_field(gm.ConnectivityName.V2E), simple_grid.connectivities[dims.V2EDim]
    )
    assert np.allclose(
        parser.int_field(gm.ConnectivityName.V2C), simple_grid.connectivities[dims.V2CDim]
    )


# TODO @magdalena add test cases for hexagon vertices v2e2v
# v2e2v: grid,???


# v2e: exists in serial, simple, grid
@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_gridmanager_eval_v2e(caplog, grid_savepoint, grid_file):
    caplog.set_level(logging.DEBUG)
    file = utils.resolve_file_from_gridfile_name(grid_file)
    grid = grid_manager(file).grid
    seralized_v2e = grid_savepoint.v2e()
    # there are vertices at the boundary of a local domain or at a pentagon point that have less than
    # 6 neighbors hence there are "Missing values" in the grid file
    # they get substituted by the "last valid index" in preprocessing step in icon.
    assert not has_invalid_index(seralized_v2e)
    v2e_table = grid.get_offset_provider("V2E").table
    assert has_invalid_index(v2e_table)
    reset_invalid_index(seralized_v2e)
    assert np.allclose(v2e_table, seralized_v2e)


@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
@pytest.mark.parametrize("dim", [dims.CellDim, dims.EdgeDim, dims.VertexDim])
def test_refin_ctrl(grid_savepoint, grid_file, experiment, dim):
    file = utils.resolve_file_from_gridfile_name(grid_file)
    gm = grid_manager(file)
    start_index, end_index, refin_ctrl, refin_ctrl_max = gm._read_grid_refinement_information(
        gm._dataset
    )
    refin_ctrl_serialized = grid_savepoint.refin_ctrl(dim)
    assert np.all(refin_ctrl_serialized.ndarray == refin.to_unnested(refin_ctrl[dim], dim))


# v2c: exists in serial, simple, grid
@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_gridmanager_eval_v2c(caplog, grid_savepoint, grid_file):
    caplog.set_level(logging.DEBUG)
    file = utils.resolve_file_from_gridfile_name(grid_file)
    grid = grid_manager(file).grid
    serialized_v2c = grid_savepoint.v2c()
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
        index_array[i, max(index) + 1 :] = gm.GridFile.INVALID_INDEX


# e2v: exists in serial, simple, grid
@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_gridmanager_eval_e2v(caplog, grid_savepoint, grid_file):
    caplog.set_level(logging.DEBUG)
    file = utils.resolve_file_from_gridfile_name(grid_file)
    grid = grid_manager(file).grid

    serialized_e2v = grid_savepoint.e2v()[0 : grid.num_edges, :]
    # all vertices in the system have to neighboring edges, there no edges that point nowhere
    # hence this connectivity has no "missing values" in the grid file
    assert not has_invalid_index(serialized_e2v)
    assert not has_invalid_index(grid.get_offset_provider("E2V").table)
    assert np.allclose(grid.get_offset_provider("E2V").table, serialized_e2v)


def has_invalid_index(ar: np.ndarray):
    return np.any(invalid_index(ar))


def invalid_index(ar):
    return np.where(ar == gm.GridFile.INVALID_INDEX)


def _is_local(grid_file: str):
    return grid_file == dt_utils.REGIONAL_EXPERIMENT


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
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_gridmanager_eval_e2c(caplog, grid_savepoint, grid_file):
    caplog.set_level(logging.DEBUG)
    file = utils.resolve_file_from_gridfile_name(grid_file)
    grid = grid_manager(file).grid
    serialized_e2c = grid_savepoint.e2c()
    e2c_table = grid.get_offset_provider("E2C").table
    assert_invalid_indices(serialized_e2c, grid_file)
    assert_invalid_indices(e2c_table, grid_file)
    assert np.allclose(e2c_table, serialized_e2c)


# c2e: serial, simple, grid
@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_gridmanager_eval_c2e(caplog, grid_savepoint, grid_file):
    caplog.set_level(logging.DEBUG)
    file = utils.resolve_file_from_gridfile_name(grid_file)
    grid = grid_manager(file).grid

    serialized_c2e = grid_savepoint.c2e()
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
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_gridmanager_eval_c2e2c(caplog, grid_savepoint, grid_file):
    caplog.set_level(logging.DEBUG)
    file = utils.resolve_file_from_gridfile_name(grid_file)
    grid = grid_manager(file).grid
    assert np.allclose(
        grid.get_offset_provider("C2E2C").table,
        grid_savepoint.c2e2c(),
    )


@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_gridmanager_eval_c2e2cO(caplog, grid_savepoint, grid_file):
    caplog.set_level(logging.DEBUG)
    file = utils.resolve_file_from_gridfile_name(grid_file)
    grid = grid_manager(file).grid
    serialized_grid = grid_savepoint.construct_icon_grid(on_gpu=False)
    assert np.allclose(
        grid.get_offset_provider("C2E2CO").table,
        serialized_grid.get_offset_provider("C2E2CO").table,
    )


# e2c2e (e2c2eo) - diamond: exists in serial, simple_mesh
@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_gridmanager_eval_e2c2e(caplog, grid_savepoint, grid_file):
    caplog.set_level(logging.DEBUG)
    file = utils.resolve_file_from_gridfile_name(grid_file)
    grid = grid_manager(file).grid
    serialized_grid = grid_savepoint.construct_icon_grid(on_gpu=False)
    serialized_e2c2e = serialized_grid.get_offset_provider("E2C2E").table
    serialized_e2c2eO = serialized_grid.get_offset_provider("E2C2EO").table
    assert_invalid_indices(serialized_e2c2e, grid_file)

    e2c2e_table = grid.get_offset_provider("E2C2E").table
    e2c2e0_table = grid.get_offset_provider("E2C2EO").table

    assert_invalid_indices(e2c2e_table, grid_file)
    # ICON calculates diamond edges only from rl_start = 2 (lateral_boundary(dims.EdgeDim) + 1 for
    # boundaries all values are INVALID even though the half diamond exists (see mo_model_domimp_setup.f90 ll 163ff.)
    assert_unless_invalid(e2c2e_table, serialized_e2c2e)
    assert_unless_invalid(e2c2e0_table, serialized_e2c2eO)


def assert_unless_invalid(table, serialized_ref):
    invalid_positions = invalid_index(table)
    np.allclose(table[invalid_positions], serialized_ref[invalid_positions])


@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_gridmanager_eval_e2c2v(caplog, grid_savepoint, grid_file):
    caplog.set_level(logging.DEBUG)
    file = utils.resolve_file_from_gridfile_name(grid_file)
    gm = grid_manager(file)
    grid = gm.grid
    # the "far" (adjacent to edge normal ) is not always there, because ICON only calculates those starting from
    #   (lateral_boundary(dims.EdgeDim) + 1) to end(dims.EdgeDim)  (see mo_intp_coeffs.f90) and only for owned cells
    serialized_ref = grid_savepoint.e2c2v()
    table = grid.get_offset_provider("E2C2V").table
    assert_unless_invalid(table, serialized_ref)


@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_gridmanager_eval_c2v(caplog, grid_savepoint, grid_file):
    caplog.set_level(logging.DEBUG)
    file = utils.resolve_file_from_gridfile_name(grid_file)
    grid = grid_manager(file).grid
    c2v = grid.get_offset_provider("C2V").table
    assert np.allclose(c2v, grid_savepoint.c2v())


@pytest.mark.parametrize("dim, size", [(dims.CellDim, 18), (dims.EdgeDim, 27), (dims.VertexDim, 9)])
@pytest.mark.with_netcdf
def test_grid_manager_getsize(simple_grid_gridfile, dim, size, caplog):
    caplog.set_level(logging.DEBUG)
    manager = grid_manager(
        simple_grid_gridfile, num_levels=10, transformation=gm.IndexTransformation()
    )

    assert size == manager.get_size(dim)


def assert_up_to_order(table, diamond_table):
    assert table.shape == diamond_table.shape
    for n in range(table.shape[0]):
        assert np.all(np.in1d(table[n, :], diamond_table[n, :]))


@pytest.mark.with_netcdf
def test_grid_manager_diamond_offset(simple_grid_gridfile):
    simple_grid = simple.SimpleGrid()
    manager = grid_manager(
        simple_grid_gridfile,
        num_levels=simple_grid.num_levels,
        transformation=gm.IndexTransformation(),
    )

    table = manager.grid.get_offset_provider("E2C2V").table
    assert_up_to_order(table, simple_grid.diamond_table)


@pytest.mark.with_netcdf
def test_gridmanager_given_file_not_found_then_abort():
    fname = "./unknown_grid.nc"
    with pytest.raises(SystemExit) as error:
        manager = gm.GridManager(
            gm.IndexTransformation(), fname, v_grid.VerticalGridConfig(num_levels=80)
        )
        manager()
        assert error.type == SystemExit
        assert error.value == 1


@pytest.mark.parametrize("size", [100, 1500, 20000])
@pytest.mark.with_netcdf
def test_gt4py_transform_offset_by_1_where_valid(size):
    trafo = gm.ToZeroBasedIndexTransformation()
    rng = np.random.default_rng()
    input_field = rng.integers(-1, size, size)
    offset = trafo.get_offset_for_index_field(input_field)
    expected = np.where(input_field >= 0, -1, 0)
    assert np.allclose(expected, offset)


@pytest.mark.parametrize(
    "grid_file, global_num_cells",
    [
        (utils.R02B04_GLOBAL, R02B04_GLOBAL_NUM_CELLS),
        (dt_utils.REGIONAL_EXPERIMENT, MCH_CH_RO4B09_GLOBAL_NUM_CELLS),
    ],
)
def test_grid_level_and_root(grid_file, global_num_cells):
    file = utils.resolve_file_from_gridfile_name(grid_file)
    assert global_num_cells == grid_manager(file, num_levels=10).grid.global_num_cells


def test_c2e2c2e(simple_grid_gridfile):
    simple_grid = simple.SimpleGrid()
    manager = grid_manager(
        simple_grid_gridfile,
        num_levels=simple_grid.num_levels,
        transformation=gm.IndexTransformation(),
    )

    table = manager.grid.get_offset_provider("C2E2C2E").table
    assert_up_to_order(table, simple.SimpleGridData.c2e2c2e_table)


@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [(utils.R02B04_GLOBAL, dt_utils.JABW_EXPERIMENT)],
)
def test_gridmanager_eval_c2e2c2e(caplog, grid_savepoint, grid_file):
    caplog.set_level(logging.DEBUG)
    file = utils.resolve_file_from_gridfile_name(grid_file)
    grid = grid_manager(file).grid
    serialized_grid = grid_savepoint.construct_icon_grid(on_gpu=False)
    assert np.allclose(
        grid.get_offset_provider("C2E2C2E").table,
        serialized_grid.get_offset_provider("C2E2C2E").table,
    )
    assert grid.get_offset_provider("C2E2C2E").table.shape == (grid.num_cells, 9)


@pytest.mark.mpi
@pytest.mark.parametrize(
    "field_offset",
    [dims.C2V, dims.E2V, dims.V2C, dims.E2C, dims.C2E, dims.V2E, dims.C2E2C, dims.V2E2V],
)
def test_local_connectivities(processor_props, caplog, field_offset):  # fixture
    caplog.set_level(logging.INFO)
    file = utils.resolve_file_from_gridfile_name(utils.R02B04_GLOBAL)
    grid = grid_manager(file).grid
    partitioner = halo.SimpleMetisDecomposer()
    face_face_connectivity = grid.connectivities[dims.C2E2CDim]
    labels = partitioner(face_face_connectivity, n_part=processor_props.comm_size)
    halo_generator = halo.HaloGenerator(
        connectivities=grid.connectivities,
        run_properties=processor_props,
        rank_mapping=labels,
        num_levels=1,
    )

    decomposition_info = halo_generator()

    connectivity = gm.construct_local_connectivity(
        field_offset, decomposition_info, connectivity=grid.connectivities[field_offset.target[1]]
    )
    # there is an neighbor list for each index of the target dimension on the node
    assert (
        connectivity.shape[0]
        == decomposition_info.global_index(
            field_offset.target[0], defs.DecompositionInfo.EntryType.ALL
        ).size
    )
    # all neighbor indices are valid local indices
    assert xp.max(connectivity) == xp.max(
        decomposition_info.local_index(field_offset.source, defs.DecompositionInfo.EntryType.ALL)
    )
    # TODO what else to assert?
    # - outer halo entries have SKIP_VALUE neighbors (depends on offsets)


@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
        # (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT)
    ],
)
@pytest.mark.parametrize("dim", utils.horizontal_dim())
def test_start_end_index(processor_props, caplog, dim, experiment, grid_file):
    caplog.set_level(logging.INFO)
    file = utils.resolve_file_from_gridfile_name(grid_file)
    limited_area = experiment == dt_utils.REGIONAL_EXPERIMENT
    manager = gm.GridManager(
        gm.ToZeroBasedIndexTransformation(), file, v_grid.VerticalGridConfig(1)
    )
    manager(limited_area=limited_area)
    single_node_grid = manager.grid

    partitioner = halo.SimpleMetisDecomposer()

    manager.with_decomposer(partitioner, processor_props)  # add these args to __call__?
    manager(limited_area=limited_area)
    grid = manager.grid

    for domain in global_grid_domains(dim):
        assert grid.start_index(domain) == single_node_grid.start_index(
            domain
        ), f"start index wrong for domain {domain}"
        assert grid.end_index(domain) == single_node_grid.end_index(
            domain
        ), f"end index wrong for domain {domain}"


def global_grid_domains(dim: dims.Dimension):
    zones = [
        h_grid.Zone.END,
        h_grid.Zone.LOCAL,
        h_grid.Zone.INTERIOR,
        h_grid.Zone.HALO,
        h_grid.Zone.HALO_LEVEL_2,
    ]

    yield from _domain(dim, zones)


def _domain(dim, zones):
    for z in zones:
        yield h_grid.domain(dim)(z)


def boundary_domains(dim: dims.Dimension):
    zones = [
        h_grid.Zone.LATERAL_BOUNDARY,
        h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2,
        h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3,
        h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4,
        h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5,
        h_grid.Zone.LATERAL_BOUNDARY_LEVEL_6,
        h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7,
        h_grid.Zone.NUDGING,
        h_grid.Zone.NUDGING_LEVEL_2,
    ]
    yield from _domain(dim, zones)
