# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import typing

import numpy as np
import pytest
from gt4py.next import backend as gtx_backend

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import (
    grid_manager as gm,
    refinement as refin,
    vertical as v_grid,
)
from icon4py.model.common.grid.grid_manager import GeometryName
from icon4py.model.common.test_utils import (
    datatest_utils as dt_utils,
    grid_utils as gridtest_utils,
    helpers,
)


if typing.TYPE_CHECKING:
    import netCDF4

try:
    import netCDF4  # noqa # F401
except ImportError:
    pytest.skip("optional netcdf dependency not installed", allow_module_level=True)


from . import utils


R02B04_GLOBAL_NUM_CELLS = 20480
R02B04_GLOBAL_NUM_EDGES = 30720
R02B04_GLOBAL_NUM_VERTEX = 10242


MCH_CH_RO4B09_GLOBAL_NUM_CELLS = 83886080


ZERO_BASE = gm.ToZeroBasedIndexTransformation()

managers = {}


def _run_grid_manager(file: str, backend: gtx_backend.Backend) -> gm.GridManager:
    if not managers.get(file):
        manager = gridtest_utils.get_grid_manager(file, num_levels=1, backend=backend)
        managers[file] = manager
    return managers.get(file)


@pytest.fixture
def global_grid_file():
    return gridtest_utils.resolve_full_grid_file_name(dt_utils.R02B04_GLOBAL)


@pytest.mark.with_netcdf
def test_grid_file_dimension(global_grid_file):
    parser = gm.GridFile(str(global_grid_file))
    try:
        parser.open()
        assert parser.dimension(gm.DimensionName.CELL_NAME) == R02B04_GLOBAL_NUM_CELLS
        assert parser.dimension(gm.DimensionName.VERTEX_NAME) == R02B04_GLOBAL_NUM_VERTEX
        assert parser.dimension(gm.DimensionName.EDGE_NAME) == R02B04_GLOBAL_NUM_EDGES
    except Exception:
        pytest.fail()
    finally:
        parser.close()


@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_grid_file_vertex_cell_edge_dimensions(grid_savepoint, grid_file):
    file = gridtest_utils.resolve_full_grid_file_name(grid_file)
    parser = gm.GridFile(str(file))
    try:
        parser.open()
        assert parser.dimension(gm.DimensionName.CELL_NAME) == grid_savepoint.num(dims.CellDim)
        assert parser.dimension(gm.DimensionName.VERTEX_NAME) == grid_savepoint.num(dims.VertexDim)
        assert parser.dimension(gm.DimensionName.EDGE_NAME) == grid_savepoint.num(dims.EdgeDim)
    except Exception as error:
        pytest.fail(f"reading of dimension from netcdf failed: {error}")
    finally:
        parser.close()


# TODO is this useful?
@pytest.mark.skip
@pytest.mark.with_netcdf
@pytest.mark.parametrize("experiment", (dt_utils.GLOBAL_EXPERIMENT,))
def test_grid_file_index_fields(global_grid_file, caplog, icon_grid):
    caplog.set_level(logging.DEBUG)
    parser = gm.GridFile(str(global_grid_file))
    try:
        parser.open()
        assert np.allclose(
            parser.int_variable(gm.ConnectivityName.C2E), icon_grid.connectivities[dims.C2EDim] + 1
        )
        assert np.allclose(
            parser.int_variable(gm.ConnectivityName.E2C), icon_grid.connectivities[dims.E2CDim] + 1
        )
        assert np.allclose(
            parser.int_variable(gm.ConnectivityName.V2E), icon_grid.connectivities[dims.V2EDim] + 1
        )
        assert np.allclose(
            parser.int_variable(gm.ConnectivityName.V2C), icon_grid.connectivities[dims.V2CDim] + 1
        )
    except Exception as e:
        logging.error(e)
        pytest.fail()
    finally:
        parser.close()


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
def test_grid_manager_eval_v2e(caplog, grid_savepoint, experiment, grid_file, backend):
    caplog.set_level(logging.DEBUG)
    grid = _run_grid_manager(grid_file, backend).grid
    seralized_v2e = grid_savepoint.v2e()
    # there are vertices at the boundary of a local domain or at a pentagon point that have less than
    # 6 neighbors hence there are "Missing values" in the grid file
    # they get substituted by the "last valid index" in preprocessing step in icon.
    assert not has_invalid_index(seralized_v2e)
    v2e_table = grid.get_offset_provider("V2E").table
    assert has_invalid_index(v2e_table)
    reset_invalid_index(seralized_v2e)
    assert np.allclose(v2e_table, seralized_v2e)


@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
@pytest.mark.parametrize("dim", [dims.CellDim, dims.EdgeDim, dims.VertexDim])
def test_grid_manager_refin_ctrl(grid_savepoint, grid_file, experiment, dim, backend):
    refin_ctrl = _run_grid_manager(grid_file, backend).refinement
    refin_ctrl_serialized = grid_savepoint.refin_ctrl(dim)
    assert np.all(
        refin_ctrl_serialized.ndarray
        == refin.convert_to_unnested_refinement_values(refin_ctrl[dim], dim)
    )


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
def test_grid_manager_eval_v2c(caplog, grid_savepoint, experiment, grid_file, backend):
    caplog.set_level(logging.DEBUG)
    grid = _run_grid_manager(grid_file, backend).grid
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
def test_grid_manager_eval_e2v(caplog, grid_savepoint, grid_file, experiment, backend):
    caplog.set_level(logging.DEBUG)
    grid = _run_grid_manager(grid_file, backend).grid

    serialized_e2v = grid_savepoint.e2v()
    # all vertices in the system have to neighboring edges, there no edges that point nowhere
    # hence this connectivity has no "missing values" in the grid file
    assert not has_invalid_index(serialized_e2v)
    assert not has_invalid_index(grid.get_offset_provider("E2V").table)
    assert np.allclose(grid.get_offset_provider("E2V").table, serialized_e2v)


def has_invalid_index(ar: np.ndarray):
    return np.any(invalid_index(ar))


def invalid_index(ar):
    return np.where(ar == gm.GridFile.INVALID_INDEX)


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
    if gridtest_utils.is_regional(grid_file):
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
def test_grid_manager_eval_e2c(caplog, grid_savepoint, grid_file, experiment, backend):
    caplog.set_level(logging.DEBUG)

    grid = _run_grid_manager(grid_file, backend).grid
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
def test_grid_manager_eval_c2e(caplog, grid_savepoint, grid_file, experiment, backend):
    caplog.set_level(logging.DEBUG)
    grid = _run_grid_manager(grid_file, backend).grid

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
def test_grid_manager_eval_c2e2c(caplog, grid_savepoint, grid_file, experiment, backend):
    caplog.set_level(logging.DEBUG)
    grid = _run_grid_manager(grid_file, backend).grid
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
def test_grid_manager_eval_c2e2cO(caplog, grid_savepoint, grid_file, experiment, backend):
    caplog.set_level(logging.DEBUG)
    grid = _run_grid_manager(grid_file, backend).grid
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
def test_grid_manager_eval_e2c2e(caplog, grid_savepoint, grid_file, experiment, backend):
    caplog.set_level(logging.DEBUG)
    grid = _run_grid_manager(grid_file, backend).grid
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
def test_grid_manager_eval_e2c2v(caplog, grid_savepoint, grid_file, backend):
    caplog.set_level(logging.DEBUG)
    grid = _run_grid_manager(grid_file, backend).grid
    # the "far" (adjacent to edge normal ) is not always there, because ICON only calculates those starting from
    #   (lateral_boundary(dims.EdgeDim) + 1) to end(dims.EdgeDim)  (see mo_intp_coeffs.f90) and only for owned cells
    serialized_ref = grid_savepoint.e2c2v()
    table = grid.get_offset_provider("E2C2V").table
    assert_unless_invalid(table, serialized_ref)
    assert np.allclose(table[:, :2], grid.get_offset_provider("E2V").table)


@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_grid_manager_eval_c2v(caplog, grid_savepoint, grid_file, backend):
    caplog.set_level(logging.DEBUG)
    grid = _run_grid_manager(grid_file, backend).grid
    c2v = grid.get_offset_provider("C2V").table
    assert np.allclose(c2v, grid_savepoint.c2v())


@pytest.mark.parametrize(
    "dim, size",
    [
        (dims.CellDim, R02B04_GLOBAL_NUM_CELLS),
        (dims.EdgeDim, R02B04_GLOBAL_NUM_EDGES),
        (dims.VertexDim, R02B04_GLOBAL_NUM_VERTEX),
    ],
)
@pytest.mark.with_netcdf
def test_grid_manager_grid_size(dim, size, backend):
    grid = _run_grid_manager(utils.R02B04_GLOBAL, backend=backend).grid
    assert size == grid.size[dim]


def assert_up_to_order(table, diamond_table):
    assert table.shape == diamond_table.shape
    for n in range(table.shape[0]):
        assert np.all(np.in1d(table[n, :], diamond_table[n, :]))


@pytest.mark.with_netcdf
def test_gridmanager_given_file_not_found_then_abort():
    fname = "./unknown_grid.nc"
    with pytest.raises(FileNotFoundError) as error:
        manager = gm.GridManager(
            gm.NoTransformation(), fname, v_grid.VerticalGridConfig(num_levels=80)
        )
        manager(None)
        assert error.value == 1


@pytest.mark.parametrize("size", [100, 1500, 20000])
@pytest.mark.with_netcdf
def test_gt4py_transform_offset_by_1_where_valid(size):
    trafo = gm.ToZeroBasedIndexTransformation()
    rng = np.random.default_rng()
    input_field = rng.integers(-1, size, size)
    offset = trafo(input_field)
    expected = np.where(input_field >= 0, -1, 0)
    assert np.allclose(expected, offset)


@pytest.mark.parametrize(
    "grid_file, global_num_cells",
    [
        (utils.R02B04_GLOBAL, R02B04_GLOBAL_NUM_CELLS),
        (dt_utils.REGIONAL_EXPERIMENT, MCH_CH_RO4B09_GLOBAL_NUM_CELLS),
    ],
)
def test_grid_manager_grid_level_and_root(grid_file, global_num_cells, backend):
    assert global_num_cells == _run_grid_manager(grid_file, backend=backend).grid.global_num_cells


@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [(utils.R02B04_GLOBAL, dt_utils.JABW_EXPERIMENT)],
)
def test_grid_manager_eval_c2e2c2e(caplog, grid_savepoint, grid_file, backend):
    caplog.set_level(logging.DEBUG)
    grid = _run_grid_manager(grid_file, backend).grid
    serialized_grid = grid_savepoint.construct_icon_grid(on_gpu=False)
    assert np.allclose(
        grid.get_offset_provider("C2E2C2E").table,
        serialized_grid.get_offset_provider("C2E2C2E").table,
    )
    assert grid.get_offset_provider("C2E2C2E").table.shape == (grid.num_cells, 9)


@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
    ],
)
@pytest.mark.parametrize("dim", utils.horizontal_dim())
def test_grid_manager_start_end_index(caplog, grid_file, experiment, dim, icon_grid, backend):
    caplog.set_level(logging.INFO)
    serialized_grid = icon_grid
    grid = _run_grid_manager(grid_file, backend).grid
    for domain in utils.global_grid_domains(dim):
        assert grid.start_index(domain) == serialized_grid.start_index(
            domain
        ), f"start index wrong for domain {domain}"
        assert grid.end_index(domain) == serialized_grid.end_index(
            domain
        ), f"end index wrong for domain {domain}"

    for domain in utils.valid_boundary_zones_for_dim(dim):
        if not gridtest_utils.is_regional(grid_file):
            assert grid.start_index(domain) == 0
            assert grid.end_index(domain) == 0
        assert grid.start_index(domain) == serialized_grid.start_index(
            domain
        ), f"start index wrong for domain {domain}"
        assert grid.end_index(domain) == serialized_grid.end_index(
            domain
        ), f"end index wrong for domain {domain}"


@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_read_geometry_fields(grid_savepoint, grid_file, backend):
    manager = _run_grid_manager(grid_file, backend=backend)
    cell_area = manager.geometry[GeometryName.CELL_AREA.value]
    tangent_orientation = manager.geometry[GeometryName.TANGENT_ORIENTATION.value]

    assert helpers.dallclose(cell_area.asnumpy(), grid_savepoint.cell_areas().asnumpy())
    assert helpers.dallclose(
        tangent_orientation.asnumpy(), grid_savepoint.tangent_orientation().asnumpy()
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
@pytest.mark.parametrize("dim", (dims.CellDim, dims.EdgeDim, dims.VertexDim))
def test_coordinates(grid_savepoint, grid_file, experiment, dim, backend):
    manager = _run_grid_manager(grid_file, backend=backend)
    lat = manager.coordinates[dim]["lat"]
    lon = manager.coordinates[dim]["lon"]
    assert helpers.dallclose(lat.asnumpy(), grid_savepoint.lat(dim).asnumpy())
    assert helpers.dallclose(lon.asnumpy(), grid_savepoint.lon(dim).asnumpy())


@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_tangent_orientation(experiment, grid_file, grid_savepoint, backend):
    expected = grid_savepoint.tangent_orientation()
    manager = _run_grid_manager(grid_file, backend=backend)
    geometry_fields = manager.geometry
    assert helpers.dallclose(
        geometry_fields[GeometryName.TANGENT_ORIENTATION].asnumpy(), expected.asnumpy()
    )
