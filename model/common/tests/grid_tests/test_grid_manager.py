# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import logging
import typing

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions as defs, halo
from icon4py.model.common.grid import (
    grid_manager as gm,
    gridfile,
    horizontal as h_grid,
    refinement as refin,
    vertical as v_grid,
)
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    helpers,
)


if typing.TYPE_CHECKING:
    import netCDF4

try:
    import netCDF4  # noqa # F401
except ImportError:
    pytest.skip("optional netcdf dependency not installed", allow_module_level=True)


from . import utils


MCH_CH_RO4B09_GLOBAL_NUM_CELLS = 83886080


ZERO_BASE = gm.ToZeroBasedIndexTransformation()
vertical = v_grid.VerticalGridConfig(num_levels=80)


# TODO @magdalena add test cases for hexagon vertices v2e2v
# v2e2v: grid,???


# v2e: exists in serial, simple, grid
@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_grid_manager_eval_v2e(caplog, grid_savepoint, experiment, grid_file, backend):
    caplog.set_level(logging.DEBUG)
    grid = utils.run_grid_manager(grid_file, keep_skip_values=True, backend=backend).grid
    seralized_v2e = grid_savepoint.v2e()
    # there are vertices at the boundary of a local domain or at a pentagon point that have less than
    # 6 neighbors hence there are "Missing values" in the grid file
    # they get substituted by the "last valid index" in preprocessing step in icon.
    assert not has_invalid_index(seralized_v2e)
    v2e_table = grid.get_connectivity("V2E").asnumpy()
    assert has_invalid_index(v2e_table)
    _reset_invalid_index(seralized_v2e)
    assert np.allclose(v2e_table, seralized_v2e)


@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
@pytest.mark.parametrize("dim", [dims.CellDim, dims.EdgeDim, dims.VertexDim])
def test_grid_manager_refin_ctrl(grid_savepoint, grid_file, experiment, dim, backend):
    refin_ctrl = utils.run_grid_manager(
        grid_file, keep_skip_values=True, backend=backend
    ).grid.refinement_control
    refin_ctrl_serialized = grid_savepoint.refin_ctrl(dim)
    assert np.all(
        refin_ctrl_serialized.ndarray
        == refin.convert_to_unnested_refinement_values(refin_ctrl[dim].ndarray, dim)
    )


# v2c: exists in serial, simple, grid
@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_grid_manager_eval_v2c(caplog, grid_savepoint, experiment, grid_file, backend):
    caplog.set_level(logging.DEBUG)
    grid = utils.run_grid_manager(grid_file, keep_skip_values=True, backend=backend).grid
    serialized_v2c = grid_savepoint.v2c()
    v2c_table = grid.get_connectivity("V2C").asnumpy()
    # there are vertices that have less than 6 neighboring cells: either pentagon points or
    # vertices at the boundary of the domain for a limited area mode
    # hence in the grid file there are "missing values"
    # they get substituted by the "last valid index" in preprocessing step in icon.
    assert not has_invalid_index(serialized_v2c)
    assert has_invalid_index(v2c_table)
    _reset_invalid_index(serialized_v2c)

    assert np.allclose(v2c_table, serialized_v2c)


def _reset_invalid_index(index_array: np.ndarray):
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
        index_array[i, max(index) + 1 :] = gridfile.GridFile.INVALID_INDEX


# e2v: exists in serial, simple, grid
@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_grid_manager_eval_e2v(caplog, grid_savepoint, grid_file, experiment, backend):
    caplog.set_level(logging.DEBUG)
    grid = utils.run_grid_manager(grid_file, keep_skip_values=True, backend=backend).grid

    serialized_e2v = grid_savepoint.e2v()
    e2v_table = grid.get_connectivity("E2V").asnumpy()
    # all vertices in the system have to neighboring edges, there no edges that point nowhere
    # hence this connectivity has no "missing values" in the grid file
    assert not has_invalid_index(serialized_e2v)
    assert not has_invalid_index(e2v_table)
    assert np.allclose(e2v_table, serialized_e2v)


def has_invalid_index(ar: np.ndarray):
    return np.any(invalid_index(ar))


def invalid_index(ar: np.ndarray):
    return np.where(ar == gridfile.GridFile.INVALID_INDEX)


# e2c : exists in serial, simple, grid
@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_grid_manager_eval_e2c(caplog, grid_savepoint, grid_file, experiment, backend):
    caplog.set_level(logging.DEBUG)

    grid = utils.run_grid_manager(grid_file, keep_skip_values=True, backend=backend).grid
    serialized_e2c = grid_savepoint.e2c()
    e2c_table = grid.get_connectivity("E2C").asnumpy()
    assert has_invalid_index(serialized_e2c) == grid.limited_area
    assert has_invalid_index(e2c_table) == grid.limited_area
    assert np.allclose(e2c_table, serialized_e2c)


# c2e: serial, simple, grid
@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_grid_manager_eval_c2e(caplog, grid_savepoint, grid_file, experiment, backend):
    caplog.set_level(logging.DEBUG)
    grid = utils.run_grid_manager(grid_file, keep_skip_values=True, backend=backend).grid

    serialized_c2e = grid_savepoint.c2e()
    c2e_table = grid.get_connectivity("C2E").asnumpy()
    # no cells with less than 3 neighboring edges exist, otherwise the cell is not there in the
    # first place
    # hence there are no "missing values" in the grid file
    assert not has_invalid_index(serialized_c2e)
    assert not has_invalid_index(c2e_table)
    assert np.allclose(c2e_table, serialized_c2e)


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
    grid = utils.run_grid_manager(grid_file, keep_skip_values=True, backend=backend).grid
    assert np.allclose(
        grid.get_connectivity("C2E2C").asnumpy(),
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
    grid = utils.run_grid_manager(grid_file, keep_skip_values=True, backend=backend).grid
    serialized_grid = grid_savepoint.construct_icon_grid(on_gpu=False)
    assert np.allclose(
        grid.get_connectivity("C2E2CO").asnumpy(),
        serialized_grid.get_connectivity("C2E2CO").asnumpy(),
    )


# e2c2e (e2c2eo) - diamond: exists in serial, simple_mesh
@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_grid_manager_eval_e2c2e(caplog, grid_savepoint, grid_file, experiment, backend):
    caplog.set_level(logging.DEBUG)
    grid = utils.run_grid_manager(grid_file, keep_skip_values=True, backend=backend).grid
    serialized_grid = grid_savepoint.construct_icon_grid(on_gpu=False)
    serialized_e2c2e = serialized_grid.get_connectivity("E2C2E").asnumpy()
    serialized_e2c2eO = serialized_grid.get_connectivity("E2C2EO").asnumpy()
    assert has_invalid_index(serialized_e2c2e) == grid.limited_area

    e2c2e_table = grid.get_connectivity("E2C2E").asnumpy()
    e2c2eO_table = grid.get_connectivity("E2C2EO").asnumpy()
    assert has_invalid_index(e2c2e_table) == grid.limited_area
    # ICON calculates diamond edges only from rl_start = 2 (lateral_boundary(dims.EdgeDim) + 1 for
    # boundaries all values are INVALID even though the half diamond exists (see mo_model_domimp_setup.f90 ll 163ff.)
    start_index = grid.start_index(
        h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
    )
    # e2c2e in ICON (quad_idx) has a different neighbor ordering than the e2c2e constructed in grid_manager.py
    assert_up_to_order(e2c2e_table, serialized_e2c2e, start_index)
    assert_up_to_order(e2c2eO_table, serialized_e2c2eO, start_index)


@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_grid_manager_eval_e2c2v(caplog, grid_savepoint, grid_file, backend):
    caplog.set_level(logging.DEBUG)
    grid = utils.run_grid_manager(grid_file, keep_skip_values=True, backend=backend).grid
    serialized_ref = grid_savepoint.e2c2v()
    # the "far" (adjacent to edge normal ) is not always there, because ICON only calculates those starting from
    #   (lateral_boundary(dims.EdgeDim) + 1) to end(dims.EdgeDim)  (see mo_intp_coeffs.f90) and only for owned cells
    table = grid.get_connectivity("E2C2V").asnumpy()
    start_index = grid.start_index(
        h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    # e2c2e in ICON (quad_idx) has a different neighbor ordering than the e2c2e constructed in grid_manager.py
    assert_up_to_order(table, serialized_ref, start_index)
    assert np.allclose(table[:, :2], grid.get_connectivity("E2V").asnumpy())


@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_grid_manager_eval_c2v(caplog, grid_savepoint, grid_file, backend):
    caplog.set_level(logging.DEBUG)
    grid = utils.run_grid_manager(grid_file, keep_skip_values=True, backend=backend).grid
    c2v = grid.get_connectivity("C2V").asnumpy()
    assert np.allclose(c2v, grid_savepoint.c2v())


@pytest.mark.parametrize(
    "dim, size",
    [
        (dims.CellDim, utils.R02B04_GLOBAL_NUM_CELLS),
        (dims.EdgeDim, utils.R02B04_GLOBAL_NUM_EDGES),
        (dims.VertexDim, utils.R02B04_GLOBAL_NUM_VERTEX),
    ],
)
@pytest.mark.with_netcdf
def test_grid_manager_grid_size(dim, size, backend):
    grid = utils.run_grid_manager(
        dt_utils.R02B04_GLOBAL, keep_skip_values=True, backend=backend
    ).grid
    assert size == grid.size[dim]


def assert_up_to_order(table: np.ndarray, reference_table: np.ndarray, start_index: gtx.int = 0):
    assert table.shape == reference_table.shape, "arrays need to have the same shape"
    reduced_table = table[start_index:, :]
    reduced_reference = reference_table[start_index:, :]
    for n in range(reduced_table.shape[0]):
        assert np.all(
            np.in1d(reduced_table[n, :], reduced_reference[n, :])
        ), f"values in row {n+start_index} are not equal: {reduced_table[n, :]} vs ref= {reduced_reference[n, :]}."


@pytest.mark.with_netcdf
def test_gridmanager_given_file_not_found_then_abort():
    fname = "./unknown_grid.nc"
    with pytest.raises(FileNotFoundError) as error:
        manager = gm.GridManager(
            gm.NoTransformation(), fname, v_grid.VerticalGridConfig(num_levels=80)
        )
        manager(backend=None, keep_skip_values=True)
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
        (dt_utils.R02B04_GLOBAL, utils.R02B04_GLOBAL_NUM_CELLS),
        (dt_utils.REGIONAL_EXPERIMENT, MCH_CH_RO4B09_GLOBAL_NUM_CELLS),
    ],
)
def test_grid_manager_grid_level_and_root(grid_file, global_num_cells, backend):
    assert (
        global_num_cells
        == utils.run_grid_manager(
            grid_file, keep_skip_values=True, backend=backend
        ).grid.global_num_cells
    )


@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [(dt_utils.R02B04_GLOBAL, dt_utils.JABW_EXPERIMENT)],
)
def test_grid_manager_eval_c2e2c2e(caplog, grid_savepoint, grid_file, backend):
    caplog.set_level(logging.DEBUG)
    grid = utils.run_grid_manager(grid_file, keep_skip_values=True, backend=backend).grid
    serialized_grid = grid_savepoint.construct_icon_grid(on_gpu=False)
    assert np.allclose(
        grid.get_connectivity("C2E2C2E").asnumpy(),
        serialized_grid.get_connectivity("C2E2C2E").asnumpy(),
    )
    assert grid.get_connectivity("C2E2C2E").asnumpy().shape == (grid.num_cells, 9)


@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
    ],
)
@pytest.mark.parametrize("dim", utils.main_horizontal_dims())
def test_grid_manager_start_end_index(caplog, grid_file, experiment, dim, icon_grid, backend):
    caplog.set_level(logging.INFO)
    serialized_grid = icon_grid
    grid = utils.run_grid_manager(grid_file, keep_skip_values=True, backend=backend).grid
    for domain in utils.global_grid_domains(dim):
        if (
            dim == dims.EdgeDim
            and domain.zone == h_grid.Zone.END
            and experiment == dt_utils.GLOBAL_EXPERIMENT
        ):
            pytest.xfail(
                "FIXME: start_index in serialized data changed to 0 with unknown consequences, see also icon-exclaim output"
            )
        assert grid.start_index(domain) == serialized_grid.start_index(
            domain
        ), f"start index wrong for domain {domain}"
        assert grid.end_index(domain) == serialized_grid.end_index(
            domain
        ), f"end index wrong for domain {domain}"

    for domain in utils.valid_boundary_zones_for_dim(dim):
        if not grid.limited_area:
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
    manager = utils.run_grid_manager(grid_file, keep_skip_values=True, backend=backend)
    cell_area = manager.geometry[gridfile.GeometryName.CELL_AREA.value]
    tangent_orientation = manager.geometry[gridfile.GeometryName.TANGENT_ORIENTATION.value]

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
    manager = utils.run_grid_manager(grid_file, keep_skip_values=True, backend=backend)
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
def test_tangent_orientation(grid_file, grid_savepoint, backend):
    expected = grid_savepoint.tangent_orientation()
    manager = utils.run_grid_manager(grid_file, keep_skip_values=True, backend=backend)
    geometry_fields = manager.geometry
    assert helpers.dallclose(
        geometry_fields[gridfile.GeometryName.TANGENT_ORIENTATION].asnumpy(), expected.asnumpy()
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_edge_orientation_on_vertex(grid_file, grid_savepoint, backend):
    expected = grid_savepoint.vertex_edge_orientation()
    manager = utils.run_grid_manager(grid_file, keep_skip_values=True, backend=backend)
    geometry_fields = manager.geometry
    assert helpers.dallclose(
        geometry_fields[gridfile.GeometryName.EDGE_ORIENTATION_ON_VERTEX].asnumpy(),
        expected.asnumpy(),
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_dual_area(grid_file, grid_savepoint, backend):
    expected = grid_savepoint.vertex_dual_area()
    manager = utils.run_grid_manager(grid_file, keep_skip_values=True, backend=backend)
    geometry_fields = manager.geometry
    assert helpers.dallclose(
        geometry_fields[gridfile.GeometryName.DUAL_AREA].asnumpy(), expected.asnumpy()
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_edge_cell_distance(grid_file, grid_savepoint, backend):
    expected = grid_savepoint.edge_cell_length()
    manager = utils.run_grid_manager(grid_file, keep_skip_values=True, backend=backend)
    geometry_fields = manager.geometry

    assert helpers.dallclose(
        geometry_fields[gridfile.GeometryName.EDGE_CELL_DISTANCE].asnumpy(),
        expected.asnumpy(),
        equal_nan=True,
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_cell_normal_orientation(grid_file, grid_savepoint, backend):
    expected = grid_savepoint.edge_orientation()
    manager = utils.run_grid_manager(grid_file, keep_skip_values=True, backend=backend)
    geometry_fields = manager.geometry
    assert helpers.dallclose(
        geometry_fields[gridfile.GeometryName.CELL_NORMAL_ORIENTATION].asnumpy(), expected.asnumpy()
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_edge_vertex_distance(grid_file, grid_savepoint, backend):
    expected = grid_savepoint.edge_vert_length()
    manager = utils.run_grid_manager(grid_file, keep_skip_values=True, backend=backend)
    geometry_fields = manager.geometry

    assert helpers.dallclose(
        geometry_fields[gridfile.GeometryName.EDGE_VERTEX_DISTANCE].asnumpy(),
        expected.asnumpy(),
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "grid_file, expected", [(dt_utils.REGIONAL_EXPERIMENT, True), (dt_utils.R02B04_GLOBAL, False)]
)
def test_limited_area_on_grid(grid_file, expected):
    grid = utils.run_grid_manager(grid_file, keep_skip_values=True, backend=None).grid
    assert expected == grid.limited_area


# TODO move to mpi_tests folder
@pytest.mark.mpi
@pytest.mark.parametrize(
    "field_offset",
    [dims.C2V, dims.E2V, dims.V2C, dims.E2C, dims.C2E, dims.V2E, dims.C2E2C, dims.V2E2V],
)
def test_local_connectivities(processor_props, caplog, field_offset):  # fixture
    caplog.set_level(logging.INFO)
    grid = utils.run_grid_manager(dt_utils.R02B04_GLOBAL, backend=None).grid
    partitioner = halo.SimpleMetisDecomposer()
    face_face_connectivity = grid.connectivities[dims.C2E2CDim]
    labels = partitioner(face_face_connectivity, n_part=processor_props.comm_size)
    halo_generator = halo.HaloGenerator(
        connectivities=grid.neighbor_tables,
        run_properties=processor_props,
        rank_mapping=labels,
        num_levels=1,
    )

    decomposition_info = halo_generator()

    connectivity = gm.construct_local_connectivity(
        field_offset, decomposition_info, connectivity=grid.neighbor_tables[field_offset.target[1]]
    )
    # there is an neighbor list for each index of the target dimension on the node
    assert (
        connectivity.shape[0]
        == decomposition_info.global_index(
            field_offset.target[0], defs.DecompositionInfo.EntryType.ALL
        ).size
    )
    # all neighbor indices are valid local indices
    assert np.max(connectivity) == np.max(
        decomposition_info.local_index(field_offset.source, defs.DecompositionInfo.EntryType.ALL)
    )
    # TODO what else to assert?
    # - outer halo entries have SKIP_VALUE neighbors (depends on offsets)
