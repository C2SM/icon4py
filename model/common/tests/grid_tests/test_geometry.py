# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools

import numpy as np
import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import (
    geometry as geometry,
    geometry_attributes as attrs,
    horizontal as h_grid,
    simple as simple,
)
from icon4py.model.common.grid.geometry import as_sparse_field
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import datatest_utils as dt_utils, grid_utils, helpers


def test_geometry_raises_for_unknown_field(backend):
    geometry = grid_utils.get_grid_geometry(
        backend, dt_utils.GLOBAL_EXPERIMENT, dt_utils.R02B04_GLOBAL
    )
    with pytest.raises(ValueError) as e:
        geometry.get("foo")
        assert "'foo'" in e.value
        assert "'GridGeometry'" in e.value


@pytest.mark.parametrize(
    "grid_file, experiment, rtol",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 1e-8),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 3e-12),
    ],
)
@pytest.mark.datatest
def test_edge_control_area(backend, grid_savepoint, grid_file, experiment, rtol):
    expected = grid_savepoint.edge_areas()
    geometry_source = grid_utils.get_grid_geometry(backend, experiment, grid_file)
    result = geometry_source.get(attrs.EDGE_AREA)
    assert helpers.dallclose(expected.asnumpy(), result.asnumpy(), rtol)


@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
@pytest.mark.datatest
def test_coriolis_parameter(backend, grid_savepoint, grid_file, experiment):
    geometry_source = grid_utils.get_grid_geometry(backend, experiment, grid_file)
    expected = grid_savepoint.f_e()

    result = geometry_source.get(attrs.CORIOLIS_PARAMETER)
    assert helpers.dallclose(expected.asnumpy(), result.asnumpy())


@pytest.mark.parametrize(
    "grid_file, experiment, rtol",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 1e-9),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 1e-12),
    ],
)
@pytest.mark.datatest
def test_compute_edge_length(backend, grid_savepoint, grid_file, experiment, rtol):
    geometry_source = grid_utils.get_grid_geometry(backend, experiment, grid_file)
    expected = grid_savepoint.primal_edge_length()
    result = geometry_source.get(attrs.EDGE_LENGTH)
    assert helpers.dallclose(result.asnumpy(), expected.asnumpy(), rtol=rtol)


@pytest.mark.parametrize(
    "grid_file, experiment, rtol",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 1e-9),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 1e-12),
    ],
)
@pytest.mark.datatest
def test_compute_inverse_edge_length(backend, grid_savepoint, grid_file, experiment, rtol):
    expected = grid_savepoint.inverse_primal_edge_lengths()
    geometry_source = grid_utils.get_grid_geometry(backend, experiment, grid_file)
    computed = geometry_source.get(f"inverse_of_{attrs.EDGE_LENGTH}")

    assert helpers.dallclose(computed.asnumpy(), expected.asnumpy(), rtol=rtol)


@pytest.mark.parametrize(
    "grid_file, experiment, rtol",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 1e-8),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 1e-11),
    ],
)
@pytest.mark.datatest
def test_compute_dual_edge_length(backend, grid_savepoint, grid_file, experiment, rtol):
    grid_geometry = grid_utils.get_grid_geometry(backend, experiment, grid_file)

    expected = grid_savepoint.dual_edge_length()
    result = grid_geometry.get(attrs.DUAL_EDGE_LENGTH)
    assert helpers.dallclose(result.asnumpy(), expected.asnumpy(), rtol=rtol)


@pytest.mark.parametrize(
    "grid_file, experiment, rtol",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 5e-9),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 1e-11),
    ],
)
@pytest.mark.datatest
def test_compute_inverse_dual_edge_length(backend, grid_savepoint, grid_file, experiment, rtol):
    grid_geometry = grid_utils.get_grid_geometry(backend, experiment, grid_file)
    expected = grid_savepoint.inv_dual_edge_length()
    result = grid_geometry.get(f"inverse_of_{attrs.DUAL_EDGE_LENGTH}")

    # compared to ICON we overcompute, so we only compare the values from LATERAL_BOUNDARY_LEVEL_2
    level = h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    start_index = grid_geometry.grid.start_index(level)
    assert helpers.dallclose(
        result.asnumpy()[start_index:], expected.asnumpy()[start_index:], rtol=rtol
    )


@pytest.mark.parametrize(
    "grid_file, experiment, rtol",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 5e-10),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 1e-12),
    ],
)
@pytest.mark.datatest
def test_compute_inverse_vertex_vertex_length(backend, grid_savepoint, grid_file, experiment, rtol):
    grid_geometry = grid_utils.get_grid_geometry(backend, experiment, grid_file)

    expected = grid_savepoint.inv_vert_vert_length()
    result = grid_geometry.get(attrs.INVERSE_VERTEX_VERTEX_LENGTH)
    assert helpers.dallclose(result.asnumpy(), expected.asnumpy(), rtol=rtol)


@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_compute_coordinates_of_edge_tangent_and_normal(
    backend, grid_savepoint, grid_file, experiment
):
    grid_geometry = grid_utils.get_grid_geometry(backend, experiment, grid_file)
    x_normal = grid_geometry.get(attrs.EDGE_NORMAL_X)
    y_normal = grid_geometry.get(attrs.EDGE_NORMAL_Y)
    z_normal = grid_geometry.get(attrs.EDGE_NORMAL_Z)
    x_tangent = grid_geometry.get(attrs.EDGE_TANGENT_X)
    y_tangent = grid_geometry.get(attrs.EDGE_TANGENT_Y)
    z_tangent = grid_geometry.get(attrs.EDGE_TANGENT_Z)

    x_normal_ref = grid_savepoint.primal_cart_normal_x()
    y_normal_ref = grid_savepoint.primal_cart_normal_y()
    z_normal_ref = grid_savepoint.primal_cart_normal_z()
    x_tangent_ref = grid_savepoint.dual_cart_normal_x()
    y_tangent_ref = grid_savepoint.dual_cart_normal_y()
    z_tangent_ref = grid_savepoint.dual_cart_normal_z()
    assert helpers.dallclose(x_tangent.asnumpy(), x_tangent_ref.asnumpy(), atol=1e-12)
    assert helpers.dallclose(y_tangent.asnumpy(), y_tangent_ref.asnumpy(), atol=1e-12)
    assert helpers.dallclose(z_tangent.asnumpy(), z_tangent_ref.asnumpy(), atol=1e-12)
    assert helpers.dallclose(x_normal.asnumpy(), x_normal_ref.asnumpy(), atol=1e-13)  # 1e-16
    assert helpers.dallclose(z_normal.asnumpy(), z_normal_ref.asnumpy(), atol=1e-13)
    assert helpers.dallclose(y_normal.asnumpy(), y_normal_ref.asnumpy(), atol=1e-12)


@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_compute_primal_normals(backend, grid_savepoint, grid_file, experiment):
    grid_geometry = grid_utils.get_grid_geometry(backend, experiment, grid_file)
    primal_normal_u = grid_geometry.get(attrs.EDGE_NORMAL_U)
    primal_normal_v = grid_geometry.get(attrs.EDGE_NORMAL_V)

    primal_normal_u_ref = grid_savepoint.primal_normal_v1()
    primal_normal_v_ref = grid_savepoint.primal_normal_v2()

    assert helpers.dallclose(primal_normal_u.asnumpy(), primal_normal_u_ref.asnumpy(), atol=1e-12)
    assert helpers.dallclose(primal_normal_v.asnumpy(), primal_normal_v_ref.asnumpy(), atol=1e-12)


@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_tangent_orientation(backend, grid_savepoint, grid_file, experiment):
    grid_geometry = grid_utils.get_grid_geometry(backend, experiment, grid_file)
    result = grid_geometry.get(attrs.TANGENT_ORIENTATION)
    expected = grid_savepoint.tangent_orientation()

    assert helpers.dallclose(result.asnumpy(), expected.asnumpy())


@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_cell_area(backend, grid_savepoint, experiment, grid_file):
    grid_geometry = grid_utils.get_grid_geometry(backend, experiment, grid_file)
    result = grid_geometry.get(attrs.CELL_AREA)
    expected = grid_savepoint.cell_areas()

    assert helpers.dallclose(result.asnumpy(), expected.asnumpy())


@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_primal_normal_cell(backend, grid_savepoint, grid_file, experiment):
    grid_geometry = grid_utils.get_grid_geometry(backend, experiment, grid_file)
    primal_normal_cell_u_ref = grid_savepoint.primal_normal_cell_x().asnumpy()
    primal_normal_cell_v_ref = grid_savepoint.primal_normal_cell_y().asnumpy()
    primal_normal_cell_u = grid_geometry.get(attrs.EDGE_NORMAL_CELL_U)
    primal_normal_cell_v = grid_geometry.get(attrs.EDGE_NORMAL_CELL_V)

    assert helpers.dallclose(primal_normal_cell_u.asnumpy(), primal_normal_cell_u_ref, atol=1e-12)
    assert helpers.dallclose(primal_normal_cell_v.asnumpy(), primal_normal_cell_v_ref, atol=1e-12)


@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_dual_normal_cell(backend, grid_savepoint, grid_file, experiment):
    grid_geometry = grid_utils.get_grid_geometry(backend, experiment, grid_file)
    dual_normal_cell_u_ref = grid_savepoint.dual_normal_cell_x().asnumpy()
    dual_normal_cell_v_ref = grid_savepoint.dual_normal_cell_y().asnumpy()
    dual_normal_cell_u = grid_geometry.get(attrs.EDGE_TANGENT_CELL_U)
    dual_normal_cell_v = grid_geometry.get(attrs.EDGE_TANGENT_CELL_V)

    assert helpers.dallclose(dual_normal_cell_u.asnumpy(), dual_normal_cell_u_ref, atol=1e-12)
    assert helpers.dallclose(dual_normal_cell_v.asnumpy(), dual_normal_cell_v_ref, atol=1e-12)


@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_primal_normal_vert(backend, grid_savepoint, grid_file, experiment):
    grid_geometry = grid_utils.get_grid_geometry(backend, experiment, grid_file)
    primal_normal_vert_u_ref = grid_savepoint.primal_normal_vert_x().asnumpy()
    primal_normal_vert_v_ref = grid_savepoint.primal_normal_vert_y().asnumpy()
    primal_normal_vert_u = grid_geometry.get(attrs.EDGE_NORMAL_VERTEX_U)
    primal_normal_vert_v = grid_geometry.get(attrs.EDGE_NORMAL_VERTEX_V)

    assert helpers.dallclose(primal_normal_vert_u.asnumpy(), primal_normal_vert_u_ref, atol=1e-12)
    assert helpers.dallclose(primal_normal_vert_v.asnumpy(), primal_normal_vert_v_ref, atol=1e-12)


@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_dual_normal_vert(backend, grid_savepoint, grid_file, experiment):
    grid_geometry = grid_utils.get_grid_geometry(backend, experiment, grid_file)
    dual_normal_vert_u_ref = grid_savepoint.dual_normal_vert_x().asnumpy()
    dual_normal_vert_v_ref = grid_savepoint.dual_normal_vert_y().asnumpy()
    dual_normal_vert_u = grid_geometry.get(attrs.EDGE_TANGENT_VERTEX_U)
    dual_normal_vert_v = grid_geometry.get(attrs.EDGE_TANGENT_VERTEX_V)

    assert helpers.dallclose(dual_normal_vert_u.asnumpy(), dual_normal_vert_u_ref, atol=1e-12)
    assert helpers.dallclose(dual_normal_vert_v.asnumpy(), dual_normal_vert_v_ref, atol=1e-12)


def test_sparse_fields_creator():
    grid = simple.SimpleGrid()
    f1 = data_alloc.random_field(grid, dims.EdgeDim)
    f2 = data_alloc.random_field(grid, dims.EdgeDim)
    g1 = data_alloc.random_field(grid, dims.EdgeDim)
    g2 = data_alloc.random_field(grid, dims.EdgeDim)

    sparse = as_sparse_field((dims.EdgeDim, dims.E2CDim), [(f1, f2), (g1, g2)])
    sparse_e2c = functools.partial(as_sparse_field, (dims.EdgeDim, dims.E2CDim))
    sparse2 = sparse_e2c(((f1, f2), (g1, g2)))
    assert sparse[0].asnumpy().shape == (grid.num_edges, 2)
    assert helpers.dallclose(sparse[0].asnumpy(), sparse2[0].asnumpy())
    assert helpers.dallclose(sparse[1].asnumpy(), sparse2[1].asnumpy())


@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_create_auxiliary_orientation_coordinates(backend, grid_savepoint, grid_file):
    gm = grid_utils.get_grid_manager(grid_file, backend=backend, num_levels=1)
    grid = gm.grid
    coordinates = gm.coordinates

    cell_lat = coordinates[dims.CellDim]["lat"]
    cell_lon = coordinates[dims.CellDim]["lon"]
    edge_lat = coordinates[dims.EdgeDim]["lat"]
    edge_lon = coordinates[dims.EdgeDim]["lon"]
    lat_0, lon_0, lat_1, lon_1 = geometry.create_auxiliary_coordinate_arrays_for_orientation(
        grid, cell_lat, cell_lon, edge_lat, edge_lon
    )
    connectivity = data_alloc.as_numpy(grid.connectivities[dims.E2CDim])
    has_boundary_edges = np.count_nonzero(connectivity == -1)
    if has_boundary_edges == 0:
        assert helpers.dallclose(lat_0.asnumpy(), cell_lat.asnumpy()[connectivity[:, 0]])
        assert helpers.dallclose(lat_1.asnumpy(), cell_lat.asnumpy()[connectivity[:, 1]])
        assert helpers.dallclose(lon_0.asnumpy(), cell_lon.asnumpy()[connectivity[:, 0]])
        assert helpers.dallclose(lon_1.asnumpy(), cell_lon.asnumpy()[connectivity[:, 1]])

    edge_coordinates_0 = np.where(connectivity[:, 0] < 0)
    edge_coordinates_1 = np.where(connectivity[:, 1] < 0)
    cell_coordinates_0 = np.where(connectivity[:, 0] >= 0)
    cell_coordinates_1 = np.where(connectivity[:, 1] >= 0)
    assert helpers.dallclose(
        lat_0.asnumpy()[edge_coordinates_0], edge_lat.asnumpy()[edge_coordinates_0]
    )
    assert helpers.dallclose(
        lat_0.asnumpy()[cell_coordinates_0], cell_lat.asnumpy()[connectivity[cell_coordinates_0, 0]]
    )

    assert helpers.dallclose(
        lon_0.asnumpy()[edge_coordinates_0], edge_lon.asnumpy()[edge_coordinates_0]
    )
    assert helpers.dallclose(
        lon_0.asnumpy()[cell_coordinates_0], cell_lon.asnumpy()[connectivity[cell_coordinates_0, 0]]
    )

    assert helpers.dallclose(
        lat_1.asnumpy()[edge_coordinates_1], edge_lat.asnumpy()[edge_coordinates_1]
    )
    assert helpers.dallclose(
        lat_1.asnumpy()[cell_coordinates_1], cell_lat.asnumpy()[connectivity[cell_coordinates_1, 1]]
    )
    assert helpers.dallclose(
        lon_1.asnumpy()[edge_coordinates_1], edge_lon.asnumpy()[edge_coordinates_1]
    )
    assert helpers.dallclose(
        lon_1.asnumpy()[cell_coordinates_1], cell_lon.asnumpy()[connectivity[cell_coordinates_1, 1]]
    )
