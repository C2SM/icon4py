# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import numpy as np
import pytest

import icon4py.model.common.constants as constants
import icon4py.model.common.dimension as dims
import icon4py.model.common.grid.geometry as geometry
import icon4py.model.common.math.helpers as math_helpers
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.test_utils import datatest_utils as dt_utils, helpers

from . import utils


@pytest.mark.parametrize("experiment", [dt_utils.GLOBAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT])
@pytest.mark.datatest
def test_dual_edge_length(experiment, grid_savepoint):
    if experiment == dt_utils.REGIONAL_EXPERIMENT:
        pytest.mark.xfail(f"FIXME: single precision error for '{experiment}'")
    expected = grid_savepoint.dual_edge_length().asnumpy()
    grid = grid_savepoint.construct_icon_grid(on_gpu=False)

    lat = grid_savepoint.lat(dims.CellDim)
    lon = grid_savepoint.lon(dims.CellDim)
    start = grid.start_index(h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    end = grid.end_index(h_grid.domain(dims.EdgeDim)(h_grid.Zone.END))

    result_arc = helpers.zero_field(grid, dims.EdgeDim)
    result_tendon = helpers.zero_field(grid, dims.EdgeDim)
    buffer = np.vstack((np.ones(grid.num_edges), -1 * np.ones(grid.num_edges))).T
    subtraction_coeff = gtx.as_field((dims.EdgeDim, dims.E2CDim), data=buffer)

    geometry.dual_edge_length.with_backend(None)(
        lat,
        lon,
        subtraction_coeff,
        constants.EARTH_RADIUS,
        offset_provider={"E2C": grid.get_offset_provider("E2C")},
        out=(result_arc, result_tendon),
        domain={dims.EdgeDim: (start, end)},

    )

    arch_array = result_arc.asnumpy()
    tendon_array = result_tendon.asnumpy()
    rel_error = np.abs(arch_array - expected) / expected
    assert np.max(rel_error < 1e-12)
    assert helpers.dallclose(arch_array, expected)


@pytest.mark.parametrize("experiment", [dt_utils.GLOBAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT])
@pytest.mark.datatest
def test_primal_edge_length(experiment, grid_savepoint):
    if experiment == dt_utils.REGIONAL_EXPERIMENT:
        pytest.mark.xfail(f"FIXME: single precision error for '{experiment}'")
    grid = grid_savepoint.construct_icon_grid(on_gpu=False)
    expected = grid_savepoint.primal_edge_length().asnumpy()
    lat = grid_savepoint.lat(dims.VertexDim)
    lon = grid_savepoint.lon(dims.VertexDim)
    start = grid.start_index(h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY))
    end = grid.end_index(h_grid.domain(dims.EdgeDim)(h_grid.Zone.END))
    arc_result = helpers.zero_field(grid, dims.EdgeDim)
    tendon_result = helpers.zero_field(grid, dims.EdgeDim)
    buffer = np.vstack((np.ones(grid.num_edges), -1 * np.ones(grid.num_edges))).T
    subtract_coeff = gtx.as_field((dims.EdgeDim, dims.E2VDim), data=buffer)
    geometry.primal_edge_length.with_backend(None)(
        lat,
        lon,
        subtract_coeff,
        constants.EARTH_RADIUS,
        offset_provider={"E2V": grid.get_offset_provider("E2V")},
        out=(arc_result, tendon_result),
        domain = {dims.EdgeDim:(start, end)},
    )
    rel_error = np.abs(arc_result.asnumpy() - expected) / expected
    assert np.max(rel_error < 1e-12)

    assert helpers.dallclose(arc_result.asnumpy(), expected)



@pytest.mark.parametrize("experiment", [dt_utils.GLOBAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT])
@pytest.mark.datatest
def test_edge_control_area(grid_savepoint):
    grid = grid_savepoint.construct_icon_grid(on_gpu=False)

    expected = grid_savepoint.edge_areas()
    owner_mask = grid_savepoint.e_owner_mask()
    primal_edge_length = grid_savepoint.primal_edge_length()
    dual_edge_length = grid_savepoint.dual_edge_length()
    result = helpers.zero_field(grid, dims.EdgeDim)
    geometry.edge_control_area(
        owner_mask, primal_edge_length, dual_edge_length, offset_provider={}, out=result
    )
    assert helpers.dallclose(expected.asnumpy(), result.asnumpy())


@pytest.mark.parametrize("experiment", [dt_utils.GLOBAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT])
@pytest.mark.datatest
def test_coriolis_parameter(grid_savepoint, icon_grid):
    expected = grid_savepoint.f_e()
    result = helpers.zero_field(icon_grid, dims.EdgeDim)
    lat = grid_savepoint.lat(dims.EdgeDim)
    geometry.coriolis_parameter_on_edges(
        lat, constants.EARTH_ANGULAR_VELOCITY, offset_provider={}, out=result
    )
    assert helpers.dallclose(expected.asnumpy(), result.asnumpy())



@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
@pytest.mark.datatest
def test_vertex_vertex_length(experiment, grid_savepoint, grid_file):
    if experiment == dt_utils.REGIONAL_EXPERIMENT:
        pytest.mark.xfail(f"FIXME: single precision error for '{experiment}'")
    gm = utils.run_grid_manager(grid_file)
    grid = gm.grid
    expected = grid_savepoint.inv_vert_vert_length()
    result = helpers.zero_field(grid, dims.EdgeDim)

    lat = gtx.as_field((dims.VertexDim,), gm.coordinates(dims.VertexDim)["lat"], dtype=float)
    lon = gtx.as_field((dims.VertexDim,), gm.coordinates(dims.VertexDim)["lon"], dtype=float)
    edge_domain = h_grid.domain(dims.EdgeDim)
    start = grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    end = grid.end_index(edge_domain(h_grid.Zone.END))
    geometry.vertex_vertex_length(
        lat,
        lon,
        constants.EARTH_RADIUS,
        out=result,
        offset_provider={"E2C2V": grid.get_offset_provider("E2C2V")},
        domain={dims.EdgeDim: (start, end)},
    )
    math_helpers.invert(result, offset_provider={}, out=result)

    assert helpers.dallclose(expected.asnumpy(), result.asnumpy())


@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_primal_normal_cell_primal_normal_vertex(grid_file, experiment, grid_savepoint, backend):
    edge_domain = h_grid.domain(dims.EdgeDim)
    grid = grid_savepoint.construct_icon_grid(on_gpu=False)
    x = grid_savepoint.primal_cart_normal_x()
    y = grid_savepoint.primal_cart_normal_y()
    z = grid_savepoint.primal_cart_normal_z()

    cell_lat = grid_savepoint.lat(dims.CellDim)
    cell_lon = grid_savepoint.lon(dims.CellDim)
    vertex_lat = grid_savepoint.verts_vertex_lat()
    vertex_lon  =grid_savepoint.verts_vertex_lon()

    u1_cell_ref = grid_savepoint.primal_normal_cell_x().asnumpy()[:, 0]
    u2_cell_ref = grid_savepoint.primal_normal_cell_x().asnumpy()[:, 1]
    v1_cell_ref = grid_savepoint.primal_normal_cell_y().asnumpy()[:, 0]
    v2_cell_ref = grid_savepoint.primal_normal_cell_y().asnumpy()[:, 1]

    u1_vertex_ref = grid_savepoint.primal_normal_vert_x().asnumpy()[:, 0]
    u2_vertex_ref = grid_savepoint.primal_normal_vert_x().asnumpy()[:, 1]
    v1_vertex_ref = grid_savepoint.primal_normal_vert_y().asnumpy()[:, 0]
    v2_vertex_ref = grid_savepoint.primal_normal_vert_y().asnumpy()[:, 1]

    v1_cell = helpers.zero_field(grid, dims.EdgeDim)
    v2_cell = helpers.zero_field(grid, dims.EdgeDim)
    u1_cell = helpers.zero_field(grid, dims.EdgeDim)
    u2_cell = helpers.zero_field(grid, dims.EdgeDim)
    v1_vertex = helpers.zero_field(grid, dims.EdgeDim)
    v2_vertex = helpers.zero_field(grid, dims.EdgeDim)
    u1_vertex = helpers.zero_field(grid, dims.EdgeDim)
    u2_vertex = helpers.zero_field(grid, dims.EdgeDim)

    start = grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    end = grid.end_index(edge_domain(h_grid.Zone.END))
    geometry.primal_normals.with_backend(None)(
        cell_lat,
        cell_lon,
        vertex_lat,
        vertex_lon,
        x,
        y,
        z,
        out=(u1_cell, u2_cell, v1_cell, v2_cell, u1_vertex, u2_vertex, v1_vertex, v2_vertex),
        offset_provider={"E2C": grid.get_offset_provider("E2C"), "E2V":grid.get_offset_provider("E2V")},
        domain={ dims.EdgeDim: (start, end)},
    )

    assert helpers.dallclose(v1_cell.asnumpy(), v1_cell_ref)
    assert helpers.dallclose(v2_cell.asnumpy(), v2_cell_ref)
    assert helpers.dallclose(u1_cell.asnumpy(), u1_cell_ref, atol = 2e-16)
    assert helpers.dallclose(u2_cell.asnumpy(), u2_cell_ref, atol = 2e-16)
    assert helpers.dallclose(v1_vertex.asnumpy(), v1_vertex_ref, atol = 2e-16)
    assert helpers.dallclose(v2_vertex.asnumpy(), v2_vertex_ref,  atol = 2e-16)
    assert helpers.dallclose(u1_vertex.asnumpy(), u1_vertex_ref,  atol = 2e-16)
    assert helpers.dallclose(u2_vertex.asnumpy(), u2_vertex_ref,  atol = 2e-16)



