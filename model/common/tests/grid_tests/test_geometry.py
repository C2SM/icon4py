# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import icon4py.model.common.constants as constants
from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions
from icon4py.model.common.grid import (
    geometry as geometry,
    geometry_attributes as attrs,
    horizontal as h_grid,
)
from icon4py.model.common.test_utils import datatest_utils as dt_utils, helpers
from icon4py.model.common.utils import gt4py_field_allocation as alloc

from . import utils


# FIXME boundary values for LAM model
@pytest.mark.parametrize(
    "grid_file, experiment, rtol",
    [
        # (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 1e-9),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 3e-12),
    ],
)
@pytest.mark.datatest
def test_edge_control_area(grid_savepoint, grid_file, backend, rtol):
    expected = grid_savepoint.edge_areas()
    geometry_source = construct_grid_geometry(backend, grid_file)
    result = geometry_source.get(attrs.EDGE_AREA)
    assert helpers.dallclose(expected.ndarray, result.ndarray, rtol)


@pytest.mark.parametrize("experiment", [dt_utils.GLOBAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT])
@pytest.mark.datatest
def test_coriolis_parameter_field_op(grid_savepoint, icon_grid, backend):
    expected = grid_savepoint.f_e()
    result = helpers.zero_field(icon_grid, dims.EdgeDim)
    lat = grid_savepoint.lat(dims.EdgeDim)
    geometry._coriolis_parameter_on_edges.with_backend(backend)(
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
def test_coriolis_parameter(grid_savepoint, grid_file, backend):
    expected = grid_savepoint.f_e()
    gm = utils.run_grid_manager(grid_file)
    grid = gm.grid
    decomposition_info = construct_decomposition_info(grid)

    geometry_source = geometry.GridGeometry(
        grid, decomposition_info, backend, gm.coordinates, attrs.attrs
    )
    geometry_source()

    result = geometry_source.get(attrs.CORIOLIS_PARAMETER)
    assert helpers.dallclose(expected.asnumpy(), result.asnumpy())


def construct_decomposition_info(grid):
    edge_indices = alloc.allocate_indices(dims.EdgeDim, grid)
    owner_mask = np.ones((grid.num_edges,), dtype=bool)
    decomposition_info = definitions.DecompositionInfo(klevels=1)
    decomposition_info.with_dimension(dims.EdgeDim, edge_indices.ndarray, owner_mask)
    return decomposition_info


@pytest.mark.parametrize(
    "grid_file, experiment, rtol",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 1e-9),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 1e-12),
    ],
)
@pytest.mark.datatest
def test_compute_edge_length_program(experiment, backend, grid_savepoint, grid_file, rtol):
    expected_edge_length = grid_savepoint.primal_edge_length()
    gm = utils.run_grid_manager(grid_file)
    grid = gm.grid
    decomposition_info = construct_decomposition_info(grid)
    geometry_source = geometry.GridGeometry(
        grid, decomposition_info, backend, gm.coordinates, attrs.attrs
    )
    geometry_source()
    # FIXME: does not run on compiled???
    # edge_length = geometry_source.get(attrs.EDGE_LENGTH)
    edge_length = helpers.zero_field(grid, dims.EdgeDim)

    vertex_lat = gm.coordinates[dims.VertexDim]["lat"]
    vertex_lon = gm.coordinates[dims.VertexDim]["lon"]

    edge_domain = h_grid.domain(dims.EdgeDim)
    start = grid.start_index(edge_domain(h_grid.Zone.LOCAL))
    end = grid.end_index(edge_domain(h_grid.Zone.LOCAL))
    geometry.compute_edge_length(
        vertex_lat,
        vertex_lon,
        constants.EARTH_RADIUS,
        edge_length,
        start,
        end,
        offset_provider={
            "E2V": grid.get_offset_provider("E2V"),
        },
    )

    assert helpers.dallclose(edge_length.asnumpy(), expected_edge_length.asnumpy(), rtol=rtol)


@pytest.mark.parametrize(
    "grid_file, experiment, rtol",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 1e-9),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 1e-12),
    ],
)
@pytest.mark.datatest
def test_compute_edge_length(experiment, backend, grid_savepoint, grid_file, rtol):
    expected = grid_savepoint.primal_edge_length()
    geometry_source = construct_grid_geometry(backend, grid_file)
    # FIXME: does not run on compiled???
    result = geometry_source.get(attrs.EDGE_LENGTH)
    assert helpers.dallclose(result.ndarray, expected.ndarray, rtol=rtol)


@pytest.mark.parametrize(
    "grid_file, experiment, rtol",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 1e-9),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 1e-12),
    ],
)
@pytest.mark.datatest
def test_compute_inverse_edge_length(experiment, backend, grid_savepoint, grid_file, rtol):
    expected = grid_savepoint.inverse_primal_edge_lengths()
    geometry_source = construct_grid_geometry(backend, grid_file)
    # FIXME: does not run on compiled???
    computed = geometry_source.get(f"inverse_of_{attrs.EDGE_LENGTH}")

    assert helpers.dallclose(computed.ndarray, expected.ndarray, rtol=rtol)


def construct_grid_geometry(backend, grid_file):
    gm = utils.run_grid_manager(grid_file)
    grid = gm.grid
    decomposition_info = construct_decomposition_info(grid)
    geometry_source = geometry.GridGeometry(
        grid, decomposition_info, backend, gm.coordinates, attrs.attrs
    )
    geometry_source()
    return geometry_source


# TODO (halungge) why does serialized reference start from index 0 even for LAM model?
@pytest.mark.parametrize(
    "grid_file, experiment, rtol",
    [
        # (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 5e-9),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 1e-11),
    ],
)
@pytest.mark.datatest
def test_compute_dual_edge_length(experiment, backend, grid_savepoint, grid_file, rtol):
    grid_geometry = construct_grid_geometry(backend, grid_file)

    expected = grid_savepoint.dual_edge_length()
    result = grid_geometry.get(attrs.DUAL_EDGE_LENGTH)
    assert helpers.dallclose(result.ndarray, expected.ndarray, rtol=rtol)


# TODO (halungge) why does serialized reference start from index 0 even for LAM model?
@pytest.mark.parametrize(
    "grid_file, experiment, rtol",
    [
        # (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 5e-9),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 1e-11),
    ],
)
@pytest.mark.datatest
def test_compute_inverse_dual_edge_length(experiment, backend, grid_savepoint, grid_file, rtol):
    grid_geometry = construct_grid_geometry(backend, grid_file)

    expected = grid_savepoint.inv_dual_edge_length()
    result = grid_geometry.get(f"inverse_of_{attrs.DUAL_EDGE_LENGTH}")
    assert helpers.dallclose(result.ndarray, expected.ndarray, rtol=rtol)


@pytest.mark.parametrize(
    "grid_file, experiment, rtol",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 5e-10),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 1e-12),
    ],
)
@pytest.mark.datatest
def test_compute_inverse_vertex_vertex_length(experiment, backend, grid_savepoint, grid_file, rtol):
    grid_geometry = construct_grid_geometry(backend, grid_file)

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
def test_primal_normal_cell_primal_normal_vertex(grid_file, experiment, grid_savepoint, backend):
    edge_domain = h_grid.domain(dims.EdgeDim)
    grid = grid_savepoint.construct_icon_grid(on_gpu=False)
    # what is this?
    x = grid_savepoint.primal_cart_normal_x()
    y = grid_savepoint.primal_cart_normal_y()
    z = grid_savepoint.primal_cart_normal_z()

    cell_lat = grid_savepoint.lat(dims.CellDim)
    cell_lon = grid_savepoint.lon(dims.CellDim)
    vertex_lat = grid_savepoint.verts_vertex_lat()
    vertex_lon = grid_savepoint.verts_vertex_lon()

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
        offset_provider={
            "E2C": grid.get_offset_provider("E2C"),
            "E2V": grid.get_offset_provider("E2V"),
        },
        domain={dims.EdgeDim: (start, end)},
    )

    assert helpers.dallclose(v1_cell.asnumpy(), v1_cell_ref)
    assert helpers.dallclose(v2_cell.asnumpy(), v2_cell_ref)
    assert helpers.dallclose(u1_cell.asnumpy(), u1_cell_ref, atol=2e-16)
    assert helpers.dallclose(u2_cell.asnumpy(), u2_cell_ref, atol=2e-16)
    assert helpers.dallclose(v1_vertex.asnumpy(), v1_vertex_ref, atol=2e-16)
    assert helpers.dallclose(v2_vertex.asnumpy(), v2_vertex_ref, atol=2e-16)
    assert helpers.dallclose(u1_vertex.asnumpy(), u1_vertex_ref, atol=2e-16)
    assert helpers.dallclose(u2_vertex.asnumpy(), u2_vertex_ref, atol=2e-16)
