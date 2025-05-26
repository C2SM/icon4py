# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import math

import numpy as np
import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import geometry_attributes as geometry_attrs, horizontal as h_grid
from icon4py.model.common.interpolation import rbf_interpolation as rbf
from icon4py.model.common.interpolation.rbf_interpolation import RBF_STENCIL_SIZE
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    grid_utils as gridtest_utils,
    helpers as test_helpers,
)


@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
    ],
)
def test_construct_rbf_matrix_offsets_tables_for_cells(grid_file, grid_savepoint, icon_grid):
    grid_manager = gridtest_utils.get_grid_manager(grid_file, 1, None)
    grid = grid_manager.grid
    offset_table = rbf.construct_rbf_matrix_offsets_tables_for_cells(grid)
    assert offset_table.shape == (
        grid.num_cells,
        rbf.RBF_STENCIL_SIZE[rbf.RBFDimension.CELL],
    )
    assert np.max(offset_table) == grid.num_edges - 1

    offset_table_savepoint = grid_savepoint.c2e2c2e()
    assert offset_table.shape == offset_table_savepoint.shape

    # Savepoint neighbors before start index may not be populated correctly,
    # ignore them.
    start_index = icon_grid.start_index(
        h_grid.domain(dims.CellDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )

    for i in range(start_index, offset_table.shape[0]):
        # Neighbors may not be in the same order. Ignore differences in order.
        assert (np.sort(offset_table[i]) == np.sort(offset_table_savepoint[i])).all()


@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
    ],
)
def test_construct_rbf_matrix_offsets_tables_for_edges(grid_file, grid_savepoint, icon_grid):
    grid_manager = gridtest_utils.get_grid_manager(grid_file, 1, None)
    grid = grid_manager.grid
    offset_table = rbf.construct_rbf_matrix_offsets_tables_for_edges(grid)
    assert offset_table.shape == (
        grid.num_edges,
        rbf.RBF_STENCIL_SIZE[rbf.RBFDimension.EDGE],
    )
    assert np.max(offset_table) == grid.num_edges - 1

    offset_table_savepoint = grid_savepoint.e2c2e()
    assert offset_table.shape == offset_table_savepoint.shape

    start_index = icon_grid.start_index(
        h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )

    for i in range(start_index, offset_table.shape[0]):
        # Neighbors may not be in the same order. Ignore differences in order.
        assert (np.sort(offset_table[i]) == np.sort(offset_table_savepoint[i])).all()


@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
    ],
)
def test_construct_rbf_matrix_offsets_tables_for_vertices(grid_file, grid_savepoint, icon_grid):
    grid_manager = gridtest_utils.get_grid_manager(grid_file, 1, None)
    grid = grid_manager.grid
    offset_table = rbf.construct_rbf_matrix_offsets_tables_for_vertices(grid)
    assert offset_table.shape == (
        grid.num_vertices,
        rbf.RBF_STENCIL_SIZE[rbf.RBFDimension.VERTEX],
    )
    assert np.max(offset_table) == grid.num_edges - 1

    offset_table_savepoint = grid_savepoint.v2e()
    assert offset_table.shape == offset_table_savepoint.shape

    start_index = icon_grid.start_index(
        h_grid.domain(dims.VertexDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )

    for i in range(start_index, offset_table.shape[0]):
        # TODO: Should this be fixed already when reading data?
        for j in range(1, offset_table_savepoint.shape[1]):
            if offset_table_savepoint[i, j] == offset_table_savepoint[i, j - 1]:
                offset_table_savepoint[i, j] = -1

        # Neighbors may not be in the same order. Ignore differences in order.
        assert (np.sort(offset_table[i]) == np.sort(offset_table_savepoint[i])).all()


@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment, atol",
    [
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 3e-9),
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 3e-2),
    ],
)
def test_rbf_interpolation_matrix_cell(
    grid_file, grid_savepoint, interpolation_savepoint, icon_grid, backend, experiment, atol
):
    geometry = gridtest_utils.get_grid_geometry(backend, experiment, grid_file)
    grid = geometry.grid
    rbf_dim = rbf.RBFDimension.CELL

    rbf_vec_coeff_c1, rbf_vec_coeff_c2 = rbf.compute_rbf_interpolation_matrix_cell(
        geometry.get(geometry_attrs.CELL_LAT),
        geometry.get(geometry_attrs.CELL_LON),
        geometry.get(geometry_attrs.CELL_CENTER_X),
        geometry.get(geometry_attrs.CELL_CENTER_Y),
        geometry.get(geometry_attrs.CELL_CENTER_Z),
        geometry.get(geometry_attrs.EDGE_CENTER_X),
        geometry.get(geometry_attrs.EDGE_CENTER_Y),
        geometry.get(geometry_attrs.EDGE_CENTER_Z),
        geometry.get(geometry_attrs.EDGE_NORMAL_X),
        geometry.get(geometry_attrs.EDGE_NORMAL_Y),
        geometry.get(geometry_attrs.EDGE_NORMAL_Z),
        rbf.construct_rbf_matrix_offsets_tables_for_cells(grid),
        rbf.InterpolationConfig.rbf_kernel[rbf_dim],
        rbf.compute_rbf_scale(math.sqrt(grid_savepoint.mean_cell_area()), rbf_dim),
        array_ns=data_alloc.import_array_ns(backend),
    )

    start_index = icon_grid.start_index(
        h_grid.domain(dims.CellDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )

    assert start_index < grid.num_cells

    rbf_vec_coeff_c1_ref = interpolation_savepoint.rbf_vec_coeff_c1()
    rbf_vec_coeff_c2_ref = interpolation_savepoint.rbf_vec_coeff_c2()

    assert rbf_vec_coeff_c1.shape == rbf_vec_coeff_c1_ref.shape
    assert rbf_vec_coeff_c2.shape == rbf_vec_coeff_c2_ref.shape
    assert rbf_vec_coeff_c1_ref.shape == (
        icon_grid.num_cells,
        RBF_STENCIL_SIZE[rbf.RBFDimension.CELL],
    )
    assert rbf_vec_coeff_c2_ref.shape == (
        icon_grid.num_cells,
        RBF_STENCIL_SIZE[rbf.RBFDimension.CELL],
    )
    assert test_helpers.dallclose(
        rbf_vec_coeff_c1[start_index:],
        rbf_vec_coeff_c1_ref.asnumpy()[start_index:],
        atol=atol,
    )
    assert test_helpers.dallclose(
        rbf_vec_coeff_c2[start_index:],
        rbf_vec_coeff_c2_ref.asnumpy()[start_index:],
        atol=atol,
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment, atol",
    [
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 3e-10),
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 2e-3),
    ],
)
def test_rbf_interpolation_matrix_vertex(
    grid_file, grid_savepoint, interpolation_savepoint, icon_grid, backend, experiment, atol
):
    geometry = gridtest_utils.get_grid_geometry(backend, experiment, grid_file)
    grid = geometry.grid
    rbf_dim = rbf.RBFDimension.VERTEX

    rbf_vec_coeff_v1, rbf_vec_coeff_v2 = rbf.compute_rbf_interpolation_matrix_vertex(
        geometry.get(geometry_attrs.VERTEX_LAT),
        geometry.get(geometry_attrs.VERTEX_LON),
        geometry.get(geometry_attrs.VERTEX_X),
        geometry.get(geometry_attrs.VERTEX_Y),
        geometry.get(geometry_attrs.VERTEX_Z),
        geometry.get(geometry_attrs.EDGE_CENTER_X),
        geometry.get(geometry_attrs.EDGE_CENTER_Y),
        geometry.get(geometry_attrs.EDGE_CENTER_Z),
        geometry.get(geometry_attrs.EDGE_NORMAL_X),
        geometry.get(geometry_attrs.EDGE_NORMAL_Y),
        geometry.get(geometry_attrs.EDGE_NORMAL_Z),
        rbf.construct_rbf_matrix_offsets_tables_for_vertices(grid),
        rbf.InterpolationConfig.rbf_kernel[rbf_dim],
        rbf.compute_rbf_scale(math.sqrt(grid_savepoint.mean_cell_area()), rbf_dim),
        array_ns=data_alloc.import_array_ns(backend),
    )

    start_index = icon_grid.start_index(
        h_grid.domain(dims.VertexDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )

    assert start_index < grid.num_vertices

    rbf_vec_coeff_v1_ref = interpolation_savepoint.rbf_vec_coeff_v1()
    rbf_vec_coeff_v2_ref = interpolation_savepoint.rbf_vec_coeff_v2()

    assert rbf_vec_coeff_v1.shape == rbf_vec_coeff_v1_ref.shape
    assert rbf_vec_coeff_v2.shape == rbf_vec_coeff_v2_ref.shape
    assert rbf_vec_coeff_v1_ref.shape == (
        icon_grid.num_vertices,
        RBF_STENCIL_SIZE[rbf.RBFDimension.VERTEX],
    )
    assert rbf_vec_coeff_v2_ref.shape == (
        icon_grid.num_vertices,
        RBF_STENCIL_SIZE[rbf.RBFDimension.VERTEX],
    )
    assert test_helpers.dallclose(
        rbf_vec_coeff_v1[start_index:],
        rbf_vec_coeff_v1_ref.asnumpy()[start_index:],
        atol=atol,
    )
    assert test_helpers.dallclose(
        rbf_vec_coeff_v2[start_index:],
        rbf_vec_coeff_v2_ref.asnumpy()[start_index:],
        atol=atol,
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment, atol",
    [
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 8e-14),
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 1e-9),
    ],
)
def test_rbf_interpolation_matrix_edge(
    grid_file,
    grid_savepoint,
    interpolation_savepoint,
    icon_grid,
    backend,
    experiment,
    atol,
):
    geometry = gridtest_utils.get_grid_geometry(backend, experiment, grid_file)
    grid = geometry.grid
    rbf_dim = rbf.RBFDimension.EDGE

    rbf_vec_coeff_e = rbf.compute_rbf_interpolation_matrix_edge(
        geometry.get(geometry_attrs.EDGE_LAT),
        geometry.get(geometry_attrs.EDGE_LON),
        geometry.get(geometry_attrs.EDGE_CENTER_X),
        geometry.get(geometry_attrs.EDGE_CENTER_Y),
        geometry.get(geometry_attrs.EDGE_CENTER_Z),
        geometry.get(geometry_attrs.EDGE_NORMAL_X),
        geometry.get(geometry_attrs.EDGE_NORMAL_Y),
        geometry.get(geometry_attrs.EDGE_NORMAL_Z),
        geometry.get(geometry_attrs.EDGE_DUAL_U),
        geometry.get(geometry_attrs.EDGE_DUAL_V),
        # NOTE: Neighbors are not in the same order. Use savepoint to make sure
        # order of coefficients computed by icon4py matches order of
        # coefficients in savepoint.
        grid_savepoint.e2c2e(),
        rbf.InterpolationConfig.rbf_kernel[rbf_dim],
        rbf.compute_rbf_scale(math.sqrt(grid_savepoint.mean_cell_area()), rbf_dim),
        array_ns=data_alloc.import_array_ns(backend),
    )

    start_index = icon_grid.start_index(
        h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )

    assert start_index < grid.num_edges

    rbf_vec_coeff_e_ref = interpolation_savepoint.rbf_vec_coeff_e()

    assert rbf_vec_coeff_e.shape == rbf_vec_coeff_e_ref.shape
    assert rbf_vec_coeff_e_ref.shape == (
        icon_grid.num_edges,
        RBF_STENCIL_SIZE[rbf.RBFDimension.EDGE],
    )
    assert test_helpers.dallclose(
        rbf_vec_coeff_e[start_index:],
        rbf_vec_coeff_e_ref.asnumpy()[start_index:],
        atol=atol,
    )
