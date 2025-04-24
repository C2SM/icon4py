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
from icon4py.model.common.grid import geometry_attributes as geometry_attrs
from icon4py.model.common.interpolation import rbf_interpolation as rbf
from icon4py.model.common.interpolation.rbf_interpolation import RBF_STENCIL_SIZE
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    grid_utils as gridtest_utils,
    helpers as test_helpers,
)


@pytest.mark.parametrize("grid_file", (dt_utils.R02B04_GLOBAL, dt_utils.REGIONAL_EXPERIMENT))
def test_construct_rbf_matrix_offsets_tables_for_cells(grid_file):
    grid_manager = gridtest_utils.get_grid_manager(grid_file, 1, None)
    grid = grid_manager.grid
    offset_table = rbf.construct_rbf_matrix_offsets_tables_for_cells(grid)
    assert offset_table.shape == (
        grid.num_cells,
        rbf.RBF_STENCIL_SIZE[rbf.RBFDimension.CELL],
    )
    assert np.max(offset_table) == grid.num_edges - 1
    # TODO: This is literally doing the same as the implementation, access directly with c2c etc. in test?
    c2e = grid.connectivities[dims.C2EDim]
    c2e2c = grid.connectivities[dims.C2E2CDim]
    for i in range(offset_table.shape[0]):
        assert (offset_table[i][:3] == c2e[c2e2c[i][0]]).all()
        assert (offset_table[i][3:6] == c2e[c2e2c[i][1]]).all()
        assert (offset_table[i][6:] == c2e[c2e2c[i][2]]).all()


@pytest.mark.parametrize("grid_file", (dt_utils.R02B04_GLOBAL, dt_utils.REGIONAL_EXPERIMENT))
def test_construct_rbf_matrix_offsets_tables_for_edges(grid_file):
    grid_manager = gridtest_utils.get_grid_manager(grid_file, 1, None)
    grid = grid_manager.grid
    offset_table = rbf.construct_rbf_matrix_offsets_tables_for_edges(grid)
    assert offset_table.shape == (
        grid.num_edges,
        rbf.RBF_STENCIL_SIZE[rbf.RBFDimension.EDGE],
    )
    assert np.max(offset_table) == grid.num_edges - 1
    e2c2e = grid.connectivities[dims.E2C2EDim]
    assert (offset_table == e2c2e).all()


@pytest.mark.parametrize("grid_file", (dt_utils.R02B04_GLOBAL, dt_utils.REGIONAL_EXPERIMENT))
def test_construct_rbf_matrix_offsets_tables_for_vertices(grid_file):
    grid_manager = gridtest_utils.get_grid_manager(grid_file, 1, None)
    grid = grid_manager.grid
    offset_table = rbf.construct_rbf_matrix_offsets_tables_for_vertices(grid)
    assert offset_table.shape == (
        grid.num_vertices,
        rbf.RBF_STENCIL_SIZE[rbf.RBFDimension.VERTEX],
    )
    assert np.max(offset_table) == grid.num_edges - 1
    v2e = grid.connectivities[dims.V2EDim]
    # for i in range(offset_table.shape[0]):
    assert (offset_table == v2e).all()


# TODO: make cupy ready
# TODO: grid_file here only for comparison?
# TODO: more experiments? at least one regional (with missing neighbors)
@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment", [(dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT)]
)
def test_rbf_interpolation_matrix_cell(
    grid_file, grid_savepoint, interpolation_savepoint, icon_grid, backend, experiment
):  # fixture
    geometry = gridtest_utils.get_grid_geometry(backend, experiment, grid_file)
    grid = geometry.grid
    rbf_dim = rbf.RBFDimension.CELL

    rbf_vec_c1, rbf_vec_c2 = rbf.compute_rbf_interpolation_matrix_cell(
        geometry.get(geometry_attrs.CELL_LAT),
        geometry.get(geometry_attrs.CELL_LON),
        geometry.get(geometry_attrs.CELL_CENTER_X),
        geometry.get(geometry_attrs.CELL_CENTER_Y),
        geometry.get(geometry_attrs.CELL_CENTER_Z),
        geometry.get(geometry_attrs.EDGE_CENTER_X),
        geometry.get(geometry_attrs.EDGE_CENTER_Y),
        geometry.get(geometry_attrs.EDGE_CENTER_Z),
        # TODO: normals not dallclose? check
        geometry.get(geometry_attrs.EDGE_NORMAL_X),
        geometry.get(geometry_attrs.EDGE_NORMAL_Y),
        geometry.get(geometry_attrs.EDGE_NORMAL_Z),
        rbf.construct_rbf_matrix_offsets_tables_for_cells(grid),
        rbf.InterpolationConfig.rbf_kernel[rbf_dim],
        rbf.compute_rbf_scale(math.sqrt(grid_savepoint.mean_cell_area()), rbf_dim),
    )

    rbf_vec_coeff_c1_ref = interpolation_savepoint.rbf_vec_coeff_c1()
    rbf_vec_coeff_c2_ref = interpolation_savepoint.rbf_vec_coeff_c2()

    assert rbf_vec_c1.shape == rbf_vec_coeff_c1_ref.shape
    assert rbf_vec_c2.shape == rbf_vec_coeff_c2_ref.shape
    assert rbf_vec_coeff_c1_ref.shape == (
        icon_grid.num_cells,
        RBF_STENCIL_SIZE[rbf.RBFDimension.CELL],
    )
    assert rbf_vec_coeff_c2_ref.shape == (
        icon_grid.num_cells,
        RBF_STENCIL_SIZE[rbf.RBFDimension.CELL],
    )
    # TODO: Why does memory usage blow up if I don't have the explicit asnumpy here?
    assert test_helpers.dallclose(
        rbf_vec_c1.asnumpy(),
        rbf_vec_coeff_c1_ref.asnumpy(),
        atol=1e-8,
    )
    assert test_helpers.dallclose(
        rbf_vec_c2.asnumpy(),
        rbf_vec_coeff_c2_ref.asnumpy(),
        atol=1e-8,
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment", [(dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT)]
)
def test_rbf_interpolation_matrix_vertex(
    grid_file, grid_savepoint, interpolation_savepoint, icon_grid, backend, experiment
):  # fixture
    geometry = gridtest_utils.get_grid_geometry(backend, experiment, grid_file)
    grid = geometry.grid
    rbf_dim = rbf.RBFDimension.VERTEX

    rbf_vec_v1, rbf_vec_v2 = rbf.compute_rbf_interpolation_matrix_vertex(
        geometry.get(geometry_attrs.VERTEX_LAT),
        geometry.get(geometry_attrs.VERTEX_LON),
        geometry.get(geometry_attrs.VERTEX_CENTER_X),
        geometry.get(geometry_attrs.VERTEX_CENTER_Y),
        geometry.get(geometry_attrs.VERTEX_CENTER_Z),
        geometry.get(geometry_attrs.EDGE_CENTER_X),
        geometry.get(geometry_attrs.EDGE_CENTER_Y),
        geometry.get(geometry_attrs.EDGE_CENTER_Z),
        geometry.get(geometry_attrs.EDGE_NORMAL_X),
        geometry.get(geometry_attrs.EDGE_NORMAL_Y),
        geometry.get(geometry_attrs.EDGE_NORMAL_Z),
        rbf.construct_rbf_matrix_offsets_tables_for_vertices(grid),
        rbf.InterpolationConfig.rbf_kernel[rbf_dim],
        rbf.compute_rbf_scale(math.sqrt(grid_savepoint.mean_cell_area()), rbf_dim),
    )

    rbf_vec_coeff_v1_ref = interpolation_savepoint.rbf_vec_coeff_v1()
    rbf_vec_coeff_v2_ref = interpolation_savepoint.rbf_vec_coeff_v2()

    assert rbf_vec_v1.shape == rbf_vec_coeff_v1_ref.shape
    assert rbf_vec_v2.shape == rbf_vec_coeff_v2_ref.shape
    assert rbf_vec_coeff_v1_ref.shape == (
        icon_grid.num_vertices,
        RBF_STENCIL_SIZE[rbf.RBFDimension.VERTEX],
    )
    assert rbf_vec_coeff_v2_ref.shape == (
        icon_grid.num_vertices,
        RBF_STENCIL_SIZE[rbf.RBFDimension.VERTEX],
    )
    assert test_helpers.dallclose(
        rbf_vec_v1.asnumpy(),
        rbf_vec_coeff_v1_ref.asnumpy(),
        atol=1e-9,
    )
    assert test_helpers.dallclose(
        rbf_vec_v2.asnumpy(),
        rbf_vec_coeff_v2_ref.asnumpy(),
        atol=1e-9,
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment", [(dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT)]
)
def test_rbf_interpolation_matrix_edge(
    grid_file, grid_savepoint, interpolation_savepoint, icon_grid, backend, experiment
):  # fixture
    geometry = gridtest_utils.get_grid_geometry(backend, experiment, grid_file)
    rbf_dim = rbf.RBFDimension.EDGE

    rbf_vec_e = rbf.compute_rbf_interpolation_matrix_edge(
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
        # TODO: neighbors are not in the same order, use savepoint for now
        grid_savepoint.e2c2e(),
        rbf.InterpolationConfig.rbf_kernel[rbf_dim],
        rbf.compute_rbf_scale(math.sqrt(grid_savepoint.mean_cell_area()), rbf_dim),
    )

    rbf_vec_coeff_e_ref = interpolation_savepoint.rbf_vec_coeff_e()

    assert rbf_vec_e.shape == rbf_vec_coeff_e_ref.shape
    assert rbf_vec_coeff_e_ref.shape == (
        icon_grid.num_edges,
        RBF_STENCIL_SIZE[rbf.RBFDimension.EDGE],
    )
    assert test_helpers.dallclose(
        rbf_vec_e.asnumpy(),
        rbf_vec_coeff_e_ref.asnumpy(),
        atol=1e-12,
    )
