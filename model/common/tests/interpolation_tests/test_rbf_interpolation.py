# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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
    assert offset_table.shape == (grid.num_cells, rbf.RBF_STENCIL_SIZE[rbf.RBFDimension.CELL])
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
    assert offset_table.shape == (grid.num_edges, rbf.RBF_STENCIL_SIZE[rbf.RBFDimension.EDGE])
    assert np.max(offset_table) == grid.num_edges - 1
    e2c2e = grid.connectivities[dims.E2C2EDim]
    assert (offset_table == e2c2e).all()


@pytest.mark.parametrize("grid_file", (dt_utils.R02B04_GLOBAL, dt_utils.REGIONAL_EXPERIMENT))
def test_construct_rbf_matrix_offsets_tables_for_vertices(grid_file):
    grid_manager = gridtest_utils.get_grid_manager(grid_file, 1, None)
    grid = grid_manager.grid
    offset_table = rbf.construct_rbf_matrix_offsets_tables_for_vertices(grid)
    assert offset_table.shape == (grid.num_vertices, rbf.RBF_STENCIL_SIZE[rbf.RBFDimension.VERTEX])
    assert np.max(offset_table) == grid.num_edges - 1
    v2e = grid.connectivities[dims.V2EDim]
    # for i in range(offset_table.shape[0]):
    assert (offset_table == v2e).all()

# TODO: make cupy ready
# TODO: grid_file here only for comparison?
# TODO: more experiments? at least one regional (with missing neighbors)
@pytest.mark.datatest
@pytest.mark.parametrize("grid_file, experiment", [(dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT)])
def test_rbf_interpolation_matrix_cell(grid_file, grid_savepoint, interpolation_savepoint, icon_grid, backend, experiment): # fixture
    geometry = gridtest_utils.get_grid_geometry(backend, experiment, grid_file)
    grid = geometry.grid
    rbf_vec_coeff_c1_ref = interpolation_savepoint.rbf_vec_coeff_c1().asnumpy()
    rbf_vec_coeff_c2_ref = interpolation_savepoint.rbf_vec_coeff_c1().asnumpy()
    assert rbf_vec_coeff_c2_ref.shape == (grid.num_cells, RBF_STENCIL_SIZE[rbf.RBFDimension.CELL])
    assert rbf_vec_coeff_c1_ref.shape == (grid.num_cells, RBF_STENCIL_SIZE[rbf.RBFDimension.CELL])

    offset_table = rbf.construct_rbf_matrix_offsets_tables_for_cells(grid)
    edge_x = geometry.get(geometry_attrs.EDGE_CENTER_X).asnumpy()
    edge_y = geometry.get(geometry_attrs.EDGE_CENTER_Y).asnumpy()
    edge_z = geometry.get(geometry_attrs.EDGE_CENTER_Z).asnumpy()
    edge_normal_x = geometry.get(geometry_attrs.EDGE_NORMAL_X).asnumpy()
    edge_normal_y = geometry.get(geometry_attrs.EDGE_NORMAL_Y).asnumpy()
    edge_normal_z = geometry.get(geometry_attrs.EDGE_NORMAL_Z).asnumpy()
    cell_center_lat = geometry.get(geometry_attrs.CELL_LAT).asnumpy()
    cell_center_lon = geometry.get(geometry_attrs.CELL_LON).asnumpy()
    cell_center_x = geometry.get(geometry_attrs.CELL_CENTER_X).asnumpy()
    cell_center_y = geometry.get(geometry_attrs.CELL_CENTER_Y).asnumpy()
    cell_center_z = geometry.get(geometry_attrs.CELL_CENTER_Z).asnumpy()

    rbf_vec_c1, rbf_vec_c2 = rbf.compute_rbf_interpolation_matrix(
        cell_center_lat,
        cell_center_lon,
        cell_center_x,
        cell_center_y,
        cell_center_z,
        edge_x,
        edge_y,
        edge_z,
        edge_normal_x,
        edge_normal_y,
        edge_normal_z,
        offset_table,
        rbf.InterpolationKernel.GAUSSIAN,
        1.0,
    )
    test_helpers.dallclose(rbf_vec_c1, rbf_vec_coeff_c1_ref)
    test_helpers.dallclose(rbf_vec_c2, rbf_vec_coeff_c2_ref)
