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
from icon4py.model.common.interpolation import rbf_interplation as rbf
from icon4py.model.common.interpolation.rbf_interplation import RBF_STENCIL_SIZE
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
    c2e = grid.connectivities[dims.C2EDim]
    c2e2c = grid.connectivities[dims.C2E2CDim]
    for i in range(offset_table.shape[0]):
        offset_table[i][:3] = c2e[c2e2c[i][0]]
        offset_table[i][3:6] = c2e[c2e2c[i][1]]
        offset_table[i][6:] = c2e[c2e2c[i][2]]


#TODO make cupy ready
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.R02B04_GLOBAL, dt_utils.JABW_EXPERIMENT),
    ],
)
def test_rbf_interpolation_matrix(grid_file, experiment, backend, interpolation_savepoint):
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
        cell_center_lat, cell_center_lon,
        cell_center_x, cell_center_y, cell_center_z,
        edge_x, edge_y, edge_z,
        edge_normal_x, edge_normal_y, edge_normal_z,
        offset_table,
        rbf.InterpolationKernel.GAUSSIAN,
        1.0
    )
    test_helpers.dallclose(rbf_vec_c1, rbf_vec_coeff_c1_ref)
    test_helpers.dallclose(rbf_vec_c2, rbf_vec_coeff_c2_ref)
