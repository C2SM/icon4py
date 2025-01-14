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
from icon4py.model.common.test_utils import datatest_utils as dt_utils, grid_utils as gridtest_utils


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


@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_rbf_interpolation_matrix(grid_file, experiment, backend):
    geometry = gridtest_utils.get_grid_geometry(backend, experiment, grid_file)
    grid = geometry.grid
    edge_x = geometry.get(geometry_attrs.EDGE_CENTER_X)
    edge_y = geometry.get(geometry_attrs.EDGE_CENTER_Y)
    edge_z = geometry.get(geometry_attrs.EDGE_CENTER_Z)
    edge_normal_x = geometry.get(geometry_attrs.EDGE_NORMAL_X)
    edge_normal_y = geometry.get(geometry_attrs.EDGE_NORMAL_Y)
    edge_normal_z = geometry.get(geometry_attrs.EDGE_NORMAL_Z)
    pytest.fail()
