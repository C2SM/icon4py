# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid.horizontal import HorizontalMarkerIndex
from icon4py.model.common.metrics.compute_coeff_gradekin import compute_coeff_gradekin
from icon4py.model.common.test_utils import datatest_utils as dt_utils
from icon4py.model.common.test_utils.helpers import dallclose


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_coeff_gradekin(icon_grid, grid_savepoint, metrics_savepoint):
    edge_cell_length = grid_savepoint.edge_cell_length().asnumpy()
    inv_dual_edge_length = grid_savepoint.inv_dual_edge_length().asnumpy()
    coeff_gradekin_ref = metrics_savepoint.coeff_gradekin()
    horizontal_start = icon_grid.get_start_index(
        dims.EdgeDim, HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 1
    )
    horizontal_end = icon_grid.num_edges

    coeff_gradekin_full = compute_coeff_gradekin(
        edge_cell_length, inv_dual_edge_length, horizontal_start, horizontal_end
    )
    assert dallclose(coeff_gradekin_ref.asnumpy(), coeff_gradekin_full.asnumpy())
