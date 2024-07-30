# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest

from icon4py.model.common.dimension import EdgeDim
from icon4py.model.common.grid.horizontal import HorizontalMarkerIndex
from icon4py.model.common.metrics.stencils.compute_coeff_gradekin import compute_coeff_gradekin
from icon4py.model.common.test_utils.helpers import dallclose


@pytest.mark.datatest
def test_compute_coeff_gradekin(icon_grid, grid_savepoint, metrics_savepoint):
    edge_cell_length = grid_savepoint.edge_cell_length().asnumpy()
    inv_dual_edge_length = grid_savepoint.inv_dual_edge_length().asnumpy()
    coeff_gradekin_ref = metrics_savepoint.coeff_gradekin()
    horizontal_start = icon_grid.get_start_index(
        EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1
    )
    horizontal_end = icon_grid.num_edges

    coeff_gradekin_full = compute_coeff_gradekin(
        edge_cell_length, inv_dual_edge_length, horizontal_start, horizontal_end
    )
    assert dallclose(coeff_gradekin_ref.asnumpy(), coeff_gradekin_full.asnumpy())
