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

import numpy as np
import pytest

from icon4py.model.common.dimension import EdgeDim
from icon4py.model.common.grid.horizontal import HorizontalMarkerIndex
from icon4py.model.common.interpolation.interpolation_fields import compute_c_lin_e


@pytest.mark.datatest
def test_compute_c_lin_e(grid_savepoint, interpolation_savepoint, icon_grid):
    inv_dual_edge_length = grid_savepoint.inv_dual_edge_length()
    edge_cell_length = grid_savepoint.edge_cell_length()
    owner_mask = grid_savepoint.e_owner_mask()
    c_lin_e_ref = interpolation_savepoint.c_lin_e()
    lateral_boundary = icon_grid.get_start_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1,
    )
    c_lin_e = compute_c_lin_e(
        edge_cell_length.asnumpy(),
        inv_dual_edge_length.asnumpy(),
        owner_mask.asnumpy(),
        lateral_boundary,
    )

    assert np.allclose(c_lin_e, c_lin_e_ref.asnumpy())
