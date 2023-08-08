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

from icon4py.common.dimension import EdgeDim
from icon4py.field_management.interpolation_fields import compute_c_lin_e
from icon4py.grid.horizontal import HorizontalMarkerIndex


def test_mo_icon_interpolation_fields_initalization(data_provider, icon_grid):
    icon_grid_save_point = data_provider.from_savepoint_grid()
    interpolation_savepoint = data_provider.from_interpolation_savepoint()
    inv_dual_edge_length = icon_grid_save_point.inv_dual_edge_length()
    edge_cell_length = icon_grid_save_point.edge_cell_length()
    owner_mask = icon_grid_save_point.e_owner_mask()
    c_lin_e_ref = interpolation_savepoint.c_lin_e()
    lateral_boundary, _ = icon_grid.get_indices_from_to(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1,
    )
    c_lin_e = compute_c_lin_e(
        np.asarray(edge_cell_length),
        np.asarray(inv_dual_edge_length),
        np.asarray(owner_mask),
        lateral_boundary,
    )

    assert np.allclose(c_lin_e, c_lin_e_ref)
