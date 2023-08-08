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


def compute_c_lin_e(
    edge_cell_length: np.array,
    inv_dual_edge_length: np.array,
    owner_mask: np.array,
    second_boundary_layer_start_index: np.int32,
) -> np.array:
    """
    Compute E2C average inverse distance.

    Args:
        edge_cell_length: numpy array, representing a Field[[EdgeDim, E2CDim], float]
        inv_dual_edge_length: inverse dual edge length, numpy array representing a Field[[EdgeDim], float]
        owner_mask: numpy array, representing a Field[[EdgeDim], bool]boolean field, True for all edges owned by this compute node
        second_boundary_layer_start_index: start index of the 2nd boundary line: c_lin_e is not calculated for the first boundary layer

    Returns: c_lin_e: numpy array  representing Field[[EdgeDim, E2CDim], float]

    """

    c_lin_e_ = edge_cell_length[:, 1] * inv_dual_edge_length[:]
    c_lin_e = np.transpose([c_lin_e_, (1.0 - c_lin_e_)])
    c_lin_e[0:second_boundary_layer_start_index, :] = 0.0
    c_lin_e[:, 0] = np.where(owner_mask, c_lin_e[:, 0], 0.0)
    c_lin_e[:, 1] = np.where(owner_mask, c_lin_e[:, 1], 0.0)
    return c_lin_e
