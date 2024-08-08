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

from icon4py.model.common.dimension import ECDim
from icon4py.model.common.test_utils.helpers import numpy_to_1D_sparse_field


def compute_coeff_gradekin(
    edge_cell_length: np.array,
    inv_dual_edge_length: np.array,
    horizontal_start: float,
    horizontal_end: float,
) -> np.array:
    """
    Compute coefficients for improved calculation of kinetic energy gradient

    Args:
        edge_cell_length: edge_cell_length
        inv_dual_edge_length: inverse of dual_edge_length
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
    """
    coeff_gradekin_0 = np.zeros_like(inv_dual_edge_length)
    coeff_gradekin_1 = np.zeros_like(inv_dual_edge_length)
    for e in range(horizontal_start, horizontal_end):
        coeff_gradekin_0[e] = (
            edge_cell_length[e, 1] / edge_cell_length[e, 0] * inv_dual_edge_length[e]
        )
        coeff_gradekin_1[e] = (
            edge_cell_length[e, 0] / edge_cell_length[e, 1] * inv_dual_edge_length[e]
        )
    coeff_gradekin_full = np.column_stack((coeff_gradekin_0, coeff_gradekin_1))
    return numpy_to_1D_sparse_field(coeff_gradekin_full, ECDim)
