# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from types import ModuleType

import numpy as np

from icon4py.model.common import dimension as dims
from icon4py.model.common.utils import data_allocation as data_alloc


def compute_coeff_gradekin(
    edge_cell_length: data_alloc.NDArray,
    inv_dual_edge_length: data_alloc.NDArray,
    horizontal_start: int,
    horizontal_end: int,
    array_ns: ModuleType = np,
) -> data_alloc.NDArray:
    """
    Compute coefficients for improved calculation of kinetic energy gradient

    Args:
        edge_cell_length: edge_cell_length
        inv_dual_edge_length: inverse of dual_edge_length
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
    """
    coeff_gradekin_0 = array_ns.zeros_like(inv_dual_edge_length)
    coeff_gradekin_1 = array_ns.zeros_like(inv_dual_edge_length)
    for e in range(horizontal_start, horizontal_end):
        coeff_gradekin_0[e] = (
            edge_cell_length[e, 1] / edge_cell_length[e, 0] * inv_dual_edge_length[e]
        )
        coeff_gradekin_1[e] = (
            edge_cell_length[e, 0] / edge_cell_length[e, 1] * inv_dual_edge_length[e]
        )
    coeff_gradekin_full = array_ns.column_stack((coeff_gradekin_0, coeff_gradekin_1))
    return data_alloc.numpy_to_1D_sparse_field(coeff_gradekin_full, dims.ECDim).asnumpy()
