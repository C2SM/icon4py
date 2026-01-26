# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import math

from icon4py.model.common.utils import data_allocation as data_alloc


def compute_sqrt(
    input_val: data_alloc.NDArray,
) -> float:
    """
    Compute the square root of input_val.
    """
    sqrt_val = math.sqrt(input_val)
    return sqrt_val
