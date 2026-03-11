# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import math

import numpy as np


def compute_sqrt(
    input_val: np.float64,
) -> np.float64:
    """
    Compute the square root of input_val.
    math.sqrt is not sufficiently typed for the validation happening in the factories.
    """
    return np.float64(math.sqrt(input_val))
