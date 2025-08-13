# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


def dallclose(
    a: np.ndarray, b: np.ndarray, rtol: float = 1.0e-12, atol: float = 0.0, equal_nan: bool = False
) -> bool:
    return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
