# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


def compute_starting_index_for_3d_divdamp(scaling_factor_for_3d_divdamp: np.array) -> int:
    return np.min(np.where(scaling_factor_for_3d_divdamp > 0.0))
