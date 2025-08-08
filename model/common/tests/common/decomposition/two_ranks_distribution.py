# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


TWO_RANKS_DISTRIBUTION: np.ndarray = np.ones(10)
TWO_RANKS_DISTRIBUTION[5:7, 10] = 0


# TODO define all the rest
