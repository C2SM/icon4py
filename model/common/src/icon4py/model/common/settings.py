# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

try:
    import cupy as xp
except ImportError:
    import numpy as xp

    print("cupy not installed, defaulting to numpy")

xp  # noqa: B018
