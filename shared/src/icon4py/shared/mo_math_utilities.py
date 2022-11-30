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


def gamma_fct(x):
    """Apply gamma function from Numerical Recipes (F77), reformulated to enable inlining and vectorisation."""
    c1 = 76.18009173
    c2 = -86.50532033
    c3 = 24.01409822
    c4 = -1.231739516
    c5 = 0.120858003e-2
    c6 = -0.536382e-5
    stp = 2.50662827465

    tmp = x + 4.5
    p = stp * (
        1.0
        + c1 / x
        + c2 / (x + 1.0)
        + c3 / (x + 2.0)
        + c4 / (x + 3.0)
        + c5 / (x + 4.0)
        + c6 / (x + 5.0)
    )
    return np.exp((x - 0.5) * np.log(tmp) - tmp + np.log(p))
