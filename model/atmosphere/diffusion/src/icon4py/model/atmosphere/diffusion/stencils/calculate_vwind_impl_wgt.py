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


def compute_vwind_impl_wgt_final(vwind_impl_wgt_k: np.array, global_exp: str, experiment: str):
    vwind_impl_wgt = (
        np.amin(vwind_impl_wgt_k, axis=1)
        if experiment == global_exp
        else np.amax(vwind_impl_wgt_k, axis=1)
    )
    return vwind_impl_wgt
