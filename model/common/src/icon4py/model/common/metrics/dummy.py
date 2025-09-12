# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import numpy as np

import icon4py.model.common.utils.data_allocation as data_alloc


def reduce_scalar_min(ar: data_alloc.NDArray) -> gtx.float:
    while ar.ndim > 0:
        ar = np.min(ar)
    return ar.item()
