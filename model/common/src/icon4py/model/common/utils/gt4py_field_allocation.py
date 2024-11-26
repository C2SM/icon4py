# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Optional

import gt4py.next as gtx
import numpy as np
from gt4py.next import backend as gt4py_backend, common as gt4py_common

from icon4py.model.common import dimension as dims, type_alias as ta


def allocate_zero_field(
    *dim: gtx.Dimension,
    grid,
    is_halfdim=False,
    dtype=ta.wpfloat,
    backend: Optional[gt4py_backend.Backend] = None,
) -> gt4py_common.Field:
    def size(local_dim: gtx.Dimension, is_half_dim: bool) -> int:
        if local_dim == dims.KDim and is_half_dim:
            return grid.size[local_dim] + 1
        else:
            return grid.size[local_dim]

    dimensions = {d: range(size(d, is_halfdim)) for d in dim}
    return gtx.zeros(dimensions, dtype=dtype, allocator=backend)


def allocate_indices(
    dim: gtx.Dimension,
    grid,
    is_halfdim=False,
    dtype=gtx.int32,
    backend: Optional[gt4py_backend.Backend] = None,
) -> gt4py_common.Field:
    shapex = grid.size[dim] + 1 if is_halfdim else grid.size[dim]
    return gtx.as_field((dim,), np.arange(shapex, dtype=dtype), allocator=backend)
