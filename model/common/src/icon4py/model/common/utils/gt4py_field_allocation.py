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
from gt4py.next import backend

from icon4py.model.common import type_alias as ta


def allocate_zero_field(
    *dims: gtx.Dimension,
    grid,
    is_halfdim=False,
    dtype=ta.wpfloat,
    backend: Optional[backend.Backend] = None,
):
    shapex = tuple(map(lambda x: grid.size[x], dims))
    if is_halfdim:
        assert len(shapex) == 2
        shapex = (shapex[0], shapex[1] + 1)
    return gtx.as_field(dims, np.zeros(shapex, dtype=dtype), allocator=backend)


def allocate_indices(
    dim: gtx.Dimension,
    grid,
    is_halfdim=False,
    dtype=gtx.int32,
    backend: Optional[backend.Backend] = None,
):
    shapex = grid.size[dim] + 1 if is_halfdim else grid.size[dim]
    return gtx.as_field((dim,), np.arange(shapex, dtype=dtype), allocator=backend)


def allocate_field_from_array(
    *dims: gtx.Dimension,
    grid,
    input_array: np.ndarray,
    is_halfdim=False,
    dtype=ta.wpfloat,
    backend: Optional[backend.Backend] = None,
):
    shapex = tuple(map(lambda x: grid.size[x], dims))
    if is_halfdim:
        assert len(shapex) == 2
        shapex = (shapex[0], shapex[1] + 1)
    return gtx.as_field(dims, np.zeros(shapex, dtype=dtype), allocator=backend)
