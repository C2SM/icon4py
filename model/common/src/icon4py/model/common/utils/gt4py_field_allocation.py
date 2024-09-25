# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx

from icon4py.model.common import type_alias as ta
from icon4py.model.common.settings import xp


def allocate_zero_field(*dims: gtx.Dimension, grid, is_halfdim=False, dtype=ta.wpfloat):
    shapex = tuple(map(lambda x: grid.size[x], dims))
    if is_halfdim:
        assert len(shapex) == 2
        shapex = (shapex[0], shapex[1] + 1)
    return gtx.as_field(dims, xp.zeros(shapex, dtype=dtype))


def allocate_indices(dim: gtx.Dimension, grid, is_halfdim=False, dtype=gtx.int32):
    shapex = grid.size[dim] + 1 if is_halfdim else grid.size[dim]
    return gtx.as_field((dim,), xp.arange(shapex, dtype=dtype))
