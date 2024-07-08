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

import gt4py.next as gtx

from icon4py.model.common.settings import xp


def _indices_field(dim: gtx.Dimension, grid, is_halfdim, dtype=int):
    shapex = grid.size[dim] + 1 if is_halfdim else grid.size[dim]
    return gtx.as_field((dim,), xp.arange(shapex, dtype=dtype))


def _zero_field(grid, *dims: gtx.Dimension, is_halfdim=False, dtype=float):
    shapex = tuple(map(lambda x: grid.size[x], dims))
    if is_halfdim:
        assert len(shapex) == 2
        shapex = (shapex[0], shapex[1] + 1)
    return gtx.as_field(dims, xp.zeros(shapex, dtype=dtype))


def allocate_zero_field(*dims: gtx.Dimension, grid, is_halfdim=False, dtype=float):
    return _zero_field(grid, *dims, is_halfdim=is_halfdim, dtype=dtype)


def allocate_indices(*dims: gtx.Dimension, grid, is_halfdim=False, dtype=gtx.int32):
    return _indices_field(*dims, grid=grid, is_halfdim=is_halfdim, dtype=dtype)
