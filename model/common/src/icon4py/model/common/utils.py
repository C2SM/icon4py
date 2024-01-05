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
from gt4py.next import Dimension, Field, as_field, broadcast, field_operator, program
from numpy import int32

from icon4py.model.common.dimension import CellDim, EdgeDim, KDim, VertexDim


def builder(func):
    """Use as decorator on builder functions."""

    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        return self

    return wrapper


@field_operator
def _scale_k(field: Field[[KDim], float], factor: float) -> Field[[KDim], float]:
    return field * factor


@program
def scale_k(field: Field[[KDim], float], factor: float, scaled_field: Field[[KDim], float]):
    _scale_k(field, factor, out=scaled_field)


@field_operator
def _set_zero_v_k() -> Field[[VertexDim, KDim], float]:
    return broadcast(0.0, (VertexDim, KDim))


@program
def set_zero_v_k(field: Field[[VertexDim, KDim], float]):
    _set_zero_v_k(out=field)


def indices_field(dim: Dimension, grid, is_halfdim, dtype=int):
    shapex = grid.size[dim] + 1 if is_halfdim else grid.size[dim]
    return as_field((dim,), np.arange(shapex, dtype=dtype))


def zero_field(grid, *dims: Dimension, is_halfdim=False, dtype=float):
    shapex = tuple(map(lambda x: grid.size[x], dims))
    if is_halfdim:
        assert len(shapex) == 2
        shapex = (shapex[0], shapex[1] + 1)
    return as_field(dims, np.zeros(shapex, dtype=dtype))


@field_operator
def _set_zero_e_k() -> Field[[EdgeDim, KDim], float]:
    return broadcast(0.0, (EdgeDim, KDim))


@program
def set_zero_e_k(
    field: Field[[EdgeDim, KDim], float],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _set_zero_e_k(
        out=field,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def _set_zero_c_k() -> Field[[CellDim, KDim], float]:
    return broadcast(0.0, (CellDim, KDim))


@program
def set_zero_c_k(
    field: Field[[CellDim, KDim], float],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _set_zero_c_k(
        out=field,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
