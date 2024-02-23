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
from gt4py.next import as_field
from gt4py.next.common import Dimension, Field
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import (
    abs,
    broadcast,
    int32,
    maximum,
)

from icon4py.model.common.dimension import CellDim, EdgeDim, KDim, VertexDim


def indices_field(dim: Dimension, grid, is_halfdim, dtype=int):
    shapex = grid.size[dim] + 1 if is_halfdim else grid.size[dim]
    return as_field((dim,), np.arange(shapex, dtype=dtype))


def zero_field(grid, *dims: Dimension, is_halfdim=False, dtype=float):
    shapex = tuple(map(lambda x: grid.size[x], dims))
    if is_halfdim:
        assert len(shapex) == 2
        shapex = (shapex[0], shapex[1] + 1)
    return as_field(dims, np.zeros(shapex, dtype=dtype))


def _allocate(*dims: Dimension, grid, is_halfdim=False, dtype=float):
    return zero_field(grid, *dims, is_halfdim=is_halfdim, dtype=dtype)


def _allocate_indices(*dims: Dimension, grid, is_halfdim=False, dtype=int32):
    return indices_field(*dims, grid=grid, is_halfdim=is_halfdim, dtype=dtype)


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


@field_operator
def _calculate_bdy_divdamp(
    scal_divdamp: Field[[KDim], float], nudge_max_coeff: float, dbl_eps: float
) -> Field[[KDim], float]:
    return 0.75 / (nudge_max_coeff + dbl_eps) * abs(scal_divdamp)


@field_operator
def _calculate_scal_divdamp(
    enh_divdamp_fac: Field[[KDim], float],
    divdamp_order: int32,
    mean_cell_area: float,
    divdamp_fac_o2: float,
) -> Field[[KDim], float]:
    enh_divdamp_fac = (
        maximum(0.0, enh_divdamp_fac - 0.25 * divdamp_fac_o2)
        if divdamp_order == 24
        else enh_divdamp_fac
    )
    return -enh_divdamp_fac * mean_cell_area**2


@field_operator
def _calculate_divdamp_fields(
    enh_divdamp_fac: Field[[KDim], float],
    divdamp_order: int32,
    mean_cell_area: float,
    divdamp_fac_o2: float,
    nudge_max_coeff: float,
    dbl_eps: float,
) -> tuple[Field[[KDim], float], Field[[KDim], float]]:
    scal_divdamp = _calculate_scal_divdamp(
        enh_divdamp_fac, divdamp_order, mean_cell_area, divdamp_fac_o2
    )
    bdy_divdamp = _calculate_bdy_divdamp(scal_divdamp, nudge_max_coeff, dbl_eps)
    return (scal_divdamp, bdy_divdamp)


@field_operator
def _compute_z_raylfac(rayleigh_w: Field[[KDim], float], dtime: float) -> Field[[KDim], float]:
    return 1.0 / (1.0 + dtime * rayleigh_w)


@program
def compute_z_raylfac(
    rayleigh_w: Field[[KDim], float], dtime: float, z_raylfac: Field[[KDim], float]
):
    _compute_z_raylfac(rayleigh_w, dtime, out=z_raylfac)
