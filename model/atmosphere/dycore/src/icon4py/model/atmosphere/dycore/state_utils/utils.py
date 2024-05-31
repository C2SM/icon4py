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
from gt4py.next.common import Dimension
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import (
    abs,
    broadcast,
    int32,
    maximum,
)
from model.common.tests import field_type_aliases as fa

from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import wpfloat


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
def _scale_k(field: fa.KfloatField, factor: float) -> fa.KfloatField:
    return field * factor


@program(backend=backend)
def scale_k(field: fa.KfloatField, factor: float, scaled_field: fa.KfloatField):
    _scale_k(field, factor, out=scaled_field)


@field_operator
def _broadcast_zero_to_three_edge_kdim_fields_wp() -> (
    tuple[
        fa.EKwpField,
        fa.EKwpField,
        fa.EKwpField,
    ]
):
    return (
        broadcast(wpfloat("0.0"), (EdgeDim, KDim)),
        broadcast(wpfloat("0.0"), (EdgeDim, KDim)),
        broadcast(wpfloat("0.0"), (EdgeDim, KDim)),
    )


@field_operator
def _calculate_bdy_divdamp(
    scal_divdamp: fa.KfloatField, nudge_max_coeff: float, dbl_eps: float
) -> fa.KfloatField:
    return 0.75 / (nudge_max_coeff + dbl_eps) * abs(scal_divdamp)


@field_operator
def _calculate_scal_divdamp(
    enh_divdamp_fac: fa.KfloatField,
    divdamp_order: int32,
    mean_cell_area: float,
    divdamp_fac_o2: float,
) -> fa.KfloatField:
    enh_divdamp_fac = (
        maximum(0.0, enh_divdamp_fac - 0.25 * divdamp_fac_o2)
        if divdamp_order == 24
        else enh_divdamp_fac
    )
    return -enh_divdamp_fac * mean_cell_area**2


@field_operator
def _calculate_divdamp_fields(
    enh_divdamp_fac: fa.KfloatField,
    divdamp_order: int32,
    mean_cell_area: float,
    divdamp_fac_o2: float,
    nudge_max_coeff: float,
    dbl_eps: float,
) -> tuple[fa.KfloatField, fa.KfloatField]:
    scal_divdamp = _calculate_scal_divdamp(
        enh_divdamp_fac, divdamp_order, mean_cell_area, divdamp_fac_o2
    )
    bdy_divdamp = _calculate_bdy_divdamp(scal_divdamp, nudge_max_coeff, dbl_eps)
    return (scal_divdamp, bdy_divdamp)


@field_operator
def _compute_z_raylfac(rayleigh_w: fa.KfloatField, dtime: float) -> fa.KfloatField:
    return 1.0 / (1.0 + dtime * rayleigh_w)


@program(backend=backend)
def compute_z_raylfac(rayleigh_w: fa.KfloatField, dtime: float, z_raylfac: fa.KfloatField):
    _compute_z_raylfac(rayleigh_w, dtime, out=z_raylfac)
