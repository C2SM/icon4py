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

from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, broadcast, int32, where

from icon4py.atm_dyn_iconam.apply_nabla2_to_w import _apply_nabla2_to_w
from icon4py.atm_dyn_iconam.apply_nabla2_to_w_in_upper_damping_layer import (
    _apply_nabla2_to_w_in_upper_damping_layer,
)
from icon4py.common.dimension import C2E2CODim, CellDim, KDim


@field_operator
def _apply_diffusion_to_w(
    area: Field[[CellDim], float],
    z_nabla2_c: Field[[CellDim, KDim], float],
    geofac_n2s: Field[[CellDim, C2E2CODim], float],
    w: Field[[CellDim, KDim], float],
    diff_multfac_w: float,
    diff_multfac_n2w: Field[[KDim], float],
    vert_idx: Field[[KDim], int32],
    nrdmax: int32,
) -> Field[[CellDim, KDim], float]:

    vert_idx = broadcast(vert_idx, (CellDim, KDim))

    w = _apply_nabla2_to_w(area, z_nabla2_c, geofac_n2s, w, diff_multfac_w)

    w = where(
        (int32(0) < vert_idx)
        & (vert_idx < nrdmax),
        _apply_nabla2_to_w_in_upper_damping_layer(
            w, diff_multfac_n2w, area, z_nabla2_c
        ),
        w,
    )

    return w


@program
def apply_diffusion_to_w(
    area: Field[[CellDim], float],
    z_nabla2_c: Field[[CellDim, KDim], float],
    geofac_n2s: Field[[CellDim, C2E2CODim], float],
    w: Field[[CellDim, KDim], float],
    diff_multfac_w: float,
    diff_multfac_n2w: Field[[KDim], float],
    vert_idx: Field[[KDim], int32],
    nrdmax: int32,
):
    _apply_diffusion_to_w(
        area,
        z_nabla2_c,
        geofac_n2s,
        w,
        diff_multfac_w,
        diff_multfac_n2w,
        vert_idx,
        nrdmax,
        out=w,
    )
