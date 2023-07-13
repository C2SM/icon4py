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

from icon4py.model.atm_dyn_iconam.apply_nabla2_to_w import _apply_nabla2_to_w
from icon4py.model.atm_dyn_iconam.apply_nabla2_to_w_in_upper_damping_layer import (
    _apply_nabla2_to_w_in_upper_damping_layer,
)
from icon4py.model.atm_dyn_iconam.calculate_horizontal_gradients_for_turbulence import (
    _calculate_horizontal_gradients_for_turbulence,
)
from icon4py.model.atm_dyn_iconam.calculate_nabla2_for_w import (
    _calculate_nabla2_for_w,
)
from icon4py.model.common.dimension import C2E2CODim, CellDim, KDim


@field_operator
def _apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulance(
    area: Field[[CellDim], float],
    geofac_n2s: Field[[CellDim, C2E2CODim], float],
    geofac_grg_x: Field[[CellDim, C2E2CODim], float],
    geofac_grg_y: Field[[CellDim, C2E2CODim], float],
    w_old: Field[[CellDim, KDim], float],
    dwdx: Field[[CellDim, KDim], float],
    dwdy: Field[[CellDim, KDim], float],
    diff_multfac_w: float,
    diff_multfac_n2w: Field[[KDim], float],
    vert_idx: Field[[KDim], int32],
    horz_idx: Field[[CellDim], int32],
    nrdmax: int32,
    interior_idx: int32,
    halo_idx: int32,
) -> tuple[
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
]:
    vert_idx = broadcast(vert_idx, (CellDim, KDim))

    dwdx, dwdy = where(
        int32(0) < vert_idx,
        _calculate_horizontal_gradients_for_turbulence(
            w_old, geofac_grg_x, geofac_grg_y
        ),
        (dwdx, dwdy),
    )

    z_nabla2_c = _calculate_nabla2_for_w(w_old, geofac_n2s)

    w = where(
        (interior_idx <= horz_idx) & (horz_idx < halo_idx),
        _apply_nabla2_to_w(area, z_nabla2_c, geofac_n2s, w_old, diff_multfac_w),
        w_old,
    )

    w = where(
        (int32(0) < vert_idx)
        & (vert_idx < nrdmax)
        & (interior_idx <= horz_idx)
        & (horz_idx < halo_idx),
        _apply_nabla2_to_w_in_upper_damping_layer(
            w, diff_multfac_n2w, area, z_nabla2_c
        ),
        w,
    )

    return w, dwdx, dwdy


@program
def apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulance(
    area: Field[[CellDim], float],
    geofac_n2s: Field[[CellDim, C2E2CODim], float],
    geofac_grg_x: Field[[CellDim, C2E2CODim], float],
    geofac_grg_y: Field[[CellDim, C2E2CODim], float],
    w_old: Field[[CellDim, KDim], float],
    w: Field[[CellDim, KDim], float],
    dwdx: Field[[CellDim, KDim], float],
    dwdy: Field[[CellDim, KDim], float],
    diff_multfac_w: float,
    diff_multfac_n2w: Field[[KDim], float],
    vert_idx: Field[[KDim], int32],
    horz_idx: Field[[CellDim], int32],
    nrdmax: int32,
    interior_idx: int32,
    halo_idx: int32,
):
    _apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulance(
        area,
        geofac_n2s,
        geofac_grg_x,
        geofac_grg_y,
        w_old,
        dwdx,
        dwdy,
        diff_multfac_w,
        diff_multfac_n2w,
        vert_idx,
        horz_idx,
        nrdmax,
        interior_idx,
        halo_idx,
        out=(w, dwdx, dwdy),
    )
