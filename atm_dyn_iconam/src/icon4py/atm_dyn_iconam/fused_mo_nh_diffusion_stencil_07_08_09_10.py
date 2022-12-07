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

from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import (
    Field,
    broadcast,
    int32,
    neighbor_sum,
    where,
)

from icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_07 import (
    _mo_nh_diffusion_stencil_07,
)
from icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_08 import (
    _mo_nh_diffusion_stencil_08,
)
from icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_09 import (
    _mo_nh_diffusion_stencil_09,
)
from icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_10 import (
    _mo_nh_diffusion_stencil_10,
)
from icon4py.common.dimension import C2E2CO, C2E2CODim, CellDim, KDim


@field_operator
def _fused_mo_nh_diffusion_stencil_07_08_09_10(
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
) -> tuple[
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
]:

    vert_idx = broadcast(vert_idx, (CellDim, KDim))

    dwdx, dwdy = where(
        vert_idx > int32(0),
        _mo_nh_diffusion_stencil_08(w_old, geofac_grg_x, geofac_grg_y),
        (dwdx, dwdy),
    )
    num_lev = 10

    # _mo_nh_diffusion_stencil_08(
    #     w_old,
    #     geofac_grg_x,
    #     geofac_grg_y,
    #     out=(dwdx, dwdy),
    #     domain={KDim: (0, num_lev)})

    z_nabla2_c = _mo_nh_diffusion_stencil_07(w_old, geofac_n2s)

    w = where(
        (horz_idx >= interior_idx) & (horz_idx < halo_idx),
        _mo_nh_diffusion_stencil_09(
            area, z_nabla2_c, geofac_n2s, w_old, diff_multfac_w
        ),
        w_old,
    )

    # _mo_nh_diffusion_stencil_07(
    #     area,
    #     z_nabla2_c,
    #     geofac_n2s,
    #     w_old,
    #     diff_multfac_w,
    #     out=w_old,
    #     domain={CellDim: (interior_idx, halo_idx)}
    # )

    w = where(
        (vert_idx > int32(0))
        & (vert_idx < nrdmax)
        & (horz_idx >= interior_idx)
        & (horz_idx < halo_idx),
        _mo_nh_diffusion_stencil_10(w, diff_multfac_n2w, area, z_nabla2_c),
        w,
    )

    # _mo_nh_diffusion_stencil_10(
    #     w_old, diff_multfac_n2w,
    #     area,
    #     z_nabla2_c,
    #     out=w_old,
    #     domain={CellDim: (interior_idx, halo_idx), KDim:(0, num_lev)}
    # )

    w = w_old
    return w, dwdx, dwdy


@program
def fused_mo_nh_diffusion_stencil_07_08_09_10(
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
    _fused_mo_nh_diffusion_stencil_07_08_09_10(
        area,
        geofac_n2s,
        geofac_grg_x,
        geofac_grg_y,
        w_old,
        w,
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
