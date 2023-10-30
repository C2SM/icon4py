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
from gt4py.next.common import Field, GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import broadcast, int32, maximum, where

from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_15 import (
    _mo_velocity_advection_stencil_15,
)
from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_16 import (
    _mo_velocity_advection_stencil_16,
)
from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_17 import (
    _mo_velocity_advection_stencil_17,
)
from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_18 import (
    _mo_velocity_advection_stencil_18,
)
from icon4py.model.common.dimension import C2E2CODim, C2EDim, CellDim, EdgeDim, KDim


@field_operator
def _fused_velocity_advection_stencil_16_to_18(
    z_w_con_c: Field[[CellDim, KDim], float],
    w: Field[[CellDim, KDim], float],
    coeff1_dwdz: Field[[CellDim, KDim], float],
    coeff2_dwdz: Field[[CellDim, KDim], float],
    ddt_w_adv: Field[[CellDim, KDim], float],
    e_bln_c_s: Field[[CellDim, C2EDim], float],
    z_v_grad_w: Field[[EdgeDim, KDim], float],
    levelmask: Field[[KDim], bool],
    cfl_clipping: Field[[CellDim, KDim], bool],
    owner_mask: Field[[CellDim], bool],
    ddqz_z_half: Field[[CellDim, KDim], float],
    area: Field[[CellDim], float],
    geofac_n2s: Field[[CellDim, C2E2CODim], float],
    horz_idx: Field[[CellDim], int32],
    vert_idx: Field[[KDim], int32],
    scalfac_exdiff: float,
    cfl_w_limit: float,
    dtime: float,
    horz_lower_bound: int32,
    horz_upper_bound: int32,
    nlev: int32,
    nrdmax: int32,
    extra_diffu: bool,
) -> Field[[CellDim, KDim], float]:

    vert_idx = broadcast(vert_idx, (CellDim, KDim))

    ddt_w_adv = where(
        (horz_lower_bound < horz_idx < horz_upper_bound) & (int32(0) < vert_idx),
        _mo_velocity_advection_stencil_16(z_w_con_c, w, coeff1_dwdz, coeff2_dwdz),
        ddt_w_adv,
    )
    ddt_w_adv = where(
        (horz_lower_bound < horz_idx < horz_upper_bound) & (int32(0) < vert_idx),
        _mo_velocity_advection_stencil_17(e_bln_c_s, z_v_grad_w, ddt_w_adv),
        ddt_w_adv,
    )
    ddt_w_adv = (
        where(
            (horz_lower_bound < horz_idx < horz_upper_bound)
            & (maximum(3, nrdmax - 2) < vert_idx < nlev - 4),
            _mo_velocity_advection_stencil_18(
                levelmask,
                cfl_clipping,
                owner_mask,
                z_w_con_c,
                ddqz_z_half,
                area,
                geofac_n2s,
                w,
                ddt_w_adv,
                scalfac_exdiff,
                cfl_w_limit,
                dtime,
            ),
            ddt_w_adv,
        )
        if extra_diffu
        else ddt_w_adv
    )

    return ddt_w_adv


@field_operator
def _fused_velocity_advection_stencil_15_to_18(
    z_w_con_c: Field[[CellDim, KDim], float],
    w: Field[[CellDim, KDim], float],
    coeff1_dwdz: Field[[CellDim, KDim], float],
    coeff2_dwdz: Field[[CellDim, KDim], float],
    ddt_w_adv: Field[[CellDim, KDim], float],
    e_bln_c_s: Field[[CellDim, C2EDim], float],
    z_v_grad_w: Field[[EdgeDim, KDim], float],
    levelmask: Field[[KDim], bool],
    cfl_clipping: Field[[CellDim, KDim], bool],
    owner_mask: Field[[CellDim], bool],
    ddqz_z_half: Field[[CellDim, KDim], float],
    area: Field[[CellDim], float],
    geofac_n2s: Field[[CellDim, C2E2CODim], float],
    horz_idx: Field[[CellDim], int32],
    vert_idx: Field[[KDim], int32],
    scalfac_exdiff: float,
    cfl_w_limit: float,
    dtime: float,
    horz_lower_bound: int32,
    horz_upper_bound: int32,
    nlev: int32,
    nrdmax: int32,
    lvn_only: bool,
    extra_diffu: bool,
) -> tuple[Field[[CellDim, KDim], float], Field[[CellDim, KDim], float]]:

    z_w_con_c_full = _mo_velocity_advection_stencil_15(z_w_con_c)
    ddt_w_adv = (
        _fused_velocity_advection_stencil_16_to_18(
            z_w_con_c,
            w,
            coeff1_dwdz,
            coeff2_dwdz,
            ddt_w_adv,
            e_bln_c_s,
            z_v_grad_w,
            levelmask,
            cfl_clipping,
            owner_mask,
            ddqz_z_half,
            area,
            geofac_n2s,
            horz_idx,
            vert_idx,
            scalfac_exdiff,
            cfl_w_limit,
            dtime,
            horz_lower_bound,
            horz_upper_bound,
            nlev,
            nrdmax,
            extra_diffu,
        )
        if not lvn_only
        else ddt_w_adv
    )

    return (z_w_con_c_full, ddt_w_adv)


@program(grid_type=GridType.UNSTRUCTURED)
def fused_velocity_advection_stencil_15_to_18(
    z_w_con_c: Field[[CellDim, KDim], float],
    w: Field[[CellDim, KDim], float],
    coeff1_dwdz: Field[[CellDim, KDim], float],
    coeff2_dwdz: Field[[CellDim, KDim], float],
    ddt_w_adv: Field[[CellDim, KDim], float],
    e_bln_c_s: Field[[CellDim, C2EDim], float],
    z_v_grad_w: Field[[EdgeDim, KDim], float],
    levelmask: Field[[KDim], bool],
    cfl_clipping: Field[[CellDim, KDim], bool],
    owner_mask: Field[[CellDim], bool],
    ddqz_z_half: Field[[CellDim, KDim], float],
    area: Field[[CellDim], float],
    geofac_n2s: Field[[CellDim, C2E2CODim], float],
    z_w_con_c_full: Field[[CellDim, KDim], float],
    horz_idx: Field[[CellDim], int32],
    vert_idx: Field[[KDim], int32],
    scalfac_exdiff: float,
    cfl_w_limit: float,
    dtime: float,
    horz_lower_bound: int32,
    horz_upper_bound: int32,
    nlev: int32,
    nrdmax: int32,
    lvn_only: bool,
    extra_diffu: bool,
):
    _fused_velocity_advection_stencil_15_to_18(
        z_w_con_c,
        w,
        coeff1_dwdz,
        coeff2_dwdz,
        ddt_w_adv,
        e_bln_c_s,
        z_v_grad_w,
        levelmask,
        cfl_clipping,
        owner_mask,
        ddqz_z_half,
        area,
        geofac_n2s,
        horz_idx,
        vert_idx,
        scalfac_exdiff,
        cfl_w_limit,
        dtime,
        horz_lower_bound,
        horz_upper_bound,
        nlev,
        nrdmax,
        lvn_only,
        extra_diffu,
        out=(z_w_con_c_full, ddt_w_adv),
    )
