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

from icon4py.model.atmosphere.dycore.interpolate_contravatiant_vertical_verlocity_to_full_levels import (
    _interpolate_contravatiant_vertical_verlocity_to_full_levels,
)
from icon4py.model.atmosphere.dycore.compute_advective_vertical_wind_tendency import (
    _compute_advective_vertical_wind_tendency,
)
from icon4py.model.atmosphere.dycore.add_interpolated_horizontal_advection_of_w import (
    _add_interpolated_horizontal_advection_of_w,
)
from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_18 import (
    _mo_velocity_advection_stencil_18,
)
from icon4py.model.common.dimension import C2E2CODim, CEDim, CellDim, EdgeDim, KDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _fused_velocity_advection_stencil_16_to_18(
    z_w_con_c: Field[[CellDim, KDim], vpfloat],
    w: Field[[CellDim, KDim], wpfloat],
    coeff1_dwdz: Field[[CellDim, KDim], vpfloat],
    coeff2_dwdz: Field[[CellDim, KDim], vpfloat],
    ddt_w_adv: Field[[CellDim, KDim], vpfloat],
    e_bln_c_s: Field[[CEDim], wpfloat],
    z_v_grad_w: Field[[EdgeDim, KDim], vpfloat],
    levelmask: Field[[KDim], bool],
    cfl_clipping: Field[[CellDim, KDim], bool],
    owner_mask: Field[[CellDim], bool],
    ddqz_z_half: Field[[CellDim, KDim], vpfloat],
    area: Field[[CellDim], wpfloat],
    geofac_n2s: Field[[CellDim, C2E2CODim], wpfloat],
    cell: Field[[CellDim], int32],
    k: Field[[KDim], int32],
    scalfac_exdiff: wpfloat,
    cfl_w_limit: vpfloat,
    dtime: wpfloat,
    cell_lower_bound: int32,
    cell_upper_bound: int32,
    nlev: int32,
    nrdmax: int32,
    extra_diffu: bool,
) -> Field[[CellDim, KDim], vpfloat]:
    k = broadcast(k, (CellDim, KDim))

    ddt_w_adv = where(
        (cell_lower_bound <= cell < cell_upper_bound) & (int32(1) <= k),
        _compute_advective_vertical_wind_tendency(z_w_con_c, w, coeff1_dwdz, coeff2_dwdz),
        ddt_w_adv,
    )
    ddt_w_adv = where(
        (cell_lower_bound <= cell < cell_upper_bound) & (int32(1) <= k),
        _add_interpolated_horizontal_advection_of_w(e_bln_c_s, z_v_grad_w, ddt_w_adv),
        ddt_w_adv,
    )
    ddt_w_adv = (
        where(
            (cell_lower_bound <= cell < cell_upper_bound)
            & (maximum(2, nrdmax - 2) <= k < nlev - 3),
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
    z_w_con_c: Field[[CellDim, KDim], vpfloat],
    w: Field[[CellDim, KDim], wpfloat],
    coeff1_dwdz: Field[[CellDim, KDim], vpfloat],
    coeff2_dwdz: Field[[CellDim, KDim], vpfloat],
    ddt_w_adv: Field[[CellDim, KDim], vpfloat],
    e_bln_c_s: Field[[CEDim], wpfloat],
    z_v_grad_w: Field[[EdgeDim, KDim], vpfloat],
    levelmask: Field[[KDim], bool],
    cfl_clipping: Field[[CellDim, KDim], bool],
    owner_mask: Field[[CellDim], bool],
    ddqz_z_half: Field[[CellDim, KDim], vpfloat],
    area: Field[[CellDim], wpfloat],
    geofac_n2s: Field[[CellDim, C2E2CODim], wpfloat],
    cell: Field[[CellDim], int32],
    k: Field[[KDim], int32],
    scalfac_exdiff: wpfloat,
    cfl_w_limit: vpfloat,
    dtime: wpfloat,
    cell_lower_bound: int32,
    cell_upper_bound: int32,
    nlev: int32,
    nrdmax: int32,
    lvn_only: bool,
    extra_diffu: bool,
) -> tuple[Field[[CellDim, KDim], vpfloat], Field[[CellDim, KDim], vpfloat]]:
    z_w_con_c_full = _interpolate_contravatiant_vertical_verlocity_to_full_levels(z_w_con_c)
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
            cell,
            k,
            scalfac_exdiff,
            cfl_w_limit,
            dtime,
            cell_lower_bound,
            cell_upper_bound,
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
    z_w_con_c: Field[[CellDim, KDim], vpfloat],
    w: Field[[CellDim, KDim], wpfloat],
    coeff1_dwdz: Field[[CellDim, KDim], vpfloat],
    coeff2_dwdz: Field[[CellDim, KDim], vpfloat],
    ddt_w_adv: Field[[CellDim, KDim], vpfloat],
    e_bln_c_s: Field[[CEDim], wpfloat],
    z_v_grad_w: Field[[EdgeDim, KDim], vpfloat],
    levelmask: Field[[KDim], bool],
    cfl_clipping: Field[[CellDim, KDim], bool],
    owner_mask: Field[[CellDim], bool],
    ddqz_z_half: Field[[CellDim, KDim], vpfloat],
    area: Field[[CellDim], wpfloat],
    geofac_n2s: Field[[CellDim, C2E2CODim], wpfloat],
    z_w_con_c_full: Field[[CellDim, KDim], vpfloat],
    cell: Field[[CellDim], int32],
    k: Field[[KDim], int32],
    scalfac_exdiff: wpfloat,
    cfl_w_limit: vpfloat,
    dtime: wpfloat,
    cell_lower_bound: int32,
    cell_upper_bound: int32,
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
        cell,
        k,
        scalfac_exdiff,
        cfl_w_limit,
        dtime,
        cell_lower_bound,
        cell_upper_bound,
        nlev,
        nrdmax,
        lvn_only,
        extra_diffu,
        out=(z_w_con_c_full, ddt_w_adv),
    )
