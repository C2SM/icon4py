# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import broadcast, maximum, where

from icon4py.model.atmosphere.dycore.stencils.add_extra_diffusion_for_w_con_approaching_cfl import (
    _add_extra_diffusion_for_w_con_approaching_cfl,
)
from icon4py.model.atmosphere.dycore.stencils.add_interpolated_horizontal_advection_of_w import (
    _add_interpolated_horizontal_advection_of_w,
)
from icon4py.model.atmosphere.dycore.stencils.compute_advective_vertical_wind_tendency import (
    _compute_advective_vertical_wind_tendency,
)
from icon4py.model.atmosphere.dycore.stencils.interpolate_contravariant_vertical_velocity_to_full_levels import (
    _interpolate_contravariant_vertical_velocity_to_full_levels,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _fused_velocity_advection_stencil_16_to_18(
    z_w_con_c: fa.CellKField[vpfloat],
    w: fa.CellKField[wpfloat],
    coeff1_dwdz: fa.CellKField[vpfloat],
    coeff2_dwdz: fa.CellKField[vpfloat],
    ddt_w_adv: fa.CellKField[vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], wpfloat],
    z_v_grad_w: fa.EdgeKField[vpfloat],
    levelmask: fa.KField[bool],
    cfl_clipping: fa.CellKField[bool],
    owner_mask: fa.CellField[bool],
    ddqz_z_half: fa.CellKField[vpfloat],
    area: fa.CellField[wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], wpfloat],
    cell: fa.CellField[gtx.int32],
    k: fa.KField[gtx.int32],
    scalfac_exdiff: wpfloat,
    cfl_w_limit: vpfloat,
    dtime: wpfloat,
    cell_lower_bound: gtx.int32,
    cell_upper_bound: gtx.int32,
    nlev: gtx.int32,
    nrdmax: gtx.int32,
) -> fa.CellKField[vpfloat]:
    k = broadcast(k, (dims.CellDim, dims.KDim))

    ddt_w_adv = where(
        (cell_lower_bound <= cell < cell_upper_bound) & (1 <= k),
        _compute_advective_vertical_wind_tendency(z_w_con_c, w, coeff1_dwdz, coeff2_dwdz),
        ddt_w_adv,
    )
    ddt_w_adv = where(
        (cell_lower_bound <= cell < cell_upper_bound) & (1 <= k),
        _add_interpolated_horizontal_advection_of_w(e_bln_c_s, z_v_grad_w, ddt_w_adv),
        ddt_w_adv,
    )
    ddt_w_adv = (
        where(
            (cell_lower_bound <= cell < cell_upper_bound)
            & ((maximum(3, nrdmax - 2) - 1) <= k < nlev - 3),
            _add_extra_diffusion_for_w_con_approaching_cfl(
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
    )

    return ddt_w_adv


@field_operator
def _fused_velocity_advection_stencil_15_to_18(
    z_w_con_c: fa.CellKField[vpfloat],
    w: fa.CellKField[wpfloat],
    coeff1_dwdz: fa.CellKField[vpfloat],
    coeff2_dwdz: fa.CellKField[vpfloat],
    ddt_w_adv: fa.CellKField[vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], wpfloat],
    z_v_grad_w: fa.EdgeKField[vpfloat],
    levelmask: fa.KField[bool],
    cfl_clipping: fa.CellKField[bool],
    owner_mask: fa.CellField[bool],
    ddqz_z_half: fa.CellKField[vpfloat],
    area: fa.CellField[wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], wpfloat],
    cell: fa.CellField[gtx.int32],
    k: fa.KField[gtx.int32],
    z_w_con_c_full: fa.CellKField[wpfloat],
    scalfac_exdiff: wpfloat,
    cfl_w_limit: vpfloat,
    dtime: wpfloat,
    cell_lower_bound: gtx.int32,
    cell_upper_bound: gtx.int32,
    nlev: gtx.int32,
    nrdmax: gtx.int32,
    skip_compute_predictor_vertical_advection: bool,
    start_cell_lateral_boundary: gtx.int32,
    end_cell_halo: gtx.int32,
) -> tuple[fa.CellKField[vpfloat], fa.CellKField[vpfloat]]:
    z_w_con_c_full = where(
        start_cell_lateral_boundary <= cell < end_cell_halo,
        _interpolate_contravariant_vertical_velocity_to_full_levels(z_w_con_c),
        z_w_con_c_full,
    )
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
        )
        if not skip_compute_predictor_vertical_advection
        else ddt_w_adv
    )

    return (z_w_con_c_full, ddt_w_adv)


@program(grid_type=GridType.UNSTRUCTURED)
def fused_velocity_advection_stencil_15_to_18(
    z_w_con_c: fa.CellKField[vpfloat],
    w: fa.CellKField[wpfloat],
    coeff1_dwdz: fa.CellKField[vpfloat],
    coeff2_dwdz: fa.CellKField[vpfloat],
    ddt_w_adv: fa.CellKField[vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], wpfloat],
    z_v_grad_w: fa.EdgeKField[vpfloat],
    levelmask: fa.KField[bool],
    cfl_clipping: fa.CellKField[bool],
    owner_mask: fa.CellField[bool],
    ddqz_z_half: fa.CellKField[vpfloat],
    area: fa.CellField[wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], wpfloat],
    z_w_con_c_full: fa.CellKField[vpfloat],
    cell: fa.CellField[gtx.int32],
    k: fa.KField[gtx.int32],
    scalfac_exdiff: wpfloat,
    cfl_w_limit: vpfloat,
    dtime: wpfloat,
    cell_lower_bound: gtx.int32,
    cell_upper_bound: gtx.int32,
    nlev: gtx.int32,
    nrdmax: gtx.int32,
    skip_compute_predictor_vertical_advection: bool,
    start_cell_lateral_boundary: gtx.int32,
    end_cell_halo: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
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
        z_w_con_c_full,
        scalfac_exdiff,
        cfl_w_limit,
        dtime,
        cell_lower_bound,
        cell_upper_bound,
        nlev,
        nrdmax,
        skip_compute_predictor_vertical_advection,
        start_cell_lateral_boundary,
        end_cell_halo,
        out=(z_w_con_c_full, ddt_w_adv),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
