# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.ffront.fbuiltins import where

from icon4py.model.atmosphere.dycore.stencils.add_interpolated_horizontal_advection_of_w import (
    _add_interpolated_horizontal_advection_of_w,
)
from icon4py.model.atmosphere.dycore.stencils.compute_advective_vertical_wind_tendency import (
    _compute_advective_vertical_wind_tendency,
)
from icon4py.model.atmosphere.dycore.stencils.compute_contravariant_correction import (
    _compute_contravariant_correction,
)
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_kinetic_energy import (
    _compute_horizontal_kinetic_energy,
)
from icon4py.model.atmosphere.dycore.stencils.compute_maximum_cfl_and_clip_contravariant_vertical_velocity import (
    _compute_maximum_cfl_and_clip_contravariant_vertical_velocity,
)
from icon4py.model.atmosphere.dycore.stencils.copy_cell_kdim_field_to_vp import (
    _copy_cell_kdim_field_to_vp,
)
from icon4py.model.atmosphere.dycore.stencils.correct_contravariant_vertical_velocity import (
    _correct_contravariant_vertical_velocity,
)
from icon4py.model.atmosphere.dycore.stencils.extrapolate_at_top import _extrapolate_at_top
from icon4py.model.atmosphere.dycore.stencils.init_cell_kdim_field_with_zero_vp import (
    _init_cell_kdim_field_with_zero_vp,
)
from icon4py.model.atmosphere.dycore.stencils.interpolate_to_cell_center import (
    _interpolate_to_cell_center,
)
from icon4py.model.atmosphere.dycore.stencils.interpolate_to_half_levels_vp import (
    _interpolate_to_half_levels_vp,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa


@gtx.field_operator
def _fused_stencils_4_5(
    vn: fa.EdgeKField[float],
    vt: fa.EdgeKField[float],
    vn_ie: fa.EdgeKField[float],
    z_vt_ie: fa.EdgeKField[float],
    z_kin_hor_e: fa.EdgeKField[float],
    ddxn_z_full: fa.EdgeKField[float],
    ddxt_z_full: fa.EdgeKField[float],
    z_w_concorr_me: fa.EdgeKField[float],
    k_field: fa.KField[gtx.int32],
    nflatlev_startindex: gtx.int32,
    nlev: gtx.int32,
) -> tuple[
    fa.EdgeKField[float],
    fa.EdgeKField[float],
    fa.EdgeKField[float],
    fa.EdgeKField[float],
]:
    z_w_concorr_me = where(
        (k_field >= nflatlev_startindex) & (k_field < nlev),
        _compute_contravariant_correction(vn, ddxn_z_full, ddxt_z_full, vt),
        z_w_concorr_me,
    )

    (vn_ie, z_vt_ie, z_kin_hor_e) = where(
        k_field == 0,
        _compute_horizontal_kinetic_energy(vn, vt),
        (vn_ie, z_vt_ie, z_kin_hor_e),
    )

    return z_w_concorr_me, vn_ie, z_vt_ie, z_kin_hor_e


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def fused_stencils_4_5(
    vn: fa.EdgeKField[float],
    vt: fa.EdgeKField[float],
    vn_ie: fa.EdgeKField[float],
    z_vt_ie: fa.EdgeKField[float],
    z_kin_hor_e: fa.EdgeKField[float],
    ddxn_z_full: fa.EdgeKField[float],
    ddxt_z_full: fa.EdgeKField[float],
    z_w_concorr_me: fa.EdgeKField[float],
    k_field: fa.KField[gtx.int32],
    nflatlev_startindex: gtx.int32,
    nlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_contravariant_correction(
        vn,
        ddxn_z_full,
        ddxt_z_full,
        vt,
        out=z_w_concorr_me,
        domain = {
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (nflatlev_startindex, vertical_end),
        },
    )
    _compute_horizontal_kinetic_energy(
        vn,
        vt,
        out = (vn_ie, z_vt_ie, z_kin_hor_e),
        domain = {
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_start + 1),
        },
    )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def extrapolate_at_top(
    wgtfacq_e: fa.EdgeKField[float],
    vn: fa.EdgeKField[float],
    vn_ie: fa.EdgeKField[float],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _extrapolate_at_top(
        wgtfacq_e,
        vn,
        out=vn_ie,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _fused_stencils_9_10(
    z_w_concorr_me: fa.EdgeKField[float],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], float],
    local_z_w_concorr_mc: fa.CellKField[float],
    wgtfac_c: fa.CellKField[float],
    w_concorr_c: fa.CellKField[float],
    k_field: fa.KField[gtx.int32],
    nflatlev_startindex: gtx.int32,
    nlev: gtx.int32,
) -> tuple[fa.CellKField[float], fa.CellKField[float]]:
    local_z_w_concorr_mc = where(
        (k_field >= nflatlev_startindex) & (k_field < nlev),
        _interpolate_to_cell_center(z_w_concorr_me, e_bln_c_s),
        local_z_w_concorr_mc,
    )

    w_concorr_c = where(
        (k_field >= nflatlev_startindex + 1) & (k_field < nlev),
        _interpolate_to_half_levels_vp(interpolant=local_z_w_concorr_mc, wgtfac_c=wgtfac_c),
        w_concorr_c,
    )

    return local_z_w_concorr_mc, w_concorr_c


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def fused_stencils_9_10(
    z_w_concorr_me: fa.EdgeKField[float],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], float],
    local_z_w_concorr_mc: fa.CellKField[float],
    wgtfac_c: fa.CellKField[float],
    w_concorr_c: fa.CellKField[float],
    k_field: fa.KField[gtx.int32],
    nflatlev_startindex: gtx.int32,
    nlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _interpolate_to_cell_center(
        z_w_concorr_me,
        e_bln_c_s,
        out=local_z_w_concorr_mc,
        domain = {
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (nflatlev_startindex, vertical_end),
        },
    )

    _interpolate_to_half_levels_vp(
        interpolant=local_z_w_concorr_mc,
        wgtfac_c=wgtfac_c,
        out=w_concorr_c,
        domain = {
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (nflatlev_startindex + 1, vertical_end),
        },
    )


@gtx.field_operator
def _fused_stencils_11_to_13(
    w: fa.CellKField[float],
    w_concorr_c: fa.CellKField[float],
    local_z_w_con_c: fa.CellKField[float],
    k_field: fa.KField[gtx.int32],
    nflatlev_startindex: gtx.int32,
    nlev: gtx.int32,
):
    local_z_w_con_c = where(
        (k_field >= 0) & (k_field < nlev),
        _copy_cell_kdim_field_to_vp(w),
        local_z_w_con_c,
    )

    local_z_w_con_c = where(k_field == nlev, _init_cell_kdim_field_with_zero_vp(), local_z_w_con_c)

    local_z_w_con_c = where(
        (k_field >= (nflatlev_startindex + 1)) & (k_field < nlev),
        _correct_contravariant_vertical_velocity(local_z_w_con_c, w_concorr_c),
        local_z_w_con_c,
    )
    return local_z_w_con_c


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def fused_stencils_11_to_13(
    w: fa.CellKField[float],
    w_concorr_c: fa.CellKField[float],
    local_z_w_con_c: fa.CellKField[float],
    k_field: fa.KField[gtx.int32],
    nflatlev_startindex: gtx.int32,
    nlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _fused_stencils_11_to_13(
        w,
        w_concorr_c,
        local_z_w_con_c,
        k_field,
        nflatlev_startindex,
        nlev,
        out=local_z_w_con_c,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _fused_stencil_14(
    local_z_w_con_c: fa.CellKField[float],
    ddqz_z_half: fa.CellKField[float],
    cfl_w_limit: float,
    dtime: float,
):
    (
        local_cfl_clipping,
        local_vcfl,
        local_z_w_con_c,
    ) = _compute_maximum_cfl_and_clip_contravariant_vertical_velocity(
        ddqz_z_half,
        local_z_w_con_c,
        cfl_w_limit,
        dtime,
    )

    return local_cfl_clipping, local_vcfl, local_z_w_con_c


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def fused_stencil_14(
    local_z_w_con_c: fa.CellKField[float],
    ddqz_z_half: fa.CellKField[float],
    local_cfl_clipping: fa.CellKField[bool],
    local_vcfl: fa.CellKField[float],
    cfl_w_limit: float,
    dtime: float,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _fused_stencil_14(
        local_z_w_con_c,
        ddqz_z_half,
        cfl_w_limit,
        dtime,
        out=(local_cfl_clipping, local_vcfl, local_z_w_con_c),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _fused_stencils_16_to_17(
    w: fa.CellKField[float],
    local_z_v_grad_w: fa.EdgeKField[float],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], float],
    local_z_w_con_c: fa.CellKField[float],
    coeff1_dwdz: fa.CellKField[float],
    coeff2_dwdz: fa.CellKField[float],
) -> fa.CellKField[float]:
    ddt_w_adv = _compute_advective_vertical_wind_tendency(
        local_z_w_con_c, w, coeff1_dwdz, coeff2_dwdz
    )

    ddt_w_adv = _add_interpolated_horizontal_advection_of_w(e_bln_c_s, local_z_v_grad_w, ddt_w_adv)
    return ddt_w_adv


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def fused_stencils_16_to_17(
    w: fa.CellKField[float],
    local_z_v_grad_w: fa.EdgeKField[float],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], float],
    local_z_w_con_c: fa.CellKField[float],
    coeff1_dwdz: fa.CellKField[float],
    coeff2_dwdz: fa.CellKField[float],
    ddt_w_adv: fa.CellKField[float],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _fused_stencils_16_to_17(
        w,
        local_z_v_grad_w,
        e_bln_c_s,
        local_z_w_con_c,
        coeff1_dwdz,
        coeff2_dwdz,
        out=ddt_w_adv,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
