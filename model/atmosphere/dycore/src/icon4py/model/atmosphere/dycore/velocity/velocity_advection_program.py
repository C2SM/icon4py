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
import gt4py.next as gtx

from icon4py.model.atmosphere.dycore.add_interpolated_horizontal_advection_of_w import (
    _add_interpolated_horizontal_advection_of_w,
)
from icon4py.model.atmosphere.dycore.compute_advective_vertical_wind_tendency import (
    _compute_advective_vertical_wind_tendency,
)
from icon4py.model.atmosphere.dycore.compute_contravariant_correction import (
    _compute_contravariant_correction,
)
from icon4py.model.atmosphere.dycore.compute_horizontal_kinetic_energy import (
    _compute_horizontal_kinetic_energy,
)
from icon4py.model.atmosphere.dycore.compute_maximum_cfl_and_clip_contravariant_vertical_velocity import (
    _compute_maximum_cfl_and_clip_contravariant_vertical_velocity,
)
from icon4py.model.atmosphere.dycore.copy_cell_kdim_field_to_vp import _copy_cell_kdim_field_to_vp
from icon4py.model.atmosphere.dycore.correct_contravariant_vertical_velocity import (
    _correct_contravariant_vertical_velocity,
)
from icon4py.model.atmosphere.dycore.extrapolate_at_top import _extrapolate_at_top
from icon4py.model.atmosphere.dycore.init_cell_kdim_field_with_zero_vp import (
    _init_cell_kdim_field_with_zero_vp,
)
from icon4py.model.atmosphere.dycore.interpolate_to_cell_center import _interpolate_to_cell_center
from icon4py.model.atmosphere.dycore.interpolate_to_half_levels_vp import (
    _interpolate_to_half_levels_vp,
)
from icon4py.model.common.dimension import CEDim, CellDim, EdgeDim, KDim
from icon4py.model.common.settings import backend


@gtx.field_operator
def _fused_stencils_4_5(
    vn: gtx.Field[[EdgeDim, KDim], float],
    vt: gtx.Field[[EdgeDim, KDim], float],
    vn_ie: gtx.Field[[EdgeDim, KDim], float],
    z_vt_ie: gtx.Field[[EdgeDim, KDim], float],
    z_kin_hor_e: gtx.Field[[EdgeDim, KDim], float],
    ddxn_z_full: gtx.Field[[EdgeDim, KDim], float],
    ddxt_z_full: gtx.Field[[EdgeDim, KDim], float],
    z_w_concorr_me: gtx.Field[[EdgeDim, KDim], float],
    k_field: gtx.Field[[KDim], gtx.int32],
    nflatlev_startindex: gtx.int32,
    nlev: gtx.int32,
) -> tuple[
    gtx.Field[[EdgeDim, KDim], float],
    gtx.Field[[EdgeDim, KDim], float],
    gtx.Field[[EdgeDim, KDim], float],
    gtx.Field[[EdgeDim, KDim], float],
]:
    z_w_concorr_me = gtx.where(
        (k_field >= nflatlev_startindex) & (k_field < nlev),
        _compute_contravariant_correction(vn, ddxn_z_full, ddxt_z_full, vt),
        z_w_concorr_me,
    )

    (vn_ie, z_vt_ie, z_kin_hor_e) = gtx.where(
        k_field == 0,
        _compute_horizontal_kinetic_energy(vn, vt),
        (vn_ie, z_vt_ie, z_kin_hor_e),
    )

    return z_w_concorr_me, vn_ie, z_vt_ie, z_kin_hor_e


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def fused_stencils_4_5(
    vn: gtx.Field[[EdgeDim, KDim], float],
    vt: gtx.Field[[EdgeDim, KDim], float],
    vn_ie: gtx.Field[[EdgeDim, KDim], float],
    z_vt_ie: gtx.Field[[EdgeDim, KDim], float],
    z_kin_hor_e: gtx.Field[[EdgeDim, KDim], float],
    ddxn_z_full: gtx.Field[[EdgeDim, KDim], float],
    ddxt_z_full: gtx.Field[[EdgeDim, KDim], float],
    z_w_concorr_me: gtx.Field[[EdgeDim, KDim], float],
    k_field: gtx.Field[[KDim], gtx.int32],
    nflatlev_startindex: gtx.int32,
    nlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _fused_stencils_4_5(
        vn,
        vt,
        vn_ie,
        z_vt_ie,
        z_kin_hor_e,
        ddxn_z_full,
        ddxt_z_full,
        z_w_concorr_me,
        k_field,
        nflatlev_startindex,
        nlev,
        out=(z_w_concorr_me, vn_ie, z_vt_ie, z_kin_hor_e),
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def extrapolate_at_top(
    wgtfacq_e: gtx.Field[[EdgeDim, KDim], float],
    vn: gtx.Field[[EdgeDim, KDim], float],
    vn_ie: gtx.Field[[EdgeDim, KDim], float],
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
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _fused_stencils_9_10(
    z_w_concorr_me: gtx.Field[[EdgeDim, KDim], float],
    e_bln_c_s: gtx.Field[[CEDim], float],
    local_z_w_concorr_mc: gtx.Field[[CellDim, KDim], float],
    wgtfac_c: gtx.Field[[CellDim, KDim], float],
    w_concorr_c: gtx.Field[[CellDim, KDim], float],
    k_field: gtx.Field[[KDim], gtx.int32],
    nflatlev_startindex: gtx.int32,
    nlev: gtx.int32,
) -> tuple[gtx.Field[[CellDim, KDim], float], gtx.Field[[CellDim, KDim], float]]:
    local_z_w_concorr_mc = gtx.where(
        (k_field >= nflatlev_startindex) & (k_field < nlev),
        _interpolate_to_cell_center(z_w_concorr_me, e_bln_c_s),
        local_z_w_concorr_mc,
    )

    w_concorr_c = gtx.where(
        (k_field >= nflatlev_startindex + 1) & (k_field < nlev),
        _interpolate_to_half_levels_vp(interpolant=local_z_w_concorr_mc, wgtfac_c=wgtfac_c),
        w_concorr_c,
    )

    return local_z_w_concorr_mc, w_concorr_c


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def fused_stencils_9_10(
    z_w_concorr_me: gtx.Field[[EdgeDim, KDim], float],
    e_bln_c_s: gtx.Field[[CEDim], float],
    local_z_w_concorr_mc: gtx.Field[[CellDim, KDim], float],
    wgtfac_c: gtx.Field[[CellDim, KDim], float],
    w_concorr_c: gtx.Field[[CellDim, KDim], float],
    k_field: gtx.Field[[KDim], gtx.int32],
    nflatlev_startindex: gtx.int32,
    nlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _fused_stencils_9_10(
        z_w_concorr_me,
        e_bln_c_s,
        local_z_w_concorr_mc,
        wgtfac_c,
        w_concorr_c,
        k_field,
        nflatlev_startindex,
        nlev,
        out=(local_z_w_concorr_mc, w_concorr_c),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _fused_stencils_11_to_13(
    w: gtx.Field[[CellDim, KDim], float],
    w_concorr_c: gtx.Field[[CellDim, KDim], float],
    local_z_w_con_c: gtx.Field[[CellDim, KDim], float],
    k_field: gtx.Field[[KDim], gtx.int32],
    nflatlev_startindex: gtx.int32,
    nlev: gtx.int32,
):
    local_z_w_con_c = gtx.where(
        (k_field >= 0) & (k_field < nlev),
        _copy_cell_kdim_field_to_vp(w),
        local_z_w_con_c,
    )

    local_z_w_con_c = gtx.where(
        k_field == nlev, _init_cell_kdim_field_with_zero_vp(), local_z_w_con_c
    )

    local_z_w_con_c = gtx.where(
        (k_field >= (nflatlev_startindex + 1)) & (k_field < nlev),
        _correct_contravariant_vertical_velocity(local_z_w_con_c, w_concorr_c),
        local_z_w_con_c,
    )
    return local_z_w_con_c


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def fused_stencils_11_to_13(
    w: gtx.Field[[CellDim, KDim], float],
    w_concorr_c: gtx.Field[[CellDim, KDim], float],
    local_z_w_con_c: gtx.Field[[CellDim, KDim], float],
    k_field: gtx.Field[[KDim], gtx.int32],
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
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _fused_stencil_14(
    local_z_w_con_c: gtx.Field[[CellDim, KDim], float],
    ddqz_z_half: gtx.Field[[CellDim, KDim], float],
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


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def fused_stencil_14(
    local_z_w_con_c: gtx.Field[[CellDim, KDim], float],
    ddqz_z_half: gtx.Field[[CellDim, KDim], float],
    local_cfl_clipping: gtx.Field[[CellDim, KDim], bool],
    local_vcfl: gtx.Field[[CellDim, KDim], float],
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
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _fused_stencils_16_to_17(
    w: gtx.Field[[CellDim, KDim], float],
    local_z_v_grad_w: gtx.Field[[EdgeDim, KDim], float],
    e_bln_c_s: gtx.Field[[CEDim], float],
    local_z_w_con_c: gtx.Field[[CellDim, KDim], float],
    coeff1_dwdz: gtx.Field[[CellDim, KDim], float],
    coeff2_dwdz: gtx.Field[[CellDim, KDim], float],
) -> gtx.Field[[CellDim, KDim], float]:
    ddt_w_adv = _compute_advective_vertical_wind_tendency(
        local_z_w_con_c, w, coeff1_dwdz, coeff2_dwdz
    )

    ddt_w_adv = _add_interpolated_horizontal_advection_of_w(e_bln_c_s, local_z_v_grad_w, ddt_w_adv)
    return ddt_w_adv


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def fused_stencils_16_to_17(
    w: gtx.Field[[CellDim, KDim], float],
    local_z_v_grad_w: gtx.Field[[EdgeDim, KDim], float],
    e_bln_c_s: gtx.Field[[CEDim], float],
    local_z_w_con_c: gtx.Field[[CellDim, KDim], float],
    coeff1_dwdz: gtx.Field[[CellDim, KDim], float],
    coeff2_dwdz: gtx.Field[[CellDim, KDim], float],
    ddt_w_adv: gtx.Field[[CellDim, KDim], float],
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
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
