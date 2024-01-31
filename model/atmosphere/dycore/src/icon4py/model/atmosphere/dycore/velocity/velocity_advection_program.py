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
from gt4py.next.ffront.fbuiltins import int32, where

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
from icon4py.model.atmosphere.dycore.interpolate_to_cell_center import _interpolate_to_cell_center
from icon4py.model.atmosphere.dycore.interpolate_to_half_levels_vp import (
    _interpolate_to_half_levels_vp,
)
from icon4py.model.atmosphere.dycore.set_cell_kdim_field_to_zero_vp import (
    _set_cell_kdim_field_to_zero_vp,
)
from icon4py.model.common.dimension import CEDim, CellDim, EdgeDim, KDim


@field_operator
def _fused_stencils_4_5(
    vn: Field[[EdgeDim, KDim], float],
    vt: Field[[EdgeDim, KDim], float],
    vn_ie: Field[[EdgeDim, KDim], float],
    z_vt_ie: Field[[EdgeDim, KDim], float],
    z_kin_hor_e: Field[[EdgeDim, KDim], float],
    ddxn_z_full: Field[[EdgeDim, KDim], float],
    ddxt_z_full: Field[[EdgeDim, KDim], float],
    z_w_concorr_me: Field[[EdgeDim, KDim], float],
    k_field: Field[[KDim], int32],
    nflatlev_startindex: int32,
    nlev: int32,
) -> tuple[
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
]:
    z_w_concorr_me = where(
        (k_field >= nflatlev_startindex) & (k_field < nlev),
        _compute_contravariant_correction(vn, ddxn_z_full, ddxt_z_full, vt),
        z_w_concorr_me,
    )

    (vn_ie, z_vt_ie, z_kin_hor_e) = where(
        k_field == int32(0),
        _compute_horizontal_kinetic_energy(vn, vt),
        (vn_ie, z_vt_ie, z_kin_hor_e),
    )

    return z_w_concorr_me, vn_ie, z_vt_ie, z_kin_hor_e


@program(grid_type=GridType.UNSTRUCTURED)
def fused_stencils_4_5(
    vn: Field[[EdgeDim, KDim], float],
    vt: Field[[EdgeDim, KDim], float],
    vn_ie: Field[[EdgeDim, KDim], float],
    z_vt_ie: Field[[EdgeDim, KDim], float],
    z_kin_hor_e: Field[[EdgeDim, KDim], float],
    ddxn_z_full: Field[[EdgeDim, KDim], float],
    ddxt_z_full: Field[[EdgeDim, KDim], float],
    z_w_concorr_me: Field[[EdgeDim, KDim], float],
    k_field: Field[[KDim], int32],
    nflatlev_startindex: int32,
    nlev: int32,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
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


@program
def extrapolate_at_top(
    wgtfacq_e: Field[[EdgeDim, KDim], float],
    vn: Field[[EdgeDim, KDim], float],
    vn_ie: Field[[EdgeDim, KDim], float],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
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


@field_operator
def _fused_stencils_9_10(
    z_w_concorr_me: Field[[EdgeDim, KDim], float],
    e_bln_c_s: Field[[CEDim], float],
    local_z_w_concorr_mc: Field[[CellDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    k_field: Field[[KDim], int32],
    nflatlev_startindex: int32,
    nlev: int32,
) -> tuple[Field[[CellDim, KDim], float], Field[[CellDim, KDim], float]]:
    local_z_w_concorr_mc = where(
        (k_field >= nflatlev_startindex) & (k_field < nlev),
        _interpolate_to_cell_center(z_w_concorr_me, e_bln_c_s),
        local_z_w_concorr_mc,
    )

    w_concorr_c = where(
        (k_field >= nflatlev_startindex + int32(1)) & (k_field < nlev),
        _interpolate_to_half_levels_vp(interpolant=local_z_w_concorr_mc, wgtfac_c=wgtfac_c),
        w_concorr_c,
    )

    return local_z_w_concorr_mc, w_concorr_c


@program(grid_type=GridType.UNSTRUCTURED)
def fused_stencils_9_10(
    z_w_concorr_me: Field[[EdgeDim, KDim], float],
    e_bln_c_s: Field[[CEDim], float],
    local_z_w_concorr_mc: Field[[CellDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    k_field: Field[[KDim], int32],
    nflatlev_startindex: int32,
    nlev: int32,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
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


@field_operator
def _fused_stencils_11_to_13(
    w: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    local_z_w_con_c: Field[[CellDim, KDim], float],
    k_field: Field[[KDim], int32],
    nflatlev_startindex: int32,
    nlev: int32,
):
    local_z_w_con_c = where(
        (k_field >= int32(0)) & (k_field < nlev),
        _copy_cell_kdim_field_to_vp(w),
        local_z_w_con_c,
    )

    local_z_w_con_c = where(k_field == nlev, _set_cell_kdim_field_to_zero_vp(), local_z_w_con_c)

    local_z_w_con_c = where(
        (k_field >= (nflatlev_startindex + int32(1))) & (k_field < nlev),
        _correct_contravariant_vertical_velocity(local_z_w_con_c, w_concorr_c),
        local_z_w_con_c,
    )
    return local_z_w_con_c


@program(grid_type=GridType.UNSTRUCTURED)
def fused_stencils_11_to_13(
    w: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    local_z_w_con_c: Field[[CellDim, KDim], float],
    k_field: Field[[KDim], int32],
    nflatlev_startindex: int32,
    nlev: int32,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
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


@field_operator
def _fused_stencil_14(
    local_z_w_con_c: Field[[CellDim, KDim], float],
    ddqz_z_half: Field[[CellDim, KDim], float],
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


@program
def fused_stencil_14(
    local_z_w_con_c: Field[[CellDim, KDim], float],
    ddqz_z_half: Field[[CellDim, KDim], float],
    local_cfl_clipping: Field[[CellDim, KDim], bool],
    local_vcfl: Field[[CellDim, KDim], float],
    cfl_w_limit: float,
    dtime: float,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
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


@field_operator
def _fused_stencils_16_to_17(
    w: Field[[CellDim, KDim], float],
    local_z_v_grad_w: Field[[EdgeDim, KDim], float],
    e_bln_c_s: Field[[CEDim], float],
    local_z_w_con_c: Field[[CellDim, KDim], float],
    coeff1_dwdz: Field[[CellDim, KDim], float],
    coeff2_dwdz: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], float]:
    ddt_w_adv = _compute_advective_vertical_wind_tendency(
        local_z_w_con_c, w, coeff1_dwdz, coeff2_dwdz
    )

    ddt_w_adv = _add_interpolated_horizontal_advection_of_w(e_bln_c_s, local_z_v_grad_w, ddt_w_adv)
    return ddt_w_adv


@program
def fused_stencils_16_to_17(
    w: Field[[CellDim, KDim], float],
    local_z_v_grad_w: Field[[EdgeDim, KDim], float],
    e_bln_c_s: Field[[CEDim], float],
    local_z_w_con_c: Field[[CellDim, KDim], float],
    coeff1_dwdz: Field[[CellDim, KDim], float],
    coeff2_dwdz: Field[[CellDim, KDim], float],
    ddt_w_adv: Field[[CellDim, KDim], float],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
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
