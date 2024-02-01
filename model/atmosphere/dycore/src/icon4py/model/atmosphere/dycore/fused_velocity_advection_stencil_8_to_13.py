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
from icon4py.model.atmosphere.dycore.interpolate_contravariant_correct_to_interface_levels import _interpolate_contravariant_correct_to_interface_levels
from icon4py.model.atmosphere.dycore.copy_cell_kdim_field_to_vp import _copy_cell_kdim_field_to_vp
from icon4py.model.atmosphere.dycore.correct_contravariant_vertical_velocity import (
    _correct_contravariant_vertical_velocity,
)
from icon4py.model.atmosphere.dycore.interpolate_to_cell_center import _interpolate_to_cell_center
from icon4py.model.atmosphere.dycore.set_cell_kdim_field_to_zero_vp import (
    _set_cell_kdim_field_to_zero_vp,
)
from icon4py.model.common.dimension import CEDim, CellDim, EdgeDim, KDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _fused_velocity_advection_stencil_8_to_13_predictor(
    z_kin_hor_e: Field[[EdgeDim, KDim], vpfloat],
    e_bln_c_s: Field[[CEDim], wpfloat],
    z_w_concorr_me: Field[[EdgeDim, KDim], vpfloat],
    wgtfac_c: Field[[CellDim, KDim], vpfloat],
    w: Field[[CellDim, KDim], wpfloat],
    z_w_concorr_mc: Field[[CellDim, KDim], vpfloat],
    w_concorr_c: Field[[CellDim, KDim], vpfloat],
    z_ekinh: Field[[CellDim, KDim], vpfloat],
    k: Field[[KDim], int32],
    nlev: int32,
    nflatlev: int32,
) -> tuple[
    Field[[CellDim, KDim], vpfloat],
    Field[[CellDim, KDim], vpfloat],
    Field[[CellDim, KDim], vpfloat],
]:
    z_ekinh = where(
        k < nlev,
        _interpolate_to_cell_center(z_kin_hor_e, e_bln_c_s),
        z_ekinh,
    )

    z_w_concorr_mc = _interpolate_to_cell_center(z_w_concorr_me, e_bln_c_s)

    w_concorr_c = where(
        nflatlev + 1 <= k < nlev,
        _interpolate_contravariant_correct_to_interface_levels(z_w_concorr_mc, wgtfac_c),
        w_concorr_c,
    )

    z_w_con_c = where(
        k < nlev,
        _copy_cell_kdim_field_to_vp(w),
        _set_cell_kdim_field_to_zero_vp(),
    )

    z_w_con_c = where(
        nflatlev + 1 <= k < nlev,
        _correct_contravariant_vertical_velocity(z_w_con_c, w_concorr_c),
        z_w_con_c,
    )

    return z_ekinh, w_concorr_c, z_w_con_c


@field_operator
def _fused_velocity_advection_stencil_8_to_13_corrector(
    z_kin_hor_e: Field[[EdgeDim, KDim], vpfloat],
    e_bln_c_s: Field[[CEDim], wpfloat],
    z_w_concorr_me: Field[[EdgeDim, KDim], vpfloat],
    wgtfac_c: Field[[CellDim, KDim], vpfloat],
    w: Field[[CellDim, KDim], wpfloat],
    z_w_concorr_mc: Field[[CellDim, KDim], vpfloat],
    w_concorr_c: Field[[CellDim, KDim], vpfloat],
    z_ekinh: Field[[CellDim, KDim], vpfloat],
    k: Field[[KDim], int32],
    nlev: int32,
    nflatlev: int32,
) -> tuple[
    Field[[CellDim, KDim], vpfloat],
    Field[[CellDim, KDim], vpfloat],
    Field[[CellDim, KDim], vpfloat],
]:
    z_ekinh = where(
        k < nlev,
        _interpolate_to_cell_center(z_kin_hor_e, e_bln_c_s),
        z_ekinh,
    )

    z_w_con_c = where(
        k < nlev,
        _copy_cell_kdim_field_to_vp(w),
        _set_cell_kdim_field_to_zero_vp(),
    )

    z_w_con_c = where(
        nflatlev + 1 <= k < nlev,
        _correct_contravariant_vertical_velocity(z_w_con_c, w_concorr_c),
        z_w_con_c,
    )

    return z_ekinh, w_concorr_c, z_w_con_c


@field_operator
def _fused_velocity_advection_stencil_8_to_13(
    z_kin_hor_e: Field[[EdgeDim, KDim], vpfloat],
    e_bln_c_s: Field[[CEDim], wpfloat],
    z_w_concorr_me: Field[[EdgeDim, KDim], vpfloat],
    wgtfac_c: Field[[CellDim, KDim], vpfloat],
    w: Field[[CellDim, KDim], wpfloat],
    z_w_concorr_mc: Field[[CellDim, KDim], vpfloat],
    w_concorr_c: Field[[CellDim, KDim], vpfloat],
    z_ekinh: Field[[CellDim, KDim], vpfloat],
    k: Field[[KDim], int32],
    istep: int32,
    nlev: int32,
    nflatlev: int32,
) -> tuple[
    Field[[CellDim, KDim], vpfloat],
    Field[[CellDim, KDim], vpfloat],
    Field[[CellDim, KDim], vpfloat],
]:

    z_ekinh, w_concorr_c, z_w_con_c = (
        _fused_velocity_advection_stencil_8_to_13_predictor(
            z_kin_hor_e,
            e_bln_c_s,
            z_w_concorr_me,
            wgtfac_c,
            w,
            z_w_concorr_mc,
            w_concorr_c,
            z_ekinh,
            k,
            nlev,
            nflatlev,
        )
        if istep == 1
        else _fused_velocity_advection_stencil_8_to_13_corrector(
            z_kin_hor_e,
            e_bln_c_s,
            z_w_concorr_me,
            wgtfac_c,
            w,
            z_w_concorr_mc,
            w_concorr_c,
            z_ekinh,
            k,
            nlev,
            nflatlev,
        )
    )

    return z_ekinh, w_concorr_c, z_w_con_c


@field_operator
def _fused_velocity_advection_stencil_8_to_13_restricted(
    z_kin_hor_e: Field[[EdgeDim, KDim], vpfloat],
    e_bln_c_s: Field[[CEDim], wpfloat],
    z_w_concorr_me: Field[[EdgeDim, KDim], vpfloat],
    wgtfac_c: Field[[CellDim, KDim], vpfloat],
    w: Field[[CellDim, KDim], wpfloat],
    z_w_concorr_mc: Field[[CellDim, KDim], vpfloat],
    w_concorr_c: Field[[CellDim, KDim], vpfloat],
    z_ekinh: Field[[CellDim, KDim], vpfloat],
    k: Field[[KDim], int32],
    istep: int32,
    nlev: int32,
    nflatlev: int32,
) -> Field[[CellDim, KDim], vpfloat]:

    return _fused_velocity_advection_stencil_8_to_13(
        z_kin_hor_e,
        e_bln_c_s,
        z_w_concorr_me,
        wgtfac_c,
        w,
        z_w_concorr_mc,
        w_concorr_c,
        z_ekinh,
        k,
        istep,
        nlev,
        nflatlev,
    )[2]


@program(grid_type=GridType.UNSTRUCTURED)
def fused_velocity_advection_stencil_8_to_13(
    z_kin_hor_e: Field[[EdgeDim, KDim], vpfloat],
    e_bln_c_s: Field[[CEDim], wpfloat],
    z_w_concorr_me: Field[[EdgeDim, KDim], vpfloat],
    wgtfac_c: Field[[CellDim, KDim], vpfloat],
    w: Field[[CellDim, KDim], wpfloat],
    z_w_concorr_mc: Field[[CellDim, KDim], vpfloat],
    w_concorr_c: Field[[CellDim, KDim], vpfloat],
    z_ekinh: Field[[CellDim, KDim], vpfloat],
    z_w_con_c: Field[[CellDim, KDim], vpfloat],
    k: Field[[KDim], int32],
    istep: int32,
    nlev: int32,
    nflatlev: int32,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _fused_velocity_advection_stencil_8_to_13(
        z_kin_hor_e,
        e_bln_c_s,
        z_w_concorr_me,
        wgtfac_c,
        w,
        z_w_concorr_mc,
        w_concorr_c,
        z_ekinh,
        k,
        istep,
        nlev,
        nflatlev,
        out=(z_ekinh, w_concorr_c, z_w_con_c),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end - 1),
        },
    )
    _fused_velocity_advection_stencil_8_to_13_restricted(
        z_kin_hor_e,
        e_bln_c_s,
        z_w_concorr_me,
        wgtfac_c,
        w,
        z_w_concorr_mc,
        w_concorr_c,
        z_ekinh,
        k,
        istep,
        nlev,
        nflatlev,
        out=z_w_con_c,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_end - 1, vertical_end),
        },
    )
