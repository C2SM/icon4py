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
from gt4py.next.ffront.fbuiltins import int32, maximum, where

from icon4py.model.atmosphere.dycore.compute_maximum_cfl_and_clip_contravariant_vertical_velocity import (
    _compute_maximum_cfl_and_clip_contravariant_vertical_velocity,
)
from icon4py.model.atmosphere.dycore.copy_cell_kdim_field_to_vp import _copy_cell_kdim_field_to_vp
from icon4py.model.atmosphere.dycore.correct_contravariant_vertical_velocity import (
    _correct_contravariant_vertical_velocity,
)
from icon4py.model.atmosphere.dycore.init_cell_kdim_field_with_zero_vp import (
    _init_cell_kdim_field_with_zero_vp,
)
from icon4py.model.atmosphere.dycore.interpolate_to_cell_center import _interpolate_to_cell_center
from icon4py.model.atmosphere.dycore.interpolate_to_half_levels_vp import (
    _interpolate_to_half_levels_vp,
)
from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import CEDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _fused_velocity_advection_stencil_8_to_14(
    z_kin_hor_e: fa.EdgeKField[vpfloat],
    e_bln_c_s: Field[[CEDim], wpfloat],
    z_w_concorr_me: fa.EdgeKField[vpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
    w: fa.CellKField[wpfloat],
    ddqz_z_half: fa.CellKField[vpfloat],
    cfl_clipping: fa.CellKField[bool],
    pre_levelmask: fa.CellKField[bool],
    vcfl: fa.CellKField[vpfloat],
    z_w_concorr_mc: fa.CellKField[vpfloat],
    w_concorr_c: fa.CellKField[vpfloat],
    z_ekinh: fa.CellKField[vpfloat],
    k: fa.KField[int32],
    istep: int32,
    cfl_w_limit: vpfloat,
    dtime: wpfloat,
    nlevp1: int32,
    nlev: int32,
    nflatlev: int32,
    nrdmax: int32,
) -> tuple[
    fa.CellKField[vpfloat],
    fa.CellKField[bool],
    fa.CellKField[bool],
    fa.CellKField[vpfloat],
    fa.CellKField[vpfloat],
]:
    z_ekinh = where(
        k < nlev,
        _interpolate_to_cell_center(z_kin_hor_e, e_bln_c_s),
        z_ekinh,
    )

    z_w_concorr_mc = (
        where(
            nflatlev < k < nlev,
            _interpolate_to_cell_center(z_w_concorr_me, e_bln_c_s),
            z_w_concorr_mc,
        )
        if istep == 1
        else z_w_concorr_mc
    )

    w_concorr_c = (
        where(
            nflatlev + 1 < k < nlev,
            _interpolate_to_half_levels_vp(interpolant=z_w_concorr_mc, wgtfac_c=wgtfac_c),
            w_concorr_c,
        )
        if istep == 1
        else w_concorr_c
    )

    z_w_con_c = where(
        k < nlevp1,
        _copy_cell_kdim_field_to_vp(w),
        _init_cell_kdim_field_with_zero_vp(),
    )

    z_w_con_c = where(
        nflatlev + 1 < k < nlev,
        _correct_contravariant_vertical_velocity(z_w_con_c, w_concorr_c),
        z_w_con_c,
    )
    cfl_clipping, vcfl, z_w_con_c = where(
        maximum(3, nrdmax - 2) < k < nlev - 3,
        _compute_maximum_cfl_and_clip_contravariant_vertical_velocity(
            ddqz_z_half, z_w_con_c, cfl_w_limit, dtime
        ),
        (cfl_clipping, vcfl, z_w_con_c),
    )

    return z_ekinh, cfl_clipping, pre_levelmask, vcfl, z_w_con_c


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def fused_velocity_advection_stencil_8_to_14(
    z_kin_hor_e: fa.EdgeKField[vpfloat],
    e_bln_c_s: Field[[CEDim], wpfloat],
    z_w_concorr_me: fa.EdgeKField[vpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
    w: fa.CellKField[wpfloat],
    ddqz_z_half: fa.CellKField[vpfloat],
    cfl_clipping: fa.CellKField[bool],
    pre_levelmask: fa.CellKField[bool],
    vcfl: fa.CellKField[vpfloat],
    z_w_concorr_mc: fa.CellKField[vpfloat],
    w_concorr_c: fa.CellKField[vpfloat],
    z_ekinh: fa.CellKField[vpfloat],
    z_w_con_c: fa.CellKField[vpfloat],
    k: fa.KField[int32],
    istep: int32,
    cfl_w_limit: wpfloat,
    dtime: wpfloat,
    nlevp1: int32,
    nlev: int32,
    nflatlev: int32,
    nrdmax: int32,
):
    _fused_velocity_advection_stencil_8_to_14(
        z_kin_hor_e,
        e_bln_c_s,
        z_w_concorr_me,
        wgtfac_c,
        w,
        ddqz_z_half,
        cfl_clipping,
        pre_levelmask,
        vcfl,
        z_w_concorr_mc,
        w_concorr_c,
        z_ekinh,
        k,
        istep,
        cfl_w_limit,
        dtime,
        nlevp1,
        nlev,
        nflatlev,
        nrdmax,
        out=(z_ekinh, cfl_clipping, pre_levelmask, vcfl, z_w_con_c),
    )
