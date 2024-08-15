# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import Field, GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import int32, where

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
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


# TODO: this will have to be removed once domain allows for imports
CellDim = dims.CellDim
KDim = dims.KDim


@field_operator
def _fused_velocity_advection_stencil_8_to_13_predictor(
    z_kin_hor_e: fa.EdgeKField[vpfloat],
    e_bln_c_s: Field[[dims.CEDim], wpfloat],
    z_w_concorr_me: fa.EdgeKField[vpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
    w: fa.CellKField[wpfloat],
    z_w_concorr_mc: fa.CellKField[vpfloat],
    w_concorr_c: fa.CellKField[vpfloat],
    z_ekinh: fa.CellKField[vpfloat],
    k: fa.KField[int32],
    nlev: int32,
    nflatlev: int32,
) -> tuple[
    fa.CellKField[vpfloat],
    fa.CellKField[vpfloat],
    fa.CellKField[vpfloat],
]:
    z_ekinh = where(
        k < nlev,
        _interpolate_to_cell_center(z_kin_hor_e, e_bln_c_s),
        z_ekinh,
    )

    z_w_concorr_mc = _interpolate_to_cell_center(z_w_concorr_me, e_bln_c_s)

    w_concorr_c = where(
        nflatlev + 1 <= k < nlev,
        _interpolate_to_half_levels_vp(interpolant=z_w_concorr_mc, wgtfac_c=wgtfac_c),
        w_concorr_c,
    )

    z_w_con_c = where(
        k < nlev,
        _copy_cell_kdim_field_to_vp(w),
        _init_cell_kdim_field_with_zero_vp(),
    )

    z_w_con_c = where(
        nflatlev + 1 <= k < nlev,
        _correct_contravariant_vertical_velocity(z_w_con_c, w_concorr_c),
        z_w_con_c,
    )

    return z_ekinh, w_concorr_c, z_w_con_c


@field_operator
def _fused_velocity_advection_stencil_8_to_13_corrector(
    z_kin_hor_e: fa.EdgeKField[vpfloat],
    e_bln_c_s: Field[[dims.CEDim], wpfloat],
    z_w_concorr_me: fa.EdgeKField[vpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
    w: fa.CellKField[wpfloat],
    z_w_concorr_mc: fa.CellKField[vpfloat],
    w_concorr_c: fa.CellKField[vpfloat],
    z_ekinh: fa.CellKField[vpfloat],
    k: fa.KField[int32],
    nlev: int32,
    nflatlev: int32,
) -> tuple[
    fa.CellKField[vpfloat],
    fa.CellKField[vpfloat],
    fa.CellKField[vpfloat],
]:
    z_ekinh = where(
        k < nlev,
        _interpolate_to_cell_center(z_kin_hor_e, e_bln_c_s),
        z_ekinh,
    )

    z_w_con_c = where(
        k < nlev,
        _copy_cell_kdim_field_to_vp(w),
        _init_cell_kdim_field_with_zero_vp(),
    )

    z_w_con_c = where(
        nflatlev + 1 <= k < nlev,
        _correct_contravariant_vertical_velocity(z_w_con_c, w_concorr_c),
        z_w_con_c,
    )

    return z_ekinh, w_concorr_c, z_w_con_c


@field_operator
def _fused_velocity_advection_stencil_8_to_13(
    z_kin_hor_e: fa.EdgeKField[vpfloat],
    e_bln_c_s: Field[[dims.CEDim], wpfloat],
    z_w_concorr_me: fa.EdgeKField[vpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
    w: fa.CellKField[wpfloat],
    z_w_concorr_mc: fa.CellKField[vpfloat],
    w_concorr_c: fa.CellKField[vpfloat],
    z_ekinh: fa.CellKField[vpfloat],
    k: fa.KField[int32],
    istep: int32,
    nlev: int32,
    nflatlev: int32,
) -> tuple[
    fa.CellKField[vpfloat],
    fa.CellKField[vpfloat],
    fa.CellKField[vpfloat],
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
    z_kin_hor_e: fa.EdgeKField[vpfloat],
    e_bln_c_s: Field[[dims.CEDim], wpfloat],
    z_w_concorr_me: fa.EdgeKField[vpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
    w: fa.CellKField[wpfloat],
    z_w_concorr_mc: fa.CellKField[vpfloat],
    w_concorr_c: fa.CellKField[vpfloat],
    z_ekinh: fa.CellKField[vpfloat],
    k: fa.KField[int32],
    istep: int32,
    nlev: int32,
    nflatlev: int32,
) -> fa.CellKField[vpfloat]:
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


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def fused_velocity_advection_stencil_8_to_13(
    z_kin_hor_e: fa.EdgeKField[vpfloat],
    e_bln_c_s: Field[[dims.CEDim], wpfloat],
    z_w_concorr_me: fa.EdgeKField[vpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
    w: fa.CellKField[wpfloat],
    z_w_concorr_mc: fa.CellKField[vpfloat],
    w_concorr_c: fa.CellKField[vpfloat],
    z_ekinh: fa.CellKField[vpfloat],
    z_w_con_c: fa.CellKField[vpfloat],
    k: fa.KField[int32],
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
