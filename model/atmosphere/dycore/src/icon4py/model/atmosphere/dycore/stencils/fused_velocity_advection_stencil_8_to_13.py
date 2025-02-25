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
from gt4py.next.ffront.fbuiltins import astype, broadcast, where

from icon4py.model.atmosphere.dycore.stencils.correct_contravariant_vertical_velocity import (
    _correct_contravariant_vertical_velocity,
)
from icon4py.model.atmosphere.dycore.stencils.interpolate_to_cell_center import (
    _interpolate_to_cell_center,
)
from icon4py.model.atmosphere.dycore.stencils.interpolate_to_half_levels_vp import (
    _interpolate_to_half_levels_vp,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _fused_velocity_advection_stencil_8_to_13_predictor(
    z_kin_hor_e: fa.EdgeKField[vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], wpfloat],
    z_w_concorr_me: fa.EdgeKField[vpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
    w: fa.CellKField[wpfloat],
    z_w_concorr_mc: fa.CellKField[vpfloat],
    z_w_con_c: fa.CellKField[vpfloat],
    w_concorr_c: fa.CellKField[vpfloat],
    z_ekinh: fa.CellKField[vpfloat],
    k: fa.KField[gtx.int32],
    nlev: gtx.int32,
    nflatlev: gtx.int32,
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

    z_w_concorr_mc = where(
        (k >= nflatlev) & (k < nlev),
        _interpolate_to_cell_center(z_w_concorr_me, e_bln_c_s),
        z_w_concorr_mc,
    )

    w_concorr_c = where(
        nflatlev + 1 <= k < nlev,
        _interpolate_to_half_levels_vp(wgtfac_c=wgtfac_c, interpolant=z_w_concorr_mc),
        w_concorr_c,
    )

    z_w_con_c = where(
        k < nlev, astype(w, vpfloat), broadcast(vpfloat("0.0"), (dims.CellDim, dims.KDim))
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
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], wpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
    w: fa.CellKField[wpfloat],
    z_w_con_c: fa.CellKField[vpfloat],
    w_concorr_c: fa.CellKField[vpfloat],
    z_ekinh: fa.CellKField[vpfloat],
    k: fa.KField[gtx.int32],
    nlev: gtx.int32,
    nflatlev: gtx.int32,
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
        astype(w, vpfloat),
        z_w_con_c,
    )

    z_w_con_c = where(k == nlev, 0.0, z_w_con_c)

    z_w_con_c = where(
        nflatlev + 1 <= k < nlev,
        _correct_contravariant_vertical_velocity(z_w_con_c, w_concorr_c),
        z_w_con_c,
    )

    return z_ekinh, w_concorr_c, z_w_con_c


@field_operator
def _fused_velocity_advection_stencil_8_to_13(
    z_kin_hor_e: fa.EdgeKField[vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], wpfloat],
    z_w_concorr_me: fa.EdgeKField[vpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
    w: fa.CellKField[wpfloat],
    z_w_concorr_mc: fa.CellKField[vpfloat],
    z_w_con_c: fa.CellKField[vpfloat],
    w_concorr_c: fa.CellKField[vpfloat],
    z_ekinh: fa.CellKField[vpfloat],
    k: fa.KField[gtx.int32],
    cell: fa.CellField[gtx.int32],
    istep: gtx.int32,
    nlev: gtx.int32,
    nflatlev: gtx.int32,
    lateral_boundary_3: gtx.int32,
    lateral_boundary_4: gtx.int32,
    end_halo: gtx.int32,
) -> tuple[
    fa.CellKField[vpfloat],
    fa.CellKField[vpfloat],
    fa.CellKField[vpfloat],
]:
    z_ekinh, w_concorr_c, z_w_con_c = (
        where(
            lateral_boundary_4 <= cell < end_halo,
            _fused_velocity_advection_stencil_8_to_13_predictor(
                z_kin_hor_e,
                e_bln_c_s,
                z_w_concorr_me,
                wgtfac_c,
                w,
                z_w_concorr_mc,
                z_w_con_c,
                w_concorr_c,
                z_ekinh,
                k,
                nlev,
                nflatlev,
            ),
            (z_ekinh, w_concorr_c, z_w_con_c),
        )
        if istep == 1
        else where(
            lateral_boundary_3 <= cell < end_halo,
            _fused_velocity_advection_stencil_8_to_13_corrector(
                z_kin_hor_e,
                e_bln_c_s,
                wgtfac_c,
                w,
                z_w_con_c,
                w_concorr_c,
                z_ekinh,
                k,
                nlev,
                nflatlev,
            ),
            (z_ekinh, w_concorr_c, z_w_con_c),
        )
    )

    return z_ekinh, w_concorr_c, z_w_con_c


@field_operator
def _restricted_set_zero(
    k: fa.KField[gtx.int32],
    z_w_con_c: fa.CellKField[vpfloat],
    nlev: gtx.int32,
) -> fa.CellKField[vpfloat]:
    z_w_con_c = where(k == nlev, 0.0, z_w_con_c)
    return z_w_con_c


@program(grid_type=GridType.UNSTRUCTURED)
def fused_velocity_advection_stencil_8_to_13_predictor(
    z_kin_hor_e: fa.EdgeKField[vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], wpfloat],
    z_w_concorr_me: fa.EdgeKField[vpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
    w: fa.CellKField[wpfloat],
    z_w_concorr_mc: fa.CellKField[vpfloat],
    w_concorr_c: fa.CellKField[vpfloat],
    z_ekinh: fa.CellKField[vpfloat],
    z_w_con_c: fa.CellKField[vpfloat],
    k: fa.KField[gtx.int32],
    nlev: gtx.int32,
    nflatlev: gtx.int32,
    lateral_boundary_3: gtx.int32,
    lateral_boundary_4: gtx.int32,
    end_halo: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _fused_velocity_advection_stencil_8_to_13_predictor(
        z_kin_hor_e,
        e_bln_c_s,
        z_w_concorr_me,
        wgtfac_c,
        w,
        z_w_concorr_mc,
        z_w_con_c,
        w_concorr_c,
        z_ekinh,
        k,
        nlev,
        nflatlev,
        out=(z_ekinh, w_concorr_c, z_w_con_c),
        domain={
            dims.CellDim: (lateral_boundary_4, end_halo),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )

    _restricted_set_zero(
        k,
        z_w_con_c,
        nlev,
        out=z_w_con_c,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_end - 1, vertical_end),
        },
    )


@program(grid_type=GridType.UNSTRUCTURED)
def fused_velocity_advection_stencil_8_to_13_corrector(
    z_kin_hor_e: fa.EdgeKField[vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], wpfloat],
    z_w_concorr_me: fa.EdgeKField[vpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
    w: fa.CellKField[wpfloat],
    w_concorr_c: fa.CellKField[vpfloat],
    z_ekinh: fa.CellKField[vpfloat],
    z_w_con_c: fa.CellKField[vpfloat],
    k: fa.KField[gtx.int32],
    nlev: gtx.int32,
    nflatlev: gtx.int32,
    lateral_boundary_3: gtx.int32,
    lateral_boundary_4: gtx.int32,
    end_halo: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _fused_velocity_advection_stencil_8_to_13_corrector(
        z_kin_hor_e,
        e_bln_c_s,
        wgtfac_c,
        w,
        z_w_con_c,
        w_concorr_c,
        z_ekinh,
        k,
        nlev,
        nflatlev,
        out=(z_ekinh, w_concorr_c, z_w_con_c),
        domain={
            dims.CellDim: (lateral_boundary_3, end_halo),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )

    _restricted_set_zero(
        k,
        z_w_con_c,
        nlev,
        out=z_w_con_c,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_end - 1, vertical_end),
        },
    )
