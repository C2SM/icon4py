# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.interpolation.stencils.interpolate_to_cell_center import (
    _interpolate_to_cell_center,
)
from icon4py.model.common.math.differential_operators import (
    cell_horizontal_gradients_by_green_gauss_method_on_half_levels,
    divergence,
)
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _temporary_fields_for_turbulence_diagnostics(
    kh_smag: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    diff_multfac_smag: gtx.Field[gtx.Dims[dims.KDim], vpfloat],
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    kh_smag_wp, diff_multfac_smag_wp = astype((kh_smag, diff_multfac_smag), wpfloat)

    kh_c_wp = _interpolate_to_cell_center(kh_smag_wp, e_bln_c_s) / diff_multfac_smag_wp
    div_wp = divergence(vn, geofac_div)
    return kh_c_wp, div_wp


@gtx.field_operator
def _flow_deformation(
    div: fa.CellKField[wpfloat],
    kh_c: fa.CellKField[wpfloat],
    wgtfac_c: fa.CellKHalfField[vpfloat],
) -> tuple[fa.CellKHalfField[vpfloat], fa.CellKHalfField[vpfloat]]:
    wgtfac_c_wp = astype(wgtfac_c, wpfloat)

    div_ic_wp = astype(wgtfac_c * div(dims.KHalfDim + 0.5), wpfloat) + (
        wpfloat("1.0") - wgtfac_c_wp
    ) * div(dims.KHalfDim - 0.5)
    hdef_ic_wp = astype(wgtfac_c * kh_c(dims.KHalfDim + 0.5), wpfloat) + (
        wpfloat("1.0") - wgtfac_c_wp
    ) * kh_c(dims.KHalfDim - 0.5)
    hdef_ic_wp = hdef_ic_wp * hdef_ic_wp

    return astype((div_ic_wp, hdef_ic_wp), vpfloat)


@gtx.field_operator
def _compute_diagnostic_quantities_for_turbulence(
    kh_smag: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    diff_multfac_smag: gtx.Field[gtx.Dims[dims.KDim], vpfloat],
    wgtfac_c: fa.CellKHalfField[vpfloat],
    w_old: fa.CellKHalfField[wpfloat],
    geofac_grg_x: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], wpfloat],
    geofac_grg_y: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], wpfloat],
    compute_hgrad_w: bool,
    dwdx: fa.CellKHalfField[vpfloat],
    dwdy: fa.CellKHalfField[vpfloat],
) -> tuple[
    fa.CellKHalfField[vpfloat],
    fa.CellKHalfField[vpfloat],
    fa.CellKHalfField[vpfloat],
    fa.CellKHalfField[vpfloat],
]:
    kh_c, div = _temporary_fields_for_turbulence_diagnostics(
        kh_smag, vn, e_bln_c_s, geofac_div, diff_multfac_smag
    )
    div_ic, hdef_ic = _flow_deformation(div, kh_c, wgtfac_c)

    dwdx, dwdy = (
        cell_horizontal_gradients_by_green_gauss_method_on_half_levels(
            w_old, geofac_grg_x, geofac_grg_y
        )
        if compute_hgrad_w
        else (dwdx, dwdy)
    )

    return div_ic, hdef_ic, dwdx, dwdy
