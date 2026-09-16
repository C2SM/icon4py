# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype, neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2E
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _temporary_fields_for_turbulence_diagnostics(
    kh_smag_ec: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    diff_multfac_smag: gtx.Field[gtx.Dims[dims.KDim], vpfloat],
) -> tuple[fa.CellKField[vpfloat], fa.CellKField[vpfloat]]:
    kh_smag_ec_wp, diff_multfac_smag_wp = astype((kh_smag_ec, diff_multfac_smag), wpfloat)

    kh_c_wp = neighbor_sum(kh_smag_ec_wp(C2E) * e_bln_c_s, axis=dims.C2EDim) / diff_multfac_smag_wp
    div_wp = neighbor_sum(vn(C2E) * geofac_div, axis=dims.C2EDim)
    return astype((kh_c_wp, div_wp), vpfloat)


@gtx.field_operator
def _diagnostics_for_turbulence(
    div: fa.CellKField[vpfloat],
    kh_c: fa.CellKField[vpfloat],
    wgtfac_c: fa.CellKHalfField[vpfloat],
) -> tuple[fa.CellKHalfField[vpfloat], fa.CellKHalfField[vpfloat]]:
    wgtfac_c_wp = astype(wgtfac_c, wpfloat)
    div_wp, kh_c_wp = astype((div, kh_c), wpfloat)

    div_ic_wp = astype(wgtfac_c * div(dims.KHalfDim + 0.5), wpfloat) + (
        wpfloat("1.0") - wgtfac_c_wp
    ) * div_wp(dims.KHalfDim - 0.5)
    hdef_ic_wp = astype(wgtfac_c * kh_c(dims.KHalfDim + 0.5), wpfloat) + (
        wpfloat("1.0") - wgtfac_c_wp
    ) * kh_c_wp(dims.KHalfDim - 0.5)
    hdef_ic_wp = hdef_ic_wp * hdef_ic_wp

    return astype((div_ic_wp, hdef_ic_wp), vpfloat)


@gtx.field_operator
def _diagnostic_quantities_for_turbulence(
    kh_smag_ec: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    diff_multfac_smag: gtx.Field[gtx.Dims[dims.KDim], vpfloat],
    wgtfac_c: fa.CellKHalfField[vpfloat],
) -> tuple[fa.CellKHalfField[vpfloat], fa.CellKHalfField[vpfloat]]:
    kh_c, div = _temporary_fields_for_turbulence_diagnostics(
        kh_smag_ec, vn, e_bln_c_s, geofac_div, diff_multfac_smag
    )
    div_ic_vp, hdef_ic_vp = _diagnostics_for_turbulence(div, kh_c, wgtfac_c)
    return div_ic_vp, hdef_ic_vp
