# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import astype

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import KHalfDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _calculate_diagnostics_for_turbulence(
    div: fa.CellKField[vpfloat],
    kh_c: fa.CellKField[vpfloat],
    wgtfac_c: fa.CellKHalfField[vpfloat],
) -> tuple[fa.CellKHalfField[vpfloat], fa.CellKHalfField[vpfloat]]:
    wgtfac_c_wp = astype(wgtfac_c, wpfloat)
    div_wp, kh_c_wp = astype((div, kh_c), wpfloat)

    div_ic_wp = astype(wgtfac_c * div(KHalfDim + 0.5), wpfloat) + (
        wpfloat("1.0") - wgtfac_c_wp
    ) * div_wp(KHalfDim - 0.5)
    hdef_ic_wp = astype(wgtfac_c * kh_c(KHalfDim + 0.5), wpfloat) + (
        wpfloat("1.0") - wgtfac_c_wp
    ) * kh_c_wp(KHalfDim - 0.5)
    hdef_ic_wp = hdef_ic_wp * hdef_ic_wp

    return astype((div_ic_wp, hdef_ic_wp), vpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def calculate_diagnostics_for_turbulence(
    div: fa.CellKField[vpfloat],
    kh_c: fa.CellKField[vpfloat],
    wgtfac_c: fa.CellKHalfField[vpfloat],
    div_ic: fa.CellKHalfField[vpfloat],
    hdef_ic: fa.CellKHalfField[vpfloat],
) -> None:
    _calculate_diagnostics_for_turbulence(
        div, kh_c, wgtfac_c, out=(div_ic[:, 1:-1], hdef_ic[:, 1:-1])
    )
