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
from gt4py.next.ffront.fbuiltins import astype

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import CellDim, KHalf2K, KHalfDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _calculate_diagnostics_for_turbulence(
    div: fa.CellKField[vpfloat],
    kh_c: fa.CellKField[vpfloat],
    wgtfac_c: fa.CellKHalfField[vpfloat],
) -> tuple[fa.CellKHalfField[vpfloat], fa.CellKHalfField[vpfloat]]:
    wgtfac_c_wp, div_wp, kh_c_wp = astype((wgtfac_c, div, kh_c), wpfloat)
    div_ic_wp = wgtfac_c_wp * div_wp(KHalf2K[0]) + (wpfloat("1.0") - wgtfac_c_wp) * div_wp(
        KHalf2K[1]
    )
    hdef_ic_wp = wgtfac_c_wp * kh_c_wp(KHalf2K[0]) + (wpfloat("1.0") - wgtfac_c_wp) * kh_c_wp(
        KHalf2K[1]
    )
    hdef_ic_wp = hdef_ic_wp * hdef_ic_wp

    return astype((div_ic_wp, hdef_ic_wp), vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def calculate_diagnostics_for_turbulence(
    div: fa.CellKField[vpfloat],
    kh_c: fa.CellKField[vpfloat],
    wgtfac_c: fa.CellKHalfField[vpfloat],
    div_ic: fa.CellKHalfField[vpfloat],
    hdef_ic: fa.CellKHalfField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _calculate_diagnostics_for_turbulence(
        div,
        kh_c,
        wgtfac_c,
        out=(div_ic, hdef_ic),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KHalfDim: (vertical_start, vertical_end + 1),
        },
    )
