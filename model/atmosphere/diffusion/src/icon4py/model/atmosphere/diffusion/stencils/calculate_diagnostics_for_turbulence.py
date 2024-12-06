# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import astype

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _calculate_diagnostics_for_turbulence(
    div: fa.CellKField[vpfloat],
    kh_c: fa.CellKField[vpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
) -> tuple[fa.CellKField[vpfloat], fa.CellKField[vpfloat]]:
    wgtfac_c_wp, div_wp, kh_c_wp = astype((wgtfac_c, div, kh_c), wpfloat)

    div_ic_wp = astype(wgtfac_c * div, wpfloat) + (wpfloat("1.0") - wgtfac_c_wp) * div_wp(Koff[-1])
    hdef_ic_wp = astype(wgtfac_c * kh_c, wpfloat) + (wpfloat("1.0") - wgtfac_c_wp) * kh_c_wp(
        Koff[-1]
    )
    hdef_ic_wp = hdef_ic_wp * hdef_ic_wp

    return astype((div_ic_wp, hdef_ic_wp), vpfloat)


@program(grid_type=GridType.UNSTRUCTURED)
def calculate_diagnostics_for_turbulence(
    div: fa.CellKField[vpfloat],
    kh_c: fa.CellKField[vpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
    div_ic: fa.CellKField[vpfloat],
    hdef_ic: fa.CellKField[vpfloat],
):
    _calculate_diagnostics_for_turbulence(div, kh_c, wgtfac_c, out=(div_ic[:, 1:], hdef_ic[:, 1:]))
