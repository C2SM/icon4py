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

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, astype

from icon4py.model.common.dimension import CellDim, KDim, Koff
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _calculate_diagnostics_for_turbulence(
    div: Field[[CellDim, KDim], vpfloat],
    kh_c: Field[[CellDim, KDim], vpfloat],
    wgtfac_c: Field[[CellDim, KDim], vpfloat],
) -> tuple[Field[[CellDim, KDim], vpfloat], Field[[CellDim, KDim], vpfloat]]:
    wgtfac_c_wp, div_wp, kh_c_wp = astype((wgtfac_c, div, kh_c), wpfloat)

    div_ic_wp = astype(wgtfac_c * div, wpfloat) + (wpfloat("1.0") - wgtfac_c_wp) * div_wp(Koff[-1])
    # TODO(magdalena): change exponent back to int (workaround for gt4py)
    hdef_ic_wp = (
        astype(wgtfac_c * kh_c, wpfloat) + (wpfloat("1.0") - wgtfac_c_wp) * kh_c_wp(Koff[-1])
    ) ** 2.0
    return astype((div_ic_wp, hdef_ic_wp), vpfloat)


@program(grid_type=GridType.UNSTRUCTURED)
def calculate_diagnostics_for_turbulence(
    div: Field[[CellDim, KDim], vpfloat],
    kh_c: Field[[CellDim, KDim], vpfloat],
    wgtfac_c: Field[[CellDim, KDim], vpfloat],
    div_ic: Field[[CellDim, KDim], vpfloat],
    hdef_ic: Field[[CellDim, KDim], vpfloat],
):
    _calculate_diagnostics_for_turbulence(div, kh_c, wgtfac_c, out=(div_ic[:, 1:], hdef_ic[:, 1:]))
