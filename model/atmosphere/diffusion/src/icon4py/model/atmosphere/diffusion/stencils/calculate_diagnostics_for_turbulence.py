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
from gt4py.next.ffront.fbuiltins import Field, astype, int32

from icon4py.model.common.dimension import CellDim, KDim, KHalf2K, KHalfDim, Koff
from icon4py.model.common.dimension import CellDim, KDim, KHalfDim, KHalf2K
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _calculate_diagnostics_for_turbulence(
    div: Field[[CellDim, KDim], vpfloat],
    kh_c: Field[[CellDim, KDim], vpfloat],
    wgtfac_c: Field[[CellDim, KHalfDim], vpfloat],
) -> tuple[Field[[CellDim, KHalfDim], vpfloat], Field[[CellDim, KHalfDim], vpfloat]]:
    wgtfac_c_wp, div_wp, kh_c_wp = astype((wgtfac_c, div, kh_c), wpfloat)
    div_wp_khalf = div_wp(KHalf2K[0])
    kh_c_wp_khalf = kh_c_wp(KHalf2K[0])

    div_ic_wp = wgtfac_c_wp * div_wp_khalf + (wpfloat("1.0") - wgtfac_c_wp) * div_wp(KHalf2K[1])
    hdef_ic_wp = wgtfac_c_wp * kh_c_wp_khalf + (wpfloat("1.0") - wgtfac_c_wp) * kh_c_wp(KHalf2K[1])
    hdef_ic_wp = hdef_ic_wp * hdef_ic_wp

    return astype((div_ic_wp, hdef_ic_wp), vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def calculate_diagnostics_for_turbulence(
    div: Field[[CellDim, KDim], vpfloat],
    kh_c: Field[[CellDim, KDim], vpfloat],
    wgtfac_c: Field[[CellDim, KHalfDim], vpfloat],
    div_ic: Field[[CellDim, KHalfDim], vpfloat],
    hdef_ic: Field[[CellDim, KHalfDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _calculate_diagnostics_for_turbulence(div, kh_c, wgtfac_c, out=(div_ic, hdef_ic),
                                          domain={
                                              CellDim: (horizontal_start, horizontal_end),
                                              KHalfDim: (vertical_start, vertical_end+1),
                                          },
                                          )
