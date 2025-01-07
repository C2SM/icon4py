# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from gt4py.next.ffront.fbuiltins import power
from icon4py.model.common import field_type_aliases as fa, type_alias as ta

@gtx.field_operator
def _dqsatdT_rho(
    qs:        fa.CellField[ta.wpfloat],             # Saturation vapor pressure (over liquid)
    t:         fa.CellField[ta.wpfloat],             # Temperature
    TMELT:     ta.wpfloat,
) -> fa.CellField[ta.wpfloat]:                       # derivative d(qsat_rho)/dT
    C3LES  = 17.269
    C4LES  = 35.86
    C5LES  = C3LES * (TMELT - C4LES)

    return qs * ( C5LES / (t-C4LES)*(t-C4LES) - 1.0/t )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def dqsatdT_rho(
    qs:        fa.CellField[ta.wpfloat],             # Saturation vapor pressure (over liquid)
    t:         fa.CellField[ta.wpfloat],             # Temperature
    TMELT:     ta.wpfloat,
    derivative: fa.CellField[ta.wpfloat]             # output
):
    _dqsatdT_rho( qs, t, TMELT, out=derivative )
