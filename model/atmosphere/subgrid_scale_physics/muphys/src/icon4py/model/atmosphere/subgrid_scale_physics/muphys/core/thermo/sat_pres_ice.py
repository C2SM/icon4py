# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from gt4py.next.ffront.fbuiltins import exp
from icon4py.model.common import field_type_aliases as fa, type_alias as ta

@gtx.field_operator
def _sat_pres_ice(
    t:         fa.CellField[ta.wpfloat],             # Temperature
    TMELT:     ta.wpfloat,
) -> fa.CellField[ta.wpfloat]:                       # Pressure
    C1ES   = 610.78
    C3IES  = 21.875
    C4IES  = 7.66

    return C1ES * exp( C3IES * ( t-TMELT ) / ( t-C4IES ) )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def sat_pres_ice(
    t:         fa.CellField[ta.wpfloat],             # Temperature
    TMELT:     ta.wpfloat,
    pressure: fa.CellField[ta.wpfloat]               # output
):
    _sat_pres_ice( t, TMELT, out=pressure )
