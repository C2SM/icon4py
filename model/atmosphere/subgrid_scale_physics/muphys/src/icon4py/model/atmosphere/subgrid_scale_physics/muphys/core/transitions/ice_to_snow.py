# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from gt4py.next.ffront.fbuiltins import where, power, maximum
from icon4py.model.common import field_type_aliases as fa, type_alias as ta

@gtx.field_operator
def _ice_to_snow(
    qi:           fa.CellField[ta.wpfloat],             # Ice specific mass
    ns:           fa.CellField[ta.wpfloat],             # Snow number
    lam:          fa.CellField[ta.wpfloat],             # Snow intercept parameter
    sticking_eff: fa.CellField[ta.wpfloat],             # Sticking efficiency
    QMIN:         ta.wpfloat,
    V0S:          ta.wpfloat,
    V1S:          ta.wpfloat,
) -> fa.CellField[ta.wpfloat]:                          # Conversion rate of ice to snow
    QI0      = 0.0                                      # Critical ice required for autoconversion
    C_IAU    = 1.0e-3                                   # Coefficient of auto conversion
    C_AGG    = 2.61*V0S                                 # Coeff of aggregation (2.610 = pi*gam(v1s+3)/4)
    B_AGG    = -(V1S + 3.0)                             # Aggregation exponent
    
    return where( (qi > QMIN), sticking_eff * (C_IAU * maximum(0.0, (qi-QI0)) + qi * (C_AGG * ns) * power(lam, B_AGG)), 0. )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def ice_to_snow(
    qi:              fa.CellField[ta.wpfloat],             # Ice specific mass
    ns:              fa.CellField[ta.wpfloat],             # Snow number
    lam:             fa.CellField[ta.wpfloat],             # Snow intercept parameter
    sticking_eff:    fa.CellField[ta.wpfloat],             # Sticking efficiency
    QMIN:            ta.wpfloat,
    V1S:             ta.wpfloat,
    V0S:             ta.wpfloat, conversion_rate: fa.CellField[ta.wpfloat],             # output
):
    _ice_to_snow(qi, ns, lam, sticking_eff, QMIN, V0S, V1S, out=conversion_rate)
