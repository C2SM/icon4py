# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from gt4py.next.ffront.fbuiltins import where, power, exp, maximum, minimum
from icon4py.model.common import field_type_aliases as fa, type_alias as ta

@gtx.field_operator
def _snow_number(
    t:        fa.CellField[ta.wpfloat],             # Temperature
    rho:      fa.CellField[ta.wpfloat],             # Ambient air density
    qs:       fa.CellField[ta.wpfloat],             # Snow specific mass
    QMIN:     ta.wpfloat,                           # 
    AMS:      ta.wpfloat,                           # 
    TMELT:    ta.wpfloat,                           # 
) -> fa.CellField[ta.wpfloat]:                      # Snow number
    TMIN = TMELT - 40.0
    TMAX = TMELT
    QSMIN = 2.0e-6
    XA1 = -1.65e+0
    XA2 = 5.45e-2
    XA3 = 3.27e-4
    XB1 = 1.42e+0
    XB2 = 1.19e-2
    XB3 = 9.60e-5
    N0S0 = 8.00e+5
    N0S1 = 13.5 * 5.65e+05
    N0S2 = -0.107
    N0S3 = 13.5
    N0S4 = 0.5 * N0S1
    N0S5 = 1.e6
    N0S6 = 1.e2 * N0S1
    N0S7 = 1.e9

    # TODO: see if these can be incorporated into WHERE statement
    tc   = maximum( minimum( t, TMAX), TMIN ) - TMELT
    alf  = power( 10.0, ( XA1 + tc * (XA2 + tc * XA3)) )
    bet  = XB1 + tc * ( XB2  + tc * XB3 )
    n0s  = N0S3 * power( ( ( qs + QSMIN ) * rho / AMS), ( 4.0 - 3.0 * bet ) ) / ( alf * alf * alf )
    y    = exp( N0S2 * tc )
    return where( qs > QMIN, minimum( minimum( N0S6 * y, N0S7), maximum( maximum( N0S4 * y, N0S5 ), N0S ) ), 0.0 )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def snow_number(
    t:        fa.CellField[ta.wpfloat],             # Temperature
    rho:      fa.CellField[ta.wpfloat],             # Ambient air density
    qs:       fa.CellField[ta.wpfloat],             # Snow specific mass
    QMIN:     ta.wpfloat,                           # 
    AMS:      ta.wpfloat,                           # 
    TMELT:    ta.wpfloat,                           # 
    snow_number: fa.CellField[ta.wpfloat]           # output
):
    _snow_number( t, rho, qs, QMIN, AMS, TMELT, out=snow_number )
