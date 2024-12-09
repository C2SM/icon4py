# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from gt4py.next.ffront.fbuiltins import where, power, maximum, minimum
from icon4py.model.common import field_type_aliases as fa, type_alias as ta

@gtx.field_operator
def _vapor_x_graupel(
    t:         fa.CellField[ta.wpfloat],             # Temperature
    p:         fa.CellField[ta.wpfloat],             # Ambient pressure
    rho:       fa.CellField[ta.wpfloat],             # Ambient density
    qg:        fa.CellField[ta.wpfloat],             # Graupel specific mass
    dvsw:      fa.CellField[ta.wpfloat],             # qv-qsat_water(T)
    dvsi:      fa.CellField[ta.wpfloat],             # qv-qsat_ice(T)
    dvsw0:     fa.CellField[ta.wpfloat],             # qv-qsat_water(T0)
    dt:        ta.wpfloat,                           # time step
    QMIN:      ta.wpfloat,
    TX:        ta.wpfloat,
    TMELT:     ta.wpfloat,
) -> fa.CellField[ta.wpfloat]:                       # Homogeneous freezing rate
    A1_VG = 0.398561
    A2_VG = -0.00152398
    A3    = 2554.99
    A4    = 2.6531E-7
    A5    = 0.153907
    A6    = -7.86703e-07
    A7    = 0.0418521
    A8    = -4.7524E-8
    B_VG  = 0.6
    result = where( (qg > QMIN) & (t < TMELT), (A1_VG + A2_VG*t + A3/p + A4*p) * dvsi * power(qg*rho, B_VG), 0. )
    result = where( (qg > QMIN) & (t >= TMELT) & (t > (TMELT - TX*dvsw0)), (A5 + A6*p) * minimum(0.0, dvsw0) * power(qg*rho, B_VG), result )
    result = where( (qg > QMIN) & (t >= TMELT) & (t <= (TMELT - TX*dvsw0)), (A7 + A8*p) * dvsw * power(qg*rho, B_VG), result )
    return maximum(result, -qg/dt)

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def vapor_x_graupel(
    t:         fa.CellField[ta.wpfloat],             # Temperature
    p:         fa.CellField[ta.wpfloat],             # Ambient pressure
    rho:       fa.CellField[ta.wpfloat],             # Ambient density
    qg:        fa.CellField[ta.wpfloat],             # Graupel specific mass
    dvsw:      fa.CellField[ta.wpfloat],             # qv-qsat_water(T)
    dvsi:      fa.CellField[ta.wpfloat],             # qv-qsat_ice(T)
    dvsw0:     fa.CellField[ta.wpfloat],             # qv-qsat_water(T0)
    dt:        ta.wpfloat,                           # time step
    QMIN:      ta.wpfloat,
    TX:        ta.wpfloat,
    TMELT:     ta.wpfloat,
    exchange_rate:     fa.CellField[ta.wpfloat],     # output
):
    _vapor_x_graupel(t, p, rho, qg, dvsw, dvsi, dvsw0, dt, QMIN, TX, TMELT, out=exchange_rate)
