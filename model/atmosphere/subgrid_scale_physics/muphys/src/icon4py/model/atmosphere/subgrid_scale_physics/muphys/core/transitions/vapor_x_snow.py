# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from gt4py.next.ffront.fbuiltins import where, sqrt, power, maximum, minimum
from icon4py.model.common import field_type_aliases as fa, type_alias as ta

@gtx.field_operator
def _vapor_x_snow(
    t:         fa.CellField[ta.wpfloat],             # Temperature
    p:         fa.CellField[ta.wpfloat],             # Ambient pressure
    rho:       fa.CellField[ta.wpfloat],             # Ambient density
    qs:        fa.CellField[ta.wpfloat],             # Snow specific mass
    ns:        fa.CellField[ta.wpfloat],             # Snow number
    lam:       fa.CellField[ta.wpfloat],             # Slope parameter (lambda) snow
    eta:       fa.CellField[ta.wpfloat],             # Deposition factor
    ice_dep:   fa.CellField[ta.wpfloat],             # Limiter for vapor dep on snow
    dvsw:      fa.CellField[ta.wpfloat],             # qv-qsat_water(T)
    dvsi:      fa.CellField[ta.wpfloat],             # qv-qsat_ice(T)
    dvsw0:     fa.CellField[ta.wpfloat],             # qv-qsat_water(T0)
    dt:        ta.wpfloat,                           # time step
    QMIN:      ta.wpfloat,
    TX:        ta.wpfloat,
    TMELT:     ta.wpfloat,
    V0S:       ta.wpfloat,
    V1S:       ta.wpfloat,
) -> fa.CellField[ta.wpfloat]:                       # Rate of vapor deposition to snow
    NU     = 1.75e-5;                                # kinematic viscosity of air
    A0_VS  = 1.0
    A1_VS  = 0.4182 * sqrt(V0S/NU)
    A2_VS  = -(V1S + 1.0) / 2.0
    EPS    = 1.0e-15
    QS_LIM = 1.0e-7
    CNX    = 4.0
    B_VS   = 0.8
    C1_VS  = 31282.3
    C2_VS  = 0.241897
    C3_VS  = 0.28003
    C4_VS  = -0.146293E-6

    # See if this can be incorporated into WHERE statement
    mask = (CNX * ns * eta / rho) * (A0_VS + A1_VS * power(lam, A2_VS)) * dvsi / (lam * lam + EPS)

    # GZ: This mask>0 limitation, which was missing in the original graupel scheme,
    # is crucial for numerical stability in the tropics!
    # a meaningful distinction between cloud ice and snow
    result = where( (qs > QMIN) & (t < TMELT) & (mask > 0.0), minimum(mask, dvsi/dt - ice_dep), 0.0 ) 
    result = where( (qs > QMIN) & (t < TMELT) & (qs <= QS_LIM), minimum(result, 0.0), result )
    # ELSE section
    result = where( (qs > QMIN) & (t >= TMELT) & (t > (TMELT - TX*dvsw0)), (C1_VS/p + C2_VS) * minimum(0.0, dvsw0) * power(qs*rho, B_VS), 0.0)
    result = where( (qs > QMIN) & (t >= TMELT) & (t <= (TMELT - TX*dvsw0)), (C3_VS + C4_VS*p) * dvsw * power(qs*rho, B_VS), 0.0)
    return where( (qs > QMIN), maximum(result, -qs/dt), 0.0)

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def vapor_x_snow(
    t:         fa.CellField[ta.wpfloat],             # Temperature
    p:         fa.CellField[ta.wpfloat],             # Ambient pressure
    rho:       fa.CellField[ta.wpfloat],             # Ambient density
    qs:        fa.CellField[ta.wpfloat],             # Snow specific mass
    ns:        fa.CellField[ta.wpfloat],             # Snow number
    lam:       fa.CellField[ta.wpfloat],             # Slope parameter (lambda) snow
    eta:       fa.CellField[ta.wpfloat],             # Deposition factor
    ice_dep:   fa.CellField[ta.wpfloat],             # Limiter for vapor dep on snow
    dvsw:      fa.CellField[ta.wpfloat],             # qv-qsat_water(T)
    dvsi:      fa.CellField[ta.wpfloat],             # qv-qsat_ice(T)
    dvsw0:     fa.CellField[ta.wpfloat],             # qv-qsat_water(T0)
    dt:        ta.wpfloat,                           # time step
    QMIN:      ta.wpfloat,
    TX:        ta.wpfloat,
    TMELT:     ta.wpfloat,
    V0S:       ta.wpfloat,
    V1S:       ta.wpfloat,
    vapor_deposition_rate:     fa.CellField[ta.wpfloat],     # output
):
    _vapor_x_snow(t, p, rho, qs, ns, lam, eta, ice_dep, dvsw, dvsi, dvsw0, dt, QMIN, TX, TMELT, V0S, V1S, out=vapor_deposition_rate)
