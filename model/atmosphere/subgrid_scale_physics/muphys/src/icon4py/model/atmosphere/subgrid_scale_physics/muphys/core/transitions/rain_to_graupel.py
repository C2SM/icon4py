# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from gt4py.next.ffront.fbuiltins import where, exp, power, minimum
from icon4py.model.common import field_type_aliases as fa, type_alias as ta

@gtx.field_operator
def _rain_to_graupel(
    t:         fa.CellField[ta.wpfloat],             # Temperature
    rho:       fa.CellField[ta.wpfloat],             # Ambient density
    qc:        fa.CellField[ta.wpfloat],             # Cloud specific mass
    qr:        fa.CellField[ta.wpfloat],             # Specific humidity of rain
    qi:        fa.CellField[ta.wpfloat],             # Ice specific mass
    qs:        fa.CellField[ta.wpfloat],             # Snow specific mass
    mi:        fa.CellField[ta.wpfloat],             # Ice crystal mass
    dvsw:      ta.wpfloat,                           # qv-qsat_water (T)
    dt:        ta.wpfloat,                           # time step
    QMIN:      ta.wpfloat,
    TFRZ_HOM:  ta.wpfloat,
    TMELT:     ta.wpfloat,
) -> fa.CellField[ta.wpfloat]:                       # Conversion rate from graupel to rain

    TFRZ_RAIN = TMELT - 2.0
    A1        = 9.95e-5            # coefficient for immersion raindrop freezing: alpha_if
    B1        = 1.75               # coefficient for immersion raindrop freezing: a_if
    C1        = 1.68               # coefficient for raindrop freezing
    C2        = 0.66               # coefficient for immersion raindrop freezing: a_if
    C3        = 1.0                # coefficient for immersion raindrop freezing: a_if
    C4        = 0.1                # coefficient for immersion raindrop freezing: a_if
    A2        = 1.24e-3            # (PI/24)*EIR*V0R*Gamma(6.5)*AR**(-5/8)
    B2        = 1.625              # TBD
    QS_CRIT   = 1.0e-7             # TBD

    result = where( (qr > QMIN) & (t < TFRZ_RAIN) & (t > TFRZ_HOM) & (dvsw+qc <= 0.0), (exp(C2*(TFRZ_RAIN-t))-C3) * (A1 * power((qr * rho), B1)), 0. )
    result = where( (qr > QMIN) & (t < TFRZ_RAIN) & (t <= TFRZ_HOM), qr/dt, 0. )

    return where( (minimum(qi,qr) > QMIN) & (qs > QS_CRIT), result + A2*(qi/mi)*power((rho*qr), B2), result)

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def rain_to_graupel(
    t:               fa.CellField[ta.wpfloat],       # Temperature
    rho:             fa.CellField[ta.wpfloat],       # Ambient density
    qc:              fa.CellField[ta.wpfloat],       # Cloud specific mass
    qr:              fa.CellField[ta.wpfloat],       # Specific humidity of rain
    qi:              fa.CellField[ta.wpfloat],       # Ice specific mass
    qs:              fa.CellField[ta.wpfloat],       # Snow specific mass
    mi:              fa.CellField[ta.wpfloat],       # Ice crystal mass
    dvsw:            ta.wpfloat,                     # qv-qsat_water (T)
    dt:              ta.wpfloat,                     # time step
    QMIN:            ta.wpfloat,
    TFRZ_HOM:        ta.wpfloat,
    TMELT:           ta.wpfloat,
    conversion_rate: fa.CellField[ta.wpfloat],       # output
):
    _rain_to_graupel(t, rho, qc, qr, qi, qs, mi, dvsw, dt, QMIN, TFRZ_HOM, TMELT, out=conversion_rate)
