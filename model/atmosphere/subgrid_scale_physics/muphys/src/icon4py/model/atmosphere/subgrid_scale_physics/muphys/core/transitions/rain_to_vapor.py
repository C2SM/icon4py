# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from gt4py.next.ffront.fbuiltins import where, power, minimum
from icon4py.model.common import field_type_aliases as fa, type_alias as ta

@gtx.field_operator
def _rain_to_vapor(
    t:         fa.CellField[ta.wpfloat],             # Temperature
    rho:       fa.CellField[ta.wpfloat],             # Ambient density
    qc:        fa.CellField[ta.wpfloat],             # Cloud-specific humidity
    qr:        fa.CellField[ta.wpfloat],             # Rain-specific humidity 
    dvsw:      fa.CellField[ta.wpfloat],             # qv-qsat_water (T)
    dt:        ta.wpfloat,                           # time step
    QMIN:      ta.wpfloat,
    TMELT:     ta.wpfloat,
) -> fa.CellField[ta.wpfloat]:                       # Conversion rate from graupel to rain

    B1_RV     =  0.16667           # exponent in power-law relation for mass density
    B2_RV     =  0.55555           # TBD
    C1_RV     =  0.61              # TBD
    C2_RV     = -0.0163            # TBD
    C3_RV     =  1.111e-4          # TBD
    A1_RV     =  1.536e-3          # TBD
    A2_RV     =  1.0e0             # TBD
    A3_RV     = 19.0621e0          # TBD

    # TO-DO: move as much as possible into WHERE statement
    tc = t - TMELT
    evap_max = (C1_RV + tc * (C2_RV + C3_RV*tc)) * (-dvsw) / dt 
    return where( (qr > QMIN) & (dvsw+qc <= 0.0), minimum(A1_RV * (A2_RV+A3_RV*power(qr*rho,B1_RV)) * (-dvsw) * power(qr*rho,B2_RV), evap_max), 0. )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def rain_to_vapor(
    t:         fa.CellField[ta.wpfloat],             # Temperature
    rho:       fa.CellField[ta.wpfloat],             # Ambient density
    qc:        fa.CellField[ta.wpfloat],             # Cloud-specific humidity
    qr:        fa.CellField[ta.wpfloat],             # Rain-specific humidity
    dvsw:      fa.CellField[ta.wpfloat],             # qv-qsat_water (T)
    dt:        ta.wpfloat,                           # time step
    QMIN:      ta.wpfloat,
    TMELT:     ta.wpfloat,
    conversion_rate: fa.CellField[ta.wpfloat],       # output
):
    _rain_to_vapor(t, rho, qc, qr, dvsw, dt, QMIN, TMELT, out=conversion_rate)
