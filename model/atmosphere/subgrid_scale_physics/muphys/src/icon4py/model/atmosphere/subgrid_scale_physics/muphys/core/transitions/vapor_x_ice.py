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
def _vapor_x_ice(
    qi:        fa.CellField[ta.wpfloat],             # Specific humidity of ice
    mi:        fa.CellField[ta.wpfloat],             # Ice crystal mass
    eta:       fa.CellField[ta.wpfloat],             # Deposition factor
    dvsi:      fa.CellField[ta.wpfloat],             # Vapor excess qv-qsat_ice(T)
    rho:       fa.CellField[ta.wpfloat],             # Ambient density
    dt:        ta.wpfloat,                           # time step
    QMIN:      ta.wpfloat,
) -> fa.CellField[ta.wpfloat]:                       # Rate of vapor deposition to ice
    AMI    = 130.0                 # Formfactor for mass-size relation of cold ice
    B_EXP  = 7.0                   # exp. for conv. (-1 + 0.33) of ice mass to sfc area
    A_FACT = 4.0 * AMI**(-1.0/3.0)  

    # TO-DO: see if this can be folded into the WHERE statement
    mask   = (A_FACT * eta) * rho * qi * power(mi, B_EXP) * dvsi
    return where( mask > 0.0, minimum(mask, dvsi/dt), maximum(maximum(mask, dvsi/dt), -qi/dt) )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def vapor_x_ice(
    qi:        fa.CellField[ta.wpfloat],             # Specific humidity of ice
    mi:        fa.CellField[ta.wpfloat],             # Ice crystal mass
    eta:       fa.CellField[ta.wpfloat],             # Deposition factor
    dvsi:      fa.CellField[ta.wpfloat],             # Vapor excess qv-qsat_ice(T)
    rho:       fa.CellField[ta.wpfloat],             # Ambient density
    dt:        ta.wpfloat,                           # time step
    QMIN:      ta.wpfloat,
    vapor_deposition_rate: fa.CellField[ta.wpfloat]  # output
):
    _vapor_x_ice(qi, mi, eta, dvsi, rho, dt, QMIN, out=vapor_deposition_rate)
