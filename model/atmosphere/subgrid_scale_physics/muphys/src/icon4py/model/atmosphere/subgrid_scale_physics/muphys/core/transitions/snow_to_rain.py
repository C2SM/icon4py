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
def _snow_to_rain(
    t:        fa.CellField[ta.wpfloat],             # Temperature
    p:        fa.CellField[ta.wpfloat],             # Ambient pressure
    rho:      fa.CellField[ta.wpfloat],             # Ambient density
    dvsw0:    fa.CellField[ta.wpfloat],             # qv-qsat_water(T0)
    qs:       fa.CellField[ta.wpfloat],             # Snow specific mass
    QMIN:     ta.wpfloat,
    TX:       ta.wpfloat,
    TMELT:    ta.wpfloat,
) -> fa.CellField[ta.wpfloat]:                      # Return: Riming snow rate
    C1_SR = 79.6863                                 # Constants in melting formula
    C2_SR = 0.612654e-3                             # Constants in melting formula
    A_SR  = TX - 389.5                              # Melting prefactor
    B_SR  = 0.8                                     # Melting exponent
    return where( (t > maximum(TMELT, TMELT-TX*dvsw0)) & (qs > QMIN), (C1_SR/p + C2_SR) * (t - TMELT + A_SR*dvsw0) * power(qs*rho, B_SR), 0.0)

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def snow_to_rain(
    t:        fa.CellField[ta.wpfloat],             # Temperature
    p:        fa.CellField[ta.wpfloat],             # Ambient pressure
    rho:      fa.CellField[ta.wpfloat],             # Ambient density
    dvsw0:    fa.CellField[ta.wpfloat],             # qv-qsat_water(T0)
    qs:       fa.CellField[ta.wpfloat],             # Snow specific mass
    QMIN:     ta.wpfloat,
    TX:       ta.wpfloat,
    TMELT:    ta.wpfloat,
    conversion_rate: fa.CellField[ta.wpfloat],      # output
):
    _snow_to_rain(t, p, rho, dvsw0, qs, QMIN, TX, TMELT, out=conversion_rate)
