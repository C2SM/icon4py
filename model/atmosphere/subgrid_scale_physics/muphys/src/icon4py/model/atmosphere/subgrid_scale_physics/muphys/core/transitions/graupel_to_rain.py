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
def _graupel_to_rain(
    t:       fa.CellField[ta.wpfloat],             # Ambient temperature
    p:       fa.CellField[ta.wpfloat],             # Ambient pressue
    rho:     fa.CellField[ta.wpfloat],             # Ambient density
    dvsw0:   fa.CellField[ta.wpfloat],             # qv-qsat_water(T0)
    qg:      fa.CellField[ta.wpfloat],             # Graupel specific mass
    QMIN:    ta.wpfloat,
    TX:      ta.wpfloat,
    TMELT:   ta.wpfloat,
) -> fa.CellField[ta.wpfloat]:                     # Return: rain rate
    A_MELT  = TX - 389.5                           # melting prefactor
    B_MELT  = 0.6                                  # melting exponent
    C1_MELT = 12.31698                             # Constants in melting formula
    C2_MELT = 7.39441e-05                          # Constants in melting formula
    return where( (t > maximum(TMELT,TMELT-TX*dvsw0)) & (qg > QMIN), (C1_MELT/p + C2_MELT)*(t-TMELT+A_MELT*dvsw0)*power(qg*rho,B_MELT), 0. )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def graupel_to_rain(
    t:       fa.CellField[ta.wpfloat],             # Ambient temperature
    p:       fa.CellField[ta.wpfloat],             # Ambient pressue
    rho:     fa.CellField[ta.wpfloat],             # Ambient density
    dvsw0:   fa.CellField[ta.wpfloat],             # qv-qsat_water(T0)
    qg:      fa.CellField[ta.wpfloat],             # Graupel specific mass
    QMIN:    ta.wpfloat,
    TX:      ta.wpfloat,
    TMELT:   ta.wpfloat,
    rain_rate:   fa.CellField[ta.wpfloat],             # output
):
    _graupel_to_rain(t, p, rho, dvsw0, qg, QMIN, TX, TMELT, out=rain_rate)
