# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from gt4py.next.ffront.fbuiltins import where, power
from icon4py.model.common import field_type_aliases as fa, type_alias as ta

@gtx.field_operator
def _ice_to_graupel(
    rho:          fa.CellField[ta.wpfloat],             # Ambient density
    qr:           fa.CellField[ta.wpfloat],             # Rain specific mass
    qg:           fa.CellField[ta.wpfloat],             # Graupel specific mass
    qi:           fa.CellField[ta.wpfloat],             # Ice specific mass
    sticking_eff: fa.CellField[ta.wpfloat],             # Sticking efficiency
    QMIN:         ta.wpfloat,
) -> fa.CellField[ta.wpfloat]:                          # Aggregation of ice by graupel
    A_CT     = 1.72                                     # (15/32)*(PI**0.5)*(EIR/RHOW)*V0R*AR**(1/8)    
    B_CT     = 0.875                                    # Exponent = 7/8
    C_AGG_CT = 2.46
    B_AGG_CT = 0.94878                                  # Exponent 
    result = where( (qi > QMIN) & (qg > QMIN), sticking_eff * qi * C_AGG_CT * power(rho*qg, B_AGG_CT), 0. )
    result = where( (qi > QMIN) & (qr > QMIN), result + A_CT*qi*power(rho*qr, B_CT), result )
    return result

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def ice_to_graupel(
    rho:          fa.CellField[ta.wpfloat],             # Ambient density
    qr:           fa.CellField[ta.wpfloat],             # Rain specific mass
    qg:           fa.CellField[ta.wpfloat],             # Graupel specific mass
    qi:           fa.CellField[ta.wpfloat],             # Ice specific mass
    sticking_eff: fa.CellField[ta.wpfloat],             # Sticking efficiency
    QMIN:         ta.wpfloat,
    aggregation:  fa.CellField[ta.wpfloat],             # output
):
    _ice_to_graupel(rho, qr, qg, qi, sticking_eff, QMIN, out=aggregation)
