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
def _snow_lambda(
    rho:        fa.CellField[ta.wpfloat],           # Ambient density
    qs:         fa.CellField[ta.wpfloat],           # Snow specific mass
    ns:         fa.CellField[ta.wpfloat],           # Snow number
    QMIN:       ta.wpfloat,                         # 
    AMS:        ta.wpfloat,                         # 
    BMS:        ta.wpfloat,                         # 
) -> fa.CellField[ta.wpfloat]:                      # Riming snow rate
    A2     = AMS * 2.0            # (with ams*gam(bms+1.0_wp) where gam(3) = 2)
    LMD_0  = 1.0e+10              # no snow value of lambda
    BX     = 1.0 / ( BMS + 1.0 )  # Exponent
    QSMIN  = 0.0e-6               # TODO: Check with Georgiana that this value is correct

    return where( qs > QMIN, power( (A2*ns / ((qs + QSMIN) * rho)), BX ), LMD_0 )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def snow_lambda(
    rho:        fa.CellField[ta.wpfloat],           # Ambient density
    qs:         fa.CellField[ta.wpfloat],           # Snow specific mass
    ns:         fa.CellField[ta.wpfloat],           # Snow number
    QMIN:       ta.wpfloat,                         # 
    AMS:        ta.wpfloat,                         # 
    BMS:        ta.wpfloat,                         # 
    riming_snow_rate: fa.CellField[ta.wpfloat]  # output
):
    _snow_lambda( rho, qs, ns, QMIN, AMS, BMS, out=riming_snow_rate )
