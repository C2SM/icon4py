# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from gt4py.next.ffront.fbuiltins import power
from icon4py.model.common import field_type_aliases as fa, type_alias as ta

@gtx.field_operator
def _deposition_factor(
    t:            fa.CellField[ta.wpfloat],             # Temperature
    qvsi:         fa.CellField[ta.wpfloat],             # Saturation (ice) specific vapor mass
    ice_dep:      fa.CellField[ta.wpfloat],             # Rate of ice deposition (some to snow)
    QMIN:         ta.wpfloat,
    ALS:          ta.wpfloat,
    RD:           ta.wpfloat, 
    RV:           ta.wpfloat, 
    TMELT:        ta.wpfloat,
) -> fa.CellField[ta.wpfloat]:                          # Deposition factor
    KAPPA    = 2.40e-2                                  # Thermal conductivity of dry air
    B        = 1.94                                     # Exponent
    A        = ALS*ALS / (KAPPA*RV)                     # TBD
    CX       = 2.22e-5 * power(TMELT, (-B)) * 101325.0  # TBD

    return  ( CX / RD * power(t, B-1.0) ) / (1.0 + A*x*qvsi / (t*t) )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def deposition_factor(
    t:            fa.CellField[ta.wpfloat],             # Temperature
    qvsi:         fa.CellField[ta.wpfloat],             # Saturation (ice) specific vapor mass
    ice_dep:      fa.CellField[ta.wpfloat],             # Rate of ice deposition (some to snow) 
    QMIN:         ta.wpfloat,
    ALS:          ta.wpfloat,
    RD:	      	  ta.wpfloat,
    RV:	      	  ta.wpfloat,
    TMELT:        ta.wpfloat,
    deposition_factor: fa.CellField[ta.wpfloat],        # output
):
    _deposition_factor(t, qvsi, ice_dep, QMIN, ALS, RD, RV, TMELT, out=deposition_factor)
