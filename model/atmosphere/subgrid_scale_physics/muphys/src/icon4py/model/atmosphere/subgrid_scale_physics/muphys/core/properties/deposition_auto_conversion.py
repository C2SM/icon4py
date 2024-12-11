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
def _deposition_auto_conversion(
    qi:           fa.CellField[ta.wpfloat],             # Ice specific mass
    m_ice:        fa.CellField[ta.wpfloat],             # Ice crystal mass
    ice_dep:      fa.CellField[ta.wpfloat],             # Rate of ice deposition (some to snow)
    QMIN:         ta.wpfloat,
) -> fa.CellField[ta.wpfloat]:                          # Conversion rate
    M0_S     = 3.0e-9                                   # Initial mass of snow crystals
    B_DEP    = 0.66666666666667                         # Exponent
    XCRIT    = 1.0                                      # Critical threshold parameter
    
    return where( (qi > QMIN), maximum(0.0, ice_dep) * B_DEP / (power((M0_S/m_ice), B_DEP) - XCRIT), 0.0)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def deposition_auto_conversion(
    qi:           fa.CellField[ta.wpfloat],             # Ice specific mass
    m_ice:        fa.CellField[ta.wpfloat],             # Ice crystal mass
    ice_dep:      fa.CellField[ta.wpfloat],             # Rate of ice deposition (some to snow)
    QMIN:         ta.wpfloat,
    conversion_rate: fa.CellField[ta.wpfloat],          # output
):
    _deposition_auto_conversion(qi, m_ice, ice_dep, QMIN, out=conversion_rate)
