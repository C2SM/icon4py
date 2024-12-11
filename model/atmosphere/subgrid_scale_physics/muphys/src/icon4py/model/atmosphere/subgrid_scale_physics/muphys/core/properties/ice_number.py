# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from gt4py.next.ffront.fbuiltins import exp, minimum
from icon4py.model.common import field_type_aliases as fa, type_alias as ta

@gtx.field_operator
def _ice_number(
    t:         fa.CellField[ta.wpfloat],             # Ambient temperature
    rho:       fa.CellField[ta.wpfloat],             # Ambient density
    TMELT:     ta.wpfloat,
) -> fa.CellField[ta.wpfloat]:                       # Ice number
    A_COOP = 5.000                                   # Parameter in cooper fit
    B_COOP = 0.304                                   # Parameter in cooper fit
    NIMAX  = 250.e+3                                 # Maximal number of ice crystals
    return minimum(NIMAX, A_COOP * exp( B_COOP * (TMELT - t) ) ) / rho

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def ice_number(
    t:         fa.CellField[ta.wpfloat],             # Ambient temperature
    rho:       fa.CellField[ta.wpfloat],             # Ambient density
    TMELT:     ta.wpfloat,
    ice_number: fa.CellField[ta.wpfloat]  # output
):
    _ice_number( qi, ni, M0_ICE, out=ice_number )
