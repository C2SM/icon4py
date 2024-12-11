# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from gt4py.next.ffront.fbuiltins import exp, maximum, minimum
from icon4py.model.common import field_type_aliases as fa, type_alias as ta

@gtx.field_operator
def _ice_sticking(
    t:        fa.CellField[ta.wpfloat],             # Temperature
    TMELT:    ta.wpfloat,
) -> fa.CellField[ta.wpfloat]:                      # Ice sticking
    A_FREEZ   = 0.09         # Scale factor for freezing depression
    B_MAX_EXP = 1.00         # Maximum for exponential temperature factor
    EFF_MIN   = 0.075        # Minimum sticking efficiency
    EFF_FAC   = 3.5e-3       # Scaling factor [1/K] for cloud ice sticking efficiency
    TCRIT     = TMELT - 85.0 # Temperature at which cloud ice autoconversion starts

    return maximum( minimum( exp( A_FREEZ * (t - TMELT), B_MAX_EXP ), EFF_MIN ), EFF_FAC * (t - TCRIT) )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def ice_sticking(
    t:        fa.CellField[ta.wpfloat],             # Temperature
    TMELT:    ta.wpfloat,
    ice_sticking: fa.CellField[ta.wpfloat]  # output
):
    _ice_sticking( qi, ni, M0_ICE, out=ice_sticking )
