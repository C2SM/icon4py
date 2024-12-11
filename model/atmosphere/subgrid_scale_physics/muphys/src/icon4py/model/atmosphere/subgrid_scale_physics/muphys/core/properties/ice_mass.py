# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from gt4py.next.ffront.fbuiltins import maximum, minimum
from icon4py.model.common import field_type_aliases as fa, type_alias as ta

@gtx.field_operator
def _ice_mass(
    qi:        fa.CellField[ta.wpfloat],             # Specific humidity of ice
    ni:        fa.CellField[ta.wpfloat],             # Ice crystal number
    M0_ICE:    ta.wpfloat,
) -> fa.CellField[ta.wpfloat]:                       # Ice mass
    MI_MAX = 1.0e-9
    return maximum(M0_ICE*ni, minimum(qi/ni, MI_MAX))

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def ice_mass(
    qi:        fa.CellField[ta.wpfloat],             # Specific humidity of ice
    ni:        fa.CellField[ta.wpfloat],             # Ice crystal number
    M0_ICE:    ta.wpfloat,
    ice_mass: fa.CellField[ta.wpfloat]  # output
):
    _ice_mass( qi, ni, M0_ICE, out=ice_mass )
