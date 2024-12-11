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
def _fall_speed(
    density:      fa.CellField[ta.wpfloat],             # Density of 
    prefactor:    ta.wpfloat,
    offset:       ta.wpfloat,
    exponent:     ta.wpfloat, 
) -> fa.CellField[ta.wpfloat]:                          # Fall speed

    return prefactor * power((density+offset), exponent)

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def fall_speed(
    density:      fa.CellField[ta.wpfloat],             # Density of
    prefactor:    ta.wpfloat,
    offset:       ta.wpfloat,
    exponent:     ta.wpfloat,              
    fall_speed: fa.CellField[ta.wpfloat],        # output
):
    _fall_speed(density, prefactor, offset, exponent, out=fall_speed)
