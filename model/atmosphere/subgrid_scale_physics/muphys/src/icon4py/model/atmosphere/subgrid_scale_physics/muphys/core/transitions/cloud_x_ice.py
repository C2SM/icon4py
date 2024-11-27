# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from gt4py.next.ffront.fbuiltins import where
from icon4py.model.common import field_type_aliases as fa, type_alias as ta

@gtx.field_operator
def _cloud_x_ice(
    t:         fa.CellField[ta.wpfloat],             # Temperature
    qc:        fa.CellField[ta.wpfloat],             # Cloud specific mass
    qi:        fa.CellField[ta.wpfloat],             # Ice specific mass
    dt:        ta.wpfloat,                           # time step
    TFRZ_HOM:  ta.wpfloat,
    QMIN:      ta.wpfloat,
    TMELT:     ta.wpfloat,
) -> fa.CellField[ta.wpfloat]:                       # Homogeneous freezing rate

    result = where( (qc > QMIN) & (t < TFRZ_HOM), qc / dt, 0. )
    result = where( (qi > QMIN) & (t > TMELT), -qi/dt, result )
    return result

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def cloud_x_ice(
    t:                 fa.CellField[ta.wpfloat],     # Temperature
    qc:                fa.CellField[ta.wpfloat],     # Cloud specific mass
    qi:                fa.CellField[ta.wpfloat],     # Ice specific mass
    dt:                ta.wpfloat,                   # time step
    TFRZ_HOM:          ta.wpfloat,
    QMIN:              ta.wpfloat,
    TMELT:             ta.wpfloat,
    freezing_rate:     fa.CellField[ta.wpfloat],     # output
):
    _cloud_x_ice(t, qc, qi, dt, TFRZ_HOM, QMIN, TMELT, out=freezing_rate)
