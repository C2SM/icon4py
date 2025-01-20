# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from gt4py.next.ffront.fbuiltins import where, power, exp, maximum, minimum
from icon4py.model.common import field_type_aliases as fa, type_alias as ta

@gtx.field_operator
def _vel_scale_factor_i(
    xrho:     fa.CellField[ta.wpfloat],             # sqrt(rho_00/rho)
    B_I:      ta.wpfloat,                           # 2/3
) -> fa.CellField[ta.wpfloat]:                      # Snow number
    return power( xrho, B_I )

@gtx.field_operator
def _vel_scale_factor_s(
    xrho:     fa.CellField[ta.wpfloat],             # sqrt(rho_00/rho)
    rho:      fa.CellField[ta.wpfloat],             # Density of condensate
    t:        fa.CellField[ta.wpfloat],             # Temperature
    qx:       fa.CellField[ta.wpfloat],             # Specific mass
    B_S:      ta.wpfloat,                           # 
    QMIN:     ta.wpfloat,                           #
    AMS:      ta.wpfloat,                           #
    TMELT:    ta.wpfloat,                           #
) -> fa.CellField[ta.wpfloat]:                      # Scale factor
    return xrho * power( snow_number( t, rho, qx, QMIN, AMS, TMELT ),  B_S )

@gtx.field_operator
def _vel_scale_factor_others(
    xrho:     fa.CellField[ta.wpfloat],             # sqrt(rho_00/rho)
) -> fa.CellField[ta.wpfloat]:                      # Scale factor
    return xrho

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def vel_scale_factor_lqi(
    xrho:     fa.CellField[ta.wpfloat],             # sqrt(rho_00/rho)
    B_I:      ta.wpfloat,                           # 2/3
    scale_factor: fa.CellField[ta.wpfloat]          # output
}:
    _vel_scale_factor_lqi( xrho, B_I, out=scale_factor )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def vel_scale_factor_lqs(
    xrho:     fa.CellField[ta.wpfloat],             # sqrt(rho_00/rho)
    rho:      fa.CellField[ta.wpfloat],             # Density of condensate
    t:        fa.CellField[ta.wpfloat],             # Temperature
    qx:       fa.CellField[ta.wpfloat],             # Specific mass
    B_S:      ta.wpfloat,                           #
    QMIN:     ta.wpfloat,                           #
    AMS:      ta.wpfloat,                           #
    TMELT:    ta.wpfloat,                           #
    scale_factor: fa.CellField[ta.wpfloat]          # output
):
    _vel_scale_factor_lqs( xrho, rho, t, qx, B_S, QMIN, AMS, TMELT, out=scale_factor )
