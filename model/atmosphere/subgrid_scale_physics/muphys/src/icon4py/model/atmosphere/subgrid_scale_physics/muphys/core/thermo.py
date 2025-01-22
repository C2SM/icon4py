# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from gt4py.next.ffront.fbuiltins import exp
from icon4py.model.common import field_type_aliases as fa, type_alias as ta

@gtx.field_operator
def _qsat_ice_rho(
    t:         fa.CellKField[ta.wpfloat],             # Temperature
    rho:       fa.CellKField[ta.wpfloat],             # Density
    TMELT:     ta.wpfloat,
    RV:        ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:                       # Pressure
    C1ES   = 610.78
    C3IES  = 21.875
    C4IES  = 7.66

    return ( C1ES * exp( C3IES * ( t-TMELT ) / ( t-C4IES ) ) ) / (rho * RV * t)

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def qsat_ice_rho(
    t:         fa.CellKField[ta.wpfloat],             # Temperature
    rho:       fa.CellKField[ta.wpfloat],             # Density
    TMELT:     ta.wpfloat,
    RV:        ta.wpfloat,
    pressure: fa.CellKField[ta.wpfloat]               # output
):
    _qsat_ice_rho( t, rho, TMELT, RV, out=pressure )

@gtx.field_operator
def _qsat_rho(
    t:         fa.CellKField[ta.wpfloat],             # Temperature
    rho:       fa.CellKField[ta.wpfloat],             # Density
    TMELT:     ta.wpfloat,
    RV:        ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:                       # Pressure
    C1ES   = 610.78
    C3LES  = 17.269
    C4LES  = 35.86

    return ( C1ES * exp( C3LES * ( t-TMELT ) / ( t-C4LES ) ) ) / ( rho * RV * t )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def qsat_rho(
    t:         fa.CellKField[ta.wpfloat],             # Temperature
    rho:       fa.CellKField[ta.wpfloat],             # Density
    TMELT:     ta.wpfloat,
    RV:        ta.wpfloat,
    pressure: fa.CellKField[ta.wpfloat]               # output
):
    _qsat_rho( t, rho, TMELT, RV, out=pressure )


@gtx.field_operator
def _dqsatdT_rho(
    qs:        fa.CellKField[ta.wpfloat],             # Saturation vapor pressure (over liquid)
    t:         fa.CellKField[ta.wpfloat],             # Temperature
    TMELT:     ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:                       # derivative d(qsat_rho)/dT
    C3LES  = 17.269
    C4LES  = 35.86
    C5LES  = C3LES * (TMELT - C4LES)

    return qs * ( C5LES / ( (t-C4LES)*(t-C4LES) ) - 1.0/t )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def dqsatdT_rho(
    qs:        fa.CellKField[ta.wpfloat],             # Saturation vapor pressure (over liquid)
    t:         fa.CellKField[ta.wpfloat],             # Temperature
    TMELT:     ta.wpfloat,
    derivative: fa.CellKField[ta.wpfloat]             # output
):
    _dqsatdT_rho( qs, t, TMELT, out=derivative )
