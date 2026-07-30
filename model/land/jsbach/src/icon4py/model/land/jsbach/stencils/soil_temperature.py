# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Soil energy: temperature back-substitution.

Back-substitution half of the JSBACH Richtmyer-Morton soil energy solve
(calc_soil_temperature, mo_sse_process.f90:487-504). Given the lagged tridiagonal
coefficients produced by the previous step's forward elimination, reconstruct the
soil temperature column top-down:

    soil_temperature(0)   = surface_temperature          # Dirichlet top boundary
    soil_temperature(k)   = acoef(k-1) + bcoef(k-1) * soil_temperature(k-1)

acoef(k)/bcoef(k) are the coefficients of the transition from layer k to k+1, so
the coefficients of the bottom layer are unused (no outgoing transition).
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.type_alias import wpfloat


@gtx.scan_operator(
    axis=dims.KDim,
    forward=True,
    init=(wpfloat("0.0"), wpfloat("0.0"), wpfloat("0.0"), gtx.int32(0)),
)
def _back_substitution_scan(
    state: tuple[ta.wpfloat, ta.wpfloat, ta.wpfloat, gtx.int32],
    acoef: ta.wpfloat,
    bcoef: ta.wpfloat,
    surface_temperature: ta.wpfloat,
) -> tuple[ta.wpfloat, ta.wpfloat, ta.wpfloat, gtx.int32]:
    # State carries the layer above: its temperature and the coefficients of the
    # transition into the current layer, plus the layer index k.
    temperature_above, acoef_above, bcoef_above, k = state
    temperature = surface_temperature if k == 0 else acoef_above + bcoef_above * temperature_above
    return (temperature, acoef, bcoef, k + 1)


@gtx.field_operator
def _soil_temperature_back_substitution(
    acoef: fa.CellKField[ta.wpfloat],
    bcoef: fa.CellKField[ta.wpfloat],
    surface_temperature: fa.CellField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    temperature, _, _, _ = _back_substitution_scan(acoef, bcoef, surface_temperature)
    return temperature


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def soil_temperature_back_substitution(
    acoef: fa.CellKField[ta.wpfloat],
    bcoef: fa.CellKField[ta.wpfloat],
    surface_temperature: fa.CellField[ta.wpfloat],
    soil_temperature: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _soil_temperature_back_substitution(
        acoef,
        bcoef,
        surface_temperature,
        out=soil_temperature,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
