# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Snow energy: temperature back-substitution.

Back-substitution half of the JSBACH snow energy solve (calc_snow_temperature,
mo_sse_process.f90:796-838). The snowpack occupies a variable set of layers: the
top layer index `itop` (1-based, as in the Fortran) marks the uppermost present
layer, and layers above it are snow-free. Layers with (0-based) index k < itop
keep the surface temperature; deeper layers are reconstructed from the lagged R&M
coefficients.

The coefficient re-seeding for newly-formed layers (:809-822) and the coefficient
build (calc_snow_abcoeff) are NOT ported here: they use data-dependent indexing
(t_snow_acoef(:,itop_old), grnd_hflx at is=itop) that needs a deliberate
gather/scatter design decision. Field names follow the JSBACH source.
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.type_alias import wpfloat


@gtx.scan_operator(
    axis=dims.KDim,
    forward=True,
    init=(wpfloat("0.0"), wpfloat("0.0"), wpfloat("0.0"), gtx.int32(0)),
)
def _snow_temperature_back_substitution_scan(
    state: tuple[ta.wpfloat, ta.wpfloat, ta.wpfloat, gtx.int32],
    t_snow_acoef: ta.wpfloat,
    t_snow_bcoef: ta.wpfloat,
    t_srf: ta.wpfloat,
    itop: gtx.int32,
) -> tuple[ta.wpfloat, ta.wpfloat, ta.wpfloat, gtx.int32]:
    t_snow_above, t_snow_acoef_above, t_snow_bcoef_above, k = state
    # Absent/top present layers (k < itop) keep the surface temperature; since
    # itop >= 1 this also covers the always-surface top layer k == 0.
    t_snow = t_srf if k < itop else t_snow_acoef_above + t_snow_bcoef_above * t_snow_above
    return (t_snow, t_snow_acoef, t_snow_bcoef, k + 1)


@gtx.field_operator
def _snow_temperature_back_substitution(
    t_snow_acoef: fa.CellKField[ta.wpfloat],
    t_snow_bcoef: fa.CellKField[ta.wpfloat],
    t_srf: fa.CellField[ta.wpfloat],
    itop: fa.CellField[gtx.int32],
) -> fa.CellKField[ta.wpfloat]:
    """Reconstruct the snow temperature column below the variable top layer.

    Args:
        t_snow_acoef:  R&M A coefficient, source-layer indexed [K]
        t_snow_bcoef:  R&M B coefficient, source-layer indexed [-]
        t_srf:         surface temperature at the top of the snowpack [K]
        itop:          1-based index of the uppermost present snow layer

    Returns:
        snow temperature per layer [K]
    """
    t_snow, _, _, _ = _snow_temperature_back_substitution_scan(
        t_snow_acoef, t_snow_bcoef, t_srf, itop
    )
    return t_snow


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def snow_temperature_back_substitution(
    t_snow_acoef: fa.CellKField[ta.wpfloat],
    t_snow_bcoef: fa.CellKField[ta.wpfloat],
    t_srf: fa.CellField[ta.wpfloat],
    itop: fa.CellField[gtx.int32],
    t_snow: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _snow_temperature_back_substitution(
        t_snow_acoef,
        t_snow_bcoef,
        t_srf,
        itop,
        out=t_snow,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
