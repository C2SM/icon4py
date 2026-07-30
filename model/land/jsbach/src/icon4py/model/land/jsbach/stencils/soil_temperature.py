# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Soil energy: temperature back-substitution.

Back-substitution half of the JSBACH Richtmyer-Morton soil energy solve
(calc_soil_temperature, mo_sse_process.f90:487-504). Field names follow the
JSBACH source for traceability against the serialized reference.
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.type_alias import wpfloat


@gtx.scan_operator(
    axis=dims.KDim,
    forward=True,
    init=(wpfloat("0.0"), wpfloat("0.0"), wpfloat("0.0"), gtx.int32(0)),
)
def _soil_temperature_back_substitution_scan(
    state: tuple[ta.wpfloat, ta.wpfloat, ta.wpfloat, gtx.int32],
    t_soil_acoef: ta.wpfloat,
    t_soil_bcoef: ta.wpfloat,
    t_soil_top: ta.wpfloat,
) -> tuple[ta.wpfloat, ta.wpfloat, ta.wpfloat, gtx.int32]:
    # State carries the layer above: its temperature and the coefficients of the
    # transition into the current layer (a/b are indexed by their source layer),
    # plus the layer index k.
    t_soil_above, t_soil_acoef_above, t_soil_bcoef_above, k = state
    t_soil = t_soil_top if k == 0 else t_soil_acoef_above + t_soil_bcoef_above * t_soil_above
    return (t_soil, t_soil_acoef, t_soil_bcoef, k + 1)


@gtx.field_operator
def _soil_temperature_back_substitution(
    t_soil_acoef: fa.CellKField[ta.wpfloat],
    t_soil_bcoef: fa.CellKField[ta.wpfloat],
    t_soil_top: fa.CellField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """Reconstruct the soil temperature column from the lagged R&M coefficients.

    t_soil_sl(0)   = t_soil_top                                  (Dirichlet top BC)
    t_soil_sl(k)   = t_soil_acoef(k-1) + t_soil_bcoef(k-1) * t_soil_sl(k-1)

    a/b of the bottom layer are unused (no outgoing transition).

    Args:
        t_soil_acoef:  R&M A coefficient, source-layer indexed [K]
        t_soil_bcoef:  R&M B coefficient, source-layer indexed [-]
        t_soil_top:    surface temperature, the column top boundary [K]

    Returns:
        soil temperature per layer [K]
    """
    t_soil_sl, _, _, _ = _soil_temperature_back_substitution_scan(
        t_soil_acoef, t_soil_bcoef, t_soil_top
    )
    return t_soil_sl


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def soil_temperature_back_substitution(
    t_soil_acoef: fa.CellKField[ta.wpfloat],
    t_soil_bcoef: fa.CellKField[ta.wpfloat],
    t_soil_top: fa.CellField[ta.wpfloat],
    t_soil_sl: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _soil_temperature_back_substitution(
        t_soil_acoef,
        t_soil_bcoef,
        t_soil_top,
        out=t_soil_sl,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.scan_operator(
    axis=dims.KDim,
    forward=False,
    init=(
        wpfloat("0.0"),  # t_soil_acoef of the layer below
        wpfloat("0.0"),  # t_soil_bcoef of the layer below
        wpfloat("0.0"),  # zdz2 of the layer below
        wpfloat("0.0"),  # zdz1 of the layer below
        wpfloat("0.0"),  # t_soil_sl of the layer below
        gtx.int32(0),  # number of layers already processed below this one
    ),
)
def _soil_temperature_coefficients_scan(
    state: tuple[ta.wpfloat, ta.wpfloat, ta.wpfloat, ta.wpfloat, ta.wpfloat, gtx.int32],
    t_soil_sl: ta.wpfloat,
    vol_heat_cap: ta.wpfloat,
    heat_cond: ta.wpfloat,
    dz: ta.wpfloat,
    zd1: ta.wpfloat,
    delta_time: ta.wpfloat,
) -> tuple[ta.wpfloat, ta.wpfloat, ta.wpfloat, ta.wpfloat, ta.wpfloat, gtx.int32]:
    (
        t_soil_acoef_below,
        t_soil_bcoef_below,
        zdz2_below,
        zdz1_below,
        t_soil_below,
        n_below,
    ) = state
    zdz2 = dz * vol_heat_cap / delta_time
    zdz1 = zd1 * heat_cond
    # n_below == 0: bottom layer, its coefficients are unused.
    # n_below == 1: the layer below is the bottom -> division form (Fortran :722-727).
    # n_below >= 2: interior -> reciprocal-multiply form (Fortran :735-739).
    denom_bottom = zdz2_below + zdz1
    z1_interior = 1.0 / (zdz2_below + zdz1 + zdz1_below * (1.0 - t_soil_bcoef_below))
    t_soil_acoef = (
        wpfloat("0.0")
        if n_below == 0
        else (
            zdz2_below * t_soil_below / denom_bottom
            if n_below == 1
            else (t_soil_below * zdz2_below + zdz1_below * t_soil_acoef_below) * z1_interior
        )
    )
    t_soil_bcoef = (
        wpfloat("0.0")
        if n_below == 0
        else (zdz1 / denom_bottom if n_below == 1 else zdz1 * z1_interior)
    )
    return (t_soil_acoef, t_soil_bcoef, zdz2, zdz1, t_soil_sl, n_below + 1)


@gtx.field_operator
def _soil_temperature_coefficients(
    t_soil_sl: fa.CellKField[ta.wpfloat],
    vol_heat_cap: fa.CellKField[ta.wpfloat],
    heat_cond: fa.CellKField[ta.wpfloat],
    dz: fa.KField[ta.wpfloat],
    zd1: fa.KField[ta.wpfloat],
    delta_time: ta.wpfloat,
) -> tuple[fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    """Build the next-step R&M A/B coefficients by upward forward elimination.

    A reverse (bottom-up) sweep of the tridiagonal soil energy system. Coefficient
    of layer k is the transition k -> k+1, so the bottom layer's coefficients are
    unused. The bottom layer uses a division form, the interior a reciprocal form
    (kept distinct for bit-fidelity with the Fortran).

    Args:
        t_soil_sl:     soil temperature per layer [K]
        vol_heat_cap:  volumetric heat capacity per layer [J/m^3/K]
        heat_cond:     heat conductivity per layer [W/m/K]
        dz:            soil layer thickness [m]
        zd1:           inverse spacing between layer mid-depths, 1/(mids(k+1)-mids(k));
                       the bottom-layer entry is unused and must be zero [1/m]
        delta_time:    time step [s]

    Returns:
        (t_soil_acoef [K], t_soil_bcoef [-])
    """
    t_soil_acoef, t_soil_bcoef, _, _, _, _ = _soil_temperature_coefficients_scan(
        t_soil_sl, vol_heat_cap, heat_cond, dz, zd1, delta_time
    )
    return t_soil_acoef, t_soil_bcoef


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def soil_temperature_coefficients(
    t_soil_sl: fa.CellKField[ta.wpfloat],
    vol_heat_cap: fa.CellKField[ta.wpfloat],
    heat_cond: fa.CellKField[ta.wpfloat],
    dz: fa.KField[ta.wpfloat],
    zd1: fa.KField[ta.wpfloat],
    delta_time: ta.wpfloat,
    t_soil_acoef: fa.CellKField[ta.wpfloat],
    t_soil_bcoef: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _soil_temperature_coefficients(
        t_soil_sl,
        vol_heat_cap,
        heat_cond,
        dz,
        zd1,
        delta_time,
        out=(t_soil_acoef, t_soil_bcoef),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _soil_ground_heat_flux(
    t_soil_sl: fa.CellKField[ta.wpfloat],
    t_soil_acoef: fa.CellKField[ta.wpfloat],
    t_soil_bcoef: fa.CellKField[ta.wpfloat],
    vol_heat_cap: fa.CellKField[ta.wpfloat],
    heat_cond: fa.CellKField[ta.wpfloat],
    dz: fa.KField[ta.wpfloat],
    zd1: fa.KField[ta.wpfloat],
    delta_time: ta.wpfloat,
) -> tuple[fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    """Surface diffusive ground heat flux and ground heat capacity (Fortran :748-751).

    Evaluated per level from that level's R&M coefficients; the caller restricts the
    vertical domain to the top (ground) layer, where these surface quantities live.

    Args:
        t_soil_sl:     soil temperature per layer [K]
        t_soil_acoef:  R&M A coefficient [K]
        t_soil_bcoef:  R&M B coefficient [-]
        vol_heat_cap:  volumetric heat capacity per layer [J/m^3/K]
        heat_cond:     heat conductivity per layer [W/m/K]
        dz:            soil layer thickness [m]
        zd1:           inverse spacing between layer mid-depths [1/m]
        delta_time:    time step [s]

    Returns:
        (grnd_hflx [W/m^2], hcap_grnd [J/m^2/K])
    """
    zdz1 = zd1 * heat_cond
    zdz2 = dz * vol_heat_cap / delta_time
    grnd_hflx = zdz1 * (t_soil_acoef + (t_soil_bcoef - 1.0) * t_soil_sl)
    hcap_grnd = zdz2 * delta_time + delta_time * (1.0 - t_soil_bcoef) * zdz1
    return grnd_hflx, hcap_grnd


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def soil_ground_heat_flux(
    t_soil_sl: fa.CellKField[ta.wpfloat],
    t_soil_acoef: fa.CellKField[ta.wpfloat],
    t_soil_bcoef: fa.CellKField[ta.wpfloat],
    vol_heat_cap: fa.CellKField[ta.wpfloat],
    heat_cond: fa.CellKField[ta.wpfloat],
    dz: fa.KField[ta.wpfloat],
    zd1: fa.KField[ta.wpfloat],
    delta_time: ta.wpfloat,
    grnd_hflx: fa.CellKField[ta.wpfloat],
    hcap_grnd: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _soil_ground_heat_flux(
        t_soil_sl,
        t_soil_acoef,
        t_soil_bcoef,
        vol_heat_cap,
        heat_cond,
        dz,
        zd1,
        delta_time,
        out=(grnd_hflx, hcap_grnd),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
