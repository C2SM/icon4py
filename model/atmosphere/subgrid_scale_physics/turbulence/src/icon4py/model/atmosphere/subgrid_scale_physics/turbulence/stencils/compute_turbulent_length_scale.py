# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The turbulent master length scale.

Translated from 'icon/src/atm_phy_schemes/turb_diffusion.f90', SUBROUTINE 'turbdiff', the closing
block of section 0), lines 1087-1147 at icon commit 26d6b98cce, the commit that produced the
reference capture. Raschendorfer heads it "Berechnung der turbulenten Laengenscalen" --
"Calculation of the turbulent length scales" -- and its second half "Uebergang von der maximalen
turbulenten Laengenskala zur effektiven turbulenten Laengenskala" -- "Transition from the maximal
turbulent length scale to the effective turbulent length scale". The scientific commentary in that
file is by Matthias Raschendorfer (DWD).

    len_scale(i,ke1) = gz0(i)*edgrav
    DO k=kcm-1,1,-1
      len_scale(i,k) = dicke(i,k) + len_scale(i,k+1)
    DO k=ke1,1,-1
      len_scale(i,k) = akt*MAX( len_min, l_scal(i)*len_scale(i,k)/(l_scal(i)+len_scale(i,k)) )

WHAT IT IS. The raw length scale at a half level is its height above the aerodynamic surface: the
roughness length 'z0 = gz0/g' plus every layer depth between it and the surface. An eddy can be
no larger than its distance from the ground, so this is the mixing length of a neutral,
unbounded surface layer -- which is what 'akt', von Karman's constant, then scales. The harmonic
combination with 'l_scal' caps it at the asymptotic maximum 'tur_len', reduced where the
horizontal grid spacing is too coarse to resolve eddies of that size ('l_scal' is
'MIN(l_hori/2, tur_len)', turb_utilities.f90:341).

THIS ONE IS A SCAN, and the only one in this section. The cumulative sum is a genuine recurrence
-- row 'k' needs the value at 'k+1' -- and its result is NOT the telescoped difference
'hhl(k) - hhl(ke1) + z0' that the layer depths sum to algebraically, because the two round
differently. Section 1a) is what the length scale feeds, and its gate is bit-exact, so the
accumulation is reproduced and not shortcut.

THE ROUGHNESS-LAYER LOOP IS DEAD. Between the surface value and the accumulation the Fortran has
a second loop, 'DO k=ke,kcm,-1', which limits the length scale by the free space between the
roughness elements. Raschendorfer's own note on it (turb_diffusion.f90:1105-1107):

    "US: Up to now it is kcm = ke+1 and the next vertical loop will not be executed!! If a canopy
     layer is implemented, kcm will be <= ke."

Measured against this capture: 'kcm = 81 = ke + 1', so the loop body never runs, and 'c_big' and
'r_air' are not passed to 'turbdiff' at all. The accumulation below therefore starts at 'ke' and
not at 'kcm - 1'. (The savepoint reader's docstring for 'kcm()' says "ke1 + 1 when the canopy is
switched off"; the measured value is 'ke1'.)
"""

import gt4py.next as gtx
from gt4py.next import maximum
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.thermodynamic_functions import (
    ThermoConstants,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.scan_operator(axis=dims.KDim, forward=False, init=wpfloat("0.0"))
def _accumulate_upward_from_the_surface(carry: wpfloat, increment: wpfloat) -> wpfloat:
    """Running sum from the surface row upward: 'state(k) = increment(k) + state(k+1)'.

    Started at zero rather than at the roughness length, because a scan's initial state is a
    scalar and the roughness length is not: the caller puts 'gz0*edgrav' into the surface row of
    the increment instead, and '0 + x' is exactly 'x'.
    """
    return increment + carry


@gtx.field_operator
def _effective_length_scale(
    maximal_length_scale: fa.CellKField[wpfloat],
    horizontal_length_scale_limit: fa.CellField[wpfloat],
    von_karman_constant: wpfloat,
    minimal_length_scale: wpfloat,
) -> fa.CellKField[wpfloat]:
    """The harmonic cap on the accumulated height, times von Karman's constant.

    'akt*MAX( len_min, l_scal*len_scale/(l_scal+len_scale) )'. A separate operator only because
    naming the scan's result in the caller and reading it twice makes gt4py 1.1.10 emit a C++
    identifier containing the non-ASCII uniquifier 'U+141E', which gcc rejects with "stray
    '\341' in program"; as a parameter of its own operator the same value survives.
    """
    return von_karman_constant * maximum(
        minimal_length_scale,
        horizontal_length_scale_limit
        * maximal_length_scale
        / (horizontal_length_scale_limit + maximal_length_scale),
    )


@gtx.field_operator
def _compute_turbulent_length_scale(
    layer_depth: fa.CellKField[wpfloat],
    roughness_length_times_gravity: fa.CellField[wpfloat],
    horizontal_length_scale_limit: fa.CellField[wpfloat],
    nlev: gtx.int32,
    von_karman_constant: wpfloat,
    minimal_length_scale: wpfloat,
) -> fa.CellKField[wpfloat]:
    """The effective turbulent length scale on every half level [m].

    Run the program with 'vertical_start = 0', 'vertical_end = nlev + 1'; 'nlev' must be that
    same last row, since it is what the roughness length is placed on. The whole column is
    written, model top included, which is why nothing here needs a boundary domain.

    Args:
        layer_depth: 'dicke' as section 0) leaves it, the geometric depth of the main layers [m];
            read on rows 0 to 'nlev - 1' only, which is where 'compute_layer_depth' defines it
        roughness_length_times_gravity: 'gz0' [m2/s2]
        horizontal_length_scale_limit: 'l_scal', the reduced asymptotic maximum [m]
        nlev: 'ke', the row of the surface half level
        von_karman_constant: 'akt' [-]
        minimal_length_scale: 'len_min' [m]

    Returns:
        'len_scale', the turbulent master length scale [m]
    """
    increment = concat_where(
        dims.KDim == nlev, roughness_length_times_gravity * ThermoConstants.EDGRAV, layer_depth
    )
    return _effective_length_scale(
        _accumulate_upward_from_the_surface(increment),
        horizontal_length_scale_limit,
        von_karman_constant,
        minimal_length_scale,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_turbulent_length_scale(
    layer_depth: fa.CellKField[wpfloat],
    roughness_length_times_gravity: fa.CellField[wpfloat],
    horizontal_length_scale_limit: fa.CellField[wpfloat],
    nlev: gtx.int32,
    von_karman_constant: wpfloat,
    minimal_length_scale: wpfloat,
    turbulent_length_scale: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_turbulent_length_scale(
        layer_depth=layer_depth,
        roughness_length_times_gravity=roughness_length_times_gravity,
        horizontal_length_scale_limit=horizontal_length_scale_limit,
        nlev=nlev,
        von_karman_constant=von_karman_constant,
        minimal_length_scale=minimal_length_scale,
        out=turbulent_length_scale,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
