# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The TKE profile the vertical TKE diffusion starts from.

Translated from 'icon/src/atm_phy_schemes/turb_diffusion.f90', SUBROUTINE 'turbdiff', from the
section whose banner reads

    "6) Berechnung der Diffusionstendenz von q=SQRT(2*TKE) einschliesslich der q-Tendenz durch
        den Zirkulationsterm"
    -- "Calculation of the diffusion tendency of q = SQRT(2*TKE), including the q-tendency due
       to the circulation term"

(:2118-2310 at icon commit 26d6b98cce, the commit that produced the reference capture), from the
block Raschendorfer heads

    "Vorbereitung zur Bestimmung der zugehoerigen Incremente von TKE=(q**2)/2:"
    -- "Preparation for determining the corresponding increments of TKE = q**2/2"

at :2122-2124. The scientific commentary in that file is by Matthias Raschendorfer (DWD).
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_saved_tke_profile(
    turbulent_velocity_scale: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """Turbulent kinetic energy from the turbulent velocity scale, 'TKE = q**2 / 2'.

    THE SQUARE IS WRITTEN AS A PRODUCT, NOT AS '**2'. Fortran's integer-exponent '**' is a
    multiplication, while GT4Py lowers Python's 'x**2' to 'math.pow(x, 2)' and CUDA's 'pow'
    carries up to 2 ulp. See '_compute_mechanical_forcing', which is where that was measured.

    The half is applied to the square rather than to one factor, mirroring the Fortran's
    'z1d2*tke(i,k,ntur)**2' in which '**' binds tighter than '*'. (Both associations happen to
    give the same answer here, since multiplying by 0.5 is exact away from the subnormals, but
    the port mirrors the operation order rather than arguing about when it may not.)
    """
    return wpfloat("0.5") * (turbulent_velocity_scale * turbulent_velocity_scale)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_saved_tke_profile(
    turbulent_velocity_scale: fa.CellKField[wpfloat],
    saved_tke_profile: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Save the pre-diffusion TKE profile 'sav_prof' on half levels (turb_diffusion.f90:2180-2185).

    'turbdiff' carries the turbulent velocity scale 'q = SQRT(2*TKE)' in the array named 'tke'
    throughout, which is what Raschendorfer's note at :2210-2211 is about:

        "Das Feld 'tke' enthaelt nicht TKE sondern q=SQRT(2TKE)!"
        -- "The field 'tke' does NOT contain TKE but q = SQRT(2*TKE)!"

    Section 9) diffuses this profile and section 10) forms the tendency as the difference
    between the diffused profile and this saved one, so it has to exist as a separate array
    before the solver overwrites anything. It is also the reference profile of the virtual TKE
    profile that section 8) builds for the circulation term.

    WHY THERE IS NO 'imode_tkediff == 1' BRANCH HERE. The Fortran has one -- at
    'imode_tkediff == 1' the diffusion is formulated in 'q' rather than in TKE, so the saved
    profile is 'q' itself and the discretisation momentum 'dicke' picks up an extra factor 'q'
    (:2187-2207). 'imode_tkediff' is frozen at its compiled-in default 2 in this port
    ('turbulence.py', FROZEN_SWITCHES; port spec 4.3: not one of the 22 "which formulation"
    switches is set in any of the 641 configurations under 'icon/run/'), so only the TKE
    formulation is ported. Its signature in the reference data is that 'dicke' is byte-identical
    across this section, which 'test_the_capture_diffuses_tke_and_not_q' asserts.

    Args:
        turbulent_velocity_scale: 'tke(:,:,ntur)', 'q = SQRT(2*TKE)' [m/s], half levels, as
            sections 3) and 4) leave it.
        saved_tke_profile: Output, 'sav_prof' = 'zaux(:,:,2)', turbulent kinetic energy
            [m2/s2], half levels.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First half level; 1, mirroring Fortran 'k=2'. The model top is not
            written -- 'turbdiff' has no TKE-diffusion level there.
        vertical_end: End of the half levels; 'ke1', mirroring Fortran 'k=...,ke1'. The surface
            half level IS written, unlike in most other sections.
    """
    _compute_saved_tke_profile(
        turbulent_velocity_scale=turbulent_velocity_scale,
        out=saved_tke_profile,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
