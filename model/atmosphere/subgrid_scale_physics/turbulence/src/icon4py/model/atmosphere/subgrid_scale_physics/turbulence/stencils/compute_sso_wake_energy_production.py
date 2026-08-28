# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The kinetic energy the mean flow loses to the SSO wakes, at main levels.

Translated from 'icon/src/atm_phy_schemes/turb_diffusion.f90', SUBROUTINE 'turbdiff', section
2a), :1557-1568 at icon commit 26d6b98cce, the commit that produced the reference capture. The
scientific commentary in that file is by Matthias Raschendorfer (DWD).

    hlp(i,k) = ut_sso(i,k)*u(i,k)+vt_sso(i,k)*v(i,k)

    !Note:
    !Horizontal wind components and SSO-tendencies refer to horizontal mass centeres here.

The SSO scheme's wind tendency projected onto the wind itself is the rate at which the
sub-grid orography drains kinetic energy from the resolved flow, per unit mass [m2/s3]. It is
negative wherever the SSO scheme is doing its job -- the drag opposes the wind -- which is why
its consumer negates it and floors it at zero.

MAIN LEVELS, NOT HALF LEVELS. 'u', 'v', 'ut_sso' and 'vt_sso' are all main-level fields and the
Fortran loop runs 'DO k=1,kem', one row further up than everything else in the section. The
result is interpolated onto the half levels by its consumer,
'compute_total_mechanical_forcing', which is where the vertical staggering is resolved.

IT LANDS IN THE SCRATCH ARRAY 'hlp', overwriting the separated-shear TKE source that
'compute_separated_horizontal_shear_tke_source' left there, and it is what 'hlp' holds at the
section's exit savepoint. Overwriting it is safe because the shear source has already been
copied to 'tket_hshr' and added to 'frm'; in the port the two quantities are separate fields and
the ordering constraint disappears with the aliasing that caused it.

THIS EXPRESSION IS AN FMA CANARY. It is 'a*b + c*d', the pattern that distinguishes a
contracted build from an uncontracted one, so a bit-exact result here is evidence that neither
side is fusing -- the same role section 1b)'s 'compute_thermal_forcing' plays.
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_sso_wake_energy_production(
    sso_tendency_u: fa.CellKField[wpfloat],
    sso_tendency_v: fa.CellKField[wpfloat],
    wind_u: fa.CellKField[wpfloat],
    wind_v: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """The scalar product of the SSO wind tendency with the wind [m2/s3]."""
    return sso_tendency_u * wind_u + sso_tendency_v * wind_v


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_sso_wake_energy_production(
    sso_tendency_u: fa.CellKField[wpfloat],
    sso_tendency_v: fa.CellKField[wpfloat],
    wind_u: fa.CellKField[wpfloat],
    wind_v: fa.CellKField[wpfloat],
    sso_wake_energy_production: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Compute 'hlp' as section 2a) leaves it [m2/s3], on MAIN levels.

    The whole block is guarded by 'ltkemcsso .OR. loutmcsso' (:1553) and by '.NOT. lini'
    (:1550), the latter because 'ut_sso' and 'vt_sso' may not have been computed yet during
    initialisation. 'ltkemcsso' requires the SSO scheme to be running and its tendencies to be
    passed; 'loutmcsso' additionally requires 'tket_sso', which the ICON interfaces never pass,
    so in practice the block runs exactly when the SSO scheme does.

    Args:
        sso_tendency_u: 'ut_sso' [m/s2], main levels.
        sso_tendency_v: 'vt_sso' [m/s2], main levels.
        wind_u: 'u' [m/s], main levels at the mass centre.
        wind_v: 'v' [m/s], main levels at the mass centre.
        sso_wake_energy_production: Output, 'hlp' [m2/s3], main levels. Negative where the SSO
            drag decelerates the flow.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First main level; 0, mirroring Fortran 'k=1'. One row above everything
            else in this section.
        vertical_end: End of the main levels; 'ke', mirroring Fortran 'k=...,kem'.
    """
    _compute_sso_wake_energy_production(
        sso_tendency_u=sso_tendency_u,
        sso_tendency_v=sso_tendency_v,
        wind_u=wind_u,
        wind_v=wind_v,
        out=sso_wake_energy_production,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
