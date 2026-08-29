# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit flux density of the semi-implicit vertical TKE diffusion.

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90', SUBROUTINE
'calc_impl_vert_diff' (:2951-2977 at icon commit 26d6b98cce), reached from 'turbdiff' section
9) (turb_diffusion.f90:2407-2489, the call at :2433-2441). The scientific commentary in those
files is by Matthias Raschendorfer (DWD).

Two Fortran loops, into one array:

    DO k = k_tp+2, k_sf
       eff_flux(i,k) = expl_mom(i,k) * (rhs_prof(i,k) - rhs_prof(i,k-1))
    IF (.NOT. lsflucond) &
       eff_flux(i,k_sf) = eff_flux(i,k_sf) + impl_mom(i,k_sf) * rhs_prof(i,k_sf-1)

The first is the explicit part of the diffusive flux across a flux level, positive upward: the
explicit diffusion momentum times the drop of the profile across the level. The second, headed
"Achtung: Korrektur: Richtige Behandlung der unteren Konzentrations-Randbedingung"
("Correction: correct treatment of the lower concentration boundary condition"), closes the
lower boundary. The surface value of the profile is prescribed rather than solved for, so the
implicit part of the surface flux is a known quantity and belongs on the right-hand side; the
Fortran's note at :2971-2976 is that this is what makes the level above the surface
semi-implicit. 'impl_mom' at the surface is unscaled there and vanishes altogether under a
surface-flux condition, which the TKE diffusion does not use.

Same field, different expression at the surface row: one program, with the row selected by
'concat_where' -- the boundary-row convention in the package README.

THE SIGN. The 'eff_flux' storage carries three different quantities in the course of
'calc_impl_vert_diff', and the declaration (turb_utilities.f90:2901-2913) names them: an
explicit flux density positive UPWARD, then the full right-hand side of the semi-implicit
equation, and finally -- only if 'leff_flux' -- the effective flux density of pure diffusion,
positive DOWNWARD. This program produces the first.
"""

import gt4py.next as gtx
from gt4py.next.experimental import concat_where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_explicit_tke_flux_density(
    explicit_diffusion_momentum: fa.CellKField[wpfloat],
    implicit_diffusion_momentum: fa.CellKField[wpfloat],
    current_tke_profile: fa.CellKField[wpfloat],
    nlev: gtx.int32,
) -> fa.CellKField[wpfloat]:
    """Explicit diffusive flux across a flux level, closed at the surface."""
    explicit_flux = explicit_diffusion_momentum * (
        current_tke_profile - current_tke_profile(Koff[-1])
    )
    return concat_where(
        dims.KDim == nlev,
        explicit_flux + implicit_diffusion_momentum * current_tke_profile(Koff[-1]),
        explicit_flux,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_explicit_tke_flux_density(
    explicit_diffusion_momentum: fa.CellKField[wpfloat],
    implicit_diffusion_momentum: fa.CellKField[wpfloat],
    current_tke_profile: fa.CellKField[wpfloat],
    nlev: gtx.int32,
    explicit_tke_flux_density: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Compute the explicit TKE flux density on flux levels, positive upward.

    The vertical range is the Fortran 'DO k = k_tp+2, k_sf' with 'k_tp = 1' and 'k_sf = ke1',
    i.e. 'k = 3..ke1', which is 'vertical_start = 2, vertical_end = nlev + 1' here. 'nlev' must
    be that same last row, since it is what selects the surface expression.

    At the surface flux level the Fortran reads the full diffusion momentum out of 'expl_mom',
    which is why 'subtract_implicit_part_of_tke_diffusion_momentum' stops one level higher; pass the same
    field it wrote.

    IN 'turbdiff' THIS DOES NOT GET A STORAGE OF ITS OWN. The Fortran computes it in place in
    the array it will overwrite with the right-hand side, which is the 'len_scale' storage
    (turb_diffusion.f90:2431). Only the surface row survives that overwrite. GT4Py cannot read
    and write one field across a vertical shift, so here it is a field of its own and
    'compute_tke_diffusion_right_hand_side' copies the surface row back.

    Args:
        explicit_diffusion_momentum: 'expl_mom' [kg/m2/s] on flux levels, from
            'subtract_implicit_part_of_tke_diffusion_momentum' above the surface and still the full
            diffusion momentum at the surface flux level.
        implicit_diffusion_momentum: 'impl_mom' [kg/m2/s]; read at the surface row only.
        current_tke_profile: 'cur_prof', the profile being diffused, on half levels. Under
            'lcircterm' this is section 8)'s virtual TKE profile, whose diffusion tendency
            carries the circulation term; otherwise it is the saved TKE profile itself.
            [m2/s2] at 'imode_tkediff = 2', [m/s] at 1.
        nlev: 'ke1' as a zero-based row, the surface half level -- the one row that takes the
            boundary expression.
        explicit_tke_flux_density: Output, 'eff_flux' in its first role [kg/m/s3 or kg/m2/s2].
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First flux level; 2, mirroring Fortran 'k = 3'.
        vertical_end: End of the flux levels; 'nlev + 1', mirroring Fortran 'k = ..,ke1'.
    """
    _compute_explicit_tke_flux_density(
        explicit_diffusion_momentum=explicit_diffusion_momentum,
        implicit_diffusion_momentum=implicit_diffusion_momentum,
        current_tke_profile=current_tke_profile,
        nlev=nlev,
        out=explicit_tke_flux_density,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
