# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Right-hand side of the tridiagonal system of the semi-implicit vertical TKE diffusion.

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90', SUBROUTINE
'calc_impl_vert_diff', the block headed "Resultant right-hand side flux" (:2979-2996 at icon
commit 26d6b98cce), reached from 'turbdiff' section 9) (turb_diffusion.f90:2407-2489, the call
at :2433-2441). The scientific commentary in those files is by Matthias Raschendorfer (DWD).

Two Fortran statements into the array that already holds the explicit flux density:

    k = k_tp+1
    eff_flux(i,k) = disc_mom(i,k) * old_prof(i,k) + eff_flux(i,k+1)
    DO k = k_tp+2, k_sf-1
       eff_flux(i,k) = disc_mom(i,k) * old_prof(i,k) + eff_flux(i,k+1) - eff_flux(i,k)

i.e. the mass of the layer times its current value, plus the convergence of the explicit flux
across it. The uppermost diffused half level has no flux level above it -- the Fortran's own
note is "Zero flux condition just below top level" -- so its expression is the same one with
that term dropped, which is why the two rows are one program with the row selected by
'concat_where' (the boundary-row convention in the package README).

WHAT ENDS UP IN THE 'len_scale' STORAGE, AND WHAT IT IS NOT
-----------------------------------------------------------
Section 9) points 'eff_flux' at the 'len_scale' array (turb_diffusion.f90:2431), so from here
on that storage no longer holds the turbulent master length scale. The Fortran's closing note
(:2448-2450) says it holds "die effektiven Flussdichten (positiv abwaerts) der
(semi-)impliziten Vertikaldiffusion" -- "the effective flux densities (positive downward) of
the (semi-)implicit vertical diffusion". THAT NOTE DESCRIBES A BRANCH THIS CONFIGURATION DOES
NOT TAKE. The vertical integration that would produce the effective flux runs only under
'leff_flux', which section 9) passes as 'kcm <= ke' (:2434); 'kcm' is the upper bound of the
resolved canopy and ICON-NWP leaves it at 'ke+1' (turb_diffusion.f90:1099-1100, "Up to now it
is kcm = ke+1"), the same condition that makes the roughness-layer section 2b) dead. Measured
against the reference capture: 'kcm = 81' with 'ke = 80', and the exit state of the storage is
reproduced bit-exactly without that block.

So what the storage actually holds at the exit of section 9) is two things:

  * the right-hand side of the tridiagonal system, on the diffused half levels;
  * at the surface row, the explicit flux density that
    'compute_explicit_tke_flux_density' left there, which this block does not reach.

Neither is a length scale and neither is an effective flux, which is why the output of this
program is named for the right-hand side and the surface row is spelled out below.
"""

import gt4py.next as gtx
from gt4py.next.experimental import concat_where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_tke_diffusion_right_hand_side(
    discretisation_momentum: fa.CellKField[wpfloat],
    current_tke_profile: fa.CellKField[wpfloat],
    explicit_tke_flux_density: fa.CellKField[wpfloat],
    uppermost_diffused_level: gtx.int32,
    nlev: gtx.int32,
) -> fa.CellKField[wpfloat]:
    """Flux convergence plus the layer's current content, with both vertical boundaries."""
    right_hand_side = concat_where(
        dims.KDim < nlev,
        discretisation_momentum * current_tke_profile
        + explicit_tke_flux_density(Koff[1])
        - explicit_tke_flux_density,
        explicit_tke_flux_density,
    )
    return concat_where(
        dims.KDim == uppermost_diffused_level,
        discretisation_momentum * current_tke_profile + explicit_tke_flux_density(Koff[1]),
        right_hand_side,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_tke_diffusion_right_hand_side(
    discretisation_momentum: fa.CellKField[wpfloat],
    current_tke_profile: fa.CellKField[wpfloat],
    explicit_tke_flux_density: fa.CellKField[wpfloat],
    uppermost_diffused_level: gtx.int32,
    nlev: gtx.int32,
    right_hand_side: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Compute the right-hand side of the semi-implicit TKE diffusion, on half levels.

    The vertical range covers both what the Fortran writes here and the surface row it leaves
    to 'compute_explicit_tke_flux_density', because the two share one storage: run it with
    'vertical_start = 1' (Fortran 'k_tp+1 = 2') and 'vertical_end = nlev + 1'. Rows below
    'nlev' get the right-hand side; row 'nlev' is copied through from the explicit flux, so
    that the field this program writes is exactly what 'turbdiff' has in 'len_scale' when
    section 9) ends. The model top is never written.

    'uppermost_diffused_level' must equal 'vertical_start' and 'nlev' must equal
    'vertical_end - 1'; nothing checks that, and the section datatest is what pins it.

    Args:
        discretisation_momentum: 'disc_mom', 'rho_n * dz / dt_tke' [kg/m2/s] on half levels,
            from section 1a).
        current_tke_profile: 'cur_prof', the profile being diffused, on half levels; the same
            field 'compute_explicit_tke_flux_density' differenced.
        explicit_tke_flux_density: The explicit flux density on flux levels, positive upward,
            from 'compute_explicit_tke_flux_density'. Read one level down, which is why it
            cannot be the same field as 'right_hand_side'.
        uppermost_diffused_level: Zero-based row of the uppermost diffused half level, Fortran
            'k_tp+1'; the row whose flux level above carries no flux.
        nlev: 'ke1' as a zero-based row, the surface half level; the row that keeps the
            explicit flux instead of a right-hand side.
        right_hand_side: Output, the 'len_scale' storage as section 9) leaves it.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: Uppermost diffused half level; 1.
        vertical_end: End of the half levels; 'nlev + 1'.
    """
    _compute_tke_diffusion_right_hand_side(
        discretisation_momentum=discretisation_momentum,
        current_tke_profile=current_tke_profile,
        explicit_tke_flux_density=explicit_tke_flux_density,
        uppermost_diffused_level=uppermost_diffused_level,
        nlev=nlev,
        out=right_hand_side,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
