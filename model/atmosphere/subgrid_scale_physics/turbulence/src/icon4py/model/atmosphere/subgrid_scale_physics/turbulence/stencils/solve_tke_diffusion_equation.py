# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Thomas solve of the semi-implicit vertical TKE diffusion.

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90', SUBROUTINE
'calc_impl_vert_diff', the blocks headed "Forward substitution" (:3024-3042 at icon commit
26d6b98cce) and "Backward substitution" (:3043-3052), reached from 'turbdiff' section 9)
(turb_diffusion.f90:2407-2489, the call at :2433-2441). The scientific commentary in those
files is by Matthias Raschendorfer (DWD).

    k = k_tp+1
    upd_prof(i,k) = eff_flux(i,k) * invs_mom(i,k)
    DO k = k_tp+2, k_sf-1
       upd_prof(i,k) = (eff_flux(i,k) + impl_mom(i,k) * upd_prof(i,k-1)) * invs_mom(i,k)
    DO k = k_sf-2, k_tp+1, -1
       upd_prof(i,k) = upd_prof(i,k) + invs_fac(i,k+1) * upd_prof(i,k+1)

The matrix was already factorised by 'compute_inverted_diffusion_momentum' and
'compute_diffusion_inversion_factor', so this is the two substitutions only. Both are genuine
vertical recurrences on their own output and both are scans; they are one program because they
are one solve, and because the intermediate profile between them is of no interest to anybody
-- ICON does not serialize it, since the Fortran overwrites it in place.

Under 'lprecondi' the Fortran scales the right-hand side and the solution by 'scal_fac' on the
way in and out (:2998-3010, :3054-3064). 'lprecnd' defaults to '.FALSE.'
(mo_turbdiff_config.f90:102) and no configuration under 'icon/run/' sets it; the reference
capture confirms it, in that the 'frm' storage 'scal_fac' would occupy is byte-identical
across section 9). Preconditioning is not translated.

'itndcon' is 0 at both of section 9)'s call sites (turb_diffusion.f90:2387, :2394), so the
'old_prof'/'rhs_prof' selection at the head of 'calc_impl_vert_diff' resolves to 'cur_prof'
for both, the explicit-tendency copy at :3012-3022 is skipped, and 'cur_prof' keeps its input
profile. The current tendency is therefore not a parameter of this port; it enters, when it
enters at all, through the virtual profile section 8) builds.
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.scan_operator(axis=dims.KDim, forward=True, init=(wpfloat("0.0"), True))
def _eliminate_downward(
    state: tuple[wpfloat, bool],
    right_hand_side: wpfloat,
    implicit_diffusion_momentum: wpfloat,
    inverted_diffusion_momentum: wpfloat,
) -> tuple[wpfloat, bool]:
    """One forward-substitution step, carrying the eliminated value of the level above.

    The carried flag marks the first row of the scan, where the Fortran drops the sub-diagonal
    term because there is no level above; as in 'compute_inverted_diffusion_momentum' both
    expressions are written out rather than folded, since 'impl_mom' at that row belongs to a
    quantity this section does not write.
    """
    eliminated_above, at_the_top = state
    eliminated = (
        right_hand_side * inverted_diffusion_momentum
        if at_the_top
        else (right_hand_side + implicit_diffusion_momentum * eliminated_above)
        * inverted_diffusion_momentum
    )
    return eliminated, False


@gtx.scan_operator(axis=dims.KDim, forward=False, init=(wpfloat("0.0"), True))
def _substitute_upward(
    state: tuple[wpfloat, bool],
    eliminated: wpfloat,
    inversion_factor_below: wpfloat,
) -> tuple[wpfloat, bool]:
    """One back-substitution step, carrying the solution of the level below.

    The Fortran back substitution starts one level above the lowest diffused half level
    ('DO k = k_sf-2, ...'), so the lowest row keeps its forward value: it is the last unknown
    of the eliminated system and needs no substitution. That is what the flag selects here.
    """
    solution_below, at_the_bottom = state
    solution = eliminated if at_the_bottom else eliminated + inversion_factor_below * solution_below
    return solution, False


@gtx.field_operator
def _solve_tke_diffusion_equation(
    right_hand_side: fa.CellKField[wpfloat],
    implicit_diffusion_momentum: fa.CellKField[wpfloat],
    inverted_diffusion_momentum: fa.CellKField[wpfloat],
    inversion_factor: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """Eliminate downward, then substitute upward."""
    eliminated, _ = _eliminate_downward(
        right_hand_side, implicit_diffusion_momentum, inverted_diffusion_momentum
    )
    solution, _ = _substitute_upward(eliminated, inversion_factor(Koff[1]))
    return solution


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def solve_tke_diffusion_equation(
    right_hand_side: fa.CellKField[wpfloat],
    implicit_diffusion_momentum: fa.CellKField[wpfloat],
    inverted_diffusion_momentum: fa.CellKField[wpfloat],
    inversion_factor: fa.CellKField[wpfloat],
    updated_tke_profile: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Solve the tridiagonal TKE diffusion system for the updated profile, on half levels.

    The vertical range is Fortran 'k = k_tp+1 .. k_sf-1' with 'k_tp = 1' and 'k_sf = ke1', i.e.
    'k = 2..ke', which is 'vertical_start = 1, vertical_end = nlev' here. The surface half
    level is not an unknown -- its TKE is the boundary value the transfer scheme set, and it
    entered the system through the right-hand side.

    Args:
        right_hand_side: The right-hand side on half levels, from
            'compute_tke_diffusion_right_hand_side'. Only the diffused rows are read; the
            surface row of that field holds something else and is outside the range.
        implicit_diffusion_momentum: 'impl_mom' [kg/m2/s] on flux levels, the sub-diagonal.
            Rows 'vertical_start + 1' onward are read; the row at 'vertical_start' is not.
        inverted_diffusion_momentum: 'invs_mom' [m2 s /kg], the reciprocal pivot, from
            'compute_inverted_diffusion_momentum'.
        inversion_factor: 'invs_fac' [-], the elimination multiplier, from
            'compute_diffusion_inversion_factor'; read one level down, so its row at
            'vertical_end' is touched but never used -- the lowest row takes the other branch.
        updated_tke_profile: Output, 'upd_prof': the profile updated by the diffusion tendency.
            A TKE [m2/s2] at 'imode_tkediff = 2', a turbulent velocity 'q' [m/s] at 1. Under
            'lcircterm' this is still the virtual profile; 'add_virtual_diffusion_increment'
            turns it into the true one.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: Uppermost diffused half level; 1, mirroring Fortran 'k_tp+1 = 2'.
        vertical_end: End of the diffused half levels; 'nlev', mirroring Fortran 'k_sf-1 = ke'.
    """
    _solve_tke_diffusion_equation(
        right_hand_side=right_hand_side,
        implicit_diffusion_momentum=implicit_diffusion_momentum,
        inverted_diffusion_momentum=inverted_diffusion_momentum,
        inversion_factor=inversion_factor,
        out=updated_tke_profile,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
