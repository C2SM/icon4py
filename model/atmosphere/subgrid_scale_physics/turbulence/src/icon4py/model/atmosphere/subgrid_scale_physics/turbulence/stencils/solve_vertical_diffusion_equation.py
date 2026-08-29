# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Thomas solve of the semi-implicit vertical diffusion of a first-order model variable.

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90', SUBROUTINE
'calc_impl_vert_diff' (:3024-3052 at icon commit 26d6b98cce), the blocks headed "Forward
substitution" and "Backward substitution", reached from 'vertdiff' through 'vert_grad_diff'
(turb_vertdiff.f90:747-763). The scientific commentary in those files is by Matthias
Raschendorfer (DWD).

    k = k_tp+1
    upd_prof(i,k) = eff_flux(i,k)*invs_mom(i,k)
    DO k = k_tp+2, k_sf-1
       upd_prof(i,k) = ( eff_flux(i,k) + impl_mom(i,k)*upd_prof(i,k-1) )*invs_mom(i,k)
    DO k = k_sf-2, k_tp+1, -1
       upd_prof(i,k) = upd_prof(i,k) + invs_fac(i,k+1)*upd_prof(i,k+1)

Both are genuine vertical recurrences on their own output, so both are 'scan_operator's; they
are one program because they are one solve and the intermediate profile between them is of no
interest -- ICON overwrites it in place and does not serialize it.

'solve_tke_diffusion_equation' is the same pair of substitutions for the TKE. The two are
separate programs because that one is named and documented for the TKE, whose unknowns sit on
the half levels between 'k_tp = 1' and 'ke' rather than on the main levels between 'k_tp = 0'
and 'ke'; the arithmetic is identical.

THE MATRIX IS BUILT ONCE PER VARIABLE TYPE, NOT PER VARIABLE. 'vert_grad_diff' factorises only
under 'linisetup .OR. lnewvtype' (turb_utilities.f90:2401), so all three scalars -- temperature,
water vapour, cloud water, and every 'ndtr' passive tracer beyond them -- are solved with one
'invs_mom' and one 'invs_fac'. Only the right-hand side changes. The caller is what has to
respect that; this program takes the factorisation as an argument and does not rebuild it.

Preconditioning ('lprecondi') scales the right-hand side and the solution by 'scal_fac' on the
way in and out (:2993-3005, :3054-3064). 'lprecnd' defaults to '.FALSE.'
(mo_turbdiff_config.f90:102), no configuration under 'icon/run/' sets it, and the reference
capture confirms it: the 'frm' storage that 'scal_fac' would occupy holds no scaling factor.
Preconditioning is not translated.
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

    The carried flag marks the first row, where the Fortran drops the sub-diagonal term because
    there is no level above it. Both expressions are written out rather than folded into one:
    'impl_mom' on that row belongs to the flux level above the uppermost unknown, which is
    outside the system, and multiplying it by zero would be a numerical accident rather than a
    boundary condition.
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

    The Fortran back substitution starts one level above the lowest unknown
    ('DO k = k_sf-2, ...'), so the lowest row keeps its forward value: it is the last unknown of
    the eliminated system and needs no substitution. That is what the flag selects.
    """
    solution_below, at_the_bottom = state
    solution = eliminated if at_the_bottom else eliminated + inversion_factor_below * solution_below
    return solution, False


@gtx.field_operator
def _solve_vertical_diffusion_equation(
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
def solve_vertical_diffusion_equation(
    right_hand_side: fa.CellKField[wpfloat],
    implicit_diffusion_momentum: fa.CellKField[wpfloat],
    inverted_diffusion_momentum: fa.CellKField[wpfloat],
    inversion_factor: fa.CellKField[wpfloat],
    updated_profile: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Solve the tridiagonal system for the profile after diffusion, on the main levels.

    The vertical range is the Fortran 'k = k_tp+1 .. k_sf-1' with 'k_tp = 0' and 'k_sf = ke1',
    i.e. 'vertical_start = 0, vertical_end = nlev'. The surface value is not an unknown: it
    entered through the right-hand side, as a prescribed concentration for the wind components
    and as a prescribed flux for the scalars.

    Args:
        right_hand_side: From 'compute_diffusion_right_hand_side'; only the diffused rows are
            read, and the surface row of that field holds something else.
        implicit_diffusion_momentum: 'impl_mom' [kg/m2/s], the sub-diagonal. Rows
            'vertical_start + 1' onward are read; the row at 'vertical_start' is not.
        inverted_diffusion_momentum: 'invs_mom' [m2 s/kg], the reciprocal pivot, from
            'compute_inverted_diffusion_momentum' and, for a surface-flux condition,
            'invert_diffusion_momentum_at_the_surface_flux_level'.
        inversion_factor: 'invs_fac' [-], the elimination multiplier, from
            'compute_diffusion_inversion_factor'; read one level down, so its row at
            'vertical_end' is touched but never used -- the lowest row takes the other branch.
        updated_profile: Output, 'upd_prof': the variable profile after diffusion, in the
            variable's own units (potential temperature for the temperature solve).
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: Uppermost diffused main level; 0.
        vertical_end: End of the diffused main levels; 'nlev'.
    """
    _solve_vertical_diffusion_equation(
        right_hand_side=right_hand_side,
        implicit_diffusion_momentum=implicit_diffusion_momentum,
        inverted_diffusion_momentum=inverted_diffusion_momentum,
        inversion_factor=inversion_factor,
        out=updated_profile,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
