# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""'calc_impl_vert_diff' for the TKE: the right-hand side of section 9)'s system, and its solve.

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90', SUBROUTINE 'calc_impl_vert_diff'
(:2880-3060 at icon commit 26d6b98cce, the commit that produced the reference capture), as
'turbdiff' section 9) calls it (turb_diffusion.f90:2429-2437). The scientific commentary in those
files is by Matthias Raschendorfer (DWD).

ICON USES ONE SUBROUTINE FOR THE TKE AND FOR THE MODEL VARIABLES, and so does the port: this is
the TKE flavour, 'calc_impl_vert_diff' proper is 'vertdiff's.

THREE STATEMENTS: the explicit flux density across every flux level, the right-hand side of the
tridiagonal system, and the Thomas solve -- one forward elimination and one back substitution,
two 'scan_operator's inside one field operator.

Each of the three was a '@gtx.program' of its own until the stencil merge, and the merge is the
direction that genuinely fuses: a scan's PRODUCERS can be inlined into it, a scan's consumers
cannot (merge plan §2.1).

ONE VERTICAL PAIR, THREE RANGES DERIVED FROM IT. The pair the granule binds is the solve's own
'k_tp+1 .. k_sf-1', which is '(1, nlev)'. The explicit flux density runs
'(vertical_start + 1, vertical_end + 1)', Fortran 'k = k_tp+2 .. k_sf'; the right-hand side runs
'(vertical_start, vertical_end + 1)', because it also carries through the surface row that the
flux density owns; the solve runs the bound pair.

'uppermost_diffused_level' AND 'nlev' STAY RUNTIME ARGUMENTS even though both equal a domain
bound the granule already binds. That is deliberate and is 'gt4py-01': a 'concat_where' whose
split point is static AT THE SAME TIME as the domain bounds miscompiles on 'dace_cpu', and both
statements that select a boundary row here use one. 'Turbulence._program' binds the bounds, so
the split points must be passed at call time. See the docstring of 'Turbulence._program'.

WHAT THE 'len_scale' STORAGE HOLDS AFTERWARDS. Section 9) points 'eff_flux' at the master length
scale's array (turb_diffusion.f90:2431) and the length scale is gone from there on. What replaces
it is the right-hand side on the diffused rows and the explicit surface flux on the surface row,
NOT an effective flux density: the vertical integration that would make it one runs only under
'leff_flux', which section 9) passes as 'kcm <= ke', and ICON-NWP leaves 'kcm' at 'ke+1'.
Measured on this capture in 'test_the_len_scale_storage_is_not_the_effective_tke_flux'.

THE EXPLICIT FLUX DENSITY KEEPS A FIELD OF ITS OWN, which the Fortran does not give it: the
Fortran computes it in place in the storage it is about to overwrite with the right-hand side,
and only the surface row survives. GT4Py cannot read and write one field across a vertical shift
-- the right-hand side reads the flux one level down -- so here it is a separate field and the
right-hand-side statement copies the surface row back. Merging the two into one program does not
change that: they are still two statements writing two fields.

THE ALIASING RULES THIS PROGRAM MEETS (measured in 'solve_turb_budgets'): statement 2 reads
'explicit_tke_flux_density', which statement 1 wrote, at 'Koff[+1]' and pointwise, and statement
3 reads 'right_hand_side', which statement 2 wrote. Reading a parameter an EARLIER statement
wrote is ordinary dataflow and is correct at any offset. No statement writes a parameter it
reads, and nothing here is aliased by the caller: 'right_hand_side' and 'updated_tke_profile' are
different storages in 'turbdiff' too.
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
def calc_impl_vert_diff_for_the_tke(
    explicit_diffusion_momentum: fa.CellKField[wpfloat],
    implicit_diffusion_momentum: fa.CellKField[wpfloat],
    inverted_diffusion_momentum: fa.CellKField[wpfloat],
    inversion_factor: fa.CellKField[wpfloat],
    discretisation_momentum: fa.CellKField[wpfloat],
    current_tke_profile: fa.CellKField[wpfloat],
    uppermost_diffused_level: gtx.int32,
    nlev: gtx.int32,
    explicit_tke_flux_density: fa.CellKField[wpfloat],
    right_hand_side: fa.CellKField[wpfloat],
    updated_tke_profile: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Build the right-hand side of the TKE diffusion system and solve it, in three statements.

    Args:
        explicit_diffusion_momentum: 'expl_mom' [kg/m2/s] on flux levels, as
            'prep_impl_vert_diff_for_the_tke' left it: the explicit remainder above the surface
            and still the full diffusion momentum at the surface flux level.
        implicit_diffusion_momentum: 'impl_mom' [kg/m2/s] on flux levels, the sub-diagonal.
        inverted_diffusion_momentum: 'invs_mom' [m2 s/kg], the eliminated pivot's reciprocal.
        inversion_factor: 'invs_fac' [-], the multiplier of the forward elimination.
        discretisation_momentum: 'disc_mom' [kg/m2/s] on half levels, from section 1a).
        current_tke_profile: 'cur_prof', the profile being diffused, on half levels. Under
            'lcircterm' this is section 8)'s virtual TKE profile, whose diffusion tendency
            carries the circulation term; otherwise the saved TKE profile itself.
        uppermost_diffused_level: Zero-based row of the uppermost diffused half level, Fortran
            'k_tp+1'; equal to 'vertical_start' and passed separately for 'gt4py-01'.
        nlev: 'ke1' as a zero-based row, the surface half level; equal to 'vertical_end' and
            passed separately for the same reason.
        explicit_tke_flux_density: Output, the explicit diffusive flux across each flux level,
            positive upward. The Fortran gives it no storage of its own; see the module
            docstring.
        right_hand_side: Output, the 'len_scale' storage as section 9) leaves it: the right-hand
            side on the diffused rows, the explicit surface flux on the surface row.
        updated_tke_profile: Output, 'upd_prof' on the diffused half levels.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: Uppermost diffused half level, Fortran 'k_tp+1'; 1.
        vertical_end: End of the diffused half levels, Fortran 'k_sf-1'; 'nlev'.
    """
    _compute_explicit_tke_flux_density(
        explicit_diffusion_momentum=explicit_diffusion_momentum,
        implicit_diffusion_momentum=implicit_diffusion_momentum,
        current_tke_profile=current_tke_profile,
        nlev=nlev,
        out=explicit_tke_flux_density,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start + 1, vertical_end + 1),
        },
    )
    _compute_tke_diffusion_right_hand_side(
        discretisation_momentum=discretisation_momentum,
        current_tke_profile=current_tke_profile,
        explicit_tke_flux_density=explicit_tke_flux_density,
        uppermost_diffused_level=uppermost_diffused_level,
        nlev=nlev,
        out=right_hand_side,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end + 1),
        },
    )
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
