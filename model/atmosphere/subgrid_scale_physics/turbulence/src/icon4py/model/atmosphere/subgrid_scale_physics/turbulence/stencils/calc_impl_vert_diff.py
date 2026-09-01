# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The semi-implicit vertical diffusion of ONE model variable, solved.

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90', SUBROUTINE 'calc_impl_vert_diff'
(:2951-3052 at icon commit 26d6b98cce), which 'vertdiff' reaches through 'vert_grad_diff'
(turb_vertdiff.f90:747-763). The scientific commentary in those files is by Matthias
Raschendorfer (DWD).

'prep_impl_vert_diff' factorises the matrix once per variable TYPE; this solves with it once per
VARIABLE. Merging the two would run the factorisation five times instead of twice, which is the
one place in the stencil merge where a merge would be a measurable pessimisation.

ONE PROGRAM, FIVE STATEMENTS, FOUR DOMAINS. In source order,

  1. the upper boundary condition, 'eff_flux(:,k_tp+1) = 0';
  2. the explicit flux density, flux levels 2..k_sf;
  3. the implicit surface coupling added to the surface row -- for the MOMENTUM type only;
  4. the right-hand side, main levels 1..k_sf-1;
  5. the Thomas solve, forward elimination then back substitution, over the same range.

STATEMENT 3 IS A SWITCH EXPRESSED AS A DOMAIN. The Fortran guards it with
'IF (.NOT.lsflucond)' -- the momentum type takes a surface-CONCENTRATION condition and so has an
implicit coupling to the ground, the scalar type takes a surface-FLUX condition and does not.
Here 'surface_addition_start' is 'vertical_end' when it applies and 'vertical_end + 1' when it
does not, and an EMPTY vertical domain is a no-op. Measured on all four compiled backends
before it was relied on --
'docs/superpowers/notes/2026-09-01-gt4py-empty-vertical-domain.py' in the workspace -- and the
datatest exercises both types on every one of them. It is the same device 'prep_impl_vert_diff' uses for 'elimination_end', and it is what
lets one program serve both types.

THE FLUX IS BUILT IN PLACE AND THAT IS ADMISSIBLE. Statement 3 reads 'explicit_flux_density'
POINTWISE and writes it -- variant A3 of the aliasing measurement in 'solve_turb_budgets' --
which is correct on every backend. Its OTHER read, 'cur_prof(k-1)', is of a different field.

THE RIGHT-HAND SIDE IS A DIFFERENT FIELD FROM THE FLUX, and must stay one. The Fortran overwrites
'eff_flux' in place while reading 'eff_flux(k+1)', a row the downward loop has not reached, so
every read is of the flux as it was; GT4Py has no row order to rely on and writing them to one
field would make the answer depend on the order it visits rows in.

THE SURFACE ROW OF THE RIGHT-HAND SIDE IS THE CALLER'S JOB. Statement 4 stops one row short of
the surface, and the row it does not write must end up holding the explicit surface flux, which
is what 'zvari(:,ke1,m)' holds at the exit savepoint. The Fortran gets that for free from the
in-place overwrite; the port copies it, AFTER this program rather than in the middle of it --
which is exactly equivalent, because nothing here reads the right-hand side's surface row: the
solve reads rows 'k_tp+1..k_sf-1' only.
"""

import gt4py.next as gtx
from gt4py.next import broadcast

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_explicit_flux_density(
    explicit_diffusion_momentum: fa.CellKField[wpfloat],
    current_profile: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'expl_mom(k)*(cur_prof(k) - cur_prof(k-1))'."""
    return explicit_diffusion_momentum * (current_profile - current_profile(Koff[-1]))


@gtx.field_operator
def _no_flux_through_the_model_top() -> fa.CellKField[wpfloat]:
    """The upper boundary condition of the diffusion equation: no flux above the top level."""
    return broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim))


@gtx.field_operator
def _add_implicit_surface_flux_to_the_explicit_flux_density(
    explicit_flux_density: fa.CellKField[wpfloat],
    implicit_diffusion_momentum: fa.CellKField[wpfloat],
    current_profile: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'eff_flux(ke1) + impl_mom(ke1)*cur_prof(ke)'."""
    return explicit_flux_density + implicit_diffusion_momentum * current_profile(Koff[-1])


@gtx.field_operator
def _compute_diffusion_right_hand_side(
    discretisation_momentum: fa.CellKField[wpfloat],
    current_profile: fa.CellKField[wpfloat],
    explicit_flux_density: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'disc_mom(k)*cur_prof(k) + flux(k+1) - flux(k)', left to right as the Fortran writes it."""
    return (
        discretisation_momentum * current_profile
        + explicit_flux_density(Koff[1])
        - explicit_flux_density
    )


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
def calc_impl_vert_diff(
    explicit_diffusion_momentum: fa.CellKField[wpfloat],
    current_profile: fa.CellKField[wpfloat],
    implicit_diffusion_momentum: fa.CellKField[wpfloat],
    discretisation_momentum: fa.CellKField[wpfloat],
    inverted_diffusion_momentum: fa.CellKField[wpfloat],
    inversion_factor: fa.CellKField[wpfloat],
    surface_addition_start: gtx.int32,
    explicit_flux_density: fa.CellKField[wpfloat],
    right_hand_side: fa.CellKField[wpfloat],
    updated_profile: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Diffuse one variable through the matrix its type left standing.

    THE FOUR VERTICAL RANGES, all arithmetic on the one pair the granule binds.
    '(vertical_start, vertical_end)' is the diffused main levels, Fortran 'k_tp+1' to 'k_sf-1'
    with 'k_tp = 0' and 'k_sf = ke1', i.e. '(0, nlev)', and statements 4 and 5 use it unchanged.
    The others are

      * '(vertical_start, vertical_start + 1)' for the upper boundary condition, the row above
        the uppermost flux level;
      * '(vertical_start + 1, vertical_end + 1)' for the explicit flux, Fortran
        'DO k = k_tp+2, k_sf';
      * '(surface_addition_start, vertical_end + 1)' for the implicit surface coupling, which the
        caller makes empty by passing 'vertical_end + 1' for a variable type with a
        surface-flux condition.

    Args:
        explicit_diffusion_momentum: 'expl_mom' [kg/m2/s], reduced by the implicit part on the
            interior levels and FULL on the surface row -- which is what makes that row the
            explicit surface flux.
        current_profile: 'cur_prof', the variable profile including its lower boundary value;
            read one level up as well.
        implicit_diffusion_momentum: 'impl_mom' [kg/m2/s]; read on the surface row by statement 3
            and by the elimination.
        discretisation_momentum: 'disc_mom' [kg/m2/s].
        inverted_diffusion_momentum: 'invs_mom' [m2 s/kg], the factorisation.
        inversion_factor: 'invs_fac' [-], the factorisation's multiplier; read one level down.
        surface_addition_start: 'vertical_end' for a variable type with a surface-CONCENTRATION
            condition (the momentum type), 'vertical_end + 1' for one with a surface-FLUX
            condition (the scalars), which makes statement 3's domain empty.
        explicit_flux_density: Output, the explicit flux density [kg/m2/s per unit of the
            variable], positive upward, on every flux level including both boundaries. Its
            surface row is the explicit surface flux and the caller copies it into the
            right-hand side afterwards.
        right_hand_side: Output, 'eff_flux' in its second role. Rows 'vertical_start' to
            'vertical_end - 1' are written here; the surface row is the caller's.
        updated_profile: Output, 'upd_prof', the solved profile.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: Uppermost diffused main level; 0, mirroring Fortran 'k_tp+1'.
        vertical_end: End of the diffused main levels; 'nlev', mirroring Fortran 'k_sf-1'.
    """
    _no_flux_through_the_model_top(
        out=explicit_flux_density,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_start + 1),
        },
    )
    _compute_explicit_flux_density(
        explicit_diffusion_momentum=explicit_diffusion_momentum,
        current_profile=current_profile,
        out=explicit_flux_density,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start + 1, vertical_end + 1),
        },
    )
    _add_implicit_surface_flux_to_the_explicit_flux_density(
        explicit_flux_density=explicit_flux_density,
        implicit_diffusion_momentum=implicit_diffusion_momentum,
        current_profile=current_profile,
        out=explicit_flux_density,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (surface_addition_start, vertical_end + 1),
        },
    )
    _compute_diffusion_right_hand_side(
        discretisation_momentum=discretisation_momentum,
        current_profile=current_profile,
        explicit_flux_density=explicit_flux_density,
        out=right_hand_side,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
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
