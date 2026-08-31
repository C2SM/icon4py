# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Deliberately defective copies of the stencils the analytic tests judge.

A test that passes tells you nothing until you have seen it fail for the right reason, and a
mutation performed once by hand and then deleted is not evidence -- it is a memory. So each
invariant in this directory ships with the defect it is supposed to catch, and the test that
asserts the invariant is followed by one that asserts the defect breaks it. The pattern is
'BrokenPiecewiseParabolicMethod' of the advection convergence study
('model/atmosphere/advection/src/.../advection_vertical.py:1022' on the 'advection_convergence'
branch, commit d2dc1c00f), which ships a degraded reconstruction in the source tree and
parametrises a test case over the degraded order.

WHY THESE LIVE IN THE TEST TREE AND NOT IN 'src/'. The advection precedent puts its broken
scheme beside the real one because a reconstruction method is a scheme OPTION -- something a
configuration could legitimately select. None of these is: they are transcription errors, and a
defective stencil exported from
'icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils' would be importable by the
granule. They are test fixtures, so they are in the test package.

EACH MUTATION IS ONE PLAUSIBLE MISTAKE, not an arbitrary perturbation. A random change to a
coefficient proves only that the test notices arithmetic; a mistake a translator would actually
make proves that the test is worth running. Every one below is annotated with the mistake it
represents and with what it would take to notice it otherwise.
"""

from __future__ import annotations

import gt4py.next as gtx
from gt4py.next import broadcast, maximum, minimum, sqrt, where
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_turbulent_velocity_scale import (
    _effective_tke_forcing,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


# A stencil argument list is positional by necessity -- the domain specification of a
# 'gtx.program' names its parameters -- which is why the project already exempts
# '**/model/**/stencils/*.py' from PLR0917 in 'pyproject.toml'. These are stencils that
# happen to live in the test tree, so they need the exemption spelled out here.
# ruff: noqa: PLR0917 [too-many-positional-arguments]

__all__ = [
    "compute_explicit_flux_density_with_the_gradient_reversed",
    "compute_implicit_diffusion_momentum_with_the_complementary_weight",
    "compute_stability_lengths_with_the_buoyancy_cofactor_negated",
    "compute_stability_lengths_with_the_scalar_buoyancy_cofactor_undiminished",
    "compute_stability_lengths_with_the_shear_taken_from_the_buoyancy",
    "compute_turbulent_velocity_scale_with_the_dissipation_time_scale_inverted",
    "compute_turbulent_velocity_scale_with_the_time_smoothing_reversed",
    "smooth_tke_forcing_vertically_with_two_sided_end_rows",
    "solve_vertical_diffusion_equation_with_a_misindexed_back_substitution",
]


# ------------------------------------------------------- 1. the smoothing, with square ends ---


@gtx.field_operator
def _smooth_tke_forcing_vertically_with_two_sided_end_rows(
    tke_forcing: fa.CellKField[wpfloat],
    discretisation_momentum: fa.CellKField[wpfloat],
    smoothing_mask: fa.CellField[wpfloat],
    smoothing_weight: wpfloat,
    nlev: gtx.int32,
) -> fa.CellKField[wpfloat]:
    """'smooth_tke_forcing_vertically' with '1 - 2s' on the two one-sided rows.

    THE MISTAKE. 'vert_smooth' writes three statements and only the middle one carries
    '1 - 2*versmot'; the first and the last carry '1 - versmot', because they reach one
    neighbour instead of two. A translator who folds the three into one expression and takes
    the interior coefficient for all of them writes exactly this. Nothing about the resulting
    profile looks wrong -- the ends are still smoothed, the values are still of the right
    magnitude, and every interior row is bit-exact.

    WHAT IT COSTS. The row weights of the two end rows sum to '1 - s' instead of one, so the
    operator no longer preserves a constant and no longer conserves the mass-weighted forcing.
    Both are invisible to a spot check of the interior, and to any comparison that only looks
    at the difference between two smoothed profiles.
    """
    smoothing = smoothing_weight * smoothing_mask
    remaining = wpfloat("1.0") - wpfloat("2.0") * smoothing

    at_the_top = (
        remaining * tke_forcing
        + smoothing
        * tke_forcing(Koff[1])
        * discretisation_momentum(Koff[1])
        / discretisation_momentum
    )
    inside = (
        remaining * tke_forcing
        + smoothing
        * (
            tke_forcing(Koff[-1]) * discretisation_momentum(Koff[-1])
            + tke_forcing(Koff[1]) * discretisation_momentum(Koff[1])
        )
        / discretisation_momentum
    )
    at_the_bottom = (
        remaining * tke_forcing
        + smoothing
        * tke_forcing(Koff[-1])
        * discretisation_momentum(Koff[-1])
        / discretisation_momentum
    )

    # The same nested half-spaces as the real stencil: written as four equalities this program
    # would read one row off either end of its inputs, which on 'dace_gpu' is an out-of-bounds
    # device load. A broken stencil is still a stencil and gets the same treatment.
    smoothed = concat_where(dims.KDim < nlev, at_the_bottom, tke_forcing)
    smoothed = concat_where(dims.KDim < nlev - 1, inside, smoothed)
    smoothed = concat_where(dims.KDim < 2, at_the_top, smoothed)
    return concat_where(dims.KDim < 1, tke_forcing, smoothed)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def smooth_tke_forcing_vertically_with_two_sided_end_rows(
    tke_forcing: fa.CellKField[wpfloat],
    discretisation_momentum: fa.CellKField[wpfloat],
    smoothing_mask: fa.CellField[wpfloat],
    smoothing_weight: wpfloat,
    nlev: gtx.int32,
    smoothed_tke_forcing: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """BROKEN ON PURPOSE. Arguments are those of 'smooth_tke_forcing_vertically'."""
    _smooth_tke_forcing_vertically_with_two_sided_end_rows(
        tke_forcing=tke_forcing,
        discretisation_momentum=discretisation_momentum,
        smoothing_mask=smoothing_mask,
        smoothing_weight=smoothing_weight,
        nlev=nlev,
        out=smoothed_tke_forcing,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


# ------------------------------------------- 2. the Thomas solve, with a shifted multiplier ---


@gtx.scan_operator(axis=dims.KDim, forward=True, init=(wpfloat("0.0"), True))
def _eliminate_downward(
    state: tuple[wpfloat, bool],
    right_hand_side: wpfloat,
    implicit_diffusion_momentum: wpfloat,
    inverted_diffusion_momentum: wpfloat,
) -> tuple[wpfloat, bool]:
    """The forward substitution, unmodified: the defect below is in the backward one."""
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
    inversion_factor_of_this_level: wpfloat,
) -> tuple[wpfloat, bool]:
    """Back substitution reading 'invs_fac' at its OWN level instead of the level below."""
    solution_below, at_the_bottom = state
    solution = (
        eliminated
        if at_the_bottom
        else eliminated + inversion_factor_of_this_level * solution_below
    )
    return solution, False


@gtx.field_operator
def _solve_vertical_diffusion_equation_with_a_misindexed_back_substitution(
    right_hand_side: fa.CellKField[wpfloat],
    implicit_diffusion_momentum: fa.CellKField[wpfloat],
    inverted_diffusion_momentum: fa.CellKField[wpfloat],
    inversion_factor: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'solve_vertical_diffusion_equation' with 'invs_fac(k)' where it should read 'invs_fac(k+1)'.

    THE MISTAKE. The Fortran back substitution is

        DO k = k_sf-2, k_tp+1, -1
           upd_prof(i,k) = upd_prof(i,k) + invs_fac(i,k+1)*upd_prof(i,k+1)

    and every other quantity in the same loop nest is read at 'k'. A translator writing the
    scan carries the solution of the level below and then reaches for the multiplier that
    belongs to the same row, which is off by one. The staggering of this scheme -- flux level
    'k' lies above concentration level 'k' -- is exactly the sort of convention that makes the
    wrong choice look right.

    WHAT IT COSTS. It is not a wrong answer of the right kind: the result is no longer a
    solution of the tridiagonal system at all. The values remain finite, smooth and of the
    right order of magnitude, which is why a plot of the diffused profile does not show it.
    """
    eliminated, _ = _eliminate_downward(
        right_hand_side, implicit_diffusion_momentum, inverted_diffusion_momentum
    )
    solution, _ = _substitute_upward(eliminated, inversion_factor)
    return solution


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def solve_vertical_diffusion_equation_with_a_misindexed_back_substitution(
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
    """BROKEN ON PURPOSE. Arguments are those of 'solve_vertical_diffusion_equation'."""
    _solve_vertical_diffusion_equation_with_a_misindexed_back_substitution(
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


# --------------------------------------- 3. the stability functions, with a confused forcing ---


@gtx.field_operator
def _compute_stability_lengths_with_the_shear_taken_from_the_buoyancy(
    master_length_scale: fa.CellKField[wpfloat],
    stability_length_for_momentum: fa.CellKField[wpfloat],
    stability_length_for_scalars: fa.CellKField[wpfloat],
    mechanical_forcing: fa.CellKField[wpfloat],
    thermal_forcing: fa.CellKField[wpfloat],
    turbulent_velocity_scale: fa.CellKField[wpfloat],
    a_h: wpfloat,
    a_m: wpfloat,
    b_h: wpfloat,
    b_m: wpfloat,
    d_m: wpfloat,
    d_1: wpfloat,
    d_2: wpfloat,
    d_3: wpfloat,
    d_4: wpfloat,
    d_5: wpfloat,
    d_6: wpfloat,
    rim: wpfloat,
    frcsecu: wpfloat,
    stbsecu: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """'compute_stability_lengths' with 'a22' taking 'd_4*gh' where the closure has 'd_4*gm'.

    THE MISTAKE. The Fortran is

        a22 = d_2 + d_3*gh + d_4*gm

    -- the only coefficient of the 2x2 system that mixes the two dimensionless forcings -- and
    'gh' and 'gm' differ by one character in both languages. Writing 'd_4*gh' keeps the
    determinant positive, keeps both numerators positive over most of the state space, and
    changes nothing about the units.

    WHAT IT COSTS. The momentum stability function loses its shear dependence entirely. At
    neutral stratification, where 'gh' vanishes, 'a22' collapses to 'd_2' and 'S_m' becomes the
    constant 'a_m*b_m' instead of 'd_mom**(-1/3)' -- a 78 per cent overestimate of the neutral
    momentum diffusivity, which is what the neutral limit test measures.
    """
    forcing = _effective_tke_forcing(
        stability_length_for_momentum=stability_length_for_momentum,
        stability_length_for_scalars=stability_length_for_scalars,
        mechanical_forcing=mechanical_forcing,
        thermal_forcing=thermal_forcing,
        frcsecu=frcsecu,
        rim=rim,
    )

    turbulent_time_scale = master_length_scale / turbulent_velocity_scale
    tim2 = turbulent_time_scale * turbulent_time_scale
    gh = thermal_forcing * tim2
    gm = mechanical_forcing * tim2

    a11 = d_1 + (d_5 - d_4) * gh
    a12 = d_4 * gm
    a21 = (d_6 - d_4) * gh
    a22 = d_2 + d_3 * gh + d_4 * gh  # <-- the defect: 'gh' for 'gm'

    determinant = a11 * a22 - a12 * a21
    numerator_h = b_h * a22 - b_m * a12
    numerator_m = b_m * a11 - b_h * a21
    inverse_determinant = wpfloat("1.0") / determinant

    gam0 = stbsecu / d_m + (wpfloat("1.0") - stbsecu) * b_m / d_4
    gama = minimum(gam0, forcing * tim2 / master_length_scale)
    wert = d_4 * gama
    bb1 = (b_h - wert) * a_h
    bb2 = (b_m - wert) * a_m
    a3 = d_3 * gama * a_m
    a5 = d_5 * gama * a_h
    a6 = d_6 * gama * a_m

    val1 = (mechanical_forcing * bb2 + (a5 - a3 + bb1) * thermal_forcing) / (wpfloat("2.0") * bb1)
    val2 = val1 + sqrt(val1 * val1 - (a6 + bb2) * thermal_forcing * mechanical_forcing / bb1)
    fakt = thermal_forcing / (val2 - thermal_forcing)
    corrected_h = bb1 - a5 * fakt
    corrected_m = corrected_h * (bb2 - a6 * fakt) / (bb1 - (a5 - a3) * fakt)

    solvable = (
        (thermal_forcing >= wpfloat("0.0"))
        & (determinant > wpfloat("0.0"))
        & (numerator_h > wpfloat("0.0"))
        & (numerator_m > wpfloat("0.0"))
    )
    sh = where(solvable, numerator_h * inverse_determinant, corrected_h)
    sm = where(solvable, numerator_m * inverse_determinant, corrected_m)
    return master_length_scale * sm, master_length_scale * sh


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_stability_lengths_with_the_shear_taken_from_the_buoyancy(
    master_length_scale: fa.CellKField[wpfloat],
    stability_length_for_momentum: fa.CellKField[wpfloat],
    stability_length_for_scalars: fa.CellKField[wpfloat],
    mechanical_forcing: fa.CellKField[wpfloat],
    thermal_forcing: fa.CellKField[wpfloat],
    turbulent_velocity_scale: fa.CellKField[wpfloat],
    a_h: wpfloat,
    a_m: wpfloat,
    b_h: wpfloat,
    b_m: wpfloat,
    d_m: wpfloat,
    d_1: wpfloat,
    d_2: wpfloat,
    d_3: wpfloat,
    d_4: wpfloat,
    d_5: wpfloat,
    d_6: wpfloat,
    rim: wpfloat,
    frcsecu: wpfloat,
    stbsecu: wpfloat,
    updated_stability_length_for_momentum: fa.CellKField[wpfloat],
    updated_stability_length_for_scalars: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """BROKEN ON PURPOSE. Arguments are those of 'compute_stability_lengths'."""
    _compute_stability_lengths_with_the_shear_taken_from_the_buoyancy(
        master_length_scale=master_length_scale,
        stability_length_for_momentum=stability_length_for_momentum,
        stability_length_for_scalars=stability_length_for_scalars,
        mechanical_forcing=mechanical_forcing,
        thermal_forcing=thermal_forcing,
        turbulent_velocity_scale=turbulent_velocity_scale,
        a_h=a_h,
        a_m=a_m,
        b_h=b_h,
        b_m=b_m,
        d_m=d_m,
        d_1=d_1,
        d_2=d_2,
        d_3=d_3,
        d_4=d_4,
        d_5=d_5,
        d_6=d_6,
        rim=rim,
        frcsecu=frcsecu,
        stbsecu=stbsecu,
        out=(updated_stability_length_for_momentum, updated_stability_length_for_scalars),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


# ---------------------------------------- 4. the implicit split, with the weight complemented ---


@gtx.field_operator
def _compute_implicit_diffusion_momentum_with_the_complementary_weight(
    diffusion_momentum: fa.CellKField[wpfloat],
    implicit_weight: fa.KField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'expl_mom*(1 - impl_weight)' where 'prep_impl_vert_diff' has 'expl_mom*impl_weight'.

    THE MISTAKE. The Fortran splits the diffusion momentum by writing the IMPLICIT part first,
    'impl_mom = expl_mom*impl_weight', and then reducing the storage it came from,
    'expl_mom = expl_mom - impl_mom'. A translator who reads 'impl_weight' as the weight of the
    EXPLICIT part -- the other half of a theta-scheme, and the reading that a name like
    'weight' invites -- writes this. The two differ by nothing a dimensional check or a sign
    check can see, and both split the same total in the same place.

    WHAT IT COSTS. It runs the scheme at 'theta = 1 - impl_weight' instead of 'impl_weight':
    0.25 instead of 0.75, and -0.20 instead of the over-implicit 1.20. The column budget still
    closes EXACTLY -- the split is what the telescoping is blind to -- and so does the second
    moment of a spreading Gaussian, which does not depend on 'theta' either. Only the
    amplification factor of a mode sees it, and at 'theta = 1/2' not even that: Crank-Nicolson
    is its own complement, which is why the mutation test below excludes that weight.
    """
    return diffusion_momentum * (wpfloat("1.0") - implicit_weight)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_implicit_diffusion_momentum_with_the_complementary_weight(
    diffusion_momentum: fa.CellKField[wpfloat],
    implicit_weight: fa.KField[wpfloat],
    implicit_diffusion_momentum: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """BROKEN ON PURPOSE. Arguments are those of 'compute_implicit_diffusion_momentum'."""
    _compute_implicit_diffusion_momentum_with_the_complementary_weight(
        diffusion_momentum=diffusion_momentum,
        implicit_weight=implicit_weight,
        out=implicit_diffusion_momentum,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


# ------------------------------------------ 5. the explicit flux, with the gradient reversed ---


@gtx.field_operator
def _compute_explicit_flux_density_with_the_gradient_reversed(
    explicit_diffusion_momentum: fa.CellKField[wpfloat],
    current_profile: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'expl_mom*(cur_prof(k-1) - cur_prof(k))' where 'calc_impl_vert_diff' has the difference
    the other way round.

    THE MISTAKE. A staggered flux has two conventions to keep straight at once -- which way the
    vertical index runs, and which direction the flux is counted positive -- and this scheme
    runs 'k' DOWNWARD while counting the flux UPWARD. Getting the pair wrong exactly once
    produces this line. It is the single most common error in a vertical flux and it survives
    every check that does not look at the sign of a transport.

    WHAT IT COSTS, and the interesting part is what it does NOT cost. The scheme stays in flux
    form, so it still telescopes and the column budget still closes exactly on a closed column:
    the conservation test cannot see it. What it breaks is the operator itself: the explicit
    part becomes anti-diffusive, and the second moment of a spreading profile then grows at
    '2*K*dt*(2*theta - 1)' per step instead of '2*K*dt' -- half the correct rate at the shipped
    'impl_t = 0.75', and none at all at 'theta = 1/2'.
    """
    return explicit_diffusion_momentum * (current_profile(Koff[-1]) - current_profile)


@gtx.field_operator
def _no_flux_through_the_model_top() -> fa.CellKField[wpfloat]:
    """The upper boundary condition, unmodified: the defect above is in the interior rows."""
    return broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim))


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_explicit_flux_density_with_the_gradient_reversed(
    explicit_diffusion_momentum: fa.CellKField[wpfloat],
    current_profile: fa.CellKField[wpfloat],
    model_top_level: gtx.int32,
    explicit_flux_density: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """BROKEN ON PURPOSE. Arguments are those of 'compute_explicit_flux_density'."""
    _no_flux_through_the_model_top(
        out=explicit_flux_density,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (model_top_level, vertical_start),
        },
    )
    _compute_explicit_flux_density_with_the_gradient_reversed(
        explicit_diffusion_momentum=explicit_diffusion_momentum,
        current_profile=current_profile,
        out=explicit_flux_density,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


# ------------------------------- 6. the TKE step, with the dissipation time scale inverted ---


@gtx.field_operator
def _compute_turbulent_velocity_scale_with_the_dissipation_time_scale_inverted(
    master_length_scale: fa.CellKField[wpfloat],
    stability_length_for_momentum: fa.CellKField[wpfloat],
    stability_length_for_scalars: fa.CellKField[wpfloat],
    mechanical_forcing: fa.CellKField[wpfloat],
    thermal_forcing: fa.CellKField[wpfloat],
    previous_velocity_scale: fa.CellKField[wpfloat],
    transport_tendency: fa.CellKField[wpfloat],
    d_m: wpfloat,
    d_4: wpfloat,
    b_m: wpfloat,
    rim: wpfloat,
    frcsecu: wpfloat,
    tkesecu: wpfloat,
    tkesmot: wpfloat,
    vel_min: wpfloat,
    tke_time_step: wpfloat,
    inverse_tke_time_step: wpfloat,
) -> fa.CellKField[wpfloat]:
    """'q1 = l_dis*dt_tke' where the Fortran has 'q1 = l_dis*fr_tke', 'fr_tke = 1/dt_tke'.

    THE MISTAKE. The port takes BOTH the time step and its reciprocal as arguments, and says so
    in its own docstring: "'fr_tke', '1/dt_tke' [1/s]. Passed rather than derived because
    'turb_setup' derives it once (turb_utilities.f90:317) and the scheme multiplies by it,
    which is not the same rounding as dividing by 'dt_tke'." Two arguments that differ by a
    reciprocal, used four lines apart, is exactly the situation in which the wrong one gets
    picked, and 'q1' is the only place in the routine where the distinction is invisible to the
    units: 'l_dis' is a length, so 'l_dis/dt' and 'l_dis*dt' are equally meaningless on their
    own and only the surrounding quadratic fixes which is wanted.

    WHAT IT COSTS. 'q1' is the dissipation rate scale of the implicit step: the root solves
    'q + q*q/q1 = q2', so multiplying 'q1' by 'dt*dt' instead of dividing weakens the
    dissipation by that factor. The steady state moves from 'q = d_mom**(1/3)*l*|S|' to roughly
    'SQRT(d_mom)*dt*l*|S|', two orders of magnitude too large at any realistic time step, and
    the neutral surface layer loses the identity 'K_m = kappa*u_star*z' with it. It leaves the
    PURE DECAY almost untouched per step, because a dissipation that weak barely moves 'q' --
    which is why the decay test is not the one that catches it.
    """
    dissipation_length = master_length_scale * d_m
    forcing_length = master_length_scale * d_4 * (wpfloat("1.0") / b_m)
    forcing = _effective_tke_forcing(
        stability_length_for_momentum=stability_length_for_momentum,
        stability_length_for_scalars=stability_length_for_scalars,
        mechanical_forcing=mechanical_forcing,
        thermal_forcing=thermal_forcing,
        frcsecu=frcsecu,
        rim=rim,
    )

    q1 = dissipation_length * tke_time_step  # <-- the defect: 'dt_tke' for 'fr_tke'
    q2 = (
        maximum(wpfloat("0.0"), previous_velocity_scale + transport_tendency * tke_time_step)
        + forcing * tke_time_step
    )
    updated = (
        q1 * (sqrt(wpfloat("1.0") + wpfloat("4.0") * q2 / q1) - wpfloat("1.0")) * wpfloat("0.5")
    )

    equilibrium_floor = sqrt(forcing_length * maximum(forcing, wpfloat("0.0")))
    smoothed = tkesmot * previous_velocity_scale + (wpfloat("1.0") - tkesmot) * updated
    return maximum(maximum(tkesecu * vel_min, tkesecu * equilibrium_floor), smoothed)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_turbulent_velocity_scale_with_the_dissipation_time_scale_inverted(
    master_length_scale: fa.CellKField[wpfloat],
    stability_length_for_momentum: fa.CellKField[wpfloat],
    stability_length_for_scalars: fa.CellKField[wpfloat],
    mechanical_forcing: fa.CellKField[wpfloat],
    thermal_forcing: fa.CellKField[wpfloat],
    previous_velocity_scale: fa.CellKField[wpfloat],
    transport_tendency: fa.CellKField[wpfloat],
    d_m: wpfloat,
    d_4: wpfloat,
    b_m: wpfloat,
    rim: wpfloat,
    frcsecu: wpfloat,
    tkesecu: wpfloat,
    tkesmot: wpfloat,
    vel_min: wpfloat,
    tke_time_step: wpfloat,
    inverse_tke_time_step: wpfloat,
    turbulent_velocity_scale: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """BROKEN ON PURPOSE. Arguments are those of 'compute_turbulent_velocity_scale'."""
    _compute_turbulent_velocity_scale_with_the_dissipation_time_scale_inverted(
        master_length_scale=master_length_scale,
        stability_length_for_momentum=stability_length_for_momentum,
        stability_length_for_scalars=stability_length_for_scalars,
        mechanical_forcing=mechanical_forcing,
        thermal_forcing=thermal_forcing,
        previous_velocity_scale=previous_velocity_scale,
        transport_tendency=transport_tendency,
        d_m=d_m,
        d_4=d_4,
        b_m=b_m,
        rim=rim,
        frcsecu=frcsecu,
        tkesecu=tkesecu,
        tkesmot=tkesmot,
        vel_min=vel_min,
        tke_time_step=tke_time_step,
        inverse_tke_time_step=inverse_tke_time_step,
        out=turbulent_velocity_scale,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


# ------------------------------------------- 7. the TKE step, with the time smoothing reversed ---


@gtx.field_operator
def _compute_turbulent_velocity_scale_with_the_time_smoothing_reversed(
    master_length_scale: fa.CellKField[wpfloat],
    stability_length_for_momentum: fa.CellKField[wpfloat],
    stability_length_for_scalars: fa.CellKField[wpfloat],
    mechanical_forcing: fa.CellKField[wpfloat],
    thermal_forcing: fa.CellKField[wpfloat],
    previous_velocity_scale: fa.CellKField[wpfloat],
    transport_tendency: fa.CellKField[wpfloat],
    d_m: wpfloat,
    d_4: wpfloat,
    b_m: wpfloat,
    rim: wpfloat,
    frcsecu: wpfloat,
    tkesecu: wpfloat,
    tkesmot: wpfloat,
    vel_min: wpfloat,
    tke_time_step: wpfloat,
    inverse_tke_time_step: wpfloat,
) -> fa.CellKField[wpfloat]:
    """'tkesmot' put on the NEW value instead of on the previous one.

    THE MISTAKE. The Fortran is 'w1 = tkesmot; w2 = 1 - tkesmot' followed by
    'w1*tvs_0 + w2*tvs_u', so the SMALL weight belongs to the OLD value and the scheme is
    mostly the fresh solution. Read as "a smoothing factor", 'tkesmot = 0.15' looks like the
    amount of the new value to admit -- that is how a relaxation constant usually reads -- and
    a translator who writes the convex combination from the name rather than from 'w1' and 'w2'
    produces this. The result stays a convex combination, stays positive, stays bounded by the
    two values it interpolates, and is still 'q' at the fixed point.

    WHAT IT COSTS, and the point is how narrow it is. At a STEADY STATE the two values being
    combined are equal, so the mutation is EXACTLY invisible: the equilibrium test cannot see
    it at all, and neither can any test that only looks at a converged column. It shows up only
    in the transient, where it slows every adjustment of the turbulence by a factor of about
    '1/(1 - tkesmot)' -- which is what the pure-decay test measures.
    """
    dissipation_length = master_length_scale * d_m
    forcing_length = master_length_scale * d_4 * (wpfloat("1.0") / b_m)
    forcing = _effective_tke_forcing(
        stability_length_for_momentum=stability_length_for_momentum,
        stability_length_for_scalars=stability_length_for_scalars,
        mechanical_forcing=mechanical_forcing,
        thermal_forcing=thermal_forcing,
        frcsecu=frcsecu,
        rim=rim,
    )

    q1 = dissipation_length * inverse_tke_time_step
    q2 = (
        maximum(wpfloat("0.0"), previous_velocity_scale + transport_tendency * tke_time_step)
        + forcing * tke_time_step
    )
    updated = (
        q1 * (sqrt(wpfloat("1.0") + wpfloat("4.0") * q2 / q1) - wpfloat("1.0")) * wpfloat("0.5")
    )

    equilibrium_floor = sqrt(forcing_length * maximum(forcing, wpfloat("0.0")))
    # The defect: the weights of the two time levels are exchanged.
    smoothed = (wpfloat("1.0") - tkesmot) * previous_velocity_scale + tkesmot * updated
    return maximum(maximum(tkesecu * vel_min, tkesecu * equilibrium_floor), smoothed)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_turbulent_velocity_scale_with_the_time_smoothing_reversed(
    master_length_scale: fa.CellKField[wpfloat],
    stability_length_for_momentum: fa.CellKField[wpfloat],
    stability_length_for_scalars: fa.CellKField[wpfloat],
    mechanical_forcing: fa.CellKField[wpfloat],
    thermal_forcing: fa.CellKField[wpfloat],
    previous_velocity_scale: fa.CellKField[wpfloat],
    transport_tendency: fa.CellKField[wpfloat],
    d_m: wpfloat,
    d_4: wpfloat,
    b_m: wpfloat,
    rim: wpfloat,
    frcsecu: wpfloat,
    tkesecu: wpfloat,
    tkesmot: wpfloat,
    vel_min: wpfloat,
    tke_time_step: wpfloat,
    inverse_tke_time_step: wpfloat,
    turbulent_velocity_scale: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """BROKEN ON PURPOSE. Arguments are those of 'compute_turbulent_velocity_scale'."""
    _compute_turbulent_velocity_scale_with_the_time_smoothing_reversed(
        master_length_scale=master_length_scale,
        stability_length_for_momentum=stability_length_for_momentum,
        stability_length_for_scalars=stability_length_for_scalars,
        mechanical_forcing=mechanical_forcing,
        thermal_forcing=thermal_forcing,
        previous_velocity_scale=previous_velocity_scale,
        transport_tendency=transport_tendency,
        d_m=d_m,
        d_4=d_4,
        b_m=b_m,
        rim=rim,
        frcsecu=frcsecu,
        tkesecu=tkesecu,
        tkesmot=tkesmot,
        vel_min=vel_min,
        tke_time_step=tke_time_step,
        inverse_tke_time_step=inverse_tke_time_step,
        out=turbulent_velocity_scale,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


# ------------------------------ 8. the stability functions, with the buoyancy cofactor negated ---


@gtx.field_operator
def _compute_stability_lengths_with_the_buoyancy_cofactor_negated(
    master_length_scale: fa.CellKField[wpfloat],
    stability_length_for_momentum: fa.CellKField[wpfloat],
    stability_length_for_scalars: fa.CellKField[wpfloat],
    mechanical_forcing: fa.CellKField[wpfloat],
    thermal_forcing: fa.CellKField[wpfloat],
    turbulent_velocity_scale: fa.CellKField[wpfloat],
    a_h: wpfloat,
    a_m: wpfloat,
    b_h: wpfloat,
    b_m: wpfloat,
    d_m: wpfloat,
    d_1: wpfloat,
    d_2: wpfloat,
    d_3: wpfloat,
    d_4: wpfloat,
    d_5: wpfloat,
    d_6: wpfloat,
    rim: wpfloat,
    frcsecu: wpfloat,
    stbsecu: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """'compute_stability_lengths' with 'a21 = -(d_6 - d_4)*gh' where the closure has '+'.

    THE MISTAKE. 'a21' is the only coefficient of the 2x2 system that appears in the solution
    with a MINUS sign already -- Cramer's rule gives 'sm = b_m*a11 - b_h*a21' -- so a translator
    who folds that sign into the coefficient once, and then lets the subtraction apply it again,
    lands here. Nothing about the result looks wrong: the determinant stays positive, both
    numerators stay positive, and the whole neutral limit is unchanged.

    WHAT IT COSTS, and this is the whole reason it exists. 'a21' multiplies 'gh' and nothing
    else, so at 'Ri = 0' this mutation is EXACTLY the identity --
    'test_the_neutral_limit_is_blind_to_the_buoyancy_cofactor' in
    'test_neutral_stability_functions.py' demonstrates it. Every neutral test in this directory
    passes it: the zero-forcing limit, the neutral equilibrium constants, the log-law surface
    layer, the pure decay. Only a STRATIFIED state sees it, which is what the stable equilibrium
    of 'test_tke_steady_state.py' is for; it is the mutation that closes the blind spot the
    first batch of analytic tests wrote down and could not cover.
    """
    forcing = _effective_tke_forcing(
        stability_length_for_momentum=stability_length_for_momentum,
        stability_length_for_scalars=stability_length_for_scalars,
        mechanical_forcing=mechanical_forcing,
        thermal_forcing=thermal_forcing,
        frcsecu=frcsecu,
        rim=rim,
    )

    turbulent_time_scale = master_length_scale / turbulent_velocity_scale
    tim2 = turbulent_time_scale * turbulent_time_scale
    gh = thermal_forcing * tim2
    gm = mechanical_forcing * tim2

    a11 = d_1 + (d_5 - d_4) * gh
    a12 = d_4 * gm
    a21 = -(d_6 - d_4) * gh  # <-- the defect: the sign of the buoyancy cofactor
    a22 = d_2 + d_3 * gh + d_4 * gm

    determinant = a11 * a22 - a12 * a21
    numerator_h = b_h * a22 - b_m * a12
    numerator_m = b_m * a11 - b_h * a21
    inverse_determinant = wpfloat("1.0") / determinant

    gam0 = stbsecu / d_m + (wpfloat("1.0") - stbsecu) * b_m / d_4
    gama = minimum(gam0, forcing * tim2 / master_length_scale)
    wert = d_4 * gama
    bb1 = (b_h - wert) * a_h
    bb2 = (b_m - wert) * a_m
    a3 = d_3 * gama * a_m
    a5 = d_5 * gama * a_h
    a6 = d_6 * gama * a_m

    val1 = (mechanical_forcing * bb2 + (a5 - a3 + bb1) * thermal_forcing) / (wpfloat("2.0") * bb1)
    val2 = val1 + sqrt(val1 * val1 - (a6 + bb2) * thermal_forcing * mechanical_forcing / bb1)
    fakt = thermal_forcing / (val2 - thermal_forcing)
    corrected_h = bb1 - a5 * fakt
    corrected_m = corrected_h * (bb2 - a6 * fakt) / (bb1 - (a5 - a3) * fakt)

    solvable = (
        (thermal_forcing >= wpfloat("0.0"))
        & (determinant > wpfloat("0.0"))
        & (numerator_h > wpfloat("0.0"))
        & (numerator_m > wpfloat("0.0"))
    )
    sh = where(solvable, numerator_h * inverse_determinant, corrected_h)
    sm = where(solvable, numerator_m * inverse_determinant, corrected_m)
    return master_length_scale * sm, master_length_scale * sh


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_stability_lengths_with_the_buoyancy_cofactor_negated(
    master_length_scale: fa.CellKField[wpfloat],
    stability_length_for_momentum: fa.CellKField[wpfloat],
    stability_length_for_scalars: fa.CellKField[wpfloat],
    mechanical_forcing: fa.CellKField[wpfloat],
    thermal_forcing: fa.CellKField[wpfloat],
    turbulent_velocity_scale: fa.CellKField[wpfloat],
    a_h: wpfloat,
    a_m: wpfloat,
    b_h: wpfloat,
    b_m: wpfloat,
    d_m: wpfloat,
    d_1: wpfloat,
    d_2: wpfloat,
    d_3: wpfloat,
    d_4: wpfloat,
    d_5: wpfloat,
    d_6: wpfloat,
    rim: wpfloat,
    frcsecu: wpfloat,
    stbsecu: wpfloat,
    updated_stability_length_for_momentum: fa.CellKField[wpfloat],
    updated_stability_length_for_scalars: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """BROKEN ON PURPOSE. Arguments are those of 'compute_stability_lengths'."""
    _compute_stability_lengths_with_the_buoyancy_cofactor_negated(
        master_length_scale=master_length_scale,
        stability_length_for_momentum=stability_length_for_momentum,
        stability_length_for_scalars=stability_length_for_scalars,
        mechanical_forcing=mechanical_forcing,
        thermal_forcing=thermal_forcing,
        turbulent_velocity_scale=turbulent_velocity_scale,
        a_h=a_h,
        a_m=a_m,
        b_h=b_h,
        b_m=b_m,
        d_m=d_m,
        d_1=d_1,
        d_2=d_2,
        d_3=d_3,
        d_4=d_4,
        d_5=d_5,
        d_6=d_6,
        rim=rim,
        frcsecu=frcsecu,
        stbsecu=stbsecu,
        out=(updated_stability_length_for_momentum, updated_stability_length_for_scalars),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


# ------------------- 9. the stability functions, with the scalar buoyancy cofactor undiminished ---


@gtx.field_operator
def _compute_stability_lengths_with_the_scalar_buoyancy_cofactor_undiminished(
    master_length_scale: fa.CellKField[wpfloat],
    stability_length_for_momentum: fa.CellKField[wpfloat],
    stability_length_for_scalars: fa.CellKField[wpfloat],
    mechanical_forcing: fa.CellKField[wpfloat],
    thermal_forcing: fa.CellKField[wpfloat],
    turbulent_velocity_scale: fa.CellKField[wpfloat],
    a_h: wpfloat,
    a_m: wpfloat,
    b_h: wpfloat,
    b_m: wpfloat,
    d_m: wpfloat,
    d_1: wpfloat,
    d_2: wpfloat,
    d_3: wpfloat,
    d_4: wpfloat,
    d_5: wpfloat,
    d_6: wpfloat,
    rim: wpfloat,
    frcsecu: wpfloat,
    stbsecu: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """'compute_stability_lengths' with 'a11 = d_1 + d_5*gh' where the closure has '(d_5 - d_4)'.

    THE MISTAKE. One dropped term in one coefficient. 'a11' and 'a21' are written in the Fortran
    as '(d_5 - d_4)*gh' and '(d_6 - d_4)*gh', the same shape twice, and a transcription that
    carries the subtraction into the second and forgets it in the first lands here. Nothing about
    the result looks wrong -- the coefficient stays positive, the determinant and both numerators
    stay positive over the whole stable quadrant, so the standard branch is still selected
    everywhere the faithful stencil selects it, and no output is out of range.

    WHY IT IS THE MUTATION FOR THE PUBLISHED-CLOSURE TEST. In Mellor & Yamada (1982) Eq. (34) the
    coefficient of 'G_H' in the scalar equation is '3*A2*B2 + 12*A1*A2': a Kolmogorov scalar
    dissipation length 'B2' plus a Rotta pressure-redistribution length 'A1'. ICON's
    '(d_5 - d_4) = 3*B2 + 12*A1' is exactly that, and the mutation makes it '3*B2 + 18*A1' -- the
    right physics with the wrong weight on one of the two length scales. That is the shape of
    error the published-source oracle exists to catch, and it is invisible to the rest of this
    directory:

    - it multiplies 'gh' and nothing else, so at 'Ri = 0' it is EXACTLY the identity, as
      'test_neutral_stability_functions.py' proves of any such coefficient;
    - 'd_5' reaches the rest of ICON only through 'rim', which this stencil takes as a separate
      argument, so every constant a test could compare against -- 'sm_0', 'sh_0', 'c_tke', 'rim'
      -- is untouched. An oracle built out of 'TurbulenceParams' therefore cannot see it at all.

    It is caught by comparing against the paper, over a stratified state, and by nothing else in
    this package.
    """
    forcing = _effective_tke_forcing(
        stability_length_for_momentum=stability_length_for_momentum,
        stability_length_for_scalars=stability_length_for_scalars,
        mechanical_forcing=mechanical_forcing,
        thermal_forcing=thermal_forcing,
        frcsecu=frcsecu,
        rim=rim,
    )

    turbulent_time_scale = master_length_scale / turbulent_velocity_scale
    tim2 = turbulent_time_scale * turbulent_time_scale
    gh = thermal_forcing * tim2
    gm = mechanical_forcing * tim2

    a11 = d_1 + d_5 * gh  # <-- the defect: the '- d_4' of the scalar buoyancy cofactor is gone
    a12 = d_4 * gm
    a21 = (d_6 - d_4) * gh
    a22 = d_2 + d_3 * gh + d_4 * gm

    determinant = a11 * a22 - a12 * a21
    numerator_h = b_h * a22 - b_m * a12
    numerator_m = b_m * a11 - b_h * a21
    inverse_determinant = wpfloat("1.0") / determinant

    gam0 = stbsecu / d_m + (wpfloat("1.0") - stbsecu) * b_m / d_4
    gama = minimum(gam0, forcing * tim2 / master_length_scale)
    wert = d_4 * gama
    bb1 = (b_h - wert) * a_h
    bb2 = (b_m - wert) * a_m
    a3 = d_3 * gama * a_m
    a5 = d_5 * gama * a_h
    a6 = d_6 * gama * a_m

    val1 = (mechanical_forcing * bb2 + (a5 - a3 + bb1) * thermal_forcing) / (wpfloat("2.0") * bb1)
    val2 = val1 + sqrt(val1 * val1 - (a6 + bb2) * thermal_forcing * mechanical_forcing / bb1)
    fakt = thermal_forcing / (val2 - thermal_forcing)
    corrected_h = bb1 - a5 * fakt
    corrected_m = corrected_h * (bb2 - a6 * fakt) / (bb1 - (a5 - a3) * fakt)

    solvable = (
        (thermal_forcing >= wpfloat("0.0"))
        & (determinant > wpfloat("0.0"))
        & (numerator_h > wpfloat("0.0"))
        & (numerator_m > wpfloat("0.0"))
    )
    sh = where(solvable, numerator_h * inverse_determinant, corrected_h)
    sm = where(solvable, numerator_m * inverse_determinant, corrected_m)
    return master_length_scale * sm, master_length_scale * sh


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_stability_lengths_with_the_scalar_buoyancy_cofactor_undiminished(
    master_length_scale: fa.CellKField[wpfloat],
    stability_length_for_momentum: fa.CellKField[wpfloat],
    stability_length_for_scalars: fa.CellKField[wpfloat],
    mechanical_forcing: fa.CellKField[wpfloat],
    thermal_forcing: fa.CellKField[wpfloat],
    turbulent_velocity_scale: fa.CellKField[wpfloat],
    a_h: wpfloat,
    a_m: wpfloat,
    b_h: wpfloat,
    b_m: wpfloat,
    d_m: wpfloat,
    d_1: wpfloat,
    d_2: wpfloat,
    d_3: wpfloat,
    d_4: wpfloat,
    d_5: wpfloat,
    d_6: wpfloat,
    rim: wpfloat,
    frcsecu: wpfloat,
    stbsecu: wpfloat,
    updated_stability_length_for_momentum: fa.CellKField[wpfloat],
    updated_stability_length_for_scalars: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """BROKEN ON PURPOSE. Arguments are those of 'compute_stability_lengths'."""
    _compute_stability_lengths_with_the_scalar_buoyancy_cofactor_undiminished(
        master_length_scale=master_length_scale,
        stability_length_for_momentum=stability_length_for_momentum,
        stability_length_for_scalars=stability_length_for_scalars,
        mechanical_forcing=mechanical_forcing,
        thermal_forcing=thermal_forcing,
        turbulent_velocity_scale=turbulent_velocity_scale,
        a_h=a_h,
        a_m=a_m,
        b_h=b_h,
        b_m=b_m,
        d_m=d_m,
        d_1=d_1,
        d_2=d_2,
        d_3=d_3,
        d_4=d_4,
        d_5=d_5,
        d_6=d_6,
        rim=rim,
        frcsecu=frcsecu,
        stbsecu=stbsecu,
        out=(updated_stability_length_for_momentum, updated_stability_length_for_scalars),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
