# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Deliberately defective copies of the three stencils the analytic tests judge.

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
from gt4py.next import minimum, sqrt, where
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
    "compute_stability_lengths_with_the_shear_taken_from_the_buoyancy",
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
