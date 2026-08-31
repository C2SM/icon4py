# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""How fast the vertical diffusion spreads a blob: the moments, and the limit it converges to.

THE MOMENTS ARE EXACT, WHICH WAS NOT EXPECTED. The obvious analytic statement about a
spreading blob -- that it stays a Gaussian whose variance grows as 'sigma**2 + 2*K*t' -- is a
solution of the CONTINUOUS equation and the scheme reproduces it only up to truncation error.
Its MOMENTS are a different matter. Summing the flux-form update against '1' and against 'z'
and against 'z**2', and telescoping twice, gives for a uniform grid, one diffusivity, one
implicit weight and a profile that vanishes in the two end rows:

    SUM rho*dz*phi              unchanged                 (mass)
    SUM rho*dz*z*phi            unchanged                 (centre of mass: diffusion does not
                                                           transport)
    SUM rho*dz*z*z*phi          grows by exactly 2*K*dt   times the mass, EVERY step

The second-moment identity is exact for the DISCRETE scheme, not asymptotically: the second
Abel summation turns the discrete Laplacian applied to 'z**2' into the constant '2*dz**2', and
a theta-scheme integrates a constant rate exactly whatever 'theta' is. Measured against the
port over twenty steps: relative residual 0.0 -- not small, zero -- on all three moments.

So this file does NOT need a convergence study to have an exact statement, contrary to what a
step-spreading test would need. It has one anyway, as the second test, because the moments are
blind to the thing a convergence study is not: whether the operator is consistent with the
diffusion equation at all. A scheme that used the wrong 'dz' would satisfy every moment
identity above and converge to the wrong answer.

WHAT THE MOMENTS SEE AND WHAT THEY DO NOT. The zeroth moment is the column budget of
'test_vertical_diffusion_conservation.py' and adds nothing. The second is the informative one,
and its blind spot is stated rather than left to be discovered: '2*K*dt' contains no 'theta',
so the spreading rate is EXACTLY as blind to the implicit weight as the mass is. The mutation
that 'test_vertical_diffusion_amplification.py' catches passes this file untouched. What the
second moment does catch is the mutation below, which the conservation test cannot: an explicit
flux whose gradient runs the wrong way is still in flux form, still conserves mass exactly, and
spreads at '2*K*dt*(2*theta - 1)' -- half the correct rate at ICON's 'impl_t = 0.75'.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from icon4py.model.testing.fixtures.datatest import backend

from . import broken_stencils, utils


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing


#: The spreading column: deep enough that the Gaussian is 12 standard deviations from either
#: end, which is what makes the boundary terms of the two Abel summations vanish to rounding
#: rather than merely be small. At 12 sigma the profile is 1e-32 of its peak.
NUM_LEVELS = 200
LAYER_DEPTH = 25.0
AIR_DENSITY = 1.2
DIFFUSIVITY = 8.0
TIME_STEP = 20.0
INITIAL_WIDTH = 200.0
NUM_STEPS = 20

#: ICON's implicit weight above 1500 m, 'impl_t'. The moment identities do not depend on it,
#: which is asserted below by running the mutation at the same value.
IMPLICIT_WEIGHT = 0.75

#: Relative tolerance on the three moments. Measured: 2.2e-16, i.e. one rounding of the sum.
MOMENT_TOLERANCE = 1.0e-13

#: The convergence study. 'dt' is refined as 'dz**2' so that the diffusion number stays fixed
#: and the second-order space error is what the slope measures; the time error at
#: Crank-Nicolson is then 'O(dz**4)' and does not enter.
CONVERGENCE_LEVELS = (128, 256, 512)
CONVERGENCE_HEIGHT = 12800.0
CONVERGENCE_WIDTH = 400.0
CONVERGENCE_DIFFUSION_NUMBER = 0.2
CONVERGENCE_TIME = 1000.0

#: The mutation of the explicit flux, named once because its own name is long.
_REVERSED_GRADIENT = broken_stencils.compute_explicit_flux_density_with_the_gradient_reversed


def _level_heights(num_levels: int, layer_depth: float) -> np.ndarray:
    """The main-level coordinate, affine in the level index -- which is what the algebra needs.

    Whether it counts upward or downward is immaterial: every moment identity below is
    invariant under 'z -> a - z' once the first moment is conserved.
    """
    return (np.arange(num_levels) + 0.5) * layer_depth


def _gaussian(heights: np.ndarray, centre: float, width: float) -> np.ndarray:
    """A unit-peak Gaussian, as a '(1, num_levels)' single-column profile."""
    return np.exp(-0.5 * ((heights - centre) / width) ** 2)[np.newaxis, :]


def _moments(
    profile: np.ndarray, heights: np.ndarray, layer_depth: float, air_density: float
) -> tuple[float, float, float]:
    """The mass, the first and the second moment of one column, per unit area."""
    weights = air_density * layer_depth * profile[0]
    return (
        float(weights.sum()),
        float((weights * heights).sum()),
        float((weights * heights * heights).sum()),
    )


def _spread(
    backend: gtx_typing.Backend | None,
    *,
    num_levels: int = NUM_LEVELS,
    layer_depth: float = LAYER_DEPTH,
    time_step: float = TIME_STEP,
    width: float = INITIAL_WIDTH,
    implicit_weight: float = IMPLICIT_WEIGHT,
    steps: int = NUM_STEPS,
    explicit_flux_density=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Diffuse a Gaussian for 'steps' steps and return the level heights and the final profile.

    Each step rebuilds the column around the profile the previous one produced, which is what
    the granule does between time steps; nothing is carried but the profile.
    """
    heights = _level_heights(num_levels, layer_depth)
    column = utils.construct_uniform_diffusion_column(
        profile=_gaussian(heights, 0.5 * num_levels * layer_depth, width),
        diffusivity=DIFFUSIVITY,
        layer_depth=layer_depth,
        time_step=time_step,
        air_density=AIR_DENSITY,
        implicit_weight=implicit_weight,
    )
    kwargs = (
        {} if explicit_flux_density is None else {"explicit_flux_density": explicit_flux_density}
    )
    for _ in range(steps):
        column = utils.DiffusionRun(column, backend, **kwargs).stepped_column()
    return heights, column.current_profile[:, :num_levels]


def _assert_the_moments_follow_the_diffusion_equation(
    initial: np.ndarray,
    final: np.ndarray,
    heights: np.ndarray,
    *,
    layer_depth: float,
    diffusivity: float,
    time_step: float,
    steps: int,
) -> None:
    """The invariant itself, factored out so the mutation test can be seen to violate IT.

    Both the passing test and the broken-variant test go through this function, so the
    statement the mutation breaks is literally the statement the port is held to.
    """
    mass_0, first_0, second_0 = _moments(initial, heights, layer_depth, AIR_DENSITY)
    mass_1, first_1, second_1 = _moments(final, heights, layer_depth, AIR_DENSITY)

    assert abs(mass_1 / mass_0 - 1.0) < MOMENT_TOLERANCE, (
        f"the diffusion did not conserve the mass of the blob: relative change "
        f"{mass_1 / mass_0 - 1.0}."
    )
    assert abs(first_1 / first_0 - 1.0) < MOMENT_TOLERANCE, (
        f"the centre of mass moved: relative change {first_1 / first_0 - 1.0}. Vertical "
        f"diffusion transports nothing, so this is a flux that points the wrong way."
    )
    predicted = second_0 + 2.0 * diffusivity * time_step * steps * mass_0
    assert abs(second_1 / predicted - 1.0) < MOMENT_TOLERANCE, (
        f"the second moment did not grow at '2*K*dt' per step: predicted {predicted}, got "
        f"{second_1}, relative error {second_1 / predicted - 1.0}."
    )


def test_the_moments_of_a_spreading_blob_are_exactly_what_the_diffusion_equation_says(
    *, backend: gtx_typing.Backend | None
) -> None:
    """A blob spreads at exactly '2*K' per unit time and its centre of mass does not move.

    THE PHYSICS. Vertical diffusion is the whole reason the boundary-layer scheme exists: it is
    what carries the surface fluxes of heat, moisture and momentum upward. The rate at which it
    does so is 'd(sigma**2)/dt = 2*K' -- the Einstein relation -- and the direction is nowhere:
    a diffusive flux has no preferred sign, so the centre of mass of a passive blob stays put.
    Both are asserted here on the scheme's own solve, and both are EXACT statements about the
    discrete operator rather than about its continuum limit.

    The second moment is the informative one. The zeroth is the column budget again, and the
    first is zero-by-symmetry for a symmetric profile -- except that it is not asserted by
    symmetry here, it is asserted after twenty steps of an operator that has no symmetry
    imposed on it.
    """
    heights, final = _spread(backend)
    initial = _gaussian(heights, 0.5 * NUM_LEVELS * LAYER_DEPTH, INITIAL_WIDTH)

    assert initial[0, 0] < 1.0e-20 and initial[0, -1] < 1.0e-20, (
        "the blob is not negligible at the ends of the column, so the boundary terms of the "
        "moment identities do not vanish and the prediction below does not apply."
    )
    assert np.all(np.isfinite(final)), "the repeated solve produced non-finite values."

    _assert_the_moments_follow_the_diffusion_equation(
        initial,
        final,
        heights,
        layer_depth=LAYER_DEPTH,
        diffusivity=DIFFUSIVITY,
        time_step=TIME_STEP,
        steps=NUM_STEPS,
    )

    # And the same statement in the form a meteorologist would recognise: the width of the blob.
    mass, first, second = _moments(final, heights, LAYER_DEPTH, AIR_DENSITY)
    variance = second / mass - (first / mass) ** 2
    np.testing.assert_allclose(
        variance,
        INITIAL_WIDTH * INITIAL_WIDTH + 2.0 * DIFFUSIVITY * TIME_STEP * NUM_STEPS,
        rtol=1.0e-12,
        err_msg="the variance of the blob is not 'sigma0**2 + 2*K*t'",
    )


def test_the_diffused_profile_converges_to_the_analytic_gaussian_at_second_order(
    *, backend: gtx_typing.Backend | None
) -> None:
    """Refining the grid drives the solution to the exact Gaussian at the discretisation order.

    THE PHYSICS. The moments above are exact on any grid, so they cannot tell whether the
    operator solves the diffusion equation or merely some equation with the same conservation
    structure. This test asks the other question: as the grid is refined at a fixed diffusion
    number, does the solution approach

        phi(z, t) = sigma0/SQRT(sigma0**2 + 2*K*t) * EXP( -z**2 / (2*(sigma0**2 + 2*K*t)) )

    and at what rate. Second order in 'dz' is what a centred three-point Laplacian gives, and
    the implicit weight is put at 1/2 so that the time error is second order too and does not
    contaminate the measurement.

    The pattern -- refine, collect the error norms, regress the logs, assert the SLOPE rather
    than any single error -- is David Strassmann's advection convergence study (icon4py
    'advection_convergence', commit d2dc1c00f).
    """
    weight = 0.5
    depths = []
    errors = []
    for num_levels in CONVERGENCE_LEVELS:
        layer_depth = CONVERGENCE_HEIGHT / num_levels
        time_step = CONVERGENCE_DIFFUSION_NUMBER * layer_depth * layer_depth / DIFFUSIVITY
        steps = round(CONVERGENCE_TIME / time_step)
        assert abs(steps * time_step - CONVERGENCE_TIME) < 1.0e-9 * CONVERGENCE_TIME, (
            "the refinement does not land on the same physical end time, so the errors below "
            "are not comparable."
        )
        heights, final = _spread(
            backend,
            num_levels=num_levels,
            layer_depth=layer_depth,
            time_step=time_step,
            width=CONVERGENCE_WIDTH,
            implicit_weight=weight,
            steps=steps,
        )
        variance = CONVERGENCE_WIDTH**2 + 2.0 * DIFFUSIVITY * CONVERGENCE_TIME
        centre = 0.5 * CONVERGENCE_HEIGHT
        exact = (CONVERGENCE_WIDTH / np.sqrt(variance)) * np.exp(
            -0.5 * (heights - centre) * (heights - centre) / variance
        )
        depths.append(layer_depth)
        errors.append(float(np.abs(final[0] - exact).max()))

    assert errors[0] > errors[-1] > 0.0, f"the error did not decrease under refinement: {errors}."
    # 'numpy.polyfit' rather than 'scipy.stats.linregress': the advection study uses the
    # latter, but scipy is not a declared dependency of this package and a least-squares
    # line through three points needs nothing else.
    slope = float(np.polyfit(np.log(depths), np.log(errors), 1)[0])
    assert 1.8 < slope < 2.3, (
        f"the vertical diffusion does not converge at second order in 'dz': measured slope "
        f"{slope} over layer depths {depths} with L-infinity errors {errors}."
    )


def test_a_reversed_explicit_gradient_halves_the_spreading_and_not_the_mass(
    *, backend: gtx_typing.Backend | None
) -> None:
    """The moment test has teeth, and it catches what the conservation test cannot.

    'broken_stencils.compute_explicit_flux_density_with_the_gradient_reversed' subtracts the
    two profile values the other way round -- the classic error of a staggered flux in a scheme
    whose index runs downward while its flux counts upward. The scheme stays in flux form, so:

    - the mass is still conserved EXACTLY and the column budget of
      'test_vertical_diffusion_conservation.py' still closes, which is asserted here;
    - the explicit half of the operator becomes anti-diffusive, and the second moment then
      grows at '2*K*dt*(2*theta - 1)' instead of '2*K*dt' -- half the rate at 'theta = 0.75'.

    The predicted broken rate is asserted as well as the failure, so that a mutation which
    stopped doing what this test believes it does would be caught rather than silently
    weakening the demonstration.
    """
    heights, final = _spread(backend, explicit_flux_density=_REVERSED_GRADIENT)
    initial = _gaussian(heights, 0.5 * NUM_LEVELS * LAYER_DEPTH, INITIAL_WIDTH)
    assert np.all(np.isfinite(final)), (
        "the broken flux produced non-finite values, so the moments below would be NaN rather "
        "than a measurement of the defect."
    )

    mass_0, first_0, second_0 = _moments(initial, heights, LAYER_DEPTH, AIR_DENSITY)
    mass_1, first_1, second_1 = _moments(final, heights, LAYER_DEPTH, AIR_DENSITY)

    assert abs(mass_1 / mass_0 - 1.0) < MOMENT_TOLERANCE, (
        f"the reversed gradient lost mass, relative change {mass_1 / mass_0 - 1.0}. It is "
        f"supposed not to -- that is the point of this test -- so either the mutation or the "
        f"conservation argument has changed."
    )
    assert abs(first_1 / first_0 - 1.0) < MOMENT_TOLERANCE, (
        f"the reversed gradient moved the centre of mass, relative change "
        f"{first_1 / first_0 - 1.0}; the mutation is doing more than this test claims."
    )
    predicted_broken = second_0 + 2.0 * DIFFUSIVITY * TIME_STEP * NUM_STEPS * mass_0 * (
        2.0 * IMPLICIT_WEIGHT - 1.0
    )
    np.testing.assert_allclose(
        second_1,
        predicted_broken,
        rtol=1.0e-12,
        err_msg=(
            "the reversed gradient did not spread at '2*K*dt*(2*theta - 1)', so it is not "
            "doing what this test assumes and the invariant it is supposed to break is untested"
        ),
    )

    # And the invariant itself, not a restatement of it.
    with pytest.raises(AssertionError, match="second moment did not grow"):
        _assert_the_moments_follow_the_diffusion_equation(
            initial,
            final,
            heights,
            layer_depth=LAYER_DEPTH,
            diffusivity=DIFFUSIVITY,
            time_step=TIME_STEP,
            steps=NUM_STEPS,
        )
