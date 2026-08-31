# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""What one step of the vertical diffusion does to a single Fourier mode: an exact eigenvalue.

WHY THIS IS NOT A CONVERGENCE STUDY. On a uniform grid with one density, one diffusivity and
one implicit weight, closed at both ends, the discrete cosines

    phi(k) = cos( j*pi*(k + 1/2) / nlev ),   j = 0 .. nlev-1

are EXACT eigenvectors of the one-step operator, not merely approximate ones. The three-point
Laplacian gives 'phi(k+1) - 2*phi(k) + phi(k-1) = -s*phi(k)' with
's = 4*sin(j*pi/(2*nlev))**2', and the two boundary rows are consistent with it because the
cosine is even about both half-integer end points -- which is exactly the zero-flux condition
the scheme imposes. So one step multiplies the mode by a NUMBER, and the number is

    G(j) = ( 1 - (1 - theta)*r*s ) / ( 1 + theta*r*s ),   r = K*dt/dz**2,   theta = impl_weight

with no truncation error anywhere in the statement. Measured against the port: 1e-14 absolute
on a mode of unit amplitude, over four implicit weights and two diffusion numbers.

WHAT IT TESTS THAT THE COLUMN BUDGET CANNOT. Conservation is a statement about the sum of the
rows and is blind to how the operator splits between the two time levels: 'test_vertical_
diffusion_conservation.py' passes unchanged when 'theta' is replaced by '1 - theta'. The
amplification factor is the statement that pins the split down, and the mutation below is
exactly that swap -- it conserves mass to rounding and gets every mode wrong.

THETA IS BIGGER THAN ONE NEAR THE GROUND, AND THAT IS DELIBERATE. ICON ramps 'impl_weight'
from 'impl_t = 0.75' aloft to 'impl_s = 1.20' on the surface flux level
(mo_nwp_phy_init.f90:1538-1547, mo_turbdiff_config.f90:126-127). Two consequences of 'G' are
worth having on record because both are easy to be surprised by:

- '|G| <= 1' for every 'r*s' as soon as 'theta >= 1/2', so both weights are unconditionally
  stable. They are NOT both monotone, and the difference is worth stating: 'G >= 0' needs
  '(1 - theta)*r*s <= 1', so a mode stiffer than 'r*s = 1/(1 - theta)' -- four, at
  'impl_t = 0.75' -- has its sign flipped every step on the way down. Only 'theta >= 1', which
  includes the surface weight 'impl_s = 1.20', damps monotonically at every stiffness;
- 'G -> (theta - 1)/theta' as 'r*s -> infinity'. At 'theta = 1.20' that is '1/6' and NOT zero:
  an over-implicit step does not annihilate the stiffest modes, it leaves a sixth of them
  standing for ever. That is asserted below, not assumed.

Everything here is built by 'utils.construct_uniform_diffusion_column'; no serialized data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from icon4py.model.testing.fixtures.datatest import backend

from . import broken_stencils, utils


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing


#: Number of main levels. 32 makes the coarsest mode span the column and the finest sit at the
#: grid scale, which is the range over which 's' varies by four decades.
NUM_LEVELS = 32

#: The mode numbers, one per column of the test state. 'j = 0' is left out on purpose: it is
#: the constant, 's = 0', 'G = 1', and it is already covered by the conservation test.
MODES = (1, 2, 4, 8, 16, 31)

#: The implicit weights swept. 0.5 is Crank-Nicolson, 0.75 is ICON's 'impl_t', 1.0 is fully
#: implicit and 1.20 is ICON's over-implicit 'impl_s' at the surface.
IMPLICIT_WEIGHTS = (0.5, 0.75, 1.0, 1.20)

#: Diffusion numbers 'r = K*dt/dz**2'. 0.12 is a mild step; 2.5 is well past the explicit
#: stability limit of 1/2, which is the regime the implicit scheme exists for.
DIFFUSION_NUMBERS = (0.12, 2.5)

#: The weights the complementary-split mutation is exercised at. 0.5 is excluded because
#: Crank-Nicolson is its own complement and the mutation is then the identity; 1.20 is excluded
#: because its complement is NEGATIVE, '-0.20', and the resulting diagonal
#: 'disc_mom + impl_mom(k) + impl_mom(k+1)' passes through zero at 'r = 2.5' -- a singular
#: matrix is not a wrong answer, it is no answer, and it would make the mutation test measure a
#: division by zero instead of a defect.
COMPLEMENTABLE_WEIGHTS = (0.75, 1.0)

#: Relative tolerance on the amplified mode. Measured worst case over the whole sweep and every
#: backend: 1e-14. The mutation below moves the answer by O(1), so nothing here rests on the
#: size of this number.
AMPLIFICATION_TOLERANCE = 1.0e-13

#: The tolerance the stiff-limit test needs instead, and the reason is CONDITIONING and not
#: slack. At 'r = 1e8' the diagonal of the tridiagonal system is 'disc_mom + 2.4e8*disc_mom',
#: so 'disc_mom' itself -- the whole of the time-derivative term -- is 4e-9 of the pivot and is
#: lost to rounding. Measured worst relative error 2e-9 on every backend. That is the accuracy
#: an infinitely stiff mode HAS; it is not a property of the port.
STIFF_LIMIT_TOLERANCE = 1.0e-7


#: The mutation of the implicit/explicit split, named once because its own name is long.
_COMPLEMENTARY_SPLIT = (
    broken_stencils.compute_implicit_diffusion_momentum_with_the_complementary_weight
)


def _cosine_modes(nlev: int, modes: tuple[int, ...]) -> np.ndarray:
    """One discrete cosine mode per column, '(len(modes), nlev)'.

    The half-integer offset is what makes the mode even about both ends of the column, i.e.
    what makes it satisfy the zero-flux boundary condition exactly rather than nearly.
    """
    levels = np.arange(nlev)
    return np.stack(
        [np.cos(mode * np.pi * (levels + 0.5) / nlev) for mode in modes],
        axis=0,
    )


def _amplification_factor(mode: int, nlev: int, diffusion_number: float, weight: float) -> float:
    """'G(j)' for one mode: the theta-scheme eigenvalue of the closed discrete Laplacian."""
    eigenvalue = 4.0 * np.sin(mode * np.pi / (2.0 * nlev)) ** 2
    stiffness = diffusion_number * eigenvalue
    return (1.0 - (1.0 - weight) * stiffness) / (1.0 + weight * stiffness)


def _one_step(
    diffusion_number: float,
    weight: float,
    backend: gtx_typing.Backend | None,
    *,
    implicit_split=None,
) -> tuple[utils.DiffusionRun, np.ndarray]:
    """Advance the six cosine modes one step and return the run and the initial profile.

    'K', 'dz' and 'dt' are chosen so that 'r' comes out as asked; only their combination
    reaches the stencils, as 'diffusion_momentum/discretisation_momentum'.
    """
    layer_depth = 40.0
    time_step = 30.0
    diffusivity = diffusion_number * layer_depth * layer_depth / time_step
    profile = _cosine_modes(NUM_LEVELS, MODES)
    column = utils.construct_uniform_diffusion_column(
        profile=profile,
        diffusivity=diffusivity,
        layer_depth=layer_depth,
        time_step=time_step,
        implicit_weight=weight,
    )
    kwargs = {} if implicit_split is None else {"implicit_split": implicit_split}
    return utils.DiffusionRun(column, backend, **kwargs), profile


def _assert_the_modes_are_amplified_by_the_predicted_factor(
    run: utils.DiffusionRun,
    profile: np.ndarray,
    diffusion_number: float,
    weight: float,
    tolerance: float = AMPLIFICATION_TOLERANCE,
) -> None:
    """The invariant itself, factored out so the mutation test can be seen to violate IT.

    Both the passing test and the broken-variant test go through this function, so the
    statement the mutation breaks is literally the statement the port is held to.
    """
    computed = run.updated_profile[:, : run.column.nlev]
    predicted = np.stack(
        [
            _amplification_factor(mode, run.column.nlev, diffusion_number, weight) * profile[i]
            for i, mode in enumerate(MODES)
        ],
        axis=0,
    )
    error = np.abs(computed - predicted).max() / np.abs(predicted).max()
    assert error < tolerance, (
        f"a cosine mode is not amplified by the theta-scheme factor: worst relative error "
        f"{error} at r = {diffusion_number}, theta = {weight}, tolerance {tolerance}."
    )


@pytest.mark.parametrize("weight", IMPLICIT_WEIGHTS)
@pytest.mark.parametrize("diffusion_number", DIFFUSION_NUMBERS)
def test_a_cosine_mode_is_multiplied_by_the_theta_scheme_amplification_factor(
    diffusion_number: float, weight: float, *, backend: gtx_typing.Backend | None
) -> None:
    """One step of the diffusion damps each Fourier mode by exactly the amount the theory says.

    THE PHYSICS. Diffusion damps a wave at a rate set by its wavenumber: the finer the
    structure, the faster it goes. The continuous operator gives 'exp(-K*m**2*dt)'; a
    discretisation gives something else, and WHICH something else is the whole content of a
    numerical scheme. This test states the discrete answer in closed form and checks the port
    against it, so the configuration -- the implicit weight, the diffusion number -- predicts
    the result and the code is held to the prediction.

    It is an exact statement, not an asymptotic one, so the tolerance is rounding and not
    truncation. That is what makes it able to catch a defect of a few per cent, which a
    convergence study at this resolution could not.
    """
    run, profile = _one_step(diffusion_number, weight, backend)

    assert np.all(np.isfinite(run.updated_profile[:, :NUM_LEVELS])), (
        "the solve produced non-finite values; the idealized matrix is singular."
    )
    _assert_the_modes_are_amplified_by_the_predicted_factor(run, profile, diffusion_number, weight)

    # The two structural properties of 'G' that decide whether the scheme is usable at all.
    factors = np.array(
        [_amplification_factor(mode, NUM_LEVELS, diffusion_number, weight) for mode in MODES]
    )
    assert np.all(np.abs(factors) <= 1.0), (
        f"the amplification factor left the unit disc at theta = {weight}, which would make "
        f"the scheme unstable: {factors}."
    )
    # Monotonicity is a strictly stronger property than stability and is NOT implied by
    # 'theta >= 1/2'. It holds exactly when the explicit part cannot overshoot.
    monotone = np.all(factors >= 0.0)
    stiffness = np.array(
        [diffusion_number * 4.0 * np.sin(mode * np.pi / (2.0 * NUM_LEVELS)) ** 2 for mode in MODES]
    )
    predicted_monotone = weight >= 1.0 or np.all((1.0 - weight) * stiffness <= 1.0)
    assert monotone == predicted_monotone, (
        f"the sign of the amplification factor is not what '(1 - theta)*r*s <= 1' predicts at "
        f"theta = {weight}, r = {diffusion_number}: factors {factors}."
    )


def test_the_over_implicit_surface_weight_leaves_a_sixth_of_the_stiffest_mode_standing(
    *, backend: gtx_typing.Backend | None
) -> None:
    """'theta = 1.20' does not annihilate an infinitely stiff mode; it damps it to a sixth.

    THE PHYSICS. A fully implicit step, 'theta = 1', sends 'G -> 0' as the mode gets stiffer:
    whatever the diffusion cannot resolve, it removes. Over-implicitness does not. The limit is
    '(theta - 1)/theta', which is '1/6' at ICON's surface weight, so the scheme retains a
    sixth of every unresolved structure for ever, one step at a time.

    That is a real property of the lowest levels of every ICON run and it is invisible in a
    savepoint comparison, which reproduces it faithfully in both codes. Asserted here at a
    diffusion number of 1e8, where 'r*s' is large enough for the limit to hold to eight digits
    on all six modes.
    """
    diffusion_number = 1.0e8
    weight = 1.20
    run, profile = _one_step(diffusion_number, weight, backend)
    _assert_the_modes_are_amplified_by_the_predicted_factor(
        run, profile, diffusion_number, weight, tolerance=STIFF_LIMIT_TOLERANCE
    )

    limit = (weight - 1.0) / weight
    ratios = run.updated_profile[:, 0] / profile[:, 0]
    np.testing.assert_allclose(
        ratios,
        limit,
        # The modes are not equally deep in the limit: 'r*s' runs from 1e6 for the longest to
        # 4e8 for the shortest, and 'G - (theta-1)/theta' falls off as '1/(r*s)'. The LONGEST
        # mode reaches 4.3e-6 and is what bounds this tolerance; the shortest reaches 1e-9.
        rtol=1.0e-4,
        err_msg=(
            "the stiff limit of the amplification factor is not '(theta-1)/theta'; either the "
            "diffusion number is not large enough to be in the limit, or the implicit weight "
            "does not reach the solver"
        ),
    )
    assert limit > 0.1, (
        "the over-implicit limit came out near zero, so this test is not distinguishing "
        "over-implicit from fully implicit."
    )


@pytest.mark.parametrize("weight", COMPLEMENTABLE_WEIGHTS)
@pytest.mark.parametrize("diffusion_number", DIFFUSION_NUMBERS)
def test_the_complementary_implicit_weight_breaks_the_modes_and_not_the_budget(
    diffusion_number: float, weight: float, *, backend: gtx_typing.Backend | None
) -> None:
    """The amplification test has teeth, and it catches what the conservation test cannot.

    'broken_stencils.compute_implicit_diffusion_momentum_with_the_complementary_weight' splits
    the diffusion momentum at '1 - impl_weight' instead of at 'impl_weight', i.e. it runs the
    scheme at 'theta = 0.25' where the configuration asked for 0.75. Two things are asserted,
    and the second is the reason this file exists:

    - the amplification factor is wrong, by an amount that grows with the diffusion number;
    - the COLUMN BUDGET STILL CLOSES, to the same 1e-12 the conservation test demands. The
      telescoping that makes the scheme conservative is blind to the split, so the whole of
      'test_vertical_diffusion_conservation.py' passes this mutation.

    'theta = 1/2' is excluded from the sweep because Crank-Nicolson is its own complement and
    the mutation is then the identity -- which is itself a fact worth knowing and is why the
    parametrisation names the weights it does.
    """
    correct, profile = _one_step(diffusion_number, weight, backend)
    broken, _ = _one_step(diffusion_number, weight, backend, implicit_split=_COMPLEMENTARY_SPLIT)

    assert np.all(np.isfinite(broken.updated_profile[:, :NUM_LEVELS])), (
        "the broken split produced non-finite values, so the comparison below would be a NaN "
        "rather than a measurement of the defect."
    )
    _assert_the_modes_are_amplified_by_the_predicted_factor(
        correct, profile, diffusion_number, weight
    )

    # The mutation is a theta-scheme too, at the complementary weight, so its error against the
    # requested one is predictable and is asserted to be large rather than merely nonzero.
    departure = np.abs(
        broken.updated_profile[:, :NUM_LEVELS] - correct.updated_profile[:, :NUM_LEVELS]
    ).max()
    assert departure > 1.0e-3, (
        f"the complementary implicit weight changed the answer by only {departure}, which is "
        f"too little for this to be a convincing demonstration."
    )

    # What the conservation test would say about the same run: nothing.
    assert np.all(broken.residual() < 1.0e-12), (
        f"the complementary split broke the column budget as well, worst residual "
        f"{broken.residual().max()}. It is supposed not to -- that is the point of this test -- "
        f"so either the mutation or the budget identity has changed."
    )

    # And the invariant itself, not a restatement of it.
    with pytest.raises(AssertionError, match="not amplified by the theta-scheme factor"):
        _assert_the_modes_are_amplified_by_the_predicted_factor(
            broken, profile, diffusion_number, weight
        )
