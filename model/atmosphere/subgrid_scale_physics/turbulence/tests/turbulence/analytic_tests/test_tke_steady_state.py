# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The TKE budget at equilibrium: production balances dissipation, neutral and stably stratified.

THE EQUATION THE SCHEME SOLVES. 'turb_utilities.f90:1494-1501' looks like an algebraic closed
form and is not; it is the positive root of 'x + dt*x**2/l_dis = q + dt*frc', i.e. exactly
BACKWARD EULER on

    dq/dt = frc - q**2/l_dis,    l_dis = d_mom*l

Multiplying by 'q' turns that into the TKE budget itself, 'd(TKE)/dt = q*frc - q**3/(d_mom*l)',
so the dissipation closure of the scheme is 'epsilon = q**3/(d_mom*l)' -- Mellor and Yamada's
'q**3/(B1*l)' with 'B1 = d_mom = 16.6'. The first test below asserts the backward-Euler
residual directly, which is the strongest statement about the time integration that can be made
without an oracle.

'ediss' IS NOT PORTED. 'TurbulenceDiagnosticState.edr' is a slot nothing writes, so the
dissipation cannot be compared against an output field. Everything about it therefore has to go
through the 'q' update, which is the only place 'd_mom' enters as a dissipation length -- and
that is what the tests here do.

THE NEUTRAL EQUILIBRIUM IS A CLOSED FORM. Setting 'dq/dt = 0' gives 'q**2 = d_mom*l*frc', and
at 'fh2 = 0' the self-consistency with the stability functions closes analytically:

    gm = sm_0**2 = d_mom**(-2/3),  S_m = sm_0,  q = d_mom**(1/3)*l*|S|,  K_m = l**2*|S|

so the whole equilibrium is a pure number times 'l*|S|'.

THE STABLE EQUILIBRIUM IS ALSO A CLOSED FORM, AND IT IS THE POINT OF THIS FILE. At a fixed
gradient Richardson number 'Ri = fh2/fm2' the equilibrium condition 'S_m*gm - S_h*gh = 1/d_mom'
together with Cramer's rule is a QUADRATIC in 'gm', because both sides are quadratic in it:

    (d_mom*Q - C)*gm**2 + (d_mom*P - B)*gm - A = 0

with 'A', 'B', 'C' the coefficients of 'det(gm)' and 'P', 'Q' those of
'num_m - Ri*num_h' -- all written out in '_stratified_equilibrium' below. Every one of them
contains 'gh', hence 'a21 = (d_6-d_4)*gh', the '(d_5-d_4)' of 'a11' and the 'd_3' of 'a22'.

THAT IS WHAT CLOSES THE BLIND SPOT the first batch of analytic tests recorded and could not
cover. 'test_neutral_stability_functions.py' demonstrates, with a test of its own, that at
'Ri = 0' those three coefficients are unobservable; nothing in the neutral surface layer or in
the pure decay changes that, because both are at 'gh = 0' too. The stable equilibrium here is
the only test in this directory that constrains them, and
'broken_stencils.compute_stability_lengths_with_the_buoyancy_cofactor_negated' -- which is
EXACTLY the identity at 'Ri = 0' -- is what demonstrates that it does.

THE CLIP MUST STAY OUT OF THE WAY. 'frc' is floored at 'frcsecu*rim*l*S_m*fm2', which enforces
a flux Richardson number no larger than 'Rf_c = 1 - rim = 0.19123'. The closed form above
assumes the unclipped forcing, so every stratified case below asserts 'Rf < Rf_c' before
believing its own prediction.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import pytest

from icon4py.model.testing.fixtures.datatest import backend

from . import broken_stencils, utils


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing


#: Master length scales [m] and shears [1/s], one pair per column. Three unrelated pairs, so a
#: result that depended on 'l' and '|S|' separately rather than on their product would show.
LENGTH_SCALES = (50.0, 100.0, 25.0)
SHEARS = (0.02, 0.01, 0.05)

#: Gradient Richardson numbers of the stratified equilibrium. Both are comfortably below the
#: flux-Richardson clip, which the test asserts rather than assumes.
RICHARDSON_NUMBERS = (0.05, 0.10)

#: 600 s is long against the eddy turnover time 'd_mom**(2/3)/|S|' of every column here, which
#: is what makes the iteration contract quickly. The convergence is asserted, not assumed: the
#: loop below runs until the velocity scale stops moving and fails if it has not by then.
FIXED_POINT_TIME_STEP = 600.0
MAXIMUM_ITERATIONS = 1500
CONVERGENCE_INCREMENT = 1.0e-14

#: How many iterations a MUTATED loop is given. It is a fixed count and not a convergence
#: criterion: a defective dissipation length changes the equilibrium by three orders of
#: magnitude AND makes the map contract far more slowly, so a mutated loop does not settle in
#: any budget this file would care to spend. Two hundred is far more than enough to leave the
#: correct answer behind, which is all a mutation test needs.
MUTATED_ITERATIONS = 200

#: Relative tolerance of the converged fixed point. Measured worst case 1.2e-13 over the three
#: columns and both stratifications.
EQUILIBRIUM_TOLERANCE = 1.0e-11

#: The two mutations of the TKE loop, named once because their own names are long.
_INVERTED_TIME_SCALE = (
    broken_stencils.compute_turbulent_velocity_scale_with_the_dissipation_time_scale_inverted
)
_REVERSED_SMOOTHING = (
    broken_stencils.compute_turbulent_velocity_scale_with_the_time_smoothing_reversed
)
_NEGATED_BUOYANCY_COFACTOR = (
    broken_stencils.compute_stability_lengths_with_the_buoyancy_cofactor_negated
)


def _iterate(
    state: utils.TurbulenceState,
    backend: gtx_typing.Backend | None,
    *,
    tkesmot: float = 0.15,
    require_convergence: bool = True,
    **programs,
) -> utils.TurbulenceState:
    """Run the scheme's own two programs to their fixed point, and prove that it is one.

    'require_convergence' is switched off for the mutated loops; see 'MUTATED_ITERATIONS'.
    """
    if not require_convergence:
        for _ in range(MUTATED_ITERATIONS):
            state = utils.advance_the_turbulence_state(
                state,
                backend=backend,
                time_step=FIXED_POINT_TIME_STEP,
                tkesmot=tkesmot,
                **programs,
            )
        return state

    for _ in range(MAXIMUM_ITERATIONS):
        previous = state.velocity_scale
        state = utils.advance_the_turbulence_state(
            state, backend=backend, time_step=FIXED_POINT_TIME_STEP, tkesmot=tkesmot, **programs
        )
        rows = state.interior
        increment = np.abs(state.velocity_scale[:, rows] - previous[:, rows]).max()
        if increment <= CONVERGENCE_INCREMENT * np.abs(state.velocity_scale[:, rows]).max():
            return state
    raise AssertionError(
        f"the TKE loop had not converged after {MAXIMUM_ITERATIONS} iterations: last increment "
        f"{increment}."
    )


def _stratified_equilibrium(richardson: float) -> dict[str, float]:
    """The closed-form stable equilibrium at a fixed gradient Richardson number.

    Solves the quadratic of the module docstring for 'gm', then evaluates Cramer's rule at that
    point. Returns the dimensionless forcings, the two stability functions and the flux
    Richardson number, all of which the caller turns into 'q' and the diffusivities by
    multiplying by 'l' and '|S|'.
    """
    config, params = utils.closure_constants()
    d_1, d_2, d_3, d_4, d_5, d_6 = (
        params.d_1,
        params.d_2,
        params.d_3,
        params.d_4,
        params.d_5,
        params.d_6,
    )
    b_h, b_m, d_m = params.b_h, params.b_m, config.d_mom

    # 'det(gm) = A + B*gm + C*gm**2' at 'gh = Ri*gm'.
    a_coefficient = d_1 * d_2
    b_coefficient = d_1 * (d_3 * richardson + d_4) + (d_5 - d_4) * richardson * d_2
    c_coefficient = (d_5 - d_4) * richardson * (d_3 * richardson + d_4) - d_4 * (
        d_6 - d_4
    ) * richardson
    # 'num_m - Ri*num_h = P + Q*gm'.
    p_coefficient = b_m * d_1 - richardson * b_h * d_2
    q_coefficient = richardson * (
        b_m * (d_5 - d_4) - b_h * (d_6 - d_4) - b_h * d_3 * richardson - (b_h - b_m) * d_4
    )

    quadratic = d_m * q_coefficient - c_coefficient
    linear = d_m * p_coefficient - b_coefficient
    constant = -a_coefficient
    discriminant = linear * linear - 4.0 * quadratic * constant
    assert discriminant > 0.0, f"no real TKE equilibrium at Ri = {richardson}."
    roots = [(-linear + sign * np.sqrt(discriminant)) / (2.0 * quadratic) for sign in (1.0, -1.0)]
    positive = [root for root in roots if root > 0.0]
    assert len(positive) == 1, (
        f"the equilibrium at Ri = {richardson} is not unique among the positive shears: {roots}."
    )
    shear = float(positive[0])
    buoyancy = richardson * shear

    a11 = d_1 + (d_5 - d_4) * buoyancy
    a12 = d_4 * shear
    a21 = (d_6 - d_4) * buoyancy
    a22 = d_2 + d_3 * buoyancy + d_4 * shear
    determinant = a11 * a22 - a12 * a21
    stability_h = (b_h * a22 - b_m * a12) / determinant
    stability_m = (b_m * a11 - b_h * a21) / determinant
    return {
        "gm": shear,
        "gh": buoyancy,
        "sm": stability_m,
        "sh": stability_h,
        "flux_richardson": stability_h * buoyancy / (stability_m * shear),
    }


def _assert_the_neutral_equilibrium(state: utils.TurbulenceState) -> None:
    """The invariant itself, factored out so the mutation test can be seen to violate IT.

    'q = d_mom**(1/3)*l*|S|', 'S_m = sm_0' and 'K_m = l**2*|S|' at neutral TKE equilibrium.
    """
    config, params = utils.closure_constants()
    rows = state.interior
    length = state.master_length_scale[:, rows]
    shear = np.sqrt(state.mechanical_forcing[:, rows])

    np.testing.assert_allclose(
        state.velocity_scale[:, rows],
        config.d_mom ** (1.0 / 3.0) * length * shear,
        rtol=EQUILIBRIUM_TOLERANCE,
        err_msg="the neutral equilibrium velocity scale is not 'd_mom**(1/3)*l*|S|'",
    )
    np.testing.assert_allclose(
        state.stability_function_for_momentum()[:, rows],
        params.sm_0,
        rtol=EQUILIBRIUM_TOLERANCE,
        err_msg="the neutral equilibrium momentum stability function is not 'sm_0'",
    )
    np.testing.assert_allclose(
        state.diffusion_coefficient_for_momentum()[:, rows],
        length * length * shear,
        rtol=EQUILIBRIUM_TOLERANCE,
        err_msg="the neutral equilibrium diffusivity is not the mixing-length 'l**2*|S|'",
    )


def _assert_the_stratified_equilibrium(state: utils.TurbulenceState, richardson: float) -> None:
    """The invariant itself for a stably stratified column, and the one that sees 'gh'.

    Compared against the closed form of '_stratified_equilibrium', which is the solution of the
    scheme's own equilibrium condition and not a rearrangement of the stencil.
    """
    expected = _stratified_equilibrium(richardson)
    rows = state.interior
    length = state.master_length_scale[:, rows]
    shear = np.sqrt(state.mechanical_forcing[:, rows])

    np.testing.assert_allclose(
        state.velocity_scale[:, rows],
        length * shear / np.sqrt(expected["gm"]),
        rtol=EQUILIBRIUM_TOLERANCE,
        err_msg="the stratified equilibrium velocity scale is not 'l*|S|/SQRT(gm_eq)'",
    )
    np.testing.assert_allclose(
        state.stability_function_for_momentum()[:, rows],
        expected["sm"],
        rtol=EQUILIBRIUM_TOLERANCE,
        err_msg="the stratified equilibrium momentum stability function is not the closed form",
    )
    np.testing.assert_allclose(
        state.stability_length_for_scalars[:, rows] / length,
        expected["sh"],
        rtol=EQUILIBRIUM_TOLERANCE,
        err_msg="the stratified equilibrium scalar stability function is not the closed form",
    )


def _assert_the_backward_euler_residual(
    before: utils.TurbulenceState, after: utils.TurbulenceState, time_step: float
) -> None:
    """The invariant itself for one step: 'q_new + dt*q_new**2/l_dis - q_old - dt*frc = 0'."""
    config, _ = utils.closure_constants()
    rows = after.interior
    dissipation_length = config.d_mom * after.master_length_scale[:, rows]
    new = after.velocity_scale[:, rows]
    old = before.velocity_scale[:, rows]
    forcing = after.tke_forcing()[:, rows]
    residual = new + time_step * new * new / dissipation_length - old - time_step * forcing
    worst = np.abs(residual).max() / np.abs(new).max()
    assert worst < 1.0e-13, (
        f"the TKE step is not backward Euler on 'dq/dt = frc - q**2/(d_mom*l)': worst scaled "
        f"residual {worst}."
    )


def test_the_tke_step_is_backward_euler_on_production_minus_dissipation(
    *, backend: gtx_typing.Backend | None
) -> None:
    """One step of the TKE equation solves the implicit equation it is supposed to solve.

    THE PHYSICS. The prognostic TKE equation of a level-2.5 closure is a balance between shear
    production, buoyancy destruction and viscous dissipation, and the dissipation term is what
    makes it stiff: it is quadratic in 'q' and its time scale near the ground is short against
    any model time step. Treating it implicitly is what lets the scheme use a physical time
    step at all, and the closed form in the Fortran is that implicit solution written out.

    Asserted away from equilibrium in both directions -- one column starting below its
    equilibrium and one above -- so that the residual is a statement about the step and not
    about a fixed point where both sides happen to vanish. The time smoothing is switched off,
    because it is applied AFTER the root and would otherwise be part of what is measured; the
    shipped 'tkesmot = 0.15' is exercised by 'test_tke_decay.py'.
    """
    config, _ = utils.closure_constants()
    length = np.array([50.0, 50.0])
    shear = np.array([0.02, 0.02])
    equilibrium = config.d_mom ** (1.0 / 3.0) * length * shear
    state = utils.construct_idealized_turbulence_state(master_length_scale=length, shear=shear)
    state = dataclasses.replace(
        state,
        velocity_scale=np.tile(
            (equilibrium * np.array([0.8, 1.5]))[:, np.newaxis],
            (1, state.master_length_scale.shape[1]),
        ),
    )

    time_step = 120.0
    advanced = utils.advance_the_turbulence_state(
        state, backend=backend, time_step=time_step, tkesmot=0.0
    )

    rows = advanced.interior
    # The two floors of the update must be out of the way, or what comes out is a floor and not
    # the root: 'tkesecu*vel_min' and 'tkesecu*SQRT(l_frc*frc)' with 'l_frc = l*d_4/b_m'.
    _, params = utils.closure_constants()
    forcing_length = advanced.master_length_scale[:, rows] * params.d_4 / params.b_m
    floor = np.sqrt(forcing_length * np.maximum(advanced.tke_forcing()[:, rows], 0.0))
    assert np.all(advanced.velocity_scale[:, rows] > floor * (1.0 + 1.0e-9)), (
        "the equilibrium floor is binding, so the stored value is not the backward-Euler root."
    )
    assert np.all(advanced.velocity_scale[:, rows] > config.tkesecu * config.vel_min), (
        "the minimum velocity scale is binding, so the stored value is not the root."
    )
    assert np.abs(advanced.velocity_scale[:, rows] - state.velocity_scale[:, rows]).max() > 0.05, (
        "the step barely moved 'q', so the residual below would be satisfied by doing nothing."
    )

    _assert_the_backward_euler_residual(state, advanced, time_step)


def test_the_neutral_steady_state_is_the_mixing_length_closure(
    *, backend: gtx_typing.Backend | None
) -> None:
    """Shear production balancing dissipation gives 'q = d_mom**(1/3)*l*|S|' and 'K_m = l**2*|S|'.

    THE PHYSICS. Left alone with a steady shear and no stratification, a level-2.5 closure
    settles where the turbulence it produces is exactly what it dissipates. That state is the
    Mellor-Yamada level-2 closure -- the stability functions become pure numbers -- and the
    diffusivity it gives is Prandtl's mixing-length result, with the closure constants
    cancelling out of it entirely.

    The dissipation is checked in the only way the port allows, through the budget: 'ediss' is
    not a field this granule writes, so 'epsilon = q**3/(d_mom*l)' is asserted to equal the
    shear production 'K_m*|S|**2' computed from the two fields that ARE written.
    """
    state = utils.construct_idealized_turbulence_state(
        master_length_scale=np.array(LENGTH_SCALES), shear=np.array(SHEARS)
    )
    converged = _iterate(state, backend)
    _assert_the_neutral_equilibrium(converged)

    config, _ = utils.closure_constants()
    rows = converged.interior
    length = converged.master_length_scale[:, rows]
    shear = np.sqrt(converged.mechanical_forcing[:, rows])
    velocity = converged.velocity_scale[:, rows]
    dissipation = velocity * velocity * velocity / (config.d_mom * length)
    production = converged.diffusion_coefficient_for_momentum()[:, rows] * shear * shear
    np.testing.assert_allclose(
        dissipation,
        production,
        rtol=EQUILIBRIUM_TOLERANCE,
        err_msg="at steady state the dissipation 'q**3/(d_mom*l)' does not equal 'K_m*|S|**2'",
    )
    np.testing.assert_allclose(
        dissipation,
        length * length * shear * shear * shear,
        rtol=EQUILIBRIUM_TOLERANCE,
        err_msg="the neutral equilibrium dissipation is not 'l**2*|S|**3'",
    )


@pytest.mark.parametrize("richardson", RICHARDSON_NUMBERS)
def test_the_stable_steady_state_matches_the_closed_form_of_the_stratified_equilibrium(
    richardson: float, *, backend: gtx_typing.Backend | None
) -> None:
    """With buoyancy taking energy out, the equilibrium is still closed form -- and it sees 'gh'.

    THE PHYSICS. Stable stratification consumes turbulent energy: part of what the shear
    produces goes into lifting dense air rather than into the eddies, so the equilibrium 'q',
    the diffusivities and the turbulent Prandtl number all move as the Richardson number rises.
    That is the regime a nocturnal boundary layer spends the whole night in, and it is the one
    the neutral tests of this directory say nothing about.

    Every coefficient of the 2x2 closure that multiplies 'gh' enters here, so this is the test
    that constrains 'a21', the '(d_5 - d_4)' of 'a11' and the 'd_3' of 'a22'. The prediction is
    the positive root of a quadratic in 'gm' and is evaluated in '_stratified_equilibrium';
    nothing about it is read off the stencil.
    """
    length = np.array(LENGTH_SCALES)
    shear = np.array(SHEARS)
    state = utils.construct_idealized_turbulence_state(
        master_length_scale=length,
        shear=shear,
        buoyancy_frequency_squared=richardson * shear * shear,
    )
    converged = _iterate(state, backend)

    expected = _stratified_equilibrium(richardson)
    _, params = utils.closure_constants()
    critical = 1.0 - params.rim
    assert expected["flux_richardson"] < critical, (
        f"the flux Richardson number of the predicted equilibrium, "
        f"{expected['flux_richardson']}, is at or above the clip at {critical}, so the closed "
        f"form does not describe what the scheme computes."
    )
    assert expected["gh"] > 0.0, "the case is not stratified, so it cannot see 'gh'."

    _assert_the_stratified_equilibrium(converged, richardson)

    # The equilibrium condition itself, from the stencil's own output: 'S_m*gm - S_h*gh' is
    # '1/d_mom'. This is what makes the state a TKE equilibrium rather than merely a fixed
    # point of the iteration.
    config, _ = utils.closure_constants()
    rows = converged.interior
    time_scale_squared = (
        converged.master_length_scale[:, rows] / converged.velocity_scale[:, rows]
    ) ** 2
    stability_m = converged.stability_function_for_momentum()[:, rows]
    stability_h = (
        converged.stability_length_for_scalars[:, rows] / (converged.master_length_scale[:, rows])
    )
    deviation = (
        stability_m * converged.mechanical_forcing[:, rows]
        - stability_h * converged.thermal_forcing[:, rows]
    ) * time_scale_squared
    np.testing.assert_allclose(
        deviation,
        1.0 / config.d_mom,
        rtol=EQUILIBRIUM_TOLERANCE,
        err_msg="'S_m*gm - S_h*gh' is not '1/d_mom', so this is not a TKE equilibrium",
    )


def test_an_inverted_dissipation_time_scale_breaks_the_neutral_steady_state(
    *, backend: gtx_typing.Backend | None
) -> None:
    """The steady-state test has teeth: 'l_dis*dt' for 'l_dis/dt' misses by two orders.

    'broken_stencils.compute_turbulent_velocity_scale_with_the_dissipation_time_scale_inverted'
    picks the time step where the Fortran uses its reciprocal, in the one place where the units
    do not object. The dissipation is then weaker by 'dt**2' and the equilibrium moves from
    'd_mom**(1/3)*l*|S|' to about 'SQRT(d_mom*S_m)*dt*l*|S|'.

    The same mutation is caught by the single-step backward-Euler residual, which is asserted
    here as well: a defect in 'q1' is a defect in the equation, not only in where it settles.
    """
    state = utils.construct_idealized_turbulence_state(
        master_length_scale=np.array(LENGTH_SCALES), shear=np.array(SHEARS)
    )
    broken = _iterate(
        state,
        backend,
        require_convergence=False,
        velocity_program=_INVERTED_TIME_SCALE,
    )
    config, _ = utils.closure_constants()
    rows = broken.interior
    correct = (
        config.d_mom ** (1.0 / 3.0)
        * broken.master_length_scale[:, rows]
        * np.sqrt(broken.mechanical_forcing[:, rows])
    )
    assert np.all(np.isfinite(broken.velocity_scale[:, rows])), (
        "the broken TKE step produced non-finite values, so the comparison below would be a "
        "NaN rather than a measurement of the defect."
    )
    assert np.all(broken.velocity_scale[:, rows] > 10.0 * correct), (
        f"the mutation moved the equilibrium velocity scale by a factor of only "
        f"{(broken.velocity_scale[:, rows] / correct).min()}, which is too little for this to "
        f"be a convincing demonstration."
    )

    # And the invariants themselves, not restatements of them.
    with pytest.raises(AssertionError, match="velocity scale is not 'd_mom"):
        _assert_the_neutral_equilibrium(broken)

    single = utils.advance_the_turbulence_state(
        state,
        backend=backend,
        time_step=120.0,
        tkesmot=0.0,
        velocity_program=_INVERTED_TIME_SCALE,
    )
    with pytest.raises(AssertionError, match="not backward Euler"):
        _assert_the_backward_euler_residual(state, single, 120.0)


@pytest.mark.parametrize("richardson", RICHARDSON_NUMBERS)
def test_a_negated_buoyancy_cofactor_is_caught_only_by_the_stratified_equilibrium(
    richardson: float, *, backend: gtx_typing.Backend | None
) -> None:
    """The blind spot of every neutral test, closed: the mutation that 'Ri = 0' cannot see.

    'broken_stencils.compute_stability_lengths_with_the_buoyancy_cofactor_negated' flips the
    sign of 'a21 = (d_6 - d_4)*gh'. Because 'a21' multiplies 'gh' and nothing else, it is
    EXACTLY the identity at neutral stratification -- which
    'test_neutral_stability_functions.py::test_the_neutral_limit_is_blind_to_the_buoyancy_cofactor'
    demonstrates on the algebra, and which is asserted here on the running stencil: the neutral
    equilibrium of the mutated loop passes '_assert_the_neutral_equilibrium' unchanged.

    Under stratification it does not. That asymmetry is the whole content of this test: it is
    the evidence that the stable equilibrium constrains something no neutral state does, and
    therefore that the blind spot recorded by the first batch of analytic tests is now closed.
    """
    neutral = utils.construct_idealized_turbulence_state(
        master_length_scale=np.array(LENGTH_SCALES), shear=np.array(SHEARS)
    )
    neutral_broken = _iterate(neutral, backend, stability_program=_NEGATED_BUOYANCY_COFACTOR)
    # The mutation is invisible at 'Ri = 0'. Not "small": the neutral invariant holds outright.
    _assert_the_neutral_equilibrium(neutral_broken)

    length = np.array(LENGTH_SCALES)
    shear = np.array(SHEARS)
    stratified = utils.construct_idealized_turbulence_state(
        master_length_scale=length,
        shear=shear,
        buoyancy_frequency_squared=richardson * shear * shear,
    )
    stratified_broken = _iterate(
        stratified,
        backend,
        require_convergence=False,
        stability_program=_NEGATED_BUOYANCY_COFACTOR,
    )
    rows = stratified_broken.interior
    assert np.all(np.isfinite(stratified_broken.velocity_scale[:, rows])), (
        "the broken stability functions produced non-finite values under stratification."
    )

    # And the invariant itself, not a restatement of it.
    with pytest.raises(AssertionError, match="stratified equilibrium"):
        _assert_the_stratified_equilibrium(stratified_broken, richardson)


def test_the_reversed_time_smoothing_is_invisible_at_the_steady_state(
    *, backend: gtx_typing.Backend | None
) -> None:
    """The limit of this file's claim, demonstrated rather than written down.

    'broken_stencils.compute_turbulent_velocity_scale_with_the_time_smoothing_reversed'
    exchanges the two weights of 'w1*tvs_0 + w2*tvs_u'. At a fixed point the two values being
    combined are the same number, so the mutation is EXACTLY the identity there and every
    assertion in this file passes it -- including the backward-Euler residual, which is
    evaluated at 'tkesmot = 0' where the smoothing does not exist at all.

    It is caught by 'test_tke_decay.py', which is the only test in this directory that looks at
    a transient. Recorded here so that the blind spot is on the record next to the tests that
    have it, rather than being rediscovered.
    """
    state = utils.construct_idealized_turbulence_state(
        master_length_scale=np.array(LENGTH_SCALES), shear=np.array(SHEARS)
    )
    # The loop takes about three times as many iterations to settle as the correct one, which
    # is itself the defect: exchanging the weights leaves 0.85 of the OLD value in every step,
    # so the map contracts at 0.94 per iteration instead of 0.44. It settles on the SAME
    # answer, which is the point.
    broken = _iterate(state, backend, velocity_program=_REVERSED_SMOOTHING)
    _assert_the_neutral_equilibrium(broken)
