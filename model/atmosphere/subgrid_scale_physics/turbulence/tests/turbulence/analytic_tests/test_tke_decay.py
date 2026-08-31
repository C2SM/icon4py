# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Turbulence left to die: the decay of the velocity scale with no forcing at all.

THE PHYSICS. Switch the shear and the buoyancy off and the TKE equation keeps only its
dissipation term, 'dq/dt = -q**2/l_dis' with 'l_dis = d_mom*l'. Turbulence with nothing feeding
it decays, and it decays as a POWER law rather than exponentially, because the dissipation is
quadratic: the continuous solution is 'q(t) = q0/(1 + q0*t/l_dis)'. This is the state the upper
troposphere is in almost everywhere and the state a boundary layer collapses into after sunset.

DO NOT ASSERT THE CONTINUOUS LAW. The scheme does not solve the continuous equation; it takes
backward-Euler steps, which under-decay. Measured on the port at 'l = 100 m', 'dt = 40 s',
'q0 = 1 m/s': the difference from 'q0/(1 + q0*t/l_dis)' is 0.4 per cent after one step and 5.8
per cent after twenty. A test that asserted the continuous law would have to carry a tolerance
of six per cent, which is far larger than any defect worth catching -- so this file asserts the
PER-STEP form and then asserts, as a separate statement, that the continuous law is NOT what
comes out. Both matter: the first is the test, the second is the reason the first is written
the way it is.

TWO THINGS SIT BETWEEN THE EQUATION AND THE STORED VALUE, and leaving either out silently
invalidates the test:

- the time smoothing, 'tke = w1*tvs_0 + w2*tvs_u' with 'w1 = tkesmot = 0.15'
  (mo_turbdiff_config.f90:144, and no MCH namelist overrides it). The stored value is a blend
  of the previous time level and the fresh root, not the root;
- the floors, 'tke = MAX( tkesecu*vel_min, tkesecu*SQRT(l_frc*frc), ... )'. With no forcing the
  second is zero and the first is '1*0.01 = 0.01 m/s', an absolute minimum velocity scale that
  a decaying column reaches and then never leaves. That is asserted too, in a short column
  where 'l_dis' is small enough for the floor to bite within twenty steps.

WHAT THIS FILE ADDS OVER THE STEADY STATE. It is the only test in this directory that looks at
a TRANSIENT, and that is exactly the blind spot 'test_tke_steady_state.py' records: exchanging
the two time-smoothing weights is the identity at a fixed point and is caught only here.

WHAT IT DOES NOT ADD. 'fm2 = fh2 = 0' means 'gh = gm = 0', so the stability functions are at
their zero-forcing values and every coefficient that multiplies 'gh' is unobservable, exactly
as in the neutral tests. This file does nothing for that blind spot.

A NOTE ON THE DISCARDED BRANCH. At zero forcing the modified solution of
'compute_stability_lengths' evaluates '0/0' -- 'fakt = fh2/(val2 - fh2)' with both zero -- and
produces a NaN. GT4Py evaluates both branches of a 'where' unconditionally, so the NaN is
computed and then discarded by 'solvable', which is true here. The results below are exact, so
the discard is complete; the only visible trace is a numpy RuntimeWarning on the embedded
backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from icon4py.model.testing.fixtures.datatest import backend

from . import broken_stencils, utils


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing


#: Master length scales [m] of the decaying columns. 100 m is the case the note behind this
#: file measured; 40 m decays faster and keeps the two columns from sharing a trajectory.
DECAY_LENGTH_SCALES = (100.0, 40.0)

#: Initial velocity scale [m/s]. 'q = SQRT(2*TKE)', so this is TKE = 0.5 m2/s2.
INITIAL_VELOCITY_SCALE = 1.0

#: The TKE time step [s] and the number of steps. 40 s is an MCH-scale physics step.
DECAY_TIME_STEP = 40.0
DECAY_STEPS = 20

#: The shipped time smoothing weight of the previous time level.
TIME_SMOOTHING = 0.15

#: Relative tolerance of the per-step form. Measured: exactly zero over twenty steps and every
#: backend, because the prediction is the same sequence of operations the stencil performs.
DECAY_TOLERANCE = 1.0e-14

#: The minimum relative departure of the scheme from the CONTINUOUS decay law after
#: 'DECAY_STEPS'. Measured 5.8 per cent at 'tkesmot = 0.15' and 0.63 per cent at 'tkesmot = 0';
#: three per cent is a floor under the first and well above the second, so the assertion states
#: that the shipped smoothing makes the departure large and not merely nonzero.
CONTINUOUS_LAW_DEPARTURE = 0.03

#: The mutation of the TKE step, named once because its own name is long.
_REVERSED_SMOOTHING = (
    broken_stencils.compute_turbulent_velocity_scale_with_the_time_smoothing_reversed
)


def _backward_euler_root(velocity_scale: np.ndarray, length_scale: np.ndarray) -> np.ndarray:
    """'tvs_u': the positive root of 'x + dt*x**2/l_dis = q', written as the Fortran writes it."""
    config, _ = utils.closure_constants()
    dissipation_length = config.d_mom * length_scale
    q1 = dissipation_length / DECAY_TIME_STEP
    return 0.5 * q1 * (np.sqrt(1.0 + 4.0 * velocity_scale / q1) - 1.0)


def _predicted_decay(length_scale: np.ndarray, steps: int, *, tkesmot: float) -> np.ndarray:
    """The whole stored sequence: root, time smoothing, floor, repeated.

    Returned as '(steps + 1, num_columns)' so a test can compare every step and not only the
    last -- a defect that cancelled over twenty steps would otherwise pass.
    """
    config, _ = utils.closure_constants()
    history = [np.full(length_scale.shape, INITIAL_VELOCITY_SCALE)]
    for _ in range(steps):
        previous = history[-1]
        root = _backward_euler_root(previous, length_scale)
        smoothed = tkesmot * previous + (1.0 - tkesmot) * root
        history.append(np.maximum(config.tkesecu * config.vel_min, smoothed))
    return np.stack(history, axis=0)


def _run_the_decay(
    backend: gtx_typing.Backend | None,
    *,
    length_scale: np.ndarray,
    steps: int = DECAY_STEPS,
    tkesmot: float = TIME_SMOOTHING,
    velocity_program=None,
) -> np.ndarray:
    """Advance an unforced column and return '(steps + 1, num_columns)' of the stored 'q'."""
    state = utils.construct_idealized_turbulence_state(
        master_length_scale=length_scale,
        shear=np.zeros_like(length_scale),
        velocity_scale=INITIAL_VELOCITY_SCALE,
    )
    programs = {} if velocity_program is None else {"velocity_program": velocity_program}
    row = 2
    history = [state.velocity_scale[:, row].copy()]
    for _ in range(steps):
        state = utils.advance_the_turbulence_state(
            state,
            backend=backend,
            time_step=DECAY_TIME_STEP,
            tkesmot=tkesmot,
            **programs,
        )
        history.append(state.velocity_scale[:, row].copy())
    return np.stack(history, axis=0)


def _assert_the_decay_follows_the_smoothed_backward_euler_step(
    computed: np.ndarray, length_scale: np.ndarray, *, tkesmot: float
) -> None:
    """The invariant itself, factored out so the mutation test can be seen to violate IT.

    Both the passing test and the broken-variant test go through this function, so the
    statement the mutation breaks is literally the statement the port is held to.
    """
    predicted = _predicted_decay(length_scale, computed.shape[0] - 1, tkesmot=tkesmot)
    np.testing.assert_allclose(
        computed,
        predicted,
        rtol=DECAY_TOLERANCE,
        err_msg=(
            "the unforced decay is not the time-smoothed backward-Euler step of "
            "'dq/dt = -q**2/(d_mom*l)'"
        ),
    )


def test_the_unforced_decay_is_the_backward_euler_step_and_not_the_continuous_law(
    *, backend: gtx_typing.Backend | None
) -> None:
    """Turbulence with nothing feeding it decays exactly as the implicit step says, step by step.

    THE PHYSICS. The only term left in the TKE equation is the dissipation, so this measures
    the dissipation closure on its own -- 'epsilon = q**3/(d_mom*l)', with 'd_mom = 16.6' the
    only constant that enters. 'ediss' is not a field this granule writes, so the closure can
    only be seen through what it does to 'q', which is what this is.

    Every one of the twenty steps is compared, not only the last, and the comparison is against
    the closed-form root and not against a numerically integrated ODE. The second assertion is
    the one that stops a later reader from 'simplifying' this test into the continuous law: the
    scheme under-decays by 5.8 per cent over the same twenty steps, so the two statements are
    not interchangeable at any tolerance worth having.
    """
    length_scale = np.array(DECAY_LENGTH_SCALES)
    computed = _run_the_decay(backend, length_scale=length_scale)

    config, _ = utils.closure_constants()
    assert np.all(computed > 1.5 * config.tkesecu * config.vel_min), (
        "the decay reached the minimum velocity scale, so what is compared below is a floor "
        "and not the backward-Euler step."
    )
    assert np.all(np.diff(computed, axis=0) < 0.0), (
        "the unforced velocity scale did not decrease monotonically, which the dissipation "
        "term alone cannot produce."
    )
    _assert_the_decay_follows_the_smoothed_backward_euler_step(
        computed, length_scale, tkesmot=TIME_SMOOTHING
    )

    # The root really is the root: 'x + dt*x**2/l_dis = q' to rounding, which is the statement
    # that the step is backward Euler rather than some other implicit form that happens to fit.
    unsmoothed = _run_the_decay(backend, length_scale=length_scale, tkesmot=0.0)
    dissipation_length = config.d_mom * length_scale
    residual = (
        unsmoothed[1:]
        + DECAY_TIME_STEP * unsmoothed[1:] * unsmoothed[1:] / dissipation_length
        - unsmoothed[:-1]
    )
    assert np.abs(residual).max() < 1.0e-14, (
        f"the unforced step does not solve 'x + dt*x**2/l_dis = q': worst residual "
        f"{np.abs(residual).max()}."
    )

    # And the continuous law, which is what a reader would reach for and must not.
    elapsed = DECAY_TIME_STEP * DECAY_STEPS
    continuous = INITIAL_VELOCITY_SCALE / (
        1.0 + INITIAL_VELOCITY_SCALE * elapsed / dissipation_length
    )
    departure = np.abs(computed[-1] / continuous - 1.0)
    assert np.all(departure > CONTINUOUS_LAW_DEPARTURE), (
        f"the scheme is within {departure.max()} of the continuous decay law after "
        f"{DECAY_STEPS} steps. Either the time step has been shortened until the two agree -- "
        f"in which case this test no longer distinguishes them -- or the smoothing is off."
    )
    assert np.all(computed[-1] > continuous), (
        "backward Euler under-decays, so the scheme must stay ABOVE the continuous solution; "
        "it did not."
    )


def test_the_decaying_velocity_scale_stops_at_the_minimum_and_stays_there(
    *, backend: gtx_typing.Backend | None
) -> None:
    """A column that runs out of turbulence lands on 'tkesecu*vel_min' exactly, and stays.

    THE PHYSICS. Nothing in the equation stops 'q' at a positive value -- the power-law decay
    goes to zero -- so the floor is a numerical guard, not a closure statement. It matters
    because 'q' is divided by: the turbulent time scale is 'l/q' and the stability functions
    are built from its square, so a column at 'q = 0' is a division by zero. 0.01 m/s is
    'vel_min' (mo_turbdiff_config.f90:226) and 'tkesecu = 1'.

    Reached here with a 0.1 m length scale, which is 'kappa*z' at a quarter of a metre above
    the ground -- the lowest half level of a real column, and the one that decays fastest.
    """
    config, _ = utils.closure_constants()
    length_scale = np.array([0.1, 0.05])
    computed = _run_the_decay(backend, length_scale=length_scale, steps=30)

    floor = config.tkesecu * config.vel_min
    assert np.all(computed[0] > 10.0 * floor), "the column started at the floor."
    np.testing.assert_array_equal(
        computed[-1],
        np.full_like(computed[-1], floor),
        err_msg="the decaying velocity scale did not stop exactly at 'tkesecu*vel_min'",
    )
    # It is a floor and not an equilibrium: once reached it is not left, and the unfloored
    # sequence would have gone below it.
    unfloored = _predicted_decay(length_scale, 30, tkesmot=TIME_SMOOTHING)
    assert np.all(computed[-5:] == floor), "the velocity scale left the floor again."
    assert np.all(unfloored[-1] == floor), (
        "the prediction did not reach the floor either, so this test is not exercising it."
    )


def test_the_reversed_time_smoothing_is_caught_by_the_transient(
    *, backend: gtx_typing.Backend | None
) -> None:
    """The decay test has teeth, and it catches what no equilibrium test can.

    'broken_stencils.compute_turbulent_velocity_scale_with_the_time_smoothing_reversed'
    exchanges 'w1' and 'w2', so the scheme keeps 0.85 of the previous value where it should
    keep 0.15. Everything about the result still looks right: it is a convex combination of the
    same two numbers, it is positive, it is monotone, it lies between the old value and the
    fresh root, and at any steady state it is EXACTLY the correct answer --
    'test_tke_steady_state.py::test_the_reversed_time_smoothing_is_invisible_at_the_steady_state'
    asserts that it passes the equilibrium invariant untouched.

    In the transient it slows the decay by about a factor of five, which is what this test
    measures. The predicted broken sequence is asserted as well as the failure, so a mutation
    that stopped doing what this test believes would be caught rather than quietly weakening
    the demonstration.
    """
    length_scale = np.array(DECAY_LENGTH_SCALES)
    correct = _run_the_decay(backend, length_scale=length_scale)
    broken = _run_the_decay(
        backend, length_scale=length_scale, velocity_program=_REVERSED_SMOOTHING
    )

    assert np.all(np.isfinite(broken)), "the broken TKE step produced non-finite values."
    _assert_the_decay_follows_the_smoothed_backward_euler_step(
        correct, length_scale, tkesmot=TIME_SMOOTHING
    )

    # The mutation is the same recursion with the weights exchanged, so its own sequence is
    # predictable and is asserted before the failure is.
    exchanged = _predicted_decay(length_scale, DECAY_STEPS, tkesmot=1.0 - TIME_SMOOTHING)
    np.testing.assert_allclose(
        broken,
        exchanged,
        # Two orders looser than 'DECAY_TOLERANCE', and for a reason that is not slack: the
        # stencil forms '1 - tkesmot' from the SHIPPED 0.15, which is exactly 0.85, while this
        # prediction forms it from 0.85, which is not exactly 0.15. The two recursions differ
        # in the last bit of one coefficient and drift apart at 1e-16 per step.
        rtol=1.0e-12,
        err_msg=(
            "the mutation is not the weight exchange this test assumes, so the invariant it is "
            "supposed to break is untested"
        ),
    )
    decayed_correct = 1.0 - correct[-1] / correct[0]
    decayed_broken = 1.0 - broken[-1] / broken[0]
    assert np.all(decayed_broken < 0.35 * decayed_correct), (
        f"the reversed smoothing decayed by {decayed_broken} against {decayed_correct}, which "
        f"is too close for this to be a convincing demonstration."
    )

    # And the invariant itself, not a restatement of it.
    with pytest.raises(AssertionError, match="not the time-smoothed backward-Euler step"):
        _assert_the_decay_follows_the_smoothed_backward_euler_step(
            broken, length_scale, tkesmot=TIME_SMOOTHING
        )
