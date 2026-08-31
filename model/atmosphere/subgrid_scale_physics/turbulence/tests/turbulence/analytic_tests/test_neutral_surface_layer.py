# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The neutral surface layer: an identity the closure constants cannot touch, and a value they fix.

THE IDENTITY. In a neutral surface layer the mixing length is 'L = kappa*z', the wind follows
the log law so the shear is 'S = u_star/(kappa*z)', and the turbulence is in local equilibrium:
production balances dissipation. Then

    u_star**2 = K_m*S = (q*S_m*L)*S   and   q*S_m*L*S = q*S_m*u_star
    ==>  q*S_m = u_star  ==>  K_m = q*S_m*L = kappa*u_star*z

and NOTHING about the closure survives into that answer -- 'S_m', 'd_mom', 'a_mom' all cancel.
The scheme is obliged to reproduce 'K_m = kappa*u_star*z' whatever its constants are, and by
the same cancellation 'K_m = L**2*|S|', Prandtl's 1925 mixing-length result. Those two are
tested first, and they would still hold if every constant in 'turb_setup' were changed.

THE VALUE. What the constants do fix is how much turbulent energy the layer carries:

    q = u_star/sm_0 = d_mom**(1/3)*u_star = 2.5509544*u_star
    TKE = q**2/2 = d_mom**(2/3)*u_star**2/2 = 3.2536841850*u_star**2

which is the classic Mellor-Yamada neutral surface-layer value. Asserting the identity and the
value SEPARATELY is deliberate: they fail differently. A defect in the stability functions
breaks both; a change of 'd_mom' -- a legitimate retuning -- breaks only the second, and a
reviewer needs to be able to tell those apart from the failure alone.

THE MIXING LENGTH IS NOT THE TEXTBOOK BLACKADAR FORM, which is the first test below.
'turb_diffusion.f90:1139' is

    len_scale = akt*MAX( len_min, l_scal*len_scale/(l_scal + len_scale) )

with 'len_scale' entering as 'z + z0'. Written out, 'L(z) = kappa*lambda*(z+z0)/(lambda+z+z0)':
von Karman's constant multiplies the WHOLE harmonic blend, so the free-atmosphere asymptote is
'kappa*lambda' and not 'lambda'. With 'tur_len = 500 m' that is 200 m, and with the MCH setting
'tur_len = 300 m' it is 120 m. The textbook 'l = kappa*z/(1 + kappa*z/l_inf)' has the same
surface limit and a different asymptote, so the two are indistinguishable near the ground and
differ by '1/kappa' aloft.

WHAT THIS FILE DOES NOT CLOSE. Everything here is at 'fh2 = 0'. The blind spot recorded in
'test_neutral_stability_functions.py' -- every coefficient that only multiplies 'gh' -- is
therefore still open after this file, exactly as it was before it. It is closed by the stable
equilibrium of 'test_tke_steady_state.py' and by nothing else in this directory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_turbulent_length_scale import (
    compute_turbulent_length_scale,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.thermodynamic_functions import (
    ThermoConstants,
)
from icon4py.model.testing.fixtures.datatest import backend

from . import broken_stencils, utils


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing


#: Friction velocity of the idealized surface layer [m/s]. 0.3 m/s is a brisk but ordinary
#: neutral boundary layer.
FRICTION_VELOCITY = 0.3

#: Heights above the surface [m], one per column. All far below 'kappa*tur_len', so the
#: harmonic blend is inactive and 'L = kappa*z' holds to better than a part in 1e3.
SURFACE_LAYER_HEIGHTS = (2.0, 5.0, 10.0, 20.0)

#: The time step of the fixed-point iteration and the most iterations it may take. 600 s is
#: long against the eddy turnover time 'd_mom**(2/3)/|S|' of every column here, which is what
#: makes the map contract quickly; the loop stops when it stops moving and fails if it has not.
FIXED_POINT_TIME_STEP = 600.0
MAXIMUM_ITERATIONS = 1500

#: The increment, relative to the velocity scale itself, below which the loop is called
#: converged. The map contracts at about 0.5 per iteration here, so the remaining distance to
#: the true fixed point is of the same order -- three orders below the tolerance the invariants
#: are asserted at.
CONVERGENCE_INCREMENT = 1.0e-14

#: The neutral turbulent energy level of the scheme, 'TKE/u_star**2 = d_mom**(2/3)/2', to the
#: ten digits the literature note quotes. It is NOT the number the assertion below compares
#: against -- ten digits is 1e-11 of the value and the fixed point is reached to 1e-15, so the
#: LITERAL would be what failed. The assertion uses 'd_mom**(2/3)/2' and this constant is
#: checked against that, which is the only way round that does not silently loosen the test.
NEUTRAL_TKE_LEVEL = 3.2536841850

#: The mutation of the stability functions, named once because its own name is long.
_CONFUSED_FORCING = broken_stencils.compute_stability_lengths_with_the_shear_taken_from_the_buoyancy

#: Relative tolerance of the converged fixed point. Measured worst case 2e-15 over the four
#: columns; the mutation below moves the answer by a factor of two.
FIXED_POINT_TOLERANCE = 1.0e-12


def _iterate_to_the_fixed_point(
    state: utils.TurbulenceState,
    backend: gtx_typing.Backend | None,
    *,
    tkesmot: float,
    velocity_program=None,
    stability_program=None,
) -> utils.TurbulenceState:
    """Run the TKE loop until it stops moving, and prove that it did.

    The convergence is checked, not assumed: the last increment of the velocity scale is
    asserted to be at the level of rounding, so a test that failed because the iteration had
    not finished would say so instead of blaming the closure.
    """
    programs = {}
    if velocity_program is not None:
        programs["velocity_program"] = velocity_program
    if stability_program is not None:
        programs["stability_program"] = stability_program

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


def _surface_layer_state() -> tuple[np.ndarray, utils.TurbulenceState]:
    """A column set per level to the neutral surface layer at its own height.

    Each cell is one height: 'L = kappa*z' and 'S = u_star/(kappa*z)', so 'L*S = u_star' is the
    same in every column and 'K_m' is not.
    """
    config, _ = utils.closure_constants()
    heights = np.array(SURFACE_LAYER_HEIGHTS)
    length_scale = config.akt * heights
    shear = FRICTION_VELOCITY / length_scale
    return heights, utils.construct_idealized_turbulence_state(
        master_length_scale=length_scale, shear=shear
    )


def _assert_the_log_law_diffusivity(state: utils.TurbulenceState, heights: np.ndarray) -> None:
    """The invariant itself, factored out so the mutation test can be seen to violate IT.

    'K_m = kappa*u_star*z', which holds for ANY closure constants, and the equivalent
    mixing-length form 'K_m = L**2*|S|'.
    """
    config, _ = utils.closure_constants()
    rows = state.interior
    computed = state.diffusion_coefficient_for_momentum()[:, rows]
    predicted = np.broadcast_to(
        (config.akt * FRICTION_VELOCITY * heights)[:, np.newaxis], computed.shape
    )
    np.testing.assert_allclose(
        computed,
        predicted,
        rtol=FIXED_POINT_TOLERANCE,
        err_msg="the neutral surface layer diffusivity is not 'kappa*u_star*z'",
    )
    length = state.master_length_scale[:, rows]
    shear = np.sqrt(state.mechanical_forcing[:, rows])
    np.testing.assert_allclose(
        computed,
        length * length * shear,
        rtol=FIXED_POINT_TOLERANCE,
        err_msg="the neutral surface layer diffusivity is not the mixing-length 'L**2*|S|'",
    )


def _assert_the_neutral_tke_level(state: utils.TurbulenceState) -> None:
    """The constant-dependent half of the statement, kept apart from the identity above.

    'TKE = d_mom**(2/3)*u_star**2/2'. Unlike 'K_m = kappa*u_star*z' this one moves if the
    closure constants are retuned, so a failure here and a pass above means the constants
    changed, not that the closure is wrong.
    """
    config, _ = utils.closure_constants()
    rows = state.interior
    normalised = state.velocity_scale[:, rows] / FRICTION_VELOCITY
    np.testing.assert_allclose(
        normalised,
        config.d_mom ** (1.0 / 3.0),
        rtol=FIXED_POINT_TOLERANCE,
        err_msg="the neutral surface layer velocity scale is not 'd_mom**(1/3)*u_star'",
    )
    exact = config.d_mom ** (2.0 / 3.0) / 2.0
    assert abs(exact - NEUTRAL_TKE_LEVEL) < 1.0e-10, (
        f"'d_mom**(2/3)/2' is {exact}, which no longer rounds to the {NEUTRAL_TKE_LEVEL} this "
        f"file and the literature note quote."
    )
    np.testing.assert_allclose(
        0.5 * normalised * normalised,
        exact,
        rtol=FIXED_POINT_TOLERANCE,
        err_msg="the neutral surface layer TKE is not 'd_mom**(2/3)*u_star**2/2'",
    )


@pytest.mark.uses_concat_where
def test_the_master_length_scale_is_the_harmonic_blend_with_kappa_outside(
    *, backend: gtx_typing.Backend | None
) -> None:
    """The mixing length is the height above the aerodynamic surface, capped harmonically.

    THE PHYSICS. An eddy near the ground can be no larger than its distance from it, which is
    Prandtl's 'l = kappa*z'; far from the ground it is limited instead by the depth of the
    turbulent layer, which is Blackadar's asymptote. A harmonic blend of the two is the usual
    way to write one formula for both, and this is the scheme's version of it.

    Two things are pinned down here that a savepoint comparison would reproduce without
    exhibiting: the height entering the blend is 'z + z0', the aerodynamic height, so the
    length scale does not vanish at the surface but tends to 'kappa*z0'; and 'kappa' multiplies
    the WHOLE blend, so the aloft limit is 'kappa*lambda'. The second is the departure from the
    textbook Blackadar form and it is a factor of 1/kappa = 2.5 in the free troposphere.
    """
    config, _ = utils.closure_constants()
    cells = 2
    nlev = 10
    layer_depth = 5.0
    roughness_times_gravity = np.array([0.981, 9.81])
    horizontal_limit = np.array([120.0, 300.0])
    roughness = roughness_times_gravity * float(ThermoConstants.EDGRAV.value)

    depth = np.zeros((cells, nlev + 1))
    depth[:, :nlev] = layer_depth
    computed = utils.as_cell_k_field(np.zeros((cells, nlev + 1)), backend)
    compute_turbulent_length_scale.with_backend(backend)(
        layer_depth=utils.as_cell_k_field(depth, backend),
        roughness_length_times_gravity=utils.as_cell_field(roughness_times_gravity, backend),
        horizontal_length_scale_limit=utils.as_cell_field(horizontal_limit, backend),
        nlev=gtx.int32(nlev),
        von_karman_constant=config.akt,
        minimal_length_scale=config.len_min,
        turbulent_length_scale=computed,
        horizontal_start=gtx.int32(0),
        horizontal_end=gtx.int32(cells),
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(nlev + 1),
        offset_provider=utils.KOFF,
    )

    aerodynamic_height = (
        np.arange(nlev, -1, -1)[np.newaxis, :] * layer_depth + roughness[:, np.newaxis]
    )
    limit = horizontal_limit[:, np.newaxis]
    predicted = config.akt * limit * aerodynamic_height / (limit + aerodynamic_height)
    np.testing.assert_allclose(
        computed.asnumpy(),
        predicted,
        rtol=1.0e-14,
        err_msg="the master length scale is not 'kappa*lambda*(z+z0)/(lambda+z+z0)'",
    )

    # The surface row is the roughness length alone, and the aloft limit is 'kappa*lambda' and
    # not 'lambda'. Both are the statements a textbook form would get wrong.
    np.testing.assert_allclose(
        computed.asnumpy()[:, nlev],
        config.akt * limit[:, 0] * roughness / (limit[:, 0] + roughness),
        rtol=1.0e-14,
        err_msg="the surface row of the length scale is not built from 'z0' alone",
    )
    assert np.all(predicted < config.akt * limit), (
        "the blend exceeded its own asymptote 'kappa*lambda', so it is not harmonic."
    )


def test_the_neutral_surface_layer_reproduces_the_log_law_diffusivity(
    *, backend: gtx_typing.Backend | None
) -> None:
    """A log-law wind profile in local equilibrium gives 'K_m = kappa*u_star*z', exactly.

    THE PHYSICS. This is the one result every boundary-layer closure has to get right, because
    it is what makes the surface stress consistent with the wind profile: a scheme that does
    not produce 'K_m = kappa*u_star*z' in the neutral surface layer produces the wrong drag,
    and everything downstream of the drag is then wrong too. It follows from local equilibrium
    alone and is independent of every closure constant, which is why it is a check of the
    IMPLEMENTATION rather than of the parametrisation.

    The state is driven to the equilibrium by iterating the scheme's own two programs -- the
    stability functions and the TKE step -- rather than by prescribing the answer, so what is
    asserted is the fixed point of the port and not an algebraic rearrangement of it.
    """
    heights, state = _surface_layer_state()
    converged = _iterate_to_the_fixed_point(state, backend, tkesmot=0.15)

    _assert_the_log_law_diffusivity(converged, heights)
    _assert_the_neutral_tke_level(converged)

    # The neutral Prandtl number: heat diffuses faster than momentum, by 'sm_0/sh_0'.
    _, params = utils.closure_constants()
    rows = converged.interior
    prandtl = (
        converged.diffusion_coefficient_for_momentum()[:, rows]
        / converged.diffusion_coefficient_for_scalars()[:, rows]
    )
    np.testing.assert_allclose(
        prandtl,
        params.sm_0 / params.sh_0,
        rtol=FIXED_POINT_TOLERANCE,
        err_msg="the neutral turbulent Prandtl number is not 'sm_0/sh_0'",
    )


def test_the_shear_taken_from_the_buoyancy_breaks_the_log_law(
    *, backend: gtx_typing.Backend | None
) -> None:
    """The surface-layer test has teeth: the 'a22' mutation misses the drag by a factor of two.

    'broken_stencils.compute_stability_lengths_with_the_shear_taken_from_the_buoyancy' writes
    'd_4*gh' where the closure has 'd_4*gm'. At neutral stratification that leaves 'S_m' at the
    constant 'a_mom*b_m', and the equilibrium then lands at 'q = SQRT(d_mom*a_mom*b_m)*L*|S|'
    instead of 'd_mom**(1/3)*L*|S|' -- so 'K_m' comes out 2.38 times too large and the neutral
    drag with it.

    BOTH assertion functions are exercised, and the pair is the point: the identity fails
    because the implementation is wrong, and the TKE level fails because the number it depends
    on has moved. A defect that broke only the second would be a retuning; one that broke only
    the first is not possible.
    """
    heights, state = _surface_layer_state()
    broken = _iterate_to_the_fixed_point(
        state,
        backend,
        tkesmot=0.15,
        stability_program=_CONFUSED_FORCING,
    )

    config, params = utils.closure_constants()
    rows = broken.interior
    overestimate = (
        broken.diffusion_coefficient_for_momentum()[:, rows]
        / (config.akt * FRICTION_VELOCITY * heights)[:, np.newaxis]
    )
    assert np.all(overestimate > 2.0), (
        f"the mutation moved the neutral diffusivity by only {overestimate.max()}, which is too "
        f"little for this to be a convincing demonstration."
    )
    np.testing.assert_allclose(
        broken.stability_function_for_momentum()[:, rows],
        config.a_mom * params.b_m,
        rtol=1.0e-12,
        err_msg=(
            "the mutation did not collapse 'S_m' to 'a_mom*b_m', so it is not doing what this "
            "test assumes and the invariants it is supposed to break are untested"
        ),
    )

    # And the invariants themselves, not restatements of them.
    with pytest.raises(AssertionError, match=r"diffusivity is not .kappa"):
        _assert_the_log_law_diffusivity(broken, heights)
    with pytest.raises(AssertionError, match=r"velocity scale is not .d_mom"):
        _assert_the_neutral_tke_level(broken)
