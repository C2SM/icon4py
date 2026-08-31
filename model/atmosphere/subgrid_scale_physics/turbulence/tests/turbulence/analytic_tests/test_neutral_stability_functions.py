# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The Raschendorfer stability functions in the neutral limit, from the constants in the code.

WHAT NEUTRAL MEANS HERE. 'Ri = 0' is 'fh2 = 0': no thermal forcing, so the dimensionless
buoyancy 'gh = fh2*tim2' vanishes and only the shear 'gm = fm2*tim2' is left. The 2x2 closure
of 'solve_turb_budgets' then becomes triangular, because 'a21 = (d_6 - d_4)*gh' is zero, and
Cramer's rule collapses to two expressions in one variable:

    S_m = b_m / (d_2 + d_4*gm)
    S_h = ( b_h*(d_2 + d_4*gm) - b_m*d_4*gm ) / ( d_1*(d_2 + d_4*gm) )

That is the first thing asserted below, over four decades of 'gm' and four choices of the
length and velocity scales. It is a statement about the linear algebra and it needs no closure
theory at all.

THE NEUTRAL CONSTANTS, DERIVED AND NOT REMEMBERED. Neutral stratification alone does not fix a
number, because 'S_m' still depends on the shear through 'gm'. What fixes it is neutral TKE
EQUILIBRIUM: production balances dissipation, which in this non-dimensionalisation is
'gama := S_m*gm - S_h*gh = 1/d_m'. At 'gh = 0' that is 'S_m*gm = 1/d_m', and substituting into
the two expressions above gives, with no further assumption,

    gm_eq = d_2 / (b_m*d_m - d_4)
    S_m   = (b_m - d_4/d_m)/d_2
    S_h   = (b_h - d_4/d_m)/d_1

which are exactly the 'sm_0' and 'sh_0' that 'TurbulenceParams.__post_init__' computes
(turbulence.py:932-933), the port of 'turb_setup'. Two closed forms follow from the definition
of 'c_m' in the same block, 'c_m = 1 - 1/(a_mom*c_tke) - 6*a_mom/d_mom' with
'c_tke = d_mom**(1/3)':

    b_m - d_4/d_m = 1/(a_mom*c_tke)   ==>   S_m(neutral) = 1/c_tke = d_mom**(-1/3)
    c_h = 0                           ==>   S_h(neutral) = a_heat*(1 - 6*a_mom/d_mom)

With the shipped 'd_mom = 16.6', 'a_mom = 0.92', 'a_heat = 0.74' those are 0.39201 and 0.49393,
and the neutral turbulent Prandtl number 'S_m/S_h' is 0.79366. Every number in this file comes
out of that derivation; none of it is a textbook value.

WHAT THIS TEST CANNOT SEE, stated because it bounds the claim. At 'gh = 0' every coefficient
that only multiplies 'gh' drops out: the whole of 'a21', the '(d_5 - d_4)' of 'a11' and the
'd_3' of 'a22'. A mistranslation confined to those is invisible here and needs a stratified
test -- 'test_the_neutral_limit_is_blind_to_the_buoyancy_cofactor' demonstrates exactly that,
so the blind spot is on record rather than assumed away.

The MODIFIED solution of the closure ('imode_stbcorr = 1', taken where the standard one is not
realizable) is not exercised: at 'fh2 = 0' the standard branch is always selected, which the
first test asserts rather than assumes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_diffusion_coefficients_from_stability_lengths import (
    compute_diffusion_coefficients_from_stability_lengths,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_stability_lengths import (
    compute_stability_lengths,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_stability_lengths_from_diffusion_coefficients import (
    compute_stability_lengths_from_diffusion_coefficients,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.turbulence import (
    TurbulenceConfig,
    TurbulenceParams,
)
from icon4py.model.common import constants
from icon4py.model.testing.fixtures.datatest import backend

from . import broken_stencils, utils


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing


#: Four columns, each with its own master length scale [m] and turbulent velocity scale [m/s].
#: The stability functions depend on the two only through 'tim2 = (tls/tke)**2', so a result
#: that varied between these columns after division by 'tls' would mean the closure had picked
#: up a scale it has no business having.
LENGTH_AND_VELOCITY_SCALES = ((100.0, 1.0), (50.0, 0.5), (250.0, 2.0), (10.0, 0.3))

#: The dimensionless shear the closed-form test sweeps: four decades, straddling the
#: equilibrium value of about 0.154.
SHEAR_RANGE = (0.01, 0.05, 0.154, 0.5, 2.0, 10.0)


def _closure_constants() -> tuple[TurbulenceConfig, TurbulenceParams]:
    """The shipped configuration and the closure constants 'turb_setup' derives from it."""
    return utils.closure_constants()


def _stability_arguments(params: TurbulenceParams, config: TurbulenceConfig) -> dict[str, float]:
    """The closure constants 'compute_stability_lengths' takes, as the granule binds them."""
    return utils.stability_length_arguments(params, config)


def _neutral_case(shear: np.ndarray, params: TurbulenceParams) -> dict[str, np.ndarray]:
    """A column set at 'fh2 = 0' whose row 'k' carries the dimensionless shear 'shear[k]'.

    Returns the host arrays of every field 'compute_stability_lengths' reads, plus the predicted
    'S_m' and 'S_h'. The mechanical forcing is back-calculated from the wanted 'gm' and each
    column's own 'tim2', which is what makes the four columns different inputs and the same
    answer.
    """
    rows = shear.size
    length = np.array([pair[0] for pair in LENGTH_AND_VELOCITY_SCALES])
    velocity = np.array([pair[1] for pair in LENGTH_AND_VELOCITY_SCALES])
    tim2 = (length / velocity) ** 2

    gm = np.tile(shear, (length.size, 1))
    mechanical_forcing = gm / tim2[:, np.newaxis]
    thermal_forcing = np.zeros_like(mechanical_forcing)

    denominator = params.d_2 + params.d_4 * gm
    stability_for_momentum = params.b_m / denominator
    stability_for_scalars = (params.b_h * denominator - params.b_m * params.d_4 * gm) / (
        params.d_1 * denominator
    )
    return {
        "master_length_scale": np.tile(length[:, np.newaxis], (1, rows)),
        "turbulent_velocity_scale": np.tile(velocity[:, np.newaxis], (1, rows)),
        "mechanical_forcing": mechanical_forcing,
        "thermal_forcing": thermal_forcing,
        "shear": gm,
        "stability_for_momentum": stability_for_momentum,
        "stability_for_scalars": stability_for_scalars,
    }


def _run_stability_lengths(
    case: dict[str, np.ndarray],
    params: TurbulenceParams,
    config: TurbulenceConfig,
    backend: gtx_typing.Backend | None,
    program=compute_stability_lengths,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one stability-function program over half levels 1..nlev-1 and return 'lsm', 'lsh'."""
    cells, rows = case["mechanical_forcing"].shape
    nlev = rows - 1
    length = case["master_length_scale"]
    entering_m = length * case["stability_for_momentum"]
    entering_h = length * case["stability_for_scalars"]

    updated_m = utils.as_cell_k_field(entering_m, backend)
    updated_h = utils.as_cell_k_field(entering_h, backend)
    program.with_backend(backend)(
        master_length_scale=utils.as_cell_k_field(length, backend),
        stability_length_for_momentum=utils.as_cell_k_field(entering_m, backend),
        stability_length_for_scalars=utils.as_cell_k_field(entering_h, backend),
        mechanical_forcing=utils.as_cell_k_field(case["mechanical_forcing"], backend),
        thermal_forcing=utils.as_cell_k_field(case["thermal_forcing"], backend),
        turbulent_velocity_scale=utils.as_cell_k_field(case["turbulent_velocity_scale"], backend),
        updated_stability_length_for_momentum=updated_m,
        updated_stability_length_for_scalars=updated_h,
        horizontal_start=gtx.int32(0),
        horizontal_end=gtx.int32(cells),
        vertical_start=gtx.int32(1),
        vertical_end=gtx.int32(nlev),
        offset_provider={},
        **_stability_arguments(params, config),
    )
    return updated_m.asnumpy(), updated_h.asnumpy()


def _assert_the_neutral_constants(
    computed_m: np.ndarray,
    computed_h: np.ndarray,
    case: dict[str, np.ndarray],
    params: TurbulenceParams,
) -> None:
    """The invariant itself, factored out so the mutation test can be seen to violate IT.

    A broken-variant test that restates the invariant in its own words proves only that the two
    statements disagree. This is the assertion the port is held to and the one the mutation is
    asserted to raise from.
    """
    rows = slice(1, computed_m.shape[1] - 1)
    length = case["master_length_scale"][:, rows]
    np.testing.assert_allclose(
        computed_m[:, rows] / length,
        params.sm_0,
        rtol=1.0e-14,
        err_msg="the neutral equilibrium momentum stability function is not 'sm_0'",
    )
    np.testing.assert_allclose(
        computed_h[:, rows] / length,
        params.sh_0,
        rtol=1.0e-14,
        err_msg="the neutral equilibrium scalar stability function is not 'sh_0'",
    )


def test_the_neutral_constants_are_what_the_closure_definitions_make_them() -> None:
    """'sm_0' and 'sh_0' have closed forms in the namelist constants, and the code agrees.

    THE PHYSICS. At neutral stratification and TKE equilibrium the level-2.5 closure reduces to
    Mellor-Yamada level 2, whose stability functions are pure numbers. Raschendorfer's
    parametrisation makes them 'd_mom**(-1/3)' and 'a_heat*(1 - 6*a_mom/d_mom)', because 'c_m'
    is defined in 'turb_setup' precisely so that 'b_m - d_4/d_m' collapses to '1/(a_mom*c_tke)'.

    Pure Python: this is a check of the derived constants, not of a stencil, and it is what
    stops the stencil tests below from being circular. Without it they would be comparing the
    port against 'params.sm_0', and 'params.sm_0' against nothing.
    """
    config, params = _closure_constants()

    assert params.c_h == 0.0, (
        "the scalar closure constant is no longer zero, so 'sh_0 = a_heat*(1 - 6*a_mom/d_mom)' "
        "no longer follows and this file's derivation needs redoing."
    )
    np.testing.assert_allclose(
        params.sm_0,
        config.d_mom ** (-1.0 / 3.0),
        rtol=1.0e-15,
        err_msg="the neutral momentum stability function is not 'd_mom**(-1/3)'",
    )
    np.testing.assert_allclose(
        params.sh_0,
        config.a_heat * (1.0 - 6.0 * config.a_mom / config.d_mom),
        rtol=1.0e-15,
        err_msg="the neutral scalar stability function is not 'a_heat*(1 - 6*a_mom/d_mom)'",
    )
    # The values the two derivations agree on, recorded so a change of the shipped constants
    # shows up as a failure here rather than as a drift in a forecast.
    np.testing.assert_allclose(params.sm_0, 0.3920101427669869, rtol=1.0e-14)
    np.testing.assert_allclose(params.sh_0, 0.4939277108433734, rtol=1.0e-14)
    # The neutral turbulent Prandtl number of the scheme, 'K_m/K_h = S_m/S_h'. Below one,
    # i.e. heat diffuses faster than momentum at neutral stratification, which is the
    # standard result and a number a reviewer can compare against the literature.
    np.testing.assert_allclose(params.sm_0 / params.sh_0, 0.7936589386686486, rtol=1.0e-14)


def test_the_stability_functions_at_zero_thermal_forcing_take_their_closed_form(
    *, backend: gtx_typing.Backend | None
) -> None:
    """At 'Ri = 0' the 2x2 closure is triangular and both stability functions are one-liners.

    THE PHYSICS. With no buoyancy the scalar equation stops feeding back into the momentum one,
    so the momentum stability function depends on the shear alone and the scalar one follows
    from it. Nothing about turbulence closure is needed to predict the answer -- only the
    coefficient assignment and Cramer's rule -- which is exactly what makes it a check of the
    translation rather than of the theory.

    Asserted over four decades of shear and four unrelated pairs of length and velocity scale.
    The scale independence is the second statement: after dividing by 'tls', all four columns
    must give the same number, because the closure sees only 'tim2'.
    """
    config, params = _closure_constants()
    case = _neutral_case(np.array(SHEAR_RANGE), params)
    rows = slice(1, len(SHEAR_RANGE) - 1)

    # Which branch is being exercised. The standard solution is taken where 'fh2 >= 0' and the
    # determinant and both numerators are positive; at 'gh = 0' that is 'det = d_1*a22 > 0',
    # 'num_m = b_m*d_1 > 0' and 'num_h = d_1*a22*S_h > 0', i.e. wherever 'S_h' is positive.
    assert np.all(case["stability_for_scalars"][:, rows] > 0.0), (
        "the predicted scalar stability function is not positive over the swept shear range, "
        "so the modified solution would be selected and this test would not be measuring the "
        "standard one."
    )

    computed_m, computed_h = _run_stability_lengths(case, params, config, backend)
    predicted_m = case["master_length_scale"] * case["stability_for_momentum"]
    predicted_h = case["master_length_scale"] * case["stability_for_scalars"]

    _, linf_m = utils.relative_errors(computed_m[:, rows], predicted_m[:, rows])
    _, linf_h = utils.relative_errors(computed_h[:, rows], predicted_h[:, rows])
    assert linf_m < 1.0e-14, f"'S_m*l' at zero buoyancy is not 'b_m*l/(d_2 + d_4*gm)': {linf_m}"
    assert linf_h < 1.0e-14, f"'S_h*l' at zero buoyancy does not match Cramer's rule: {linf_h}"

    # Scale independence: the dimensionless stability function is the same in every column.
    dimensionless_m = computed_m[:, rows] / case["master_length_scale"][:, rows]
    spread = dimensionless_m.max(axis=0) - dimensionless_m.min(axis=0)
    assert np.all(spread < 1.0e-14 * dimensionless_m.mean(axis=0)), (
        f"'S_m' depends on the length and velocity scales separately and not only through "
        f"'tim2': worst spread {spread.max()} across the four columns."
    )


def test_at_neutral_tke_equilibrium_the_closure_returns_the_neutral_constants(
    *, backend: gtx_typing.Backend | None
) -> None:
    """Level 2.5 reduces to level 2 at 'Ri = 0': the stability functions become pure numbers.

    THE PHYSICS. Set the shear to the value at which the TKE budget balances -- shear production
    equals dissipation, 'S_m*gm = 1/d_m' -- and the closure must return the neutral constants
    of the scheme, 'S_m = d_mom**(-1/3)' and 'S_h = a_heat*(1 - 6*a_mom/d_mom)'. This is the one
    point in the whole state space where the answer is a number rather than a function, and it
    is the anchor for every other stability value the scheme produces: get it wrong and the
    neutral surface layer has the wrong diffusivity everywhere.

    The equilibrium is verified, not assumed: 'S_m*gm*d_mom' is asserted to be one from the
    stencil's own output.
    """
    config, params = _closure_constants()
    equilibrium_shear = params.d_2 / (params.b_m * config.d_mom - params.d_4)
    case = _neutral_case(np.full(6, equilibrium_shear), params)
    rows = slice(1, 5)

    computed_m, computed_h = _run_stability_lengths(case, params, config, backend)
    _assert_the_neutral_constants(computed_m, computed_h, case, params)

    dimensionless_m = computed_m[:, rows] / case["master_length_scale"][:, rows]
    np.testing.assert_allclose(
        dimensionless_m * case["shear"][:, rows] * config.d_mom,
        1.0,
        rtol=1.0e-14,
        err_msg=(
            "the shear chosen is not the TKE equilibrium one, so the constants above were "
            "reproduced for the wrong reason"
        ),
    )


def test_the_diffusion_coefficients_round_trip_through_the_stability_lengths(
    *, backend: gtx_typing.Backend | None
) -> None:
    """'tkv -> ls -> tkv' is the identity to two ulp and not bit-exact, by construction.

    THE PHYSICS. Section 2c) divides the diffusion coefficients by the turbulent velocity scale
    to hand the turbulence model a length, and section 3) multiplies them back. The round trip
    is the identity as algebra, and at neutral equilibrium the coefficient it returns is
    'l*q*S_neutral' -- the mixing-length form of the diffusivity, with the neutral constant this
    file derived.

    IT IS NOT BIT-EXACT, and the reason is in the port on purpose: section 2c) forms the shared
    reciprocal '1/q' once and multiplies by it, as the Fortran does, so the round trip is
    'x*(1/q)*q' and carries two roundings rather than none. That is not a defect -- reproducing
    ICON is what the reciprocal is for -- but it means the two storages are not interchangeable
    and a caller must not assume a lossless conversion.
    """
    params = _closure_constants()[1]
    scales = np.array([pair[0] for pair in LENGTH_AND_VELOCITY_SCALES])
    velocity = np.array([pair[1] for pair in LENGTH_AND_VELOCITY_SCALES])
    rows = 6
    nlev = rows - 1

    length = np.tile(scales[:, np.newaxis], (1, rows))
    velocity_scale = np.tile(velocity[:, np.newaxis], (1, rows))
    stability_m = length * params.sm_0
    stability_h = length * params.sh_0

    coefficient_m = utils.as_cell_k_field(np.zeros((scales.size, rows)), backend)
    coefficient_h = utils.as_cell_k_field(np.zeros((scales.size, rows)), backend)
    bounds = {
        "horizontal_start": gtx.int32(0),
        "horizontal_end": gtx.int32(scales.size),
        "vertical_start": gtx.int32(1),
        "vertical_end": gtx.int32(nlev),
        "offset_provider": {},
    }
    compute_diffusion_coefficients_from_stability_lengths.with_backend(backend)(
        stability_length_for_momentum=utils.as_cell_k_field(stability_m, backend),
        stability_length_for_scalars=utils.as_cell_k_field(stability_h, backend),
        turbulent_velocity_scale=utils.as_cell_k_field(velocity_scale, backend),
        molecular_diffusivity_for_scalars=constants.MOLECULAR_DIFFUSIVITY_FOR_SCALARS,
        diffusion_coefficient_for_momentum=coefficient_m,
        diffusion_coefficient_for_scalars=coefficient_h,
        **bounds,
    )

    interior = slice(1, nlev)
    expected_m = (length * params.sm_0 * velocity_scale)[:, interior]
    expected_h = (length * params.sh_0 * velocity_scale)[:, interior]
    assert np.all(expected_h > constants.MOLECULAR_DIFFUSIVITY_FOR_SCALARS), (
        "the molecular floor binds on this state, so the round trip below would not be one."
    )
    np.testing.assert_allclose(
        coefficient_m.asnumpy()[:, interior],
        expected_m,
        rtol=1.0e-15,
        err_msg="the momentum diffusivity is not 'l*q*S_m' at neutral equilibrium",
    )
    np.testing.assert_allclose(
        coefficient_h.asnumpy()[:, interior],
        expected_h,
        rtol=1.0e-15,
        err_msg="the scalar diffusivity is not 'l*q*S_h' at neutral equilibrium",
    )

    returned_m = utils.as_cell_k_field(np.zeros((scales.size, rows)), backend)
    returned_h = utils.as_cell_k_field(np.zeros((scales.size, rows)), backend)
    compute_stability_lengths_from_diffusion_coefficients.with_backend(backend)(
        diffusion_coefficient_for_momentum=coefficient_m,
        diffusion_coefficient_for_scalars=coefficient_h,
        turbulent_velocity_scale=utils.as_cell_k_field(velocity_scale, backend),
        stability_length_for_momentum=returned_m,
        stability_length_for_scalars=returned_h,
        **bounds,
    )

    for returned, original, name in (
        (returned_m.asnumpy(), stability_m, "momentum"),
        (returned_h.asnumpy(), stability_h, "scalar"),
    ):
        difference = np.abs(returned[:, interior] - original[:, interior])
        ulp = np.spacing(np.abs(original[:, interior]))
        assert np.all(difference <= 2.0 * ulp), (
            f"the {name} round trip lost more than two ulp: worst {(difference / ulp).max()} "
            f"ulp. The conversion is a reciprocal and a multiplication, which costs at most a "
            f"rounding each way; more than that is a defect."
        )


def test_the_shear_taken_from_the_buoyancy_is_caught_by_the_neutral_limit(
    *, backend: gtx_typing.Backend | None
) -> None:
    """The neutral test has teeth: 'a22 = d_2 + d_3*gh + d_4*gh' fails it by 78 per cent.

    'broken_stencils.compute_stability_lengths_with_the_shear_taken_from_the_buoyancy' writes
    'gh' where the closure has 'gm' in the one coefficient that mixes the two forcings. The
    determinant stays positive, both numerators stay positive, the units are unchanged and the
    modified branch is still not selected -- so the mutation survives every structural check
    there is. At neutral it removes the shear dependence of 'S_m' entirely, leaving the constant
    'a_mom*b_m', and the neutral momentum diffusivity comes out 78 per cent too large.
    """
    config, params = _closure_constants()
    equilibrium_shear = params.d_2 / (params.b_m * config.d_mom - params.d_4)
    case = _neutral_case(np.full(6, equilibrium_shear), params)
    rows = slice(1, 5)

    broken_m, broken_h = _run_stability_lengths(
        case,
        params,
        config,
        backend,
        program=broken_stencils.compute_stability_lengths_with_the_shear_taken_from_the_buoyancy,
    )
    dimensionless_m = broken_m[:, rows] / case["master_length_scale"][:, rows]

    np.testing.assert_allclose(
        dimensionless_m,
        config.a_mom * params.b_m,
        rtol=1.0e-14,
        err_msg=(
            "the mutation did not collapse 'S_m' to 'a_mom*b_m', so it is not doing what this "
            "test assumes and the invariant it is supposed to break is untested"
        ),
    )
    overestimate = config.a_mom * params.b_m / params.sm_0 - 1.0
    assert overestimate > 0.5, (
        f"the mutation moves the neutral momentum stability function by only {overestimate}, "
        f"which is too little for this to be a convincing demonstration."
    )
    assert np.all(
        np.abs(broken_h[:, rows] / case["master_length_scale"][:, rows] - params.sh_0) > 0.1
    ), (
        "the scalar stability function was not moved by the mutation, so the neutral test would "
        "only be constraining the momentum branch."
    )

    # And the invariant itself, not a restatement of it: the assertion the neutral equilibrium
    # test makes must raise when handed the mutated stability lengths.
    with pytest.raises(AssertionError, match="momentum stability function is not 'sm_0'"):
        _assert_the_neutral_constants(broken_m, broken_h, case, params)


def test_the_neutral_limit_is_blind_to_the_buoyancy_cofactor() -> None:
    """The limit of the claim: at 'Ri = 0' nothing constrains a coefficient that multiplies 'gh'.

    'a21 = (d_6 - d_4)*gh' is zero at neutral stratification, so its sign, its coefficient and
    its very presence are unobservable in every test above. So are the '(d_5 - d_4)' of 'a11'
    and the 'd_3' of 'a22'. Demonstrated here rather than asserted in prose, because a blind
    spot that is only written down tends to be rediscovered.

    Pure numpy: the point is about the algebra at 'gh = 0', not about a backend. Catching a
    defect in those three coefficients needs a stratified reference -- the stable and unstable
    limits of the stability functions, which are the next analytic tests to write.
    """
    params = _closure_constants()[1]
    shear = np.array(SHEAR_RANGE)
    buoyancy = 0.0

    def standard(buoyancy_cofactor: float) -> tuple[np.ndarray, np.ndarray]:
        a11 = params.d_1 + (params.d_5 - params.d_4) * buoyancy
        a12 = params.d_4 * shear
        a21 = buoyancy_cofactor * buoyancy
        a22 = params.d_2 + params.d_3 * buoyancy + params.d_4 * shear
        determinant = a11 * a22 - a12 * a21
        return (
            (params.b_h * a22 - params.b_m * a12) / determinant,
            (params.b_m * a11 - params.b_h * a21) / determinant,
        )

    faithful_h, faithful_m = standard(params.d_6 - params.d_4)
    mutated_h, mutated_m = standard(-(params.d_6 - params.d_4))

    np.testing.assert_array_equal(
        np.stack([faithful_h, faithful_m]),
        np.stack([mutated_h, mutated_m]),
        err_msg=(
            "the sign of the buoyancy cofactor changed the answer at zero buoyancy, which "
            "would mean this file constrains more than it claims -- welcome, but check the "
            "arithmetic before believing it"
        ),
    )


@pytest.mark.parametrize("length_scale", [1.0e-6, 1.0e-4])
def test_only_the_scalar_diffusivity_carries_a_molecular_floor(
    length_scale: float, *, backend: gtx_typing.Backend | None
) -> None:
    """As the turbulence dies the scalar diffusivity stops at 'con_h' and the momentum one does not.

    THE PHYSICS. Molecular conduction does not switch off when the turbulence does, so ICON
    floors 'tkvh' at the molecular conductivity of dry air, 2.20e-5 m2/s. It does NOT floor
    'tkvm' in the same statement -- the momentum coefficient gets its own lower limits in
    section 4) instead. That asymmetry is easy to read as an oversight and is not one; it is
    asserted here so that a later tidy-up which "fixes" it fails a test.
    """
    params = _closure_constants()[1]
    rows = 4
    cells = 2
    length = np.full((cells, rows), length_scale)
    velocity = np.full((cells, rows), 1.0e-3)

    coefficient_m = utils.as_cell_k_field(np.zeros((cells, rows)), backend)
    coefficient_h = utils.as_cell_k_field(np.zeros((cells, rows)), backend)
    compute_diffusion_coefficients_from_stability_lengths.with_backend(backend)(
        stability_length_for_momentum=utils.as_cell_k_field(length * params.sm_0, backend),
        stability_length_for_scalars=utils.as_cell_k_field(length * params.sh_0, backend),
        turbulent_velocity_scale=utils.as_cell_k_field(velocity, backend),
        molecular_diffusivity_for_scalars=constants.MOLECULAR_DIFFUSIVITY_FOR_SCALARS,
        diffusion_coefficient_for_momentum=coefficient_m,
        diffusion_coefficient_for_scalars=coefficient_h,
        horizontal_start=gtx.int32(0),
        horizontal_end=gtx.int32(cells),
        vertical_start=gtx.int32(1),
        vertical_end=gtx.int32(rows - 1),
        offset_provider={},
    )

    interior = slice(1, rows - 1)
    turbulent_h = (length * params.sh_0 * velocity)[:, interior]
    assert np.all(turbulent_h < constants.MOLECULAR_DIFFUSIVITY_FOR_SCALARS), (
        "the turbulent scalar diffusivity is above the molecular floor on this state, so the "
        "floor is not being exercised."
    )
    np.testing.assert_array_equal(
        coefficient_h.asnumpy()[:, interior],
        np.full_like(turbulent_h, constants.MOLECULAR_DIFFUSIVITY_FOR_SCALARS),
        err_msg="the scalar diffusivity did not stop at the molecular conductivity",
    )
    np.testing.assert_allclose(
        coefficient_m.asnumpy()[:, interior],
        (length * params.sm_0 * velocity)[:, interior],
        rtol=1.0e-15,
        err_msg=(
            "the momentum diffusivity was floored as well; ICON floors only the scalar one in "
            "this statement"
        ),
    )
