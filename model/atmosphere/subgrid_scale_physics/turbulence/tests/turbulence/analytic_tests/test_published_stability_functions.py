# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The stability functions against the PUBLISHED closure, not against ICON's own constants.

WHY THIS FILE IS DIFFERENT FROM EVERY OTHER TEST IN THE PACKAGE. The savepoint tests derive
their expected value from serialized ICON output; the other analytic tests derive it from
'TurbulenceParams', i.e. from 'turb_setup' re-expressed in Python. Both answer "does the port
reproduce ICON". Neither can answer "does ICON reproduce the scheme it cites", because in both
of them ICON is on both sides of the comparison. A faithful port of a wrong implementation
passes all of them.

This file puts the paper on the other side. The oracle below is Mellor & Yamada (1982) Eqs.
(34) and (35) written out in the five published constants '(A1, A2, B1, B2, C1)' and solved
here; nothing in it is read from 'TurbulenceParams' or from any 'd_1 ... d_6'. If ICON's
implementation ever drifts from the closure it claims, this is the test that fails.

    G. L. Mellor and T. Yamada, "Development of a Turbulence Closure Model for Geophysical
    Fluid Problems", Reviews of Geophysics and Space Physics 20(4), 851-875, 1982.

MY82 p. 855, after Eqs. (33a)/(33b) define the two dimensionless forcings

    G_M = (l**2/q**2)*[(dU/dz)**2 + (dV/dz)**2]      G_H = -(l**2/q**2)*(g/theta_0)*dTheta/dz

states that "(26), (27), (28), and (29) after considerable algebra reduce to

    S_M*[6*A1*A2*G_M] + S_H*[1 - 3*A2*B2*G_H - 12*A1*A2*G_H]      = A2               (34)
    S_M*[1 + 6*A1**2*G_M - 9*A1*A2*G_H] - S_H*[12*A1**2*G_H + 9*A1*A2*G_H]
                                                                  = A1*(1 - 3*C1)    (35)

which are readily solved [...] for S_M and S_H as functions of G_M and G_H."

THE SIGN CONVENTION IS THE TRAP, and it is the single most likely way to get a green test that
means nothing. ICON's 'gh = fh2*(tls/tke)**2' is POSITIVE for stable stratification; MY82's
'G_H' carries an explicit minus sign in its definition and is NEGATIVE for stable. So

    G_H = -gh          G_M = +gm

and getting it wrong still produces plausible-looking numbers, for the wrong stratification.
'test_the_sign_convention_is_not_a_free_choice' measures what the wrong sign costs rather than
leaving the reader to trust the comment.

'C1' IS NOT 0.08 HERE. MY82 Eq. (45) prints '(A1, B1, A2, B2, C1) = (0.92, 16.6, 0.74, 10.1,
0.08)', but 0.08 is a rounding of a closed form the same paper gives, Eq. (44d) with Eq. (42a):

    gamma_1 = 1/3 - 2*A1/B1                          (42a)
    C1      = gamma_1 - 1/(3*A1*B1**(1/3))           (44d)

which is 0.08045729941653612. ICON's 'turb_utilities.f90:405' computes
'c_m = 1 - 1/(a_m*c_tke) - 6*a_m/d_m' -- three times (44d), character for character -- and
annotates it '!=3*0.08'. THAT COMMENT IS A REMINDER OF THE PUBLISHED ROUNDING, NOT A FORMULA:
'3*0.08 = 0.24' is 0.57 % away from what the line computes -- eleven orders of magnitude above
the tolerance this file asserts at. Every constant below is the closed form.
'test_the_published_constants_are_the_ones_icon_runs' asserts both halves of that: the closed
form agrees to 1e-15, and the printed rounding does not.

WHERE THE CROSS-CHECK IS VALID, AND WHERE IT IS NOT. Only on the stable half-plane
'gh >= 0', i.e. 'G_H <= 0'. There ICON solves (34)-(35) by Cramer's rule and the two agree to
machine precision. On the unstable side they are DIFFERENT MODELS, deliberately:

- MY82 avoid the singularity of (34)-(35) -- 'S_H -> infinity' as 'G_H -> 0.0338', p. 859 --
  by CLIPPING THE INPUT: "The constraint we use on (34) and (35) is G_H < 0.033 and
  G_M <= 0.825 - 25.0*G_H" (p. 862);
- ICON implements neither clause. With 'imode_stbcorr = 1' it REPLACES THE EQUATION for every
  'fh2 < 0' point, re-solving the system with the TKE equation in equilibrium form and a
  bounded deviation 'gama <= gam0' (turb_utilities.f90:1627-1667).

So a test that asserted agreement across 'G_H = 0' would be asserting that two schemes which
deliberately differ do not. 'test_the_cross_check_stops_where_the_two_schemes_part_company'
draws that line by measurement, and
'test_the_published_realizability_constraint_is_not_implemented' shows that MY82's second
clause bites on the stable side too -- at 'G_H = 0' it caps 'G_M' at 0.825 -- which is a
smaller region than "the whole stable half-plane" and is worth knowing before quoting this
file's result.

WHAT THE STANDARD BRANCH BEING SELECTED RESTS ON. ICON takes the Cramer solution only where
'fh2 >= 0' and the determinant and both numerators are positive. All three hold identically on
'gh >= 0, gm >= 0': every coefficient of the determinant as a polynomial in '(gh, gm)' is
positive, and so are those of the two numerators.
'test_the_published_system_is_nonsingular_on_the_whole_stable_quadrant' asserts it from the
published constants, so the comparison below is known to be measuring the branch it claims.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_stability_lengths import (
    compute_stability_lengths,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.turbulence import (
    TurbulenceConfig,
    TurbulenceParams,
)
from icon4py.model.testing.fixtures.datatest import backend

from . import broken_stencils, utils


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing


#: The dimensionless buoyancy 'gh = fh2*(tls/tke)**2' swept, all of it stable or neutral. In
#: MY82's convention these are 'G_H = -gh', so the sweep runs from 0 down to -100, six decades
#: of stratification. 0.0338 is in it because that is the MAGNITUDE of the singularity MY82
#: report -- theirs is at 'G_H = +0.0338', on the other side of neutral, and the point of
#: including its mirror image is that nothing happens there.
STABLE_BUOYANCY = (0.0, 1.0e-4, 1.0e-3, 1.0e-2, 0.0338, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)

#: The dimensionless shear 'gm = fm2*(tls/tke)**2' swept; 'G_M = gm' with no sign change. Six
#: decades, straddling the neutral equilibrium value 0.1537 and MY82's realizability cap of
#: 0.825 at 'G_H = 0'.
DIMENSIONLESS_SHEAR = (0.0, 1.0e-4, 1.0e-2, 0.05, 0.1537, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0)

#: Master length scale [m] and turbulent velocity scale [m/s]. The sweep is repeated once per
#: pair, so the column grid is 'shear x scale' and the four columns of a given shear hold the
#: identical dimensionless state reached from four unrelated pairs of metres and metres per
#: second. The closure sees the two only through 'tim2 = (tls/tke)**2', so a result that varied
#: between them would mean the port had picked up a scale it has no business having.
LENGTH_AND_VELOCITY_SCALES = ((100.0, 1.0), (50.0, 0.5), (250.0, 2.0), (10.0, 0.3))

#: What "agrees with the paper" means here. The measured worst relative disagreement over the
#: whole sweep is about 1e-15 on both stability functions, so this is two decades of headroom
#: and still eleven orders of magnitude tighter than the 0.57 % the '3*0.08' rounding costs.
PUBLISHED_TOLERANCE = 1.0e-13


# ------------------------------------------------------- the published closure, as an oracle ---


def published_constants() -> tuple[float, float, float, float, float]:
    """'(A1, A2, B1, B2, C1)' of MY82, with 'C1' in the closed form of Eqs. (42a) and (44d).

    The first four are read from 'TurbulenceConfig' because they are the namelist numbers whose
    identification with MY82 Eq. (45) is the thing under test -- 'a_mom', 'a_heat', 'd_mom',
    'd_heat' are A1, A2, B1, B2 digit for digit, and
    'test_the_published_constants_are_the_ones_icon_runs' asserts exactly that. Nothing derived
    is taken: 'C1' is computed here from the paper, and no 'd_1 ... d_6', 'b_m', 'b_h', 'sm_0'
    or 'sh_0' appears anywhere in this module's oracle.
    """
    config = TurbulenceConfig()
    a_1, a_2, b_1, b_2 = config.a_mom, config.a_heat, config.d_mom, config.d_heat
    gamma_1 = 1.0 / 3.0 - 2.0 * a_1 / b_1  # MY82 (42a)
    c_1 = gamma_1 - 1.0 / (3.0 * a_1 * b_1 ** (1.0 / 3.0))  # MY82 (44d)
    return a_1, a_2, b_1, b_2, c_1


def published_stability_functions(
    shear: np.ndarray | float, buoyancy: np.ndarray | float
) -> dict[str, np.ndarray]:
    """MY82 (34) and (35), solved for '(S_M, S_H)' at 'G_M = shear', 'G_H = buoyancy'.

    Written from the paper and from nothing else. The two equations are put in the order
    '(S_H, S_M)' so that the determinant comes out a POSITIVE multiple of ICON's -- 'A1*A2'
    times it -- which is what lets the sign of this determinant and of these two numerators say
    which branch ICON takes, without ICON's coefficients being consulted.

    Args:
        shear: 'G_M', MY82's dimensionless shear. Same sign as ICON's 'gm'.
        buoyancy: 'G_H', MY82's dimensionless buoyancy. The NEGATIVE of ICON's 'gh'.

    Returns:
        'momentum' and 'scalar' for 'S_M' and 'S_H', and 'determinant', 'numerator_momentum'
        and 'numerator_scalar' for the intermediate quantities the branch test needs.
    """
    # 'B1' does not appear in (34) or (35): it reaches them only through 'C1', Eq. (44d).
    a_1, a_2, _, b_2, c_1 = published_constants()
    shear = np.asarray(shear, dtype=float)
    buoyancy = np.asarray(buoyancy, dtype=float)

    # (34):  scalar_row_h * S_H + scalar_row_m * S_M = A2
    scalar_row_h = 1.0 - 3.0 * a_2 * b_2 * buoyancy - 12.0 * a_1 * a_2 * buoyancy
    scalar_row_m = 6.0 * a_1 * a_2 * shear
    # (35):  momentum_row_h * S_H + momentum_row_m * S_M = A1*(1 - 3*C1)
    momentum_row_h = -(12.0 * a_1 * a_1 * buoyancy + 9.0 * a_1 * a_2 * buoyancy)
    momentum_row_m = 1.0 + 6.0 * a_1 * a_1 * shear - 9.0 * a_1 * a_2 * buoyancy

    right_scalar = a_2
    right_momentum = a_1 * (1.0 - 3.0 * c_1)

    determinant = scalar_row_h * momentum_row_m - scalar_row_m * momentum_row_h
    numerator_scalar = right_scalar * momentum_row_m - right_momentum * scalar_row_m
    numerator_momentum = scalar_row_h * right_momentum - momentum_row_h * right_scalar
    return {
        "momentum": numerator_momentum / determinant,
        "scalar": numerator_scalar / determinant,
        "determinant": determinant,
        "numerator_momentum": numerator_momentum,
        "numerator_scalar": numerator_scalar,
    }


# ------------------------------------------------------------------- the state and the runner ---


def _stratified_case(buoyancy: np.ndarray, shear: np.ndarray) -> dict[str, np.ndarray]:
    """A state whose row 'i+1' carries 'buoyancy[i]' and whose columns are shear x scale pairs.

    One padding row above and one below, because 'compute_stability_lengths' runs the Fortran
    'DO k=k_st,k_en' and neither end row is computed; '_band' selects the rows that are.

    The columns are the outer product of the shear sweep with the four '(tls, tke)' pairs, laid
    out as 'column = scale_index*len(shear) + shear_index'. The forcings are back-calculated
    from each column's own 'tim2', so the four columns of one shear reach the identical
    dimensionless state from four unrelated pairs of metres and metres per second -- which is
    what makes the scale independence of the closure assertable rather than assumed.
    """
    scales = len(LENGTH_AND_VELOCITY_SCALES)
    cells = shear.size * scales
    rows = buoyancy.size + 2
    length = np.repeat(np.array([pair[0] for pair in LENGTH_AND_VELOCITY_SCALES]), shear.size)
    velocity = np.repeat(np.array([pair[1] for pair in LENGTH_AND_VELOCITY_SCALES]), shear.size)
    turbulent_time_squared = (length / velocity) ** 2

    padded_buoyancy = np.concatenate(([buoyancy[0]], buoyancy, [buoyancy[-1]]))
    gh = np.tile(padded_buoyancy, (cells, 1))
    gm = np.tile(np.tile(shear, scales)[:, np.newaxis], (1, rows))

    predicted = published_stability_functions(shear=gm, buoyancy=-gh)
    return {
        "master_length_scale": np.tile(length[:, np.newaxis], (1, rows)),
        "turbulent_velocity_scale": np.tile(velocity[:, np.newaxis], (1, rows)),
        "mechanical_forcing": gm / turbulent_time_squared[:, np.newaxis],
        "thermal_forcing": gh / turbulent_time_squared[:, np.newaxis],
        "buoyancy": gh,
        "shear": gm,
        "predicted_momentum": predicted["momentum"],
        "predicted_scalar": predicted["scalar"],
    }


def _band(case: dict[str, np.ndarray]) -> slice:
    """The rows 'compute_stability_lengths' actually writes, given the padding above."""
    return slice(1, case["buoyancy"].shape[1] - 1)


def _run_stability_lengths(
    case: dict[str, np.ndarray],
    backend: gtx_typing.Backend | None,
    program=compute_stability_lengths,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one stability-function program over the padded band and return 'lsm', 'lsh' [m]."""
    config = TurbulenceConfig()
    params = TurbulenceParams(config=config)
    cells, rows = case["mechanical_forcing"].shape
    length = case["master_length_scale"]

    # The entering stability lengths reach only the MODIFIED branch, through 'frc'; the standard
    # branch does not read them. ICON's own neutral constants are handed over so that nothing
    # this file predicts is fed back in as an input.
    entering_m = length * params.sm_0
    entering_h = length * params.sh_0

    updated_m = utils.as_cell_k_field(np.zeros_like(length), backend)
    updated_h = utils.as_cell_k_field(np.zeros_like(length), backend)
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
        vertical_end=gtx.int32(rows - 1),
        offset_provider={},
        **utils.stability_length_arguments(params, config),
    )
    return updated_m.asnumpy(), updated_h.asnumpy()


def _assert_the_published_stability_functions(
    computed_m: np.ndarray,
    computed_h: np.ndarray,
    case: dict[str, np.ndarray],
    rtol: float = PUBLISHED_TOLERANCE,
) -> None:
    """The invariant itself, factored out so the mutation test can be seen to violate IT.

    A broken-variant test that restates the invariant in its own words proves only that the two
    statements disagree. This is the assertion the port is held to and the one the mutation is
    asserted to raise from.

    The stencil returns 'S*l' in metres; dividing by the master length scale is what makes the
    comparison one against MY82's dimensionless 'S_M' and 'S_H'.
    """
    band = _band(case)
    length = case["master_length_scale"][:, band]
    np.testing.assert_allclose(
        computed_m[:, band] / length,
        case["predicted_momentum"][:, band],
        rtol=rtol,
        err_msg="'S_M' is not the solution of Mellor & Yamada (1982) Eq. (34)-(35)",
    )
    np.testing.assert_allclose(
        computed_h[:, band] / length,
        case["predicted_scalar"][:, band],
        rtol=rtol,
        err_msg="'S_H' is not the solution of Mellor & Yamada (1982) Eq. (34)-(35)",
    )


# --------------------------------------------------------------------------------- the tests ---


def test_the_published_constants_are_the_ones_icon_runs() -> None:
    """The five MY82 constants, and the four ICON identities that follow from them alone.

    THE POINT. Everything below this test compares ICON's stability functions against MY82's
    equations, which is only meaningful if the numbers those equations are evaluated with are
    the numbers ICON runs. That identification is asserted here, once, from the paper:

    - Eq. (45), p. 858: "(A1, B1, A2, B2, C1) = (0.92, 16.6, 0.74, 10.1, 0.08)". The first four
      are ICON's 'a_mom', 'd_mom', 'a_heat', 'd_heat' digit for digit, and ICON's own comments
      give them MY82's roles -- 'a_*' the Rotta pressure-destruction length-scale factors,
      'd_*' the Kolmogorov dissipation ones, which is Eq. (12).
    - 'C1' to the closed form of Eqs. (42a) and (44d), NOT to the printed 0.08. ICON's 'c_m' is
      three times it; the difference is 5.6e-17.
    - Eq. (44a) 'q**3 = B1*u_tau**3' gives 'q/u* = B1**(1/3)', which is ICON's 'c_tke'.
    - Eq. (47a) 'phi_M = [B1*(1-R_f)*S_M**3]**(-1/4)' at neutral 'R_f = 0, phi_M = 1' gives
      'S_M = B1**(-1/3)', which is ICON's 'sm_0'.
    - Eqs. (42a)/(42b) and p. 856 give the critical flux Richardson number
      'R_fc = gamma_1/(gamma_1 + gamma_2)' with 'gamma_2 = B2/B1 + 6*A1/B1'. ICON's 'rim' is
      '1 - R_fc' and the difference is exactly zero.

    Pure Python: this is a check of the constants, not of a stencil, and it is what stops the
    stencil test below from being a comparison of ICON with itself.
    """
    config = TurbulenceConfig()
    params = TurbulenceParams(config=config)
    a_1, a_2, b_1, b_2, c_1 = published_constants()

    # MY82 Eq. (45) against the namelist. Exact equality: these are the printed digits.
    assert (a_1, a_2, b_1, b_2) == (0.92, 0.74, 16.6, 10.1), (
        "the shipped closure constants are no longer the Mellor & Yamada (1982) Eq. (45) set, "
        "so the published oracle in this file is being applied to a different scheme"
    )

    # MY82 Eqs. (42a) + (44d): 'c_m' is three times the closed form of 'C1'.
    np.testing.assert_allclose(
        params.c_m,
        3.0 * c_1,
        rtol=1.0e-15,
        err_msg="'c_m' is not '3*C1' of Mellor & Yamada (1982) Eq. (44d)",
    )
    # And the printed rounding is NOT what the code computes: 0.57 %, eleven orders of
    # magnitude above the tolerance this file asserts at. Asserted so that a later
    # 'simplification' of 'c_m' to the literal in its own comment fails here, not in a forecast.
    rounding_error = abs(3.0 * 0.08 - params.c_m) / params.c_m
    assert 5.0e-3 < rounding_error < 6.0e-3, (
        f"the published rounding '3*0.08' is {rounding_error} away from the closed form; it was "
        f"0.0057, and this file's oracle depends on the closed form being used"
    )
    assert rounding_error > 1.0e4 * PUBLISHED_TOLERANCE, (
        "the '3*0.08' rounding is now within this file's tolerance, which would mean the test "
        "can no longer tell the closed form from the printed value"
    )

    # MY82 Eq. (44a): 'q/u* = B1**(1/3)'.
    np.testing.assert_allclose(
        params.c_tke, b_1 ** (1.0 / 3.0), rtol=1.0e-15, err_msg="'c_tke' is not 'B1**(1/3)'"
    )
    # MY82 Eq. (47a) at neutral: 'S_M = B1**(-1/3)'.
    np.testing.assert_allclose(
        params.sm_0, b_1 ** (-1.0 / 3.0), rtol=1.0e-15, err_msg="'sm_0' is not 'B1**(-1/3)'"
    )
    # MY82 Eqs. (42a)/(42b), p. 856: the critical flux Richardson number.
    gamma_1 = 1.0 / 3.0 - 2.0 * a_1 / b_1
    gamma_2 = b_2 / b_1 + 6.0 * a_1 / b_1
    critical_flux_richardson = gamma_1 / (gamma_1 + gamma_2)
    np.testing.assert_allclose(
        params.rim,
        1.0 - critical_flux_richardson,
        rtol=1.0e-15,
        err_msg="'rim' is not '1 - gamma_1/(gamma_1 + gamma_2)' of Mellor & Yamada (1982)",
    )
    # The value, recorded so that a change of the shipped constants shows up here. MY82 p. 856
    # says only "the critical Richardson number is 0.19"; this is what their constants give.
    np.testing.assert_allclose(critical_flux_richardson, 0.19123230928546775, rtol=1.0e-14)


def test_the_published_system_is_nonsingular_on_the_whole_stable_quadrant() -> None:
    """Why the comparison below measures Cramer's rule and not the fallback.

    ICON takes the standard (Cramer) solution only where 'fh2 >= 0' and the determinant and
    both numerators are positive, and takes a completely different closed form otherwise. So a
    comparison against MY82 (34)-(35) is only a comparison against MY82 (34)-(35) where those
    four conditions hold, and asserting that from ICON's own coefficients would be circular.

    It follows from the published constants instead. Writing the determinant of (34)-(35) as a
    polynomial in '(G_M, -G_H) = (gm, gh)', every coefficient is positive -- the constant term,
    the two linear ones, the 'gh**2' one, and the mixed 'gh*gm' one, whose sign is that of
    '(3*B2 + 12*A1) - (9*A2 + 12*A1)', i.e. of '3*B2 - 9*A2'. That last inequality is the only
    one that is not immediate and it is asserted below. The scalar numerator is positive for the
    same kind of reason; the momentum one is positive because its 'gh' coefficient beats the
    subtraction, which is arithmetic rather than structure, so all three are also MEASURED over
    the sweep below rather than argued.

    Pure numpy: a statement about the paper's algebra, not about a backend.
    """
    _, a_2, _, b_2, _ = published_constants()
    assert 3.0 * b_2 > 9.0 * a_2, (
        "'3*B2 > 9*A2' has stopped holding, so the mixed 'gh*gm' coefficient of the determinant "
        "may change sign and the quadrant is no longer known to be nonsingular"
    )

    gh, gm = np.meshgrid(np.array(STABLE_BUOYANCY), np.array(DIMENSIONLESS_SHEAR), indexing="ij")
    solution = published_stability_functions(shear=gm, buoyancy=-gh)

    assert np.all(solution["determinant"] > 0.0), (
        "the published 2x2 system is singular somewhere on the stable quadrant, so ICON would "
        "take its fallback branch there and this file would not be comparing what it claims"
    )
    assert np.all(solution["numerator_momentum"] > 0.0) and np.all(
        solution["numerator_scalar"] > 0.0
    ), "a numerator of the published solution is non-positive, so ICON's 'solvable' is false"
    assert np.all(solution["momentum"] > 0.0) and np.all(solution["scalar"] > 0.0), (
        "the published stability functions are not positive on the stable quadrant"
    )
    # Both fall monotonically as the stratification strengthens at fixed shear -- the physical
    # content of the closure and a cheap guard against an oracle transcribed with a sign slip
    # that the positivity checks above would not see.
    assert np.all(np.diff(solution["momentum"], axis=0) < 0.0), (
        "'S_M' does not decrease with increasing stable stratification"
    )
    assert np.all(np.diff(solution["scalar"], axis=0) < 0.0), (
        "'S_H' does not decrease with increasing stable stratification"
    )


def test_the_stability_functions_are_the_published_ones_on_the_stable_half_plane(
    *, backend: gtx_typing.Backend | None
) -> None:
    """THE TEST THIS FILE EXISTS FOR. ICON's '(S_h, S_m)' is MY82 (34)-(35), to 1e-15.

    THE PHYSICS. Level 2.5 of the Mellor-Yamada hierarchy determines the two stability functions
    algebraically from the two dimensionless forcings once the TKE is known, and (34)-(35) is
    that algebra. ICON solves the same pair by Cramer's rule after pre-multiplying row 1 by
    'a_heat' and row 2 by 'a_mom' and bundling the products into 'd_1 ... d_6'; the bundle is
    MY82's system in different variables, and this test is the statement that it is nothing
    more than that.

    Swept over 12 stable and neutral buoyancies against 11 shears -- 132 states -- with the
    dimensional inputs cycling through four unrelated pairs of length and velocity scale, so
    that neighbouring columns reach the same dimensionless state from different metres and
    metres per second.

    MEASURED 2026-08-31, and identical on 'embedded' and on 'gtfn_cpu' to the last digit: the
    normalised L-infinity disagreement is 3.2e-16 on 'S_M' and 1.9e-16 on 'S_H', and the worst
    POINTWISE relative disagreement over the 528 states is 1.0e-15 and 9.1e-16. The two sides do
    the same arithmetic in a different order -- ICON inverts the determinant once and multiplies
    twice where the oracle divides -- so exact equality is not expected and is not asserted.

    So the answer to "does ICON implement the closure it cites" is, on this half-plane, yes, to
    within one rounding of double precision.
    """
    case = _stratified_case(np.array(STABLE_BUOYANCY), np.array(DIMENSIONLESS_SHEAR))
    band = _band(case)
    assert np.all(case["thermal_forcing"] >= 0.0), (
        "the sweep left the stable half-plane, where ICON solves a different system"
    )

    computed_m, computed_h = _run_stability_lengths(case, backend)
    _assert_the_published_stability_functions(computed_m, computed_h, case)

    # What the agreement actually is, reported rather than merely bounded.
    length = case["master_length_scale"][:, band]
    _, worst_m = utils.relative_errors(
        computed_m[:, band] / length, case["predicted_momentum"][:, band]
    )
    _, worst_h = utils.relative_errors(
        computed_h[:, band] / length, case["predicted_scalar"][:, band]
    )
    assert worst_m < 1.0e-14 and worst_h < 1.0e-14, (
        f"the port and the published closure agree less well than one rounding: {worst_m}, "
        f"{worst_h}. Machine precision is the expectation here, not a tolerance."
    )

    # Scale independence, the second statement of the test: after dividing by 'tls' the four
    # columns of a given shear must give one number, because the closure sees only 'tim2'.
    dimensionless_m = computed_m[:, band] / length
    by_scale = dimensionless_m.reshape(
        len(LENGTH_AND_VELOCITY_SCALES), len(DIMENSIONLESS_SHEAR), -1
    )
    spread = by_scale.max(axis=0) - by_scale.min(axis=0)
    assert np.all(spread <= 1.0e-14 * by_scale.max(axis=0)), (
        f"'S_M' depends on the length and velocity scales separately and not only through "
        f"'tim2': worst spread {spread.max()} across the four scale pairs."
    )


def test_the_sign_convention_is_not_a_free_choice() -> None:
    """'gh = -G_H'. Getting it wrong gives a green-looking answer for the wrong stratification.

    THE TRAP. ICON's 'gh' and MY82's 'G_H' describe the same physics with opposite signs: MY82
    put the minus sign into the definition of 'G_H', so 'G_H' is negative under stable
    stratification, while ICON's 'gh = fh2*tim2' is positive there. Both are smooth functions
    of one variable that agree at zero, so a test written with the wrong sign still runs, still
    produces finite stability functions in the right ballpark for weak stratification, and
    still fails to constrain anything -- it would be comparing ICON's stable branch against the
    paper's unstable one.

    Demonstrated rather than commented. At 'G_H = +gh' the published system crosses its own
    singularity, so the disagreement is not a small bias: it is two orders of magnitude, and
    the stability functions come out NEGATIVE where ICON's are positive.

    Pure numpy: the oracle is compared with itself under the two conventions, so no backend and
    no stencil is involved.
    """
    gh, gm = np.meshgrid(np.array(STABLE_BUOYANCY), np.array(DIMENSIONLESS_SHEAR), indexing="ij")
    right = published_stability_functions(shear=gm, buoyancy=-gh)
    wrong = published_stability_functions(shear=gm, buoyancy=+gh)

    stratified = gh > 0.0
    worst = np.abs(
        (wrong["momentum"][stratified] - right["momentum"][stratified])
        / right["momentum"][stratified]
    ).max()
    assert worst > 10.0, (
        f"the two sign conventions differ by only {worst} in relative terms, so a test written "
        f"with the wrong one could pass and this file's central claim would be unsupported"
    )
    assert np.any(wrong["scalar"][stratified] < 0.0), (
        "the wrong sign convention does not even produce a negative scalar stability function, "
        "which is the cheapest symptom of having crossed the singularity"
    )
    # And the two conventions do agree at neutral, which is why the mistake survives a neutral
    # test. That blind spot is the reason this file sweeps a stratified range at all.
    np.testing.assert_array_equal(
        right["momentum"][0],
        wrong["momentum"][0],
        err_msg="the sign convention is observable at 'gh = 0', which it should not be",
    )


def test_the_published_realizability_constraint_is_not_implemented() -> None:
    """MY82 clip the INPUT; ICON replaces the EQUATION. And one clause bites on the stable side.

    MY82 p. 862 state the domain restriction they apply to (34)-(35):

        "The constraint we use on (34) and (35) is G_H < 0.033 and G_M <= 0.825 - 25.0*G_H"

    which excludes the region where 'S_H' runs to infinity as 'G_H -> 0.0338' (p. 859) and the
    region where their own Rotta hypothesis is doubtful. ICON implements neither clause: it
    guards with 'det > 0 .AND. sh > 0 .AND. sm > 0' and, for 'fh2 < 0', substitutes a different
    closed form altogether.

    THE PART WORTH RECORDING. The first clause is automatically satisfied on the stable side --
    'G_H <= 0 < 0.033' -- but THE SECOND IS NOT. At 'G_H = 0' it caps 'G_M' at 0.825, and a
    stably stratified layer with strong shear exceeds that easily. So "MY82's constraint is
    inactive on the stable side" is too strong a statement; what is true is that the constraint
    is inactive ALONG THE TKE-EQUILIBRIUM LOCUS, where the scheme actually sits: solving
    'S_M*gm - S_H*gh = 1/B1' for 'gm' at each 'gh' gives a curve that runs at 'gm ~ 5.2*gh' for
    large 'gh' and stays a factor of four or more inside '0.825 + 25*gh' at every
    stratification -- measured ratio 0.19 to 0.23 over seven decades.

    Both halves are measured here. The agreement asserted by the test above therefore extends
    over a region MY82 would decline to enter, and does so because the equations still hold
    there -- a restriction of the domain of validity is not a change of the equations.

    ONE CAVEAT ON THE READING. The clause is quoted as printed, and as printed it is a straight
    line in the '(G_H, G_M)' plane that goes on constraining 'G_M' however negative 'G_H'
    becomes. MY82 introduce it as the boundary of the shaded region of their Figure 3, whose
    plotted range is small, so whether they meant it to extend that far is not something the
    sentence settles. What the sentence does settle is that "MY82's constraint is inactive on
    the stable side" is not a safe thing to say without checking, which is why this test exists.

    Pure numpy.
    """
    _, _, b_1, _, _ = published_constants()
    gh, gm = np.meshgrid(np.array(STABLE_BUOYANCY), np.array(DIMENSIONLESS_SHEAR), indexing="ij")
    published_cap = 0.825 - 25.0 * (-gh)  # MY82 p. 862, at 'G_H = -gh'

    assert np.any(gm > published_cap), (
        "no state in the sweep violates MY82's 'G_M <= 0.825 - 25*G_H', so this test is not "
        "demonstrating anything and the claim that the clause bites on the stable side is "
        "unsupported by it"
    )
    outside = float((gm > published_cap).mean())
    assert outside > 0.2, (
        f"only {outside} of the swept states are outside MY82's stated domain; the sweep was "
        f"meant to straddle it well enough for the point to be visible"
    )

    # The TKE-equilibrium locus, where the scheme sits, and where the clause does not bite.
    # 'S_M*gm - S_H*gh = 1/B1' is neutral TKE equilibrium generalised to stratification; solve
    # it for 'gm' by bisection at each 'gh', using the published stability functions only.
    def equilibrium_shear(buoyancy: float) -> float:
        def excess(shear: float) -> float:
            solution = published_stability_functions(shear=shear, buoyancy=-buoyancy)
            return float(solution["momentum"] * shear - solution["scalar"] * buoyancy - 1.0 / b_1)

        low, high = 1.0e-8, 1.0e8
        assert excess(low) < 0.0 < excess(high), (
            f"the TKE-equilibrium shear at 'gh = {buoyancy}' is not bracketed by "
            f"[{low}, {high}], so the bisection below is not solving anything"
        )
        for _ in range(200):
            middle = 0.5 * (low + high)
            if excess(middle) < 0.0:
                low = middle
            else:
                high = middle
        return 0.5 * (low + high)

    for buoyancy in (0.0, 1.0e-3, 1.0e-2, 0.1, 1.0, 10.0, 100.0):
        shear = equilibrium_shear(buoyancy)
        cap = 0.825 + 25.0 * buoyancy
        assert shear < 0.3 * cap, (
            f"at 'gh = {buoyancy}' the TKE-equilibrium shear is {shear}, which is not "
            f"comfortably inside MY82's cap of {cap}; the claim that the constraint is inactive "
            f"where the scheme sits no longer holds"
        )
    # The neutral equilibrium point of the scheme, for orientation: 'gm = 0.15367' against a cap
    # of 0.825, a factor of 5.4. It is the same number 'test_neutral_stability_functions.py'
    # reaches from ICON's constants, which is a cross-check of both derivations.
    np.testing.assert_allclose(equilibrium_shear(0.0), 0.15367195203219336, rtol=1.0e-9)


def test_the_cross_check_stops_where_the_two_schemes_part_company(
    *, backend: gtx_typing.Backend | None
) -> None:
    """On the unstable side ICON is NOT solving (34)-(35), and this says so with numbers.

    A test that asserted agreement across 'G_H = 0' would be asserting that two schemes which
    deliberately differ do not, which is a defect in the test rather than a finding. So the
    boundary is drawn by measurement:

    - MY82's system is singular at 'G_H = 0.0327' (with 'G_M = 0'; the determinant of (34)-(35)
      has its two roots there and at 0.1632, which is where MY82's reported 'G_H -> 0.0338,
      S_H -> infinity' lives). Beyond it their solution is NEGATIVE -- unphysical -- which is
      exactly why they clip the input.
    - ICON never evaluates it there. With 'imode_stbcorr = 1' every 'fh2 < 0' point takes the
      modified solution, which is bounded by construction, and the stencil returns a finite
      positive pair.

    Asserted with the stencil, so the claim is about what the port computes and not about what
    a transcription of the Fortran would compute.
    """
    unstable = np.array([0.0010, 0.0050, 0.0100, 0.0200, 0.0327, 0.0500, 0.1000, 0.1630, 0.3000])
    case = _stratified_case(-unstable, np.array([0.0, 0.1537, 1.0, 10.0]))
    band = _band(case)
    assert np.all(case["thermal_forcing"] < 0.0), "the case is not unstable throughout"

    published = published_stability_functions(
        shear=case["shear"][:, band], buoyancy=-case["buoyancy"][:, band]
    )
    # The published determinant changes sign inside the swept range: that is the singularity.
    assert published["determinant"].max() > 0.0 and published["determinant"].min() < 0.0, (
        "the unstable sweep does not straddle the singularity of the published system, so it "
        "is not showing where the two schemes part company"
    )
    assert np.any(published["scalar"] < 0.0), (
        "the published solution stays positive over the unstable sweep, so the region MY82 "
        "exclude is not being reached"
    )

    computed_m, computed_h = _run_stability_lengths(case, backend)
    length = case["master_length_scale"][:, band]
    icon_m = computed_m[:, band] / length
    icon_h = computed_h[:, band] / length

    assert np.all(np.isfinite(icon_m)) and np.all(np.isfinite(icon_h)), (
        "ICON's modified branch produced a non-finite stability function on the unstable side, "
        "which is the failure it exists to prevent"
    )
    assert np.all(icon_m > 0.0) and np.all(icon_h > 0.0), (
        "ICON's modified branch produced a non-positive stability function on the unstable side"
    )
    # And they disagree, grossly, wherever MY82's own constraint is violated.
    beyond = -case["buoyancy"][:, band] >= 0.0327
    worst = np.abs((published["momentum"][beyond] - icon_m[beyond]) / icon_m[beyond]).max()
    assert worst > 1.0, (
        f"past its own singularity the published system differs from ICON by only {worst} in "
        f"relative terms; the two are supposed to be different models there and a test that "
        f"compared them would be meaningless"
    )
    # The published-oracle assertion must NOT be applied here. Demonstrated, not asserted in
    # prose: the same function the stable test calls raises on this state.
    with pytest.raises(AssertionError, match=r"Mellor & Yamada \(1982\) Eq\. \(34\)-\(35\)"):
        _assert_the_published_stability_functions(computed_m, computed_h, case)


def test_the_published_oracle_catches_a_coefficient_no_icon_derived_test_can_see(
    *, backend: gtx_typing.Backend | None
) -> None:
    """The test has teeth: '(d_5 - d_4)' written as 'd_5' moves 'S_M' by 15.2 per cent.

    'broken_stencils.compute_stability_lengths_with_the_scalar_buoyancy_cofactor_undiminished'
    drops the '- d_4' from the scalar equation's buoyancy cofactor. In MY82's variables the
    coefficient of 'G_H' in Eq. (34) becomes '3*A2*B2 + 18*A1*A2' where the paper has
    '3*A2*B2 + 12*A1*A2': one of the two length scales enters with one and a half times its
    published weight.

    WHY IT IS THIS MUTATION. It is the one shape of error the rest of the package is
    structurally unable to see. 'd_5' reaches ICON only through this coefficient and through
    'rim', which the stencil takes as a separate argument, so 'sm_0', 'sh_0', 'c_tke', 'c_m' and
    'rim' are all unchanged and no oracle built out of 'TurbulenceParams' moves at all. It
    multiplies 'gh' and nothing else, so at 'Ri = 0' it is exactly the identity -- asserted
    below -- and every neutral test in this directory passes it. The determinant and both
    numerators stay positive, so the standard branch is still selected and no structural check
    fires. Only the paper says it is wrong.
    """
    case = _stratified_case(np.array(STABLE_BUOYANCY), np.array(DIMENSIONLESS_SHEAR))
    band = _band(case)
    broken_m, broken_h = _run_stability_lengths(
        case,
        backend,
        program=broken_stencils.compute_stability_lengths_with_the_scalar_buoyancy_cofactor_undiminished,
    )
    length = case["master_length_scale"][:, band]
    mutated_m = broken_m[:, band] / length
    mutated_h = broken_h[:, band] / length

    # The mutation is invisible at neutral, which is why it needs a stratified oracle. Row 0 of
    # the band is 'gh = 0'; there it is bit-for-bit the faithful result.
    faithful_m, faithful_h = _run_stability_lengths(case, backend)
    neutral = case["buoyancy"][:, band][:, 0] == 0.0
    assert neutral.all(), "the first row of the band is no longer the neutral one"
    for mutated, faithful, name in (
        (mutated_m, faithful_m, "momentum"),
        (mutated_h, faithful_h, "scalar"),
    ):
        np.testing.assert_array_equal(
            mutated[:, 0],
            faithful[:, band][:, 0] / length[:, 0],
            err_msg=(
                f"the mutation is visible in the {name} stability function at 'Ri = 0', so it "
                f"is not the blind spot this test claims to close and the neutral tests would "
                f"already have caught it"
            ),
        )

    # And it is grossly visible under stratification.
    stratified = case["buoyancy"][:, band] > 0.0
    worst = np.abs(
        (mutated_m[stratified] - case["predicted_momentum"][:, band][stratified])
        / case["predicted_momentum"][:, band][stratified]
    ).max()
    assert worst > 0.1, (
        f"the mutation moves 'S_M' away from the published solution by only {worst}; it was "
        f"0.152, and too small a move makes this a weak demonstration"
    )
    assert (
        np.abs(
            (mutated_h[stratified] - case["predicted_scalar"][:, band][stratified])
            / case["predicted_scalar"][:, band][stratified]
        ).max()
        > 0.1
    ), "the scalar stability function was not moved, so only the momentum branch is constrained"

    # The invariant itself, not a restatement of it: the assertion the passing test makes must
    # raise when handed the mutated stability lengths.
    with pytest.raises(AssertionError, match=r"Mellor & Yamada \(1982\) Eq\. \(34\)-\(35\)"):
        _assert_the_published_stability_functions(broken_m, broken_h, case)
