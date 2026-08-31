# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""What 'smooth_tke_forcing_vertically' does to a profile it should be able to reproduce.

THIS IS THE ONE STENCIL OF THE PACKAGE WITH NO ICON ORACLE, AND THIS FILE IS ITS ONLY
DATA-FREE ONE. 'exp.mch_icon-ch2_small' runs at 'frcsmot = 0.0' and its 'trop_mask' is
identically zero at all 8276 computed columns, so no capture made from it exercises
'vert_smooth' at any 'frcsmot' -- Switzerland is not in the tropics. What existed before this
file was a numpy transcription of the 24 Fortran lines and a set of structural properties
measured on capture inputs ('integration_tests/test_turbdiff_section_2c.py'). A transcription
can share a misreading with the port; the properties below cannot, and they need no capture at
all.

WHAT THE OPERATOR ACTUALLY IS. Writing 's' for the per-column 'versmot = frcsmot*trop_mask' and
'dm' for the discretisation momentum, on half levels 0..'nlev':

    row 0            f(0)                                                   copied
    row 1            (1-s) *f(1)      + s* f(2)*dm(2)/dm(1)                 one-sided
    rows 2..nlev-2   (1-2s)*f(k)      + s*(f(k-1)*dm(k-1) + f(k+1)*dm(k+1))/dm(k)
    row nlev-1       (1-s) *f(nlev-1) + s* f(nlev-2)*dm(nlev-2)/dm(nlev-1)  one-sided
    row nlev         f(nlev)                                                copied

Every neighbour weight carries a MASS RATIO. That is what makes this a conservative operator
rather than an averaging one, and it is also what the three tests below separate:

  - the COLUMN sums of the weight matrix are one, for any 'dm' and any 's'. That is the
    mass-weighted conservation the datatest already measures on capture inputs.
  - the ROW sums are one only where 'dm(k-1) + dm(k+1) = 2*dm(k)' and 'dm(2) = dm(1)' and
    'dm(nlev-2) = dm(nlev-1)', i.e. on a UNIFORM grid. A constant profile therefore survives a
    uniform grid and does not survive a stretched one.

THE PORT'S OWN DOCSTRING GETS THAT DISTINCTION WRONG, and it is worth being precise about
because the sentence reads like a proof: 'smooth_tke_forcing_vertically.py' says the one-sided
ends are "what makes the smoothing conservative: the weights of each row still sum to one,
since the missing neighbour's weight is the one that is added back to the row itself". The
conclusion is right and the reason is not. The row weights of an interior row sum to
'1 - 2s + s*(dm(k-1) + dm(k+1))/dm(k)', which is one only on a uniform grid; what the one-sided
ends buy is the COLUMN sum, which is the conservation statement. See
'test_a_constant_profile_does_not_survive_a_stretched_grid'.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.smooth_tke_forcing_vertically import (
    smooth_tke_forcing_vertically,
)
from icon4py.model.testing.fixtures.datatest import backend

from . import broken_stencils, utils


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing


#: The 'frcsmot' values the tests sweep. 0.0 is what the reference capture ran and makes the
#: operator the identity; 0.2 is what 28 top-level 'exp.*' configurations set, 13 of them MCH
#: operational and 7 DWD; 0.5 and 1.0 are outside any configuration and are here because a
#: partition of unity is a partition of unity at every weight, and a defect that cancels at
#: small 's' need not cancel at large 's'.
SMOOTHING_FACTORS = (0.0, 0.2, 0.5, 1.0)


def _smooth(
    state: utils.SmoothingState,
    backend: gtx_typing.Backend | None,
    program=smooth_tke_forcing_vertically,
) -> np.ndarray:
    """Run one smoothing pass over the whole column and return the result as a host array."""
    smoothed = utils.as_cell_k_field(np.full_like(state.forcing, np.nan), backend)
    program.with_backend(backend)(
        tke_forcing=utils.as_cell_k_field(state.forcing, backend),
        discretisation_momentum=utils.as_cell_k_field(state.discretisation_momentum, backend),
        smoothing_mask=utils.as_cell_field(state.mask, backend),
        smoothing_weight=state.weight,
        nlev=gtx.int32(state.nlev),
        smoothed_tke_forcing=smoothed,
        horizontal_start=gtx.int32(0),
        horizontal_end=gtx.int32(state.num_cells),
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(state.nlev + 1),
        offset_provider=utils.KOFF,
    )
    return smoothed.asnumpy()


def _assert_a_constant_is_preserved(smoothed: np.ndarray, state: utils.SmoothingState) -> None:
    """The invariant itself, factored out so the mutation test can be seen to violate IT.

    A broken-variant test that restates the invariant in its own words proves only that the two
    statements disagree. This function is the one the passing test calls and the one the failing
    test is asserted to raise from, so there is no second version to drift.
    """
    l1, linf = utils.relative_errors(smoothed, state.forcing)
    assert linf < 1.0e-15, (
        f"a constant profile was not reproduced at frcsmot={state.weight}: relative L-inf "
        f"{linf}, L1 {l1}. On a uniform grid every row weight sums to one, so this is exact up "
        f"to rounding."
    )


@pytest.mark.uses_concat_where
@pytest.mark.parametrize("weight", SMOOTHING_FACTORS)
def test_a_constant_profile_survives_a_uniform_grid(
    weight: float, *, backend: gtx_typing.Backend | None
) -> None:
    """A partition of unity returns a constant unchanged, at any smoothing factor.

    THE PHYSICS. 'vert_smooth' redistributes a forcing between neighbouring half levels; it is
    not allowed to create or destroy any of it, and a profile with no vertical structure has
    nothing to redistribute. If a constant came back changed, the scheme would be manufacturing
    a TKE source out of the smoothing itself -- everywhere, in every column, at every timestep
    a tropical configuration runs.

    Asserted at 'frcsmot = 1.0' as well as at the operational 0.2, because the identity is
    exact in the weights and not asymptotic in them. At 1.0 the interior coefficient '1 - 2s'
    is -1, so the operator is not a positive average any more and the profile is still
    reproduced: that is the statement being made, and it is stronger than the operational one.
    """
    state = utils.construct_idealized_smoothing_state(
        grid=utils.VerticalGrid.UNIFORM, profile=utils.VerticalProfile.CONSTANT, weight=weight
    )
    _assert_a_constant_is_preserved(_smooth(state, backend), state)


@pytest.mark.uses_concat_where
@pytest.mark.parametrize("weight", [0.2, 1.0])
def test_a_constant_profile_does_not_survive_a_stretched_grid(
    weight: float, *, backend: gtx_typing.Backend | None
) -> None:
    """A FINDING, not a defect: the smoothing is mass-conservative, not a partition of unity.

    THE PHYSICS. The neighbour weights carry 'dm(k')/dm(k)', so what the operator redistributes
    is the forcing TIMES the layer mass, not the forcing. On a stretched grid a constant
    forcing is not a constant mass-weighted forcing, and it is the latter that is conserved. So
    'vert_smooth' does change a constant profile, by

        out(k) - c = c * s * ( (dm(k-1) + dm(k+1))/dm(k) - 2 )      interior
        out(1) - c = c * s * (  dm(2)/dm(1) - 1 )                   one-sided top
        out(n-1)-c = c * s * (  dm(n-2)/dm(n-1) - 1 )               one-sided bottom

    and the deviation is asserted here EXACTLY, level by level, rather than merely observed to
    be non-zero. That is what turns "the invariant fails" into a statement about the scheme.

    WHETHER IT MATTERS. It is not obviously wrong -- a mass-weighted redistribution is the
    right thing for a source term that is integrated over a layer -- but it is not what the
    stencil's own docstring claims, and on ICON's grid the effect is not small: the
    discretisation momentum changes by tens of per cent between adjacent half levels in the
    stratosphere, which at 'frcsmot = 0.2' moves a uniform forcing by a few per cent per pass.
    A reviewer should know that 'vert_smooth' is not an averaging operator.
    """
    state = utils.construct_idealized_smoothing_state(
        grid=utils.VerticalGrid.STRETCHED, profile=utils.VerticalProfile.CONSTANT, weight=weight
    )
    smoothed = _smooth(state, backend)
    predicted = state.forcing * state.row_weight_sums()

    deviation = np.abs(smoothed - state.forcing).max()
    assert deviation > 1.0e-3 * np.abs(state.forcing).max(), (
        "the stretched grid did not change the constant profile, so this test is not measuring "
        "what it claims to; the grid in 'construct_idealized_smoothing_state' is too mild."
    )
    l1, linf = utils.relative_errors(smoothed, predicted)
    assert linf < 1.0e-14, (
        f"the deviation from the constant is not the row-weight defect: relative L-inf {linf}, "
        f"L1 {l1}. The operator redistributes 'f*dm', so a constant 'f' is only preserved "
        f"where 'dm' is uniform, and the size of the failure is predictable."
    )


@pytest.mark.uses_concat_where
@pytest.mark.parametrize("weight", [0.2, 0.5])
def test_a_linear_profile_survives_the_interior_but_not_the_one_sided_ends(
    weight: float, *, backend: gtx_typing.Backend | None
) -> None:
    """The interior stencil is second order; the two one-sided rows are not, by exactly 's*b'.

    THE PHYSICS. A three-point average with equal neighbour weights reproduces any profile
    whose second difference vanishes, so on a uniform grid the interior of 'vert_smooth' leaves
    a linear forcing alone -- it is a discrete Laplacian damping and a linear profile has no
    curvature to damp. The end rows reach one neighbour instead of two, so they see a one-sided
    difference instead of a centred one and shift the profile toward that neighbour:

        out(1)      = f(1)      + s*b
        out(nlev-1) = f(nlev-1) - s*b        for f(k) = a + b*k

    This is asserted rather than tolerated. It is a real property of the operator, it is first
    order at the two ends, and it means the smoothing has a systematic sign at the boundaries
    of the smoothed range -- upward at the top and downward at the bottom -- rather than being
    a pure diffusion. On a physical profile whose forcing has a strong vertical gradient near
    the model top, which is where the stratospheric TKE forcing does, that is a bias and not a
    smoothing.
    """
    slope = -0.7
    state = utils.construct_idealized_smoothing_state(
        grid=utils.VerticalGrid.UNIFORM,
        profile=utils.VerticalProfile.LINEAR,
        weight=weight,
        slope=slope,
    )
    smoothed = _smooth(state, backend)
    nlev = state.nlev

    interior = slice(2, nlev - 1)
    _, linf = utils.relative_errors(smoothed[:, interior], state.forcing[:, interior])
    assert linf < 1.0e-14, (
        f"the interior of the smoothing does not reproduce a linear profile (relative L-inf "
        f"{linf}), so it is not second order."
    )

    for row, sign, name in ((1, +1.0, "the top"), (nlev - 1, -1.0, "the bottom")):
        expected = state.forcing[:, row] + sign * state.smoothing * slope
        np.testing.assert_allclose(
            smoothed[:, row],
            expected,
            rtol=1.0e-14,
            atol=0.0,
            err_msg=(
                f"the one-sided row at {name} does not shift a linear profile by exactly "
                f"'{'+' if sign > 0 else '-'}s*b'; the deviation of a first-order end row is "
                f"predictable and this is not it."
            ),
        )

    for row, name in ((0, "the model top"), (nlev, "the surface half level")):
        np.testing.assert_array_equal(
            smoothed[:, row],
            state.forcing[:, row],
            err_msg=f"{name} is copied through by the port and was modified",
        )


@pytest.mark.uses_concat_where
@pytest.mark.parametrize("weight", [0.2, 1.0])
def test_two_sided_end_rows_would_not_preserve_a_constant(
    weight: float, *, backend: gtx_typing.Backend | None
) -> None:
    """The constant-preservation test has teeth: the obvious mistranslation breaks it.

    'broken_stencils.smooth_tke_forcing_vertically_with_two_sided_end_rows' writes the interior
    coefficient '1 - 2s' on the two one-sided rows, which is what folding the Fortran's three
    statements into one produces. On a uniform grid the two end rows then return

        (1 - 2s)*c + s*c = (1 - s)*c

    while every interior row is still bit-exact against the real stencil. So the defect is
    invisible to a comparison of the interior, invisible to a magnitude check, and caught by
    this file's first test -- which is the reason that test is worth running.

    The 'mask = 0' column is asserted to be UNAFFECTED. There the smoothing vanishes and the
    two stencils are the same computation, so a mutation test that did not separate the columns
    could pass on a stencil that ignored its per-column weight entirely.
    """
    state = utils.construct_idealized_smoothing_state(
        grid=utils.VerticalGrid.UNIFORM, profile=utils.VerticalProfile.CONSTANT, weight=weight
    )
    broken = _smooth(
        state,
        backend,
        program=broken_stencils.smooth_tke_forcing_vertically_with_two_sided_end_rows,
    )
    nlev = state.nlev

    smoothing_vanishes = state.smoothing == 0.0
    assert smoothing_vanishes.any() and not smoothing_vanishes.all(), (
        "the synthetic mask no longer spans zero and non-zero, so this test cannot separate "
        "the identity columns from the smoothed ones."
    )

    for row in (1, nlev - 1):
        expected = state.forcing[:, row] * (1.0 - state.smoothing)
        np.testing.assert_allclose(
            broken[:, row],
            expected,
            rtol=1.0e-14,
            atol=0.0,
            err_msg=(
                f"the two-sided end row at {row} did not return '(1-s)*c'; the mutation is not "
                f"doing what this test assumes and the invariant it is supposed to break is "
                f"therefore untested."
            ),
        )
        assert np.all(broken[smoothing_vanishes, row] == state.forcing[smoothing_vanishes, row]), (
            "the broken stencil changed a column whose smoothing weight is zero, so it differs "
            "from the real one for a reason other than the end-row coefficient."
        )

    interior = slice(2, nlev - 1)
    np.testing.assert_array_equal(
        broken[:, interior],
        _smooth(state, backend)[:, interior],
        err_msg=(
            "the mutation changed the interior as well, so the constant-preservation test "
            "could be catching it there rather than at the ends."
        ),
    )

    # And the invariant itself, not a restatement of it: the assertion that the first test in
    # this file makes must raise when handed the broken profile.
    with pytest.raises(AssertionError, match="a constant profile was not reproduced"):
        _assert_a_constant_is_preserved(broken, state)
