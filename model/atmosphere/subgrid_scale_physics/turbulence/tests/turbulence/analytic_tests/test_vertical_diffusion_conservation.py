# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The column budget of the semi-implicit vertical diffusion, on a state built from nothing.

WHAT IS UNDER TEST. 'solve_vertical_diffusion_equation' takes a matrix it did not build, so a
conservation statement about it is only meaningful together with the programs that build one.
This file therefore runs the whole 'vertdiff' matrix chain -- 'utils.DiffusionRun', which is
'compute_implicit_diffusion_momentum', 'subtract_implicit_diffusion_momentum',
'compute_explicit_flux_density', 'add_implicit_surface_flux_to_the_explicit_flux_density',
'compute_inverted_diffusion_momentum', 'invert_diffusion_momentum_at_the_surface_flux_level',
'compute_diffusion_inversion_factor', 'compute_diffusion_right_hand_side' and the solve -- on an
idealized column, and asks what came out of the column.

THE DISCRETE SYSTEM. With main levels 'k = 0..nlev-1', flux level 'k' just above concentration
level 'k', and 'F(k)' positive UPWARD, the scheme solves

    A c_new = rhs,
    A(k,k)   = disc_mom(k) + impl_mom(k) + impl_mom(k+1),   A(k,k+-1) = -impl_mom(.),
    rhs(k)   = disc_mom(k)*c_old(k) + F(k+1) - F(k),        F(0) = 0.

Summing the 'nlev' equations makes every interior 'impl_mom' cancel between the two rows it
appears in, and telescopes the explicit fluxes to their surface value. What survives is

    SUM_k disc_mom(k)*(c_new(k) - c_old(k)) = e*(c_surf - c_old(nlev-1))
                                              + impl_mom(nlev)*(c_surf - c_new(nlev-1))

with 'e = expl_mom(nlev) - impl_mom(nlev)'. 'disc_mom' is 'rho*dz/dt', so the left side is the
rate of change of the column-integrated content per unit area, and the right side is the
semi-implicit surface flux: its explicit part evaluated at the OLD profile and its implicit
part at the NEW one. Nothing else enters. That is the conservation statement, and it is exact
for BOTH lower boundary conditions -- the flux condition of the scalars, where 'impl_mom(nlev)'
is absent from the matrix and the second term vanishes, and the concentration condition of the
two wind components, where it is not.

ZERO FLUX IS REACHABLE AND IS NOT WHAT THE STENCIL HARD-CODES. The upper boundary is a
structural zero: 'compute_explicit_flux_density' writes row 0 as an explicit 0.0 and there is
no 'impl_mom(0)'. The LOWER boundary is not zero and cannot be made zero by a row selection --
it is a value, 'expl_mom' on the surface flux level, which
'compute_surface_diffusion_momentum_and_depth' builds from the surface transfer velocity. Set
it to zero and the budget closes exactly; that is the first test below, and it is a genuine
no-exchange boundary rather than a contrivance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from icon4py.model.testing.fixtures.datatest import backend

from . import broken_stencils, utils


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing


#: The residual of the column budget, relative to the L1 scale of the column content, that a
#: correct Thomas solve leaves behind. Measured over the three cases below and every backend
#: run: the worst is of order 1e-16, i.e. one rounding of the summation, so 1e-12 is four
#: orders of slack and still four orders below the smallest defect this file mutates in.
CONSERVATION_TOLERANCE = 1.0e-12


def _assert_the_budget_closes(run: utils.DiffusionRun) -> None:
    """The invariant itself, factored out so the mutation test can be seen to violate IT.

    Both the passing tests and the broken-variant test go through this function, so the
    statement the mutation breaks is literally the statement the port is held to.
    """
    residual = run.residual()
    assert np.all(residual < CONSERVATION_TOLERANCE), (
        f"the column budget does not close: worst relative residual {residual.max()}, "
        f"tolerance {CONSERVATION_TOLERANCE}."
    )


def test_a_zero_surface_flux_conserves_the_column_content(
    *, backend: gtx_typing.Backend | None
) -> None:
    """With no flux through either boundary, the diffusion moves mass around and creates none.

    THE PHYSICS. Vertical diffusion is a flux-divergence process: it can only redistribute a
    conserved quantity inside the column. Close both boundaries -- the model top is closed by
    construction, the surface by setting the surface diffusion momentum to zero, which is
    exactly no turbulent exchange with the ground -- and the mass-weighted column integral of
    the diffused variable must come out of the solve unchanged, to rounding.

    This is the statement the whole implicit machinery has to respect and the one a wrong
    matrix, a wrong boundary row or a wrong solve all break. It is checked here at a surface
    momentum of exactly zero, so the right-hand side of the budget is an exact zero and the
    test is not comparing two large numbers that happen to be close.
    """
    column = utils.construct_idealized_diffusion_column(
        surface=utils.SurfaceCondition.FLUX, surface_diffusion_momentum=0.0
    )
    run = utils.DiffusionRun(column, backend)

    assert np.all(run.surface_flux() == 0.0), (
        "the surface flux is not exactly zero, so this test is not measuring a closed column."
    )
    assert np.all(np.isfinite(run.updated_profile[:, : column.nlev])), (
        "the solve produced non-finite values; the idealized matrix is singular."
    )
    _assert_the_budget_closes(run)


@pytest.mark.parametrize(
    "surface",
    [utils.SurfaceCondition.FLUX, utils.SurfaceCondition.CONCENTRATION],
    ids=lambda s: s.value,
)
def test_the_column_content_changes_by_exactly_the_surface_flux(
    surface: utils.SurfaceCondition, *, backend: gtx_typing.Backend | None
) -> None:
    """With the surface open, the column gains exactly what crosses the surface, and nothing else.

    THE PHYSICS. The same budget as the closed column, with the right-hand side no longer zero.
    A scheme can fail this in two ways that a closed column cannot see: it can lose the surface
    flux, and it can apply the wrong one. Both are the interesting failures, because the surface
    flux is the whole point of the boundary layer scheme -- the diffusion exists to move the
    surface fluxes of heat, moisture and momentum into the atmosphere.

    BOTH LOWER BOUNDARY CONDITIONS ARE CHECKED, and the flux they conserve is the same
    expression. Under the FLUX condition -- every scalar, 'tdc%lsflcnd = .TRUE.' -- there is no
    implicit momentum at the surface flux level and the whole flux is explicit. Under the
    CONCENTRATION condition -- the two wind components, "Attention: no lower flux condition for
    momentum!" -- the surface value is prescribed, part of the flux is implicit, and the budget
    closes only when that part is evaluated at the NEW lowest-level value. Getting that wrong is
    a defect in 'u_tens' and 'v_tens' with nothing upstream of it disagreeing.

    Note that the surface flux uses 'expl_mom(nlev) - impl_mom(nlev)' and NOT the surface row of
    the reduced momentum: 'subtract_implicit_diffusion_momentum' stops one row short of the
    surface, so under a concentration condition ICON's 'expl_mom' at that row still holds the
    full momentum and the implicit part is added back through the flux density instead.
    """
    column = utils.construct_idealized_diffusion_column(
        surface=surface, surface_diffusion_momentum=5.0
    )
    run = utils.DiffusionRun(column, backend)

    flux = run.surface_flux()
    assert np.all(np.abs(flux) > 0.0), (
        "the surface flux vanished, so this test degenerates into the closed-column one."
    )
    assert np.all(np.abs(flux) > 1.0e-6 * run.scale()), (
        "the surface flux is negligible against the column content, so the budget identity "
        "would hold even if the flux term were dropped entirely."
    )
    _assert_the_budget_closes(run)


@pytest.mark.parametrize(
    "surface",
    [utils.SurfaceCondition.FLUX, utils.SurfaceCondition.CONCENTRATION],
    ids=lambda s: s.value,
)
def test_a_misindexed_back_substitution_breaks_the_budget(
    surface: utils.SurfaceCondition, *, backend: gtx_typing.Backend | None
) -> None:
    """The budget test has teeth: a back substitution off by one level fails it loudly.

    'broken_stencils.solve_vertical_diffusion_equation_with_a_misindexed_back_substitution'
    reads 'invs_fac' at its own level where the Fortran reads it one level down. The result is
    still finite, still smooth and still of the right magnitude -- it is simply not a solution
    of the tridiagonal system, so the summed equations no longer telescope.

    The failure is asserted to be LARGE, not merely above the tolerance: a mutation that showed
    up at 1e-11 would leave the tolerance doing the work, and the tolerance is set from
    rounding rather than from the smallest defect anybody cares about.
    """
    column = utils.construct_idealized_diffusion_column(
        surface=surface, surface_diffusion_momentum=5.0
    )
    correct = utils.DiffusionRun(column, backend)
    broken = utils.DiffusionRun(
        column,
        backend,
        solve=broken_stencils.solve_vertical_diffusion_equation_with_a_misindexed_back_substitution,
    )

    assert np.all(np.isfinite(broken.updated_profile[:, : column.nlev])), (
        "the broken solve produced non-finite values, so the budget residual below would be a "
        "NaN rather than a measurement of the defect."
    )
    _assert_the_budget_closes(correct)

    broken_residual = broken.residual()
    assert np.all(broken_residual > 1.0e-4), (
        f"the misindexed back substitution still closes the column budget to "
        f"{broken_residual.max()}, so the conservation test does not constrain the solve."
    )
    # And the invariant itself, not a restatement of it.
    with pytest.raises(AssertionError, match="the column budget does not close"):
        _assert_the_budget_closes(broken)
