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
This file therefore runs the whole 'vertdiff' matrix chain -- 'compute_implicit_diffusion_
momentum', 'subtract_implicit_diffusion_momentum', 'compute_explicit_flux_density',
'add_implicit_surface_flux_to_the_explicit_flux_density', 'compute_inverted_diffusion_momentum',
'invert_diffusion_momentum_at_the_surface_flux_level', 'compute_diffusion_inversion_factor',
'compute_diffusion_right_hand_side' and the solve -- on an idealized column, and asks what came
out of the column.

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

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.add_implicit_surface_flux_to_the_explicit_flux_density import (
    add_implicit_surface_flux_to_the_explicit_flux_density,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_diffusion_inversion_factor import (
    compute_diffusion_inversion_factor,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_diffusion_right_hand_side import (
    compute_diffusion_right_hand_side,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_explicit_flux_density import (
    compute_explicit_flux_density,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_implicit_diffusion_momentum import (
    compute_implicit_diffusion_momentum,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_inverted_diffusion_momentum import (
    compute_inverted_diffusion_momentum,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.invert_diffusion_momentum_at_the_surface_flux_level import (
    invert_diffusion_momentum_at_the_surface_flux_level,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.solve_vertical_diffusion_equation import (
    solve_vertical_diffusion_equation,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.subtract_implicit_diffusion_momentum import (
    subtract_implicit_diffusion_momentum,
)
from icon4py.model.testing.fixtures.datatest import backend

from . import broken_stencils, utils


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing


#: The residual of the column budget, relative to the L1 scale of the column content, that a
#: correct Thomas solve leaves behind. Measured over the three cases below and every backend
#: run: the worst is of order 1e-16, i.e. one rounding of the summation, so 1e-12 is four
#: orders of slack and still four orders below the smallest defect this file mutates in.
CONSERVATION_TOLERANCE = 1.0e-12


class DiffusionRun:
    """One idealized column carried through the whole 'vertdiff' matrix chain.

    Holds the host arrays the budget is formed from. Constructing it runs nine programs on the
    backend under test, which is what makes the assertions statements about the port rather
    than about numpy.
    """

    def __init__(
        self,
        column: utils.DiffusionColumn,
        backend: gtx_typing.Backend | None,
        solve=solve_vertical_diffusion_equation,
    ) -> None:
        nlev = column.nlev
        rows = nlev + 1
        cells = column.num_cells
        bounds = {"horizontal_start": gtx.int32(0), "horizontal_end": gtx.int32(cells)}
        flux_condition = column.surface is utils.SurfaceCondition.FLUX

        discretisation_momentum = utils.as_cell_k_field(column.discretisation_momentum, backend)
        current_profile = utils.as_cell_k_field(column.current_profile, backend)
        # 'expl_mom' on entry: ICON reduces it in place, so the port is handed one field and
        # the surface row is the one it does not reduce. Two fields here, because the test also
        # needs the unreduced value to form the surface flux.
        full_momentum = utils.as_cell_k_field(column.diffusion_momentum, backend)
        explicit_momentum = utils.as_cell_k_field(column.diffusion_momentum, backend)
        implicit_momentum = utils.as_cell_k_field(np.zeros((cells, rows)), backend)
        explicit_flux = utils.as_cell_k_field(np.zeros((cells, rows)), backend)
        inverted_momentum = utils.as_cell_k_field(np.zeros((cells, rows)), backend)
        inversion_factor = utils.as_cell_k_field(np.zeros((cells, rows)), backend)
        right_hand_side = utils.as_cell_k_field(np.zeros((cells, rows)), backend)
        updated_profile = utils.as_cell_k_field(np.full((cells, rows), np.nan), backend)

        # 'DO k = k_tp+2, k_sf+1-m': to the surface flux level under a concentration condition,
        # one row short of it under a flux condition, where the surface value is not an unknown
        # and there is no sub-diagonal to it.
        compute_implicit_diffusion_momentum.with_backend(backend)(
            diffusion_momentum=full_momentum,
            implicit_weight=utils.as_k_field(column.implicit_weight, backend),
            implicit_diffusion_momentum=implicit_momentum,
            vertical_start=gtx.int32(1),
            vertical_end=gtx.int32(nlev if flux_condition else nlev + 1),
            offset_provider={},
            **bounds,
        )
        # One row short of the split above: the surface flux level keeps the WHOLE diffusion
        # momentum, which is what makes its explicit flux the explicit SURFACE flux.
        subtract_implicit_diffusion_momentum.with_backend(backend)(
            diffusion_momentum=full_momentum,
            implicit_diffusion_momentum=implicit_momentum,
            explicit_diffusion_momentum=explicit_momentum,
            vertical_start=gtx.int32(1),
            vertical_end=gtx.int32(nlev),
            offset_provider={},
            **bounds,
        )
        compute_explicit_flux_density.with_backend(backend)(
            explicit_diffusion_momentum=explicit_momentum,
            current_profile=current_profile,
            model_top_level=gtx.int32(0),
            explicit_flux_density=explicit_flux,
            vertical_start=gtx.int32(1),
            vertical_end=gtx.int32(nlev + 1),
            offset_provider=utils.KOFF,
            **bounds,
        )
        if not flux_condition:
            add_implicit_surface_flux_to_the_explicit_flux_density.with_backend(backend)(
                explicit_flux_density_at_the_surface=explicit_flux,
                implicit_diffusion_momentum=implicit_momentum,
                current_profile=current_profile,
                explicit_flux_density=explicit_flux,
                vertical_start=gtx.int32(nlev),
                vertical_end=gtx.int32(nlev + 1),
                offset_provider=utils.KOFF,
                **bounds,
            )
        compute_inverted_diffusion_momentum.with_backend(backend)(
            discretisation_momentum=discretisation_momentum,
            implicit_diffusion_momentum=implicit_momentum,
            inverted_diffusion_momentum=inverted_momentum,
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(nlev - 1 if flux_condition else nlev),
            offset_provider=utils.KOFF,
            **bounds,
        )
        if flux_condition:
            invert_diffusion_momentum_at_the_surface_flux_level.with_backend(backend)(
                discretisation_momentum=discretisation_momentum,
                implicit_diffusion_momentum=implicit_momentum,
                inverted_diffusion_momentum_above=inverted_momentum,
                inverted_diffusion_momentum=inverted_momentum,
                vertical_start=gtx.int32(nlev - 1),
                vertical_end=gtx.int32(nlev),
                offset_provider=utils.KOFF,
                **bounds,
            )
        compute_diffusion_inversion_factor.with_backend(backend)(
            inverted_diffusion_momentum=inverted_momentum,
            implicit_diffusion_momentum=implicit_momentum,
            inversion_factor=inversion_factor,
            vertical_start=gtx.int32(1),
            vertical_end=gtx.int32(nlev),
            offset_provider=utils.KOFF,
            **bounds,
        )
        compute_diffusion_right_hand_side.with_backend(backend)(
            discretisation_momentum=discretisation_momentum,
            current_profile=current_profile,
            explicit_flux_density=explicit_flux,
            right_hand_side=right_hand_side,
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(nlev),
            offset_provider=utils.KOFF,
            **bounds,
        )
        solve.with_backend(backend)(
            right_hand_side=right_hand_side,
            implicit_diffusion_momentum=implicit_momentum,
            inverted_diffusion_momentum=inverted_momentum,
            inversion_factor=inversion_factor,
            updated_profile=updated_profile,
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(nlev),
            offset_provider=utils.KOFF,
            **bounds,
        )

        self.column = column
        self.implicit_momentum = implicit_momentum.asnumpy()
        self.explicit_flux = explicit_flux.asnumpy()
        self.updated_profile = updated_profile.asnumpy()

    def mass_change(self) -> np.ndarray:
        """'SUM_k disc_mom(k)*(c_new(k) - c_old(k))' [kg/m2/s per unit of c], per column."""
        nlev = self.column.nlev
        disc = self.column.discretisation_momentum[:, :nlev]
        return (
            disc * (self.updated_profile[:, :nlev] - self.column.current_profile[:, :nlev])
        ).sum(axis=1)

    def surface_flux(self) -> np.ndarray:
        """The semi-implicit surface flux, positive upward, per column.

        Formed from the SCHEME's own quantities and the solution, not from the stencils: this
        is the right-hand side of the budget identity and computing it with the programs under
        test would make the identity circular.
        """
        nlev = self.column.nlev
        surface_value = self.column.current_profile[:, nlev]
        lowest_old = self.column.current_profile[:, nlev - 1]
        lowest_new = self.updated_profile[:, nlev - 1]
        implicit = self.implicit_momentum[:, nlev]
        explicit = self.column.diffusion_momentum[:, nlev] - implicit
        return explicit * (surface_value - lowest_old) + implicit * (surface_value - lowest_new)

    def scale(self) -> np.ndarray:
        """The L1 scale of the column content, which the residual is judged against."""
        nlev = self.column.nlev
        return np.abs(
            self.column.discretisation_momentum[:, :nlev] * self.column.current_profile[:, :nlev]
        ).sum(axis=1)

    def residual(self) -> np.ndarray:
        return np.abs(self.mass_change() - self.surface_flux()) / self.scale()


def _assert_the_budget_closes(run: DiffusionRun) -> None:
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
    run = DiffusionRun(column, backend)

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
    run = DiffusionRun(column, backend)

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
    correct = DiffusionRun(column, backend)
    broken = DiffusionRun(
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
