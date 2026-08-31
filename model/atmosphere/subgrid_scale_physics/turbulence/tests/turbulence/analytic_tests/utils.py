# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Idealized states for the analytic tests, and the error measures they are judged by.

Every state here is constructed, not read: a column with a chosen vertical grid, a chosen
profile and chosen closure constants, so that the answer follows from the equations rather
than from a capture. The shape of this module -- 'construct_idealized_*' builders, an enum per
initial condition, a frozen configuration dataclass validated in '__post_init__', and
'relative_errors' unwrapping a GT4Py field on the way in -- is taken from the advection
convergence study ('model/atmosphere/advection/tests/advection_tests/utils.py' on the
'advection_convergence' branch, commit d2dc1c00f).

THE VERTICAL INDEXING OF THIS SCHEME, once, because all three test modules need it. 'nlev' is
ICON's 'ke', the number of main levels; the fields are 'nlev + 1' rows deep. Half levels are
rows 0..'nlev', row 'nlev' being the surface. Main levels are rows 0..'nlev'-1, plus row 'nlev'
which carries the surface boundary value of a diffused variable. A FLUX level with index 'k'
lies just ABOVE the concentration level 'k' (turb_diffusion.f90:2413-2417), so flux level 0 is
the model top and flux level 'nlev' is the surface flux level.

ONE STATEMENT AT A TIME. Where a section has been merged into a single '@gtx.program', the
analytic tests still need its statements one at a time -- to iterate a pair to a fixed point,
and to substitute a deliberately defective copy for one of them. 'single_statement_stencils'
holds those one-statement programs; see its docstring.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import TYPE_CHECKING

import gt4py.next as gtx
import numpy as np

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
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.turbulence import (
    TurbulenceConfig,
    TurbulenceParams,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat

from . import single_statement_stencils


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing


__all__ = [
    "DiffusionColumn",
    "DiffusionRun",
    "SmoothingState",
    "SurfaceCondition",
    "TurbulenceState",
    "VerticalGrid",
    "VerticalProfile",
    "advance_the_turbulence_state",
    "as_cell_field",
    "as_cell_k_field",
    "as_k_field",
    "closure_constants",
    "construct_idealized_diffusion_column",
    "construct_idealized_smoothing_state",
    "construct_idealized_turbulence_state",
    "construct_uniform_diffusion_column",
    "relative_errors",
    "stability_length_arguments",
]


#: The offset provider every vertically staggered program of this package needs.
KOFF = {dims.Koff.value: dims.KDim}

#: The largest implicit weight an idealized column may carry. NOT one: ICON's own weight at the
#: surface is 'impl_s = 1.20' (mo_turbdiff_config.f90:126), deliberately OVER-implicit, and
#: 'mo_nwp_phy_init.f90:1538-1547' ramps 'impl_weight' from 'impl_t = 0.75' at 1500 m to that
#: value on the surface flux level. A theta-scheme with 'theta > 1' is still unconditionally
#: stable and monotone -- 'test_vertical_diffusion_amplification.py' asserts what it costs
#: instead: the amplification factor tends to '(theta-1)/theta' rather than to zero, so the
#: stiffest modes are not annihilated. The bound here is twice 'impl_t' and exists only to catch
#: a typo, not to express a scheme limit.
LARGEST_IMPLICIT_WEIGHT = 1.5


class VerticalGrid(enum.Enum):
    """How the layer masses vary with height.

    UNIFORM is the only grid on which a mass-weighted three-point smoothing is a partition of
    unity; STRETCHED is what an atmospheric model actually has, and telling the two apart is
    the point of 'test_vertical_smoothing_weights.py'.
    """

    #: Every layer carries the same mass; the ratios the weights divide by are all one.
    UNIFORM = "uniform"
    #: Layer mass grows downward and the growth rate itself varies, so no two neighbouring
    #: ratios coincide and a defect cannot cancel between rows.
    STRETCHED = "stretched"


class VerticalProfile(enum.Enum):
    """The initial condition of the smoothed quantity."""

    #: A single value at every level: the partition-of-unity probe.
    CONSTANT = "constant"
    #: Affine in the level index: the second-order probe.
    LINEAR = "linear"
    #: Pseudo-random, for the invariants that hold for any profile.
    RANDOM = "random"


class SurfaceCondition(enum.Enum):
    """The lower boundary condition of the vertical diffusion, ICON's 'lsflucond'.

    'vertdiff' runs the two wind components under CONCENTRATION ("Attention: no lower flux
    condition for momentum!", turb_vertdiff.f90:585-586) and every scalar under FLUX
    ('tdc%lsflcnd', frozen '.TRUE.'). The two differ by exactly one row of the matrix, and the
    conserved quantity is the same for both -- see 'test_vertical_diffusion_conservation.py'.
    """

    #: 'lsflucond = .TRUE.': the surface concentration is not an unknown and the last diffused
    #: level has no sub-diagonal below it.
    FLUX = "flux"
    #: 'lsflucond = .FALSE.': the surface concentration is prescribed and couples implicitly to
    #: the lowest main level.
    CONCENTRATION = "concentration"


# ------------------------------------------------------------------------- field allocation ---


def as_cell_k_field(values: np.ndarray, backend: gtx_typing.Backend | None) -> gtx.Field:
    """A '(CellDim, KDim)' field on the backend under test, from a host array."""
    return gtx.as_field(
        (dims.CellDim, dims.KDim), np.asarray(values, dtype=wpfloat), allocator=backend
    )


def as_cell_field(values: np.ndarray, backend: gtx_typing.Backend | None) -> gtx.Field:
    """A '(CellDim,)' field on the backend under test, from a host array."""
    return gtx.as_field((dims.CellDim,), np.asarray(values, dtype=wpfloat), allocator=backend)


def as_k_field(values: np.ndarray, backend: gtx_typing.Backend | None) -> gtx.Field:
    """A '(KDim,)' field on the backend under test, from a host array."""
    return gtx.as_field((dims.KDim,), np.asarray(values, dtype=wpfloat), allocator=backend)


# ------------------------------------------------------------------------------ error norms ---


def relative_errors(values: gtx.Field | np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """The L1 and L-infinity errors of 'values' against 'reference', each normalised.

    Both are divided by the corresponding norm of the reference, so the result is dimensionless
    and comparable between a forcing in 1/s2 and a length in metres. A GT4Py field is unwrapped
    on the way in, which is what lets a call site pass the output of a program directly and
    keeps every test running unchanged on a device backend.

    The reference must not be identically zero; where an invariant predicts zero, subtract and
    normalise by a scale the test chooses, rather than calling this.
    """
    computed = np.asarray(values.asnumpy() if hasattr(values, "asnumpy") else values)
    reference = np.asarray(reference)
    assert computed.shape == reference.shape, (
        f"shape mismatch: computed {computed.shape}, reference {reference.shape}"
    )
    difference = np.abs(computed - reference)
    l1_scale = np.abs(reference).sum()
    linf_scale = np.abs(reference).max()
    assert l1_scale > 0.0 and linf_scale > 0.0, (
        "the reference is identically zero, so a relative error is not defined"
    )
    return float(difference.sum() / l1_scale), float(difference.max() / linf_scale)


# ----------------------------------------------------------------- the smoothing test state ---


@dataclasses.dataclass(frozen=True)
class SmoothingState:
    """Everything 'smooth_tke_forcing_vertically' needs, plus the host arrays to predict with.

    'smoothing' is the per-column 'versmot = frcsmot*trop_mask' the stencil forms internally
    from 'weight' and 'mask'; it is carried here so a test can write its prediction column by
    column without repeating the product.
    """

    #: 'cur_tend' on entry, '(num_cells, nlev + 1)' host array.
    forcing: np.ndarray
    #: 'disc_mom', the same shape; strictly positive on every row the stencil reads.
    discretisation_momentum: np.ndarray
    #: 'trop_mask', one value per column in [0, 1].
    mask: np.ndarray
    #: 'frcsmot'.
    weight: float
    #: 'versmot', 'weight * mask'.
    smoothing: np.ndarray
    #: 'ke': the row of the surface half level, and the depth of the field minus one.
    nlev: int

    @property
    def num_cells(self) -> int:
        return int(self.forcing.shape[0])

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Refuse a state whose weights the stencil would not be defined on.

        The same discipline 'TurbulenceConfig._validate' applies to the real configuration.
        Each of these has a silent failure mode: a non-positive 'disc_mom' on a row the stencil
        divides by turns the whole test into a measurement of a division by zero, and a weight
        outside [0, 1] is a smoothing 'TurbulenceConfig' would itself refuse.
        """
        rows = self.nlev + 1
        if self.forcing.shape != (self.num_cells, rows):
            raise ValueError(
                f"Invalid argument 'forcing': should have shape {(self.num_cells, rows)}, "
                f"got {self.forcing.shape}."
            )
        if self.discretisation_momentum.shape != self.forcing.shape:
            raise ValueError(
                f"Invalid argument 'discretisation_momentum': should have the shape of the "
                f"forcing, {self.forcing.shape}, got {self.discretisation_momentum.shape}."
            )
        if not np.all(self.discretisation_momentum > 0.0):
            raise ValueError(
                "Invalid argument 'discretisation_momentum': the smoothing divides by it on "
                "every row it writes and reads it on both neighbours, so it must be positive "
                "throughout."
            )
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError(
                f"Invalid argument 'weight': should be a smoothing fraction in [0, 1], "
                f"got {self.weight}."
            )
        if not np.all((self.mask >= 0.0) & (self.mask <= 1.0)):
            raise ValueError(
                f"Invalid argument 'mask': 'trop_mask' is a fraction in [0, 1], got {self.mask}."
            )
        if self.nlev < 4:
            raise ValueError(
                f"Invalid argument 'nlev': the smoothing has two copied rows, two one-sided "
                f"rows and an interior, so it needs at least four half levels, got {self.nlev}."
            )

    def row_weight_sums(self) -> np.ndarray:
        """The sum of the three stencil weights of each row: what a CONSTANT is multiplied by.

        Read off 'smooth_tke_forcing_vertically' directly. Rows 0 and 'nlev' are copied through,
        so their weight is one by construction; row 1 and row 'nlev'-1 are one-sided; the rest
        are the interior form. The neighbour weights carry the mass ratio 'dm(k')/dm(k)', which
        is what makes this differ from one on a stretched grid.
        """
        s = self.smoothing[:, np.newaxis]
        dm = self.discretisation_momentum
        sums = np.ones_like(self.forcing)
        interior = slice(2, self.nlev - 1)
        sums[:, 1] = (1.0 - self.smoothing) + self.smoothing * dm[:, 2] / dm[:, 1]
        sums[:, interior] = (1.0 - 2.0 * s) + s * (
            dm[:, 1 : self.nlev - 2] + dm[:, 3 : self.nlev]
        ) / dm[:, interior]
        sums[:, self.nlev - 1] = (1.0 - self.smoothing) + self.smoothing * dm[
            :, self.nlev - 2
        ] / dm[:, self.nlev - 1]
        return sums


def construct_idealized_smoothing_state(
    *,
    num_cells: int = 5,
    nlev: int = 12,
    grid: VerticalGrid = VerticalGrid.UNIFORM,
    profile: VerticalProfile = VerticalProfile.CONSTANT,
    weight: float = 0.2,
    constant: float = 2.5,
    slope: float = -0.7,
    seed: int = 20260831,
) -> SmoothingState:
    """Build a smoothing state from nothing.

    The mask spans the whole range 'trop_mask' can take, one value per column, exactly as the
    datatest's synthetic mask does: a uniform mask could not distinguish a stencil that
    broadcast the per-column weight wrongly from one that did not, and column 0 at 'mask = 0'
    carries the identity case for free.

    'weight' defaults to the 0.2 that 28 top-level 'exp.*' configurations set, including 13 MCH
    and 7 DWD operational setups. It is NOT the Fortran default -- 'mo_turbdiff_config.f90:143'
    defaults 'frcsmot' to 0.0 -- and 0.0 is what the reference capture ran, which is why nothing
    in the capture exercises this routine at all.
    """
    rows = nlev + 1
    levels = np.arange(rows, dtype=wpfloat)

    if grid is VerticalGrid.UNIFORM:
        discretisation_momentum = np.full(rows, 7.0, dtype=wpfloat)
    else:
        # Exponential growth downward, modulated so that no two neighbouring mass ratios are
        # equal: with a purely geometric 'dm' every interior row would have the same weight
        # defect and a per-row error could hide as a global factor.
        discretisation_momentum = 5.0 * np.exp(0.35 * levels) * (1.0 + 0.1 * np.cos(levels))

    if profile is VerticalProfile.CONSTANT:
        column = np.full(rows, constant, dtype=wpfloat)
    elif profile is VerticalProfile.LINEAR:
        column = constant + slope * levels
    else:
        column = np.random.default_rng(seed).normal(scale=constant, size=rows)

    forcing = np.tile(column, (num_cells, 1))
    mask = np.linspace(0.0, 1.0, num_cells, dtype=wpfloat)
    return SmoothingState(
        forcing=forcing,
        discretisation_momentum=np.tile(discretisation_momentum, (num_cells, 1)),
        mask=mask,
        weight=weight,
        smoothing=weight * mask,
        nlev=nlev,
    )


# ----------------------------------------------------------------- the diffusion test state ---


@dataclasses.dataclass(frozen=True)
class DiffusionColumn:
    """One idealized column of the semi-implicit vertical diffusion of a first-order variable.

    Only the quantities the ICON stencils take as INPUTS are stored. Everything the scheme
    derives -- the implicit and explicit split, the flux density, the right-hand side and the
    LU factorisation -- is produced by running the ported programs, because a test that
    reproduced them here would be testing its own arithmetic.
    """

    #: 'disc_mom' = 'rho*dz/dt' [kg/m2/s] on main levels; rows 0..'nlev'-1 are read.
    discretisation_momentum: np.ndarray
    #: 'expl_mom' BEFORE the implicit part is split off, on flux levels 1..'nlev' [kg/m2/s].
    diffusion_momentum: np.ndarray
    #: 'tdc%impl_weight' [-], one value per flux level.
    implicit_weight: np.ndarray
    #: 'cur_prof', the profile the system is built around; row 'nlev' is the surface value.
    current_profile: np.ndarray
    #: Which lower boundary condition the matrix is built for.
    surface: SurfaceCondition
    nlev: int

    @property
    def num_cells(self) -> int:
        return int(self.current_profile.shape[0])

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Refuse a state the invariant would not hold for, rather than measuring a broken one.

        An idealized setup gets the same validation discipline as 'TurbulenceConfig': every one
        of these has a way of being violated silently, and each would surface as a conservation
        residual that looks like a defect in the solver.
        """
        rows = self.nlev + 1
        for name in (
            "discretisation_momentum",
            "diffusion_momentum",
            "current_profile",
        ):
            field = getattr(self, name)
            if field.shape != (self.num_cells, rows):
                raise ValueError(
                    f"Invalid argument '{name}': should have shape {(self.num_cells, rows)}, "
                    f"got {field.shape}."
                )
        if self.implicit_weight.shape != (rows,):
            raise ValueError(
                f"Invalid argument 'implicit_weight': should have shape {(rows,)}, "
                f"got {self.implicit_weight.shape}."
            )
        if not np.all(self.discretisation_momentum[:, : self.nlev] > 0.0):
            raise ValueError(
                "Invalid argument 'discretisation_momentum': every diffused main level needs a "
                "positive layer mass, or the tridiagonal matrix is singular."
            )
        if not np.all(self.diffusion_momentum[:, 1 : self.nlev] > 0.0):
            raise ValueError(
                "Invalid argument 'diffusion_momentum': the interior flux levels need a "
                "positive diffusion momentum, or the system is not diffusive."
            )
        if np.any(self.diffusion_momentum[:, self.nlev] < 0.0):
            raise ValueError(
                "Invalid argument 'diffusion_momentum': the surface flux level may be zero, "
                "which is the no-flux boundary, but not negative."
            )
        if not np.all(
            (self.implicit_weight > 0.0) & (self.implicit_weight <= LARGEST_IMPLICIT_WEIGHT)
        ):
            raise ValueError(
                f"Invalid argument 'implicit_weight': should be an implicit weight in "
                f"]0, {LARGEST_IMPLICIT_WEIGHT}], got {self.implicit_weight}."
            )


def construct_idealized_diffusion_column(
    *,
    num_cells: int = 4,
    nlev: int = 12,
    surface: SurfaceCondition = SurfaceCondition.FLUX,
    surface_diffusion_momentum: float = 5.0,
    implicit_weight: float = 0.7,
    time_step: float = 30.0,
    seed: int = 20260831,
) -> DiffusionColumn:
    """Build a diffusion column from nothing: a stretched grid, a decaying density, a rough profile.

    'surface_diffusion_momentum' is 'expl_mom' on the surface flux level, ICON's 'k_sf' row,
    which 'compute_surface_diffusion_momentum_and_depth' builds out of 'rhon', 'tkv' and the
    surface transfer velocity. Setting it to zero is a genuine NO-FLUX lower boundary -- no
    surface exchange -- and is how the conservation test reaches an exactly closed budget.

    The profile is deliberately rough rather than smooth: a smooth profile makes the flux
    divergences small, and a conservation residual is only informative when the individual
    fluxes are not.
    """
    rows = nlev + 1
    levels = np.arange(rows, dtype=wpfloat)
    rng = np.random.default_rng(seed)

    layer_depth = 20.0 * np.exp(0.15 * levels)
    air_density = 1.2 * np.exp(-0.1 * levels)
    discretisation_momentum = air_density * layer_depth / time_step

    momentum = np.zeros(rows, dtype=wpfloat)
    momentum[1:nlev] = 2.0 + 0.3 * levels[1:nlev]
    momentum[nlev] = surface_diffusion_momentum

    profile = np.empty((num_cells, rows), dtype=wpfloat)
    profile[:, :nlev] = 300.0 + 2.0 * rng.normal(size=(num_cells, nlev))
    profile[:, nlev] = 305.0

    return DiffusionColumn(
        discretisation_momentum=np.tile(discretisation_momentum, (num_cells, 1)),
        diffusion_momentum=np.tile(momentum, (num_cells, 1)),
        implicit_weight=np.full(rows, implicit_weight, dtype=wpfloat),
        current_profile=profile,
        surface=surface,
        nlev=nlev,
    )


def construct_uniform_diffusion_column(
    *,
    profile: np.ndarray,
    diffusivity: float,
    layer_depth: float,
    time_step: float,
    air_density: float = 1.2,
    implicit_weight: float = 0.75,
    surface_diffusion_momentum: float = 0.0,
) -> DiffusionColumn:
    """A column with EQUALLY SPACED levels, one density and one diffusivity, closed at both ends.

    The stretched column of 'construct_idealized_diffusion_column' is what an atmospheric model
    has, and it is the right state for an invariant that holds on any grid. It is the wrong one
    for the two invariants that name the grid: the discrete cosine modes are eigenvectors of the
    one-step operator only when 'disc_mom' and 'expl_mom' are constant in k, and the moment
    identities of the spreading test need a coordinate that is affine in the level index.

    'surface_diffusion_momentum' defaults to zero, a genuine no-exchange lower boundary: with
    the model top closed by construction, the column is then closed at both ends and the
    operator is exactly the zero-flux Laplacian whose eigenvectors are known.

    Args:
        profile: '(num_cells, nlev)' values on the diffused main levels. The surface row of
            'cur_prof' is appended as a zero; nothing reads it while the surface is closed.
        diffusivity: 'tkv', the eddy diffusivity [m2/s], one value for the whole column.
        layer_depth: 'dz' [m], the same for every layer.
        time_step: 'dt' [s].
        air_density: 'rhon' [kg/m3], one value for the whole column.
        implicit_weight: 'tdc%impl_weight' [-], one value for every flux level.
        surface_diffusion_momentum: 'expl_mom' on the surface flux level [kg/m2/s].

    Returns:
        The column, whose diffusion number 'r = K*dt/dz2' is 'diffusion_momentum/
        discretisation_momentum' on every interior flux level.
    """
    cells, nlev = profile.shape
    rows = nlev + 1

    discretisation = np.full((cells, rows), air_density * layer_depth / time_step, dtype=wpfloat)
    momentum = np.zeros((cells, rows), dtype=wpfloat)
    momentum[:, 1:nlev] = air_density * diffusivity / layer_depth
    momentum[:, nlev] = surface_diffusion_momentum

    current = np.zeros((cells, rows), dtype=wpfloat)
    current[:, :nlev] = profile

    return DiffusionColumn(
        discretisation_momentum=discretisation,
        diffusion_momentum=momentum,
        implicit_weight=np.full(rows, implicit_weight, dtype=wpfloat),
        current_profile=current,
        surface=SurfaceCondition.FLUX,
        nlev=nlev,
    )


class DiffusionRun:
    """One idealized column carried through the whole 'vertdiff' matrix chain, once.

    Holds the host arrays the invariants are formed from. Constructing it runs nine programs on
    the backend under test, which is what makes every assertion built on it a statement about
    the port rather than about numpy.

    THE THREE PROGRAM SLOTS ARE THE MUTATION POINTS. Each defaults to the shipped stencil and
    can be replaced by a defective copy from 'broken_stencils', which is how the mutation tests
    reach the same assertion functions the passing tests use. They are the three places a
    plausible transcription error in this chain would land: the implicit/explicit split, the
    flux the split feeds, and the Thomas solve.
    """

    def __init__(
        self,
        column: DiffusionColumn,
        backend: gtx_typing.Backend | None,
        solve=solve_vertical_diffusion_equation,
        implicit_split=compute_implicit_diffusion_momentum,
        explicit_flux_density=compute_explicit_flux_density,
    ) -> None:
        nlev = column.nlev
        rows = nlev + 1
        cells = column.num_cells
        bounds = {"horizontal_start": gtx.int32(0), "horizontal_end": gtx.int32(cells)}
        flux_condition = column.surface is SurfaceCondition.FLUX

        discretisation_momentum = as_cell_k_field(column.discretisation_momentum, backend)
        current_profile = as_cell_k_field(column.current_profile, backend)
        # 'expl_mom' on entry: ICON reduces it in place, so the port is handed one field and
        # the surface row is the one it does not reduce. Two fields here, because the test also
        # needs the unreduced value to form the surface flux.
        full_momentum = as_cell_k_field(column.diffusion_momentum, backend)
        explicit_momentum = as_cell_k_field(column.diffusion_momentum, backend)
        implicit_momentum = as_cell_k_field(np.zeros((cells, rows)), backend)
        explicit_flux = as_cell_k_field(np.zeros((cells, rows)), backend)
        inverted_momentum = as_cell_k_field(np.zeros((cells, rows)), backend)
        inversion_factor = as_cell_k_field(np.zeros((cells, rows)), backend)
        right_hand_side = as_cell_k_field(np.zeros((cells, rows)), backend)
        updated_profile = as_cell_k_field(np.full((cells, rows), np.nan), backend)

        # 'DO k = k_tp+2, k_sf+1-m': to the surface flux level under a concentration condition,
        # one row short of it under a flux condition, where the surface value is not an unknown
        # and there is no sub-diagonal to it.
        implicit_split.with_backend(backend)(
            diffusion_momentum=full_momentum,
            implicit_weight=as_k_field(column.implicit_weight, backend),
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
        explicit_flux_density.with_backend(backend)(
            explicit_diffusion_momentum=explicit_momentum,
            current_profile=current_profile,
            model_top_level=gtx.int32(0),
            explicit_flux_density=explicit_flux,
            vertical_start=gtx.int32(1),
            vertical_end=gtx.int32(nlev + 1),
            offset_provider=KOFF,
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
                offset_provider=KOFF,
                **bounds,
            )
        compute_inverted_diffusion_momentum.with_backend(backend)(
            discretisation_momentum=discretisation_momentum,
            implicit_diffusion_momentum=implicit_momentum,
            inverted_diffusion_momentum=inverted_momentum,
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(nlev - 1 if flux_condition else nlev),
            offset_provider=KOFF,
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
                offset_provider=KOFF,
                **bounds,
            )
        compute_diffusion_inversion_factor.with_backend(backend)(
            inverted_diffusion_momentum=inverted_momentum,
            implicit_diffusion_momentum=implicit_momentum,
            inversion_factor=inversion_factor,
            vertical_start=gtx.int32(1),
            vertical_end=gtx.int32(nlev),
            offset_provider=KOFF,
            **bounds,
        )
        compute_diffusion_right_hand_side.with_backend(backend)(
            discretisation_momentum=discretisation_momentum,
            current_profile=current_profile,
            explicit_flux_density=explicit_flux,
            right_hand_side=right_hand_side,
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(nlev),
            offset_provider=KOFF,
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
            offset_provider=KOFF,
            **bounds,
        )

        self.column = column
        self.implicit_momentum = implicit_momentum.asnumpy()
        self.explicit_flux = explicit_flux.asnumpy()
        self.updated_profile = updated_profile.asnumpy()

    def stepped_column(self) -> DiffusionColumn:
        """The same column with its profile advanced one step, ready for the next.

        The surface row of 'updated_profile' is never written -- it is not an unknown of the
        system -- so it is carried over from the column this run started from rather than
        propagating the 'nan' the output field was filled with.
        """
        nlev = self.column.nlev
        profile = self.column.current_profile.copy()
        profile[:, :nlev] = self.updated_profile[:, :nlev]
        return dataclasses.replace(self.column, current_profile=profile)

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


# ------------------------------------------------------------------ the turbulence closure ---


def closure_constants() -> tuple[TurbulenceConfig, TurbulenceParams]:
    """The shipped configuration and the closure constants 'turb_setup' derives from it."""
    config = TurbulenceConfig()
    return config, TurbulenceParams(config=config)


def stability_length_arguments(
    params: TurbulenceParams, config: TurbulenceConfig
) -> dict[str, float]:
    """The closure constants 'compute_stability_lengths' takes, as the granule binds them."""
    return {
        "a_h": config.a_heat,
        "a_m": config.a_mom,
        "b_h": params.b_h,
        "b_m": params.b_m,
        "d_m": config.d_mom,
        "d_1": params.d_1,
        "d_2": params.d_2,
        "d_3": params.d_3,
        "d_4": params.d_4,
        "d_5": params.d_5,
        "d_6": params.d_6,
        "rim": params.rim,
        "frcsecu": config.frcsecu,
        "stbsecu": config.stbsecu,
    }


@dataclasses.dataclass(frozen=True)
class TurbulenceState:
    """One column of half levels at a PRESCRIBED length scale and prescribed forcings.

    This is the state the TKE loop of 'turbdiff' sections 2c) and 3) advances: given the master
    length scale and the two squared forcing frequencies, the stability functions follow from
    the velocity scale and the velocity scale follows from the stability functions. Prescribing
    the forcings rather than deriving them from a wind and a temperature profile is what makes
    the equilibrium analytic -- the fixed point of that pair is a closed form in the closure
    constants, and it is what 'test_tke_steady_state.py' and 'test_neutral_surface_layer.py'
    assert.

    Rows 1 to 'nlev'-1 are the ones the programs write, mirroring the Fortran 'DO k = k_st,k_en'
    with 'k_st = 2' and 'k_en = ke' (turb_diffusion.f90:1814). Rows 0 and 'nlev' are the model
    top and the surface, which the transfer scheme owns.
    """

    #: 'tls' ('len_scale'), the turbulent master length scale [m], '(num_cells, nlev + 1)'.
    master_length_scale: np.ndarray
    #: 'fm2' ('frm'), the squared frequency of the mechanical forcing [1/s2].
    mechanical_forcing: np.ndarray
    #: 'fh2' ('frh'), the squared frequency of the thermal forcing [1/s2].
    thermal_forcing: np.ndarray
    #: 'tke(:,:,nvor)', the turbulent velocity scale 'q = SQRT(2*TKE)' [m/s].
    velocity_scale: np.ndarray
    #: 'lsm', the master length scale times the momentum stability function [m].
    stability_length_for_momentum: np.ndarray
    #: 'lsh', the same for scalars [m].
    stability_length_for_scalars: np.ndarray

    @property
    def num_cells(self) -> int:
        return int(self.master_length_scale.shape[0])

    @property
    def nlev(self) -> int:
        return int(self.master_length_scale.shape[1]) - 1

    #: The rows the programs write, and therefore the only rows an assertion may read.
    @property
    def interior(self) -> slice:
        return slice(1, self.nlev)

    def diffusion_coefficient_for_momentum(self) -> np.ndarray:
        """'tkvm = q*S_m*l' [m2/s], as section 3) forms it from the two stored fields."""
        return self.velocity_scale * self.stability_length_for_momentum

    def diffusion_coefficient_for_scalars(self) -> np.ndarray:
        """'tkvh = q*S_h*l' [m2/s]."""
        return self.velocity_scale * self.stability_length_for_scalars

    def stability_function_for_momentum(self) -> np.ndarray:
        """'S_m' [-]: the port stores 'l*S_m', so a dimensionless assertion divides it out."""
        return self.stability_length_for_momentum / self.master_length_scale

    def tke_forcing(self) -> np.ndarray:
        """'frc' [m/s2] BEFORE the security clip: shear production less buoyancy destruction."""
        return (
            self.stability_length_for_momentum * self.mechanical_forcing
            - self.stability_length_for_scalars * self.thermal_forcing
        )


def construct_idealized_turbulence_state(
    *,
    master_length_scale: np.ndarray,
    shear: np.ndarray,
    buoyancy_frequency_squared: np.ndarray | None = None,
    velocity_scale: float = 1.0,
    nlev: int = 5,
) -> TurbulenceState:
    """Build a turbulence state from one length scale and one shear per column.

    Every row of a column carries the same state, so the fixed point below is reached by every
    written row independently and a defect confined to one row cannot hide.

    Args:
        master_length_scale: 'tls' [m], one value per column.
        shear: '|S| = SQRT(fm2)' [1/s], one value per column. Squared here rather than by the
            caller, because 'fm2' is what the stencils take and 'S' is what the closed forms
            are written in.
        buoyancy_frequency_squared: 'fh2' [1/s2], one value per column; zero (neutral) if
            omitted.
        velocity_scale: the 'q' [m/s] the iteration starts from, the same everywhere.
        nlev: the row of the surface half level; the fields are 'nlev + 1' rows deep.
    """
    rows = nlev + 1
    length = np.tile(np.asarray(master_length_scale, dtype=wpfloat)[:, np.newaxis], (1, rows))
    mechanical = np.tile(
        (np.asarray(shear, dtype=wpfloat) * np.asarray(shear, dtype=wpfloat))[:, np.newaxis],
        (1, rows),
    )
    if buoyancy_frequency_squared is None:
        thermal = np.zeros_like(mechanical)
    else:
        thermal = np.tile(
            np.asarray(buoyancy_frequency_squared, dtype=wpfloat)[:, np.newaxis], (1, rows)
        )
    _, params = closure_constants()
    return TurbulenceState(
        master_length_scale=length,
        mechanical_forcing=mechanical,
        thermal_forcing=thermal,
        velocity_scale=np.full_like(length, velocity_scale),
        stability_length_for_momentum=length * params.sm_0,
        stability_length_for_scalars=length * params.sh_0,
    )


def advance_the_turbulence_state(
    state: TurbulenceState,
    *,
    backend: gtx_typing.Backend | None,
    time_step: float,
    tkesmot: float,
    stability_program=single_statement_stencils.compute_stability_lengths,
    velocity_program=single_statement_stencils.compute_turbulent_velocity_scale,
) -> TurbulenceState:
    """One pass of 'turbdiff' sections 2c) and 3): stability functions, then the TKE equation.

    The two programs are run in the order and with the row range the granule uses, so what this
    iterates is the scheme's own loop and not a paraphrase of it. Both are mutation points.

    'tkesmot' is an argument rather than being read from the configuration because the two
    invariants want different values of it: the bare backward-Euler step is only visible at
    'tkesmot = 0', and the shipped 0.15 is what the per-step decay law has to account for.
    """
    config, params = closure_constants()
    cells = state.num_cells
    nlev = state.nlev
    bounds = {
        "horizontal_start": gtx.int32(0),
        "horizontal_end": gtx.int32(cells),
        "vertical_start": gtx.int32(1),
        "vertical_end": gtx.int32(nlev),
    }
    length = as_cell_k_field(state.master_length_scale, backend)
    mechanical = as_cell_k_field(state.mechanical_forcing, backend)
    thermal = as_cell_k_field(state.thermal_forcing, backend)

    updated_m = as_cell_k_field(state.stability_length_for_momentum, backend)
    updated_h = as_cell_k_field(state.stability_length_for_scalars, backend)
    stability_program.with_backend(backend)(
        master_length_scale=length,
        stability_length_for_momentum=as_cell_k_field(state.stability_length_for_momentum, backend),
        stability_length_for_scalars=as_cell_k_field(state.stability_length_for_scalars, backend),
        mechanical_forcing=mechanical,
        thermal_forcing=thermal,
        turbulent_velocity_scale=as_cell_k_field(state.velocity_scale, backend),
        updated_stability_length_for_momentum=updated_m,
        updated_stability_length_for_scalars=updated_h,
        offset_provider={},
        **stability_length_arguments(params, config),
        **bounds,
    )
    stability_m = updated_m.asnumpy().copy()
    stability_h = updated_h.asnumpy().copy()

    updated_q = as_cell_k_field(state.velocity_scale, backend)
    velocity_program.with_backend(backend)(
        master_length_scale=length,
        stability_length_for_momentum=as_cell_k_field(stability_m, backend),
        stability_length_for_scalars=as_cell_k_field(stability_h, backend),
        mechanical_forcing=mechanical,
        thermal_forcing=thermal,
        previous_velocity_scale=as_cell_k_field(state.velocity_scale, backend),
        transport_tendency=as_cell_k_field(np.zeros_like(state.velocity_scale), backend),
        d_m=config.d_mom,
        d_4=params.d_4,
        b_m=params.b_m,
        rim=params.rim,
        frcsecu=config.frcsecu,
        tkesecu=config.tkesecu,
        tkesmot=tkesmot,
        vel_min=config.vel_min,
        tke_time_step=time_step,
        inverse_tke_time_step=1.0 / time_step,
        turbulent_velocity_scale=updated_q,
        offset_provider={},
        **bounds,
    )

    return dataclasses.replace(
        state,
        velocity_scale=updated_q.asnumpy().copy(),
        stability_length_for_momentum=stability_m,
        stability_length_for_scalars=stability_h,
    )
