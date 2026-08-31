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
"""

from __future__ import annotations

import dataclasses
import enum
from typing import TYPE_CHECKING

import gt4py.next as gtx
import numpy as np

from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing


__all__ = [
    "DiffusionColumn",
    "SmoothingState",
    "SurfaceCondition",
    "VerticalGrid",
    "VerticalProfile",
    "as_cell_field",
    "as_cell_k_field",
    "as_k_field",
    "construct_idealized_diffusion_column",
    "construct_idealized_smoothing_state",
    "relative_errors",
]


#: The offset provider every vertically staggered program of this package needs.
KOFF = {dims.Koff.value: dims.KDim}


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
        if not np.all((self.implicit_weight > 0.0) & (self.implicit_weight <= 1.0)):
            raise ValueError(
                f"Invalid argument 'implicit_weight': should be an implicit weight in ]0, 1], "
                f"got {self.implicit_weight}."
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
