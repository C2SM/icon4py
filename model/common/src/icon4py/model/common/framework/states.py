# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The states the driver passes between the components."""

from __future__ import annotations

import dataclasses
import datetime
from collections.abc import Iterator
from typing import TYPE_CHECKING, Final

from icon4py.model.common import field, framework as fw
from icon4py.model.common.framework import quantities as qty
from icon4py.model.common.utils import PredictorCorrectorPair


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid


class PrognosticState(fw.State):
    """ICON's t_nh_prog without the tracers: advanced once per dynamics substep."""

    rho: field.Field[qty.RhoOnCellK]
    w: field.Field[qty.WOnCellKHalf]
    vn: field.Field[qty.VnOnEdgeK]
    exner: field.Field[qty.ExnerOnCellK]
    theta_v: field.Field[qty.ThetaVOnCellK]


class TracerState(fw.State):
    """
    The tracers of ICON's t_nh_prog, advanced once per model time step.

    The leaves are optional: `TracerConfig` says which tracers are active, the others are None.
    """

    qv: field.Field[qty.QvOnCellK] | None = None
    qc: field.Field[qty.QcOnCellK] | None = None
    qi: field.Field[qty.QiOnCellK] | None = None
    qr: field.Field[qty.QrOnCellK] | None = None
    qs: field.Field[qty.QsOnCellK] | None = None
    qg: field.Field[qty.QgOnCellK] | None = None


#: the tracer names, in ICON's order (QV=0, QC=1, QI=2, QR=3, QS=4, QG=5)
TRACERS: Final[tuple[str, ...]] = ("qv", "qc", "qi", "qr", "qs", "qg")


@dataclasses.dataclass(frozen=True)
class TracerConfig:
    """
    Which tracers are active in the model configuration.

    Each boolean field indicates whether the corresponding tracer is active.
    Used instead of a raw ``ntracer: int`` to provide type-safe tracer selection.
    """

    qv: bool = False
    qc: bool = False
    qi: bool = False
    qr: bool = False
    qs: bool = False
    qg: bool = False

    @classmethod
    def all(cls) -> TracerConfig:
        return cls(qv=True, qc=True, qi=True, qr=True, qs=True, qg=True)

    @classmethod
    def none(cls) -> TracerConfig:
        return cls()

    @classmethod
    def from_ntracer(cls, ntracer: int) -> TracerConfig:
        """
        Build a ``TracerConfig`` from a Fortran ``ntracer`` count.

        Fortran ICON uses a fixed tracer ordering: QV=0, QC=1, QI=2, QR=3, QS=4, QG=5.
        The first *ntracer* entries in this order are considered active.

        Raises:
            ValueError: if ntracer is outside the valid range.
        """
        n = len(TRACERS)
        if not 0 <= ntracer <= n:
            raise ValueError(f"ntracer must be between 0 and {n}, got {ntracer}")
        return cls(**{name: i < ntracer for i, name in enumerate(TRACERS)})

    @property
    def nactive(self) -> int:
        return sum(dataclasses.asdict(self).values())

    @property
    def active_names(self) -> tuple[str, ...]:
        return tuple(name for name in TRACERS if getattr(self, name))

    def __iter__(self) -> Iterator[str]:
        return iter(self.active_names)

    def __len__(self) -> int:
        return self.nactive

    def __contains__(self, name: str) -> bool:
        return name in TRACERS and getattr(self, name)

    def __str__(self) -> str:
        names = ", ".join(self.active_names)
        return names if names else "none"


class PrepAdvection(fw.State):
    """The fluxes the dycore accumulates over the substeps for the tracer advection."""

    # TODO(jcanton, msimberg): part 02 fix this. If vn_traj keeps sharing VnOnEdgeK with vn, `collect`
    # either raises AmbiguousSource or quietly picks the wrong buffer. Discuss whether we should
    # define a new quantity.
    vn_traj: field.Field[qty.VnOnEdgeK]
    mass_flx_me: field.Field[qty.HorizontalMassFluxOnEdgeK]
    dynamical_vertical_mass_flux_at_cells_on_half_levels: field.Field[
        qty.VerticalMassFluxOnCellKHalf
    ]
    dynamical_vertical_volumetric_flux_at_cells_on_half_levels: field.Field[
        qty.VerticalVolumetricFluxOnCellKHalf
    ]


class DycoreForcing(fw.State):
    """
    The tendencies and increments the dycore reads but does not produce.

    The lateral boundary tendencies (grf_tend_*), the slow physics tendencies (ddt_*_phy) and
    the increments of the incremental analysis update (*_incr), filled by the prescribed
    tendencies, the physics or the initial condition.
    """

    exner_tendency_due_to_slow_physics: field.Field[qty.ExnerTendencyDueToSlowPhysicsOnCellK]
    normal_wind_tendency_due_to_slow_physics_process: field.Field[
        qty.NormalWindTendencyDueToSlowPhysicsOnEdgeK
    ]
    grf_tend_rho: field.Field[qty.GrfTendencyOfRhoOnCellK]
    grf_tend_thv: field.Field[qty.GrfTendencyOfThetaVOnCellK]
    grf_tend_w: field.Field[qty.GrfTendencyOfWOnCellKHalf]
    grf_tend_vn: field.Field[qty.GrfTendencyOfVnOnEdgeK]
    rho_iau_increment: field.Field[qty.RhoIauIncrementOnCellK]
    normal_wind_iau_increment: field.Field[qty.NormalWindIauIncrementOnEdgeK]
    exner_iau_increment: field.Field[qty.ExnerIauIncrementOnCellK]


class DycoreDiagnostics(fw.State):
    """
    The diagnostics the dycore keeps from one substep to the next, owned by the driver.

    The advective tendencies are predictor/corrector pairs the driver swaps between the
    substeps (`_update_time_levels_for_velocity_tendencies`); the initial condition fills the
    perturbed exner function and, on a restart, the pairs. The dycore raises `max_vertical_cfl`
    over the substeps; the driver reads and resets it. The interpolated winds, densities and
    fluxes are ICON's `t_nh_diag` fields: the next substep reads some of them (`vt`, `w_concorr_c`,
    `rho_ic`), ICON outside the dycore the others. Built by `allocate`, not by
    `framework.allocate`, because of the pairs.
    """

    perturbed_exner_at_cells_on_model_levels: field.Field[qty.PerturbedExnerOnCellK]
    exner_dynamical_increment: field.Field[qty.ExnerDynamicalIncrementOnCellK]
    normal_wind_advective_tendency: PredictorCorrectorPair[
        field.Field[qty.NormalWindAdvectiveTendencyOnEdgeK]
    ]
    vertical_wind_advective_tendency: PredictorCorrectorPair[
        field.Field[qty.VerticalWindAdvectiveTendencyOnCellKHalf]
    ]
    max_vertical_cfl: field.Field[qty.MaxVerticalCflNumber]
    tangential_wind: field.Field[qty.TangentialWindOnEdgeK]
    vn_on_half_levels: field.Field[qty.VnOnEdgeKHalf]
    contravariant_correction_at_cells_on_half_levels: field.Field[
        qty.ContravariantCorrectionOnCellKHalf
    ]
    theta_v_at_cells_on_half_levels: field.Field[qty.ThetaVOnCellKHalf]
    rho_at_cells_on_half_levels: field.Field[qty.RhoOnCellKHalf]
    mass_flux_at_edges_on_model_levels: field.Field[qty.HorizontalMassFluxOnEdgeK]

    @classmethod
    def allocate(
        cls, grid: base_grid.Grid, allocator: gtx_typing.Allocator | None
    ) -> DycoreDiagnostics:
        return cls(
            perturbed_exner_at_cells_on_model_levels=fw.zeros(
                qty.PerturbedExnerOnCellK, grid, allocator
            ),
            exner_dynamical_increment=fw.zeros(qty.ExnerDynamicalIncrementOnCellK, grid, allocator),
            normal_wind_advective_tendency=PredictorCorrectorPair(
                fw.zeros(qty.NormalWindAdvectiveTendencyOnEdgeK, grid, allocator),
                fw.zeros(qty.NormalWindAdvectiveTendencyOnEdgeK, grid, allocator),
            ),
            vertical_wind_advective_tendency=PredictorCorrectorPair(
                fw.zeros(qty.VerticalWindAdvectiveTendencyOnCellKHalf, grid, allocator),
                fw.zeros(qty.VerticalWindAdvectiveTendencyOnCellKHalf, grid, allocator),
            ),
            max_vertical_cfl=fw.zeros(qty.MaxVerticalCflNumber, grid, allocator),
            tangential_wind=fw.zeros(qty.TangentialWindOnEdgeK, grid, allocator),
            vn_on_half_levels=fw.zeros(qty.VnOnEdgeKHalf, grid, allocator),
            contravariant_correction_at_cells_on_half_levels=fw.zeros(
                qty.ContravariantCorrectionOnCellKHalf, grid, allocator
            ),
            theta_v_at_cells_on_half_levels=fw.zeros(qty.ThetaVOnCellKHalf, grid, allocator),
            rho_at_cells_on_half_levels=fw.zeros(qty.RhoOnCellKHalf, grid, allocator),
            mass_flux_at_edges_on_model_levels=fw.zeros(
                qty.HorizontalMassFluxOnEdgeK, grid, allocator
            ),
        )


class DiffusionDiagnostics(fw.State):
    """The diagnostics of the diffusion (turbulence fields of ICON's t_nh_diag), owned by the driver."""

    hdef_ic: field.Field[qty.HorizontalWindDeformationOnCellKHalf]
    div_ic: field.Field[qty.DivergenceOnCellKHalf]
    dwdx: field.Field[qty.ZonalGradientOfWOnCellKHalf]
    dwdy: field.Field[qty.MeridionalGradientOfWOnCellKHalf]


class AdvectionDiagnostics(fw.State):
    """The diagnostics of the tracer advection, owned by the driver."""

    airmass_now: field.Field[qty.AirMassOnCellK]
    airmass_new: field.Field[qty.AirMassOnCellK]
    grf_tend_tracer: field.Field[qty.GrfTendencyOfTracerOnCellK]
    hfl_tracer: field.Field[qty.HorizontalTracerFluxOnEdgeK]
    vfl_tracer: field.Field[qty.VerticalTracerFluxOnCellKHalf]


class Diagnostics(fw.State):
    """ICON's t_nh_diag fields the physics reads and the output writes."""

    pressure: field.Field[qty.PressureOnCellK]
    pressure_ifc: field.Field[qty.PressureOnCellKHalf]
    temperature: field.Field[qty.TemperatureOnCellK]
    virtual_temperature: field.Field[qty.VirtualTemperatureOnCellK]
    u: field.Field[qty.UOnCellK]
    v: field.Field[qty.VOnCellK]


@dataclasses.dataclass(frozen=True, kw_only=True)
class StepInfo:
    """The driver's time variables of one step; plain values, not quantities."""

    dtime: float
    substep_dtime: float
    ndyn_substeps: int
    step_index: int
    simulation_time: datetime.datetime
