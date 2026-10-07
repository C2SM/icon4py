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
from typing import TYPE_CHECKING, Any

from icon4py.model.common.components import framework as fw, quantities as qty
from icon4py.model.common.states.tracer_states import TracerConfig
from icon4py.model.common.utils import PredictorCorrectorPair, data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid


__all__ = [
    "Diagnostics",
    "DiffusionDiagnostics",
    "DycoreDiagnostics",
    "DycoreForcing",
    "PrepAdvection",
    "PrognosticState",
    "StepInfo",
    "TracerConfig",
    "TracerState",
]


class PrognosticState(fw.State):
    """ICON's t_nh_prog without the tracers: advanced once per dynamics substep."""

    rho: fw.Field[qty.RhoOnCellK]
    w: fw.Field[qty.WOnCellKHalf]
    vn: fw.Field[qty.VnOnEdgeK]
    exner: fw.Field[qty.ExnerOnCellK]
    theta_v: fw.Field[qty.ThetaVOnCellK]


class TracerState(fw.State):
    """
    The tracers of ICON's t_nh_prog, advanced once per model time step.

    The leaves are optional: `TracerConfig` says which tracers are active, the others are None.
    """

    qv: fw.Field[qty.QvOnCellK] | None = None
    qc: fw.Field[qty.QcOnCellK] | None = None
    qi: fw.Field[qty.QiOnCellK] | None = None
    qr: fw.Field[qty.QrOnCellK] | None = None
    qs: fw.Field[qty.QsOnCellK] | None = None
    qg: fw.Field[qty.QgOnCellK] | None = None

    def copy(self, allocator: gtx_typing.Allocator | None = None) -> TracerState:
        """A new state with a copy of each active tracer, for the other time level."""
        copies: dict[str, Any] = {
            declaration.name: fw.Field(
                declaration.quantity, data_alloc.reallocate(field.data, allocator=allocator)
            )
            for declaration, field in self.leaves()
        }
        return TracerState(**copies)


class PrepAdvection(fw.State):
    """The fluxes the dycore accumulates over the substeps for the tracer advection."""

    vn_traj: fw.Field[qty.VnOnEdgeK]
    mass_flx_me: fw.Field[qty.MassFluxOnEdgeK]
    dynamical_vertical_mass_flux_at_cells_on_half_levels: fw.Field[qty.MassFluxOnCellKHalf]
    dynamical_vertical_volumetric_flux_at_cells_on_half_levels: fw.Field[
        qty.VolumetricFluxOnCellKHalf
    ]


class DycoreForcing(fw.State):
    """
    The tendencies and increments the dycore reads but does not produce.

    The lateral boundary tendencies (grf_tend_*), the slow physics tendencies (ddt_*_phy) and
    the increments of the incremental analysis update (*_incr), filled by the prescribed
    tendencies, the physics or the initial condition.
    """

    exner_tendency_due_to_slow_physics: fw.Field[qty.ExnerTendencyDueToSlowPhysicsOnCellK]
    normal_wind_tendency_due_to_slow_physics_process: fw.Field[
        qty.NormalWindTendencyDueToSlowPhysicsOnEdgeK
    ]
    grf_tend_rho: fw.Field[qty.GrfTendencyOfRhoOnCellK]
    grf_tend_thv: fw.Field[qty.GrfTendencyOfThetaVOnCellK]
    grf_tend_w: fw.Field[qty.GrfTendencyOfWOnCellKHalf]
    grf_tend_vn: fw.Field[qty.GrfTendencyOfVnOnEdgeK]
    rho_iau_increment: fw.Field[qty.RhoIauIncrementOnCellK]
    normal_wind_iau_increment: fw.Field[qty.NormalWindIauIncrementOnEdgeK]
    exner_iau_increment: fw.Field[qty.ExnerIauIncrementOnCellK]


class DycoreDiagnostics(fw.State):
    """
    The diagnostics the dycore carries from one substep to the next, owned by the composer.

    The advective tendencies are predictor/corrector pairs the composer swaps between the
    substeps (`_update_time_levels_for_velocity_tendencies` in the driver); the initial
    condition fills the perturbed exner function and, on a restart, the pairs. Built by
    `allocate`, not by `framework.allocate`, because of the pairs.
    """

    perturbed_exner_at_cells_on_model_levels: fw.Field[qty.PerturbedExnerOnCellK]
    exner_dynamical_increment: fw.Field[qty.ExnerDynamicalIncrementOnCellK]
    normal_wind_advective_tendency: PredictorCorrectorPair[
        fw.Field[qty.NormalWindAdvectiveTendencyOnEdgeK]
    ]
    vertical_wind_advective_tendency: PredictorCorrectorPair[
        fw.Field[qty.VerticalWindAdvectiveTendencyOnCellKHalf]
    ]

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
        )


class DiffusionDiagnostics(fw.State):
    """The diagnostics of the diffusion (turbulence fields of ICON's t_nh_diag), owned by the composer."""

    hdef_ic: fw.Field[qty.HorizontalWindDeformationOnCellKHalf]
    div_ic: fw.Field[qty.DivergenceOnCellKHalf]
    dwdx: fw.Field[qty.ZonalGradientOfWOnCellKHalf]
    dwdy: fw.Field[qty.MeridionalGradientOfWOnCellKHalf]


class Diagnostics(fw.State):
    """ICON's t_nh_diag fields the physics reads and the output writes."""

    pressure: fw.Field[qty.PressureOnCellK]
    pressure_ifc: fw.Field[qty.PressureOnCellKHalf]
    temperature: fw.Field[qty.TemperatureOnCellK]
    virtual_temperature: fw.Field[qty.VirtualTemperatureOnCellK]
    u: fw.Field[qty.UOnCellK]
    v: fw.Field[qty.VOnCellK]


@dataclasses.dataclass(frozen=True, kw_only=True)
class StepInfo:
    """The driver's time variables of one step; plain values, not quantities."""

    dtime: float
    substep_dtime: float
    ndyn_substeps: int
    step_index: int
    simulation_time: datetime.datetime
