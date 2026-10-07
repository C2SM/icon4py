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
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing


__all__ = [
    "Diagnostics",
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
