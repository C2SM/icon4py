# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The tmx ComponentState adapter.

Maps the frozen ``EntryState`` facade of the PhysicsState layer to the
:class:`TmxComponent` contract. The layer already diagnoses most tmx inputs
(T, Tv, p, p_ifc, u, v) and points at the rest (w, rho, tracers); this adapter
adds the tmx-specific derived inputs (``air_mass``, ``cv_air``) and the
surface-flux seam. Stateless beyond the bindings — tmx's outputs (tendencies,
km/kh/... diagnostics) are routed by the driver into the PhysicsState layer's
sinks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gt4py.next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import (
    state_stencils,
    surface_fluxes,
    tmx_states,
)
from icon4py.model.common import dimension as dims, model_options
from icon4py.model.common.components.component_state import ComponentState
from icon4py.model.common.metrics import metrics_attributes
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.common.states import factory


class State(ComponentState):
    """The tmx ComponentState adapter: input mapping + derived inputs + flux seam."""

    def __init__(
        self,
        *,
        grid: base_grid.Grid,
        metrics: factory.FieldSource,
        surface_flux_provider: surface_fluxes.SurfaceFluxProvider | None = None,
        backend: gtx_typing.Backend | None = None,
    ) -> None:
        self._ddqz_z_full = metrics.get(metrics_attributes.DDQZ_Z_FULL)
        self._entry: Any = None

        full_horizontal = {
            "horizontal_start": gtx.int32(0),
            "horizontal_end": gtx.int32(grid.num_cells),
        }
        full_vertical = {
            "vertical_start": gtx.int32(0),
            "vertical_end": gtx.int32(grid.num_levels),
        }
        self._compute_air_mass = model_options.setup_program(
            program=state_stencils.compute_air_mass,
            backend=backend,
            horizontal_sizes=full_horizontal,
            vertical_sizes=full_vertical,
            offset_provider={},
        )
        self._compute_cv_air = model_options.setup_program(
            program=state_stencils.compute_cv_air,
            backend=backend,
            horizontal_sizes=full_horizontal,
            vertical_sizes=full_vertical,
            offset_provider={},
        )

        # tmx-specific derived inputs — computed each collect_inputs call
        self.air_mass = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)
        self.cv_air = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)

        # Surface-flux buffers: 2-D (CellDim only), granule *inputs*. The
        # surface-flux provider (phase-2 seam) fills them at the end of every
        # collect_inputs; the granule consumes them via as_component_input.
        self.evapotranspiration = data_alloc.zero_field(grid, dims.CellDim, allocator=backend)
        self.sensible_heat_flux = data_alloc.zero_field(grid, dims.CellDim, allocator=backend)
        self.u_stress = data_alloc.zero_field(grid, dims.CellDim, allocator=backend)
        self.v_stress = data_alloc.zero_field(grid, dims.CellDim, allocator=backend)
        self.q_snocpymlt = data_alloc.zero_field(grid, dims.CellDim, allocator=backend)
        self._surface_flux_provider = surface_flux_provider or surface_fluxes.ZeroFluxProvider()
        self._surface_flux_state = tmx_states.TmxSurfaceFluxState(
            evapotranspiration=self.evapotranspiration,
            sensible_heat_flux=self.sensible_heat_flux,
            u_stress=self.u_stress,
            v_stress=self.v_stress,
            q_snocpymlt=self.q_snocpymlt,
        )

    def collect_inputs(self, entry_state: Any) -> None:
        """Bind the facade and derive the tmx-specific inputs from it."""
        self._entry = entry_state

        # air mass = rho * dz (diag%airmass bound to field%mair in ICON)
        self._compute_air_mass(
            rho=entry_state.rho,
            ddqz_z_full=self._ddqz_z_full,
            air_mass=self.air_mass,
        )
        # moisture-weighted heat capacity per unit area (get_cvair port)
        tracers = entry_state.tracers
        self._compute_cv_air(
            qv=tracers.qv,
            qc=tracers.qc,
            qi=tracers.qi,
            qr=tracers.qr,
            qs=tracers.qs,
            qg=tracers.qg,
            air_mass=self.air_mass,
            cv_air=self.cv_air,
        )
        # Surface fluxes (phase-2 seam): the provider sets every flux, every step
        self._surface_flux_provider.compute(out=self._surface_flux_state)

    def as_component_input(self) -> dict[str, Any]:
        """Return exactly the 21 ``INPUTS_PROPERTIES`` keys mapped to GT4Py fields."""
        entry = self._entry
        if entry is None:
            raise RuntimeError("as_component_input called before collect_inputs")
        return {
            # Diagnosed by the PhysicsState layer
            "temperature": entry.ta,
            "virtual_temperature": entry.tv,
            "pressure": entry.pressure,
            "pressure_ifc": entry.pressure_ifc,
            "u": entry.u,
            "v": entry.v,
            # Model-state pointers via the facade (no copy)
            "w": entry.w,
            "rho": entry.rho,
            **{f"q{s}": getattr(entry.tracers, f"q{s}") for s in "vcirsg"},
            # tmx-specific derived fields
            "air_mass": self.air_mass,
            "cv_air": self.cv_air,
            # Surface-flux seam (filled by the surface-flux provider each collect)
            "evapotranspiration": self.evapotranspiration,
            "sensible_heat_flux": self.sensible_heat_flux,
            "u_stress": self.u_stress,
            "v_stress": self.v_stress,
            "q_snocpymlt": self.q_snocpymlt,
        }
