# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The ComponentState adapter of the tmx component."""

from __future__ import annotations

from typing import Any

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import data as tmx_data
from icon4py.model.common.components import component_state


class State(component_state.ComponentState):
    """Maps the PhysicsState to the input names of the tmx component.

    Every input is a field of the facade or, for the surface fluxes, a diagnostic of
    another process read by name; nothing is copied and nothing is owned. The air mass is the one the driver hands to the physics (ICON's
    ``mair``, the dycore's ``airmass_new``), so a facade without it cannot feed tmx.
    """

    def as_component_input(self, state: Any) -> dict[str, Any]:
        """The tmx inputs, mapped from the facade without copies."""
        if state.air_mass is None:
            raise ValueError("tmx needs the air mass per unit area, which the PhysicsState lacks.")
        diagnostics = state.diagnostics
        return {
            "temperature": diagnostics.temperature,
            "virtual_temperature": diagnostics.virtual_temperature,
            "pressure": diagnostics.pressure,
            "u": diagnostics.u,
            "v": diagnostics.v,
            "w": state.w,
            "rho": state.rho,
            **{f"q{s}": getattr(state.tracers, f"q{s}") for s in ("v", "c", "i", "r", "s", "g")},
            "air_mass": state.air_mass,
            **{name: state.diagnostic_output(name) for name in tmx_data.FLUX_PORTS},
        }
