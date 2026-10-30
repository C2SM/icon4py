# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The ComponentState adapter of the surface component."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from icon4py.model.common import dimension as dims
from icon4py.model.common.components import component_state


if TYPE_CHECKING:
    from icon4py.model.common.grid import base as base_grid


class State(component_state.ComponentState):
    """Maps the PhysicsState to the input names of the surface component.

    The surface scheme sees the lowest level of the atmosphere: both inputs are views of the
    physics state's diagnostics, nothing is copied and nothing is owned.
    """

    def __init__(self, *, grid: base_grid.Grid) -> None:
        self._lowest_level = dims.KDim(grid.num_levels - 1)
        self._surface_level = dims.KHalfDim(grid.num_levels)

    def as_component_input(self, state: Any) -> dict[str, Any]:
        """The surface pressure and the air temperature at the lowest level, as views."""
        return {
            "surface_pressure": state.diagnostics.pressure_ifc[self._surface_level],
            "temperature": state.diagnostics.temperature[self._lowest_level],
        }
