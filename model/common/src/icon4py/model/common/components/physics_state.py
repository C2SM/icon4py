# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Protocol for the L1 physics-state adapter that the L2 PhysicsDriver depends on.

The concrete implementations live with their process (e.g.
``icon4py.model.atmosphere.subgrid_scale_physics.muphys.state.State``); this
module only declares the interface the orchestrator relies on, so the driver
stays decoupled from any specific physics state.
"""

from __future__ import annotations

from typing import Any, Protocol


class PhysicsStateProtocol(Protocol):
    """The slice of a physics state that ``PhysicsDriver`` depends on."""

    def gather_from_prognostic(self, prognostic: Any) -> None: ...
    def as_component_input(self) -> dict[str, Any]: ...
    def scatter_to_prognostic(
        self, prognostic: Any, outputs: dict[str, Any], dt: float
    ) -> None: ...
