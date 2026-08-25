# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, Protocol


class ComponentState(Protocol):
    """Per-process state adapter: the second of the two physics state layers.

    Binds the frozen ``EntryState`` facade of the PhysicsState layer (the first
    layer, ``icon4py.model.atmosphere.subgrid_scale_physics.physics_driver.
    physics_state``) and maps it to this component's input contract, adding any
    process-specific derived inputs. Stateless beyond the bindings: the
    component's outputs are routed by the ``PhysicsDriver`` into the PhysicsState
    layer's sinks (tendency accumulators / diagnostics store) — a ComponentState
    stores nothing and never writes to the model state.

    ``entry_state`` is typed ``Any`` to keep ``common`` decoupled from the
    ``physics_driver`` package that defines the facade.
    """

    def collect_inputs(self, entry_state: Any) -> None: ...
    def as_component_input(self) -> dict[str, Any]: ...
