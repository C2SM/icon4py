# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.definitions import SPECIES
from icon4py.model.common.components.component_state import ComponentState
from icon4py.model.common.metrics import metrics_attributes


if TYPE_CHECKING:
    from icon4py.model.common.states import factory


class State(ComponentState):
    """The muphys ComponentState adapter.

    Maps the frozen ``EntryState`` facade of the PhysicsState layer to the muphys
    Component contract. The layer already diagnoses everything muphys consumes
    (T, p) and points at the rest (rho, tracers); the only process-owned input is
    ``dz``, fetched once from the metrics source. Stateless beyond the bindings —
    muphys's outputs (tendencies, precip diagnostics) are routed by the driver
    into the PhysicsState layer's sinks.
    """

    def __init__(self, *, metrics: factory.FieldSource) -> None:
        self.dz = metrics.get(metrics_attributes.DDQZ_Z_FULL)
        self._entry: Any = None

    def collect_inputs(self, entry_state: Any) -> None:
        self._entry = entry_state

    def as_component_input(self) -> dict[str, Any]:
        """The 10 muphys input fields, mapped from the facade (no copies)."""
        entry = self._entry
        if entry is None:
            raise RuntimeError("as_component_input called before collect_inputs")
        return {
            "dz": self.dz,
            "te": entry.diagnostics.temperature,
            "p": entry.diagnostics.pressure,
            "rho": entry.rho,
            **{f"q{s}": getattr(entry.tracers, f"q{s}") for s in SPECIES},
        }
