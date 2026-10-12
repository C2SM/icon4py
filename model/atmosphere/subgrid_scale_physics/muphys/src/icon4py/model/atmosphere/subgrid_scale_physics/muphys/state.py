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
    """Maps the entry state to the input names of the muphys component.

    Every non-static input is taken from the entry state; ``dz`` is the only
    field this adapter owns. It derives nothing and holds nothing else.
    """

    def __init__(self, *, metrics: factory.FieldSource) -> None:
        self.dz = metrics.get(metrics_attributes.DDQZ_Z_FULL)

    def as_component_input(self, state: Any) -> dict[str, Any]:
        """The 10 muphys input fields, mapped from the facade (no copies)."""
        return {
            "dz": self.dz,
            "te": state.diagnostics.temperature,
            "p": state.diagnostics.pressure,
            "rho": state.rho,
            **{f"q{s}": getattr(state.tracers, f"q{s}") for s in SPECIES},
        }
