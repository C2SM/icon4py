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
    """Adapter from a driver-owned state to one component's input contract.

    A driver owns the state its components share; each component consumes its own
    subset of that state, under its own argument names, plus any input only it
    derives. ``as_component_input`` is that translation: it takes the shared state
    and returns this component's input mapping, deriving on the way whatever only
    this component needs.

    The driver calls it once per step on which the component actually computes, so
    a derivation placed here never runs for a step whose result is discarded.
    """

    def as_component_input(self, state: Any) -> dict[str, Any]: ...
