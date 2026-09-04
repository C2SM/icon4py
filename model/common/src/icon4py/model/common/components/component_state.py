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
    derives. This protocol is that translation: ``collect_inputs`` binds the
    shared state, ``as_component_input`` returns the component's input mapping.

    """

    def collect_inputs(self, entry_state: Any) -> None: ...
    def as_component_input(self) -> dict[str, Any]: ...
