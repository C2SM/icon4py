# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""A physics process: a component under a time control."""

from __future__ import annotations

import dataclasses
import typing
from collections.abc import Callable
from typing import ClassVar

from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver import physics_state
from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver.process_time_control import (
    ProcessTimeControl,
)
from icon4py.model.common import framework as fw, quantity
from icon4py.model.common.grid import base as base_grid


if typing.TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

__all__ = ["PhysicsProcess"]


@dataclasses.dataclass
class PhysicsProcess[I: fw.State, O: fw.State](fw.Component[physics_state.EntryState, O]):
    """
    A component that gates its wrapped component by time control.

    Outside the time-controlled window the process is inactive: the `Tendency`
    leaves of the output State are zeroed, the other leaves left untouched.
    Inside the time-controlled window there are two options selected by
    `ProcessTimeControl.computes_at()`: compute (i.e. run the wrapped
    component) or reuse the cached tendencies from the previous compute run.
    """

    Input = physics_state.EntryState
    Output: ClassVar[type[O]]

    component: fw.Component[I, O]
    # the component's `Input` from the physics driver's entry state
    # NOTE: temporary, components MWE part 02 (collect): the process collects its component's
    # `Input` by quantity (`fw.collect`), and `collect_input` goes.
    collect_input: Callable[[physics_state.EntryState], I]
    time_control: ProcessTimeControl
    grid: base_grid.Grid
    allocator: gtx_typing.Allocator | None
    _cache: O | None = dataclasses.field(default=None, init=False)

    def run(self, input: physics_state.EntryState, output: O) -> O:
        if not self.time_control.is_in_window(input.step_start):
            for field in fw.fields(output).values():
                if issubclass(field.quantity, quantity.Tendency):
                    field.ndarray[...] = 0.0  # type: ignore[index]
            return output

        if self.time_control.computes_at(input.step_start) or self._cache is None:
            self.component.run(self.collect_input(input), output)
            if self._cache is None:
                self._cache = fw.allocate(self.component.Output, self.grid, self.allocator)
            fw.copy(output, self._cache)
        else:
            fw.copy(self._cache, output)
        return output
