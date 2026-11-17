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

from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver import (
    physics_state,
    process_time_control,
)
from icon4py.model.common import framework as fw, quantity
from icon4py.model.common.grid import base as base_grid


if typing.TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing


# NOTE: temporary, replaced in part 02 (collect): a generic `TimeGatedComponent`, once the
# physics state goes.
# Disable the default __eq__ so that PhysicsProcess compares by identity and
# enables __hash__ so that a PhysicsProcess can be used in dicts.
@dataclasses.dataclass(eq=False)
class PhysicsProcess[I: fw.State, O: fw.State](fw.Component[physics_state.PhysicsState, O]):
    """
    A component that gates its wrapped component by time control.

    Outside the time-controlled window the process is inactive: the `Tendency`
    leaves of the output State are zeroed, the other leaves left untouched.
    Inside the time-controlled window there are two options selected by
    `ProcessTimeControl.computes_at()`: compute (i.e. run the wrapped
    component) or reuse the cached tendencies from the previous compute run.
    """

    Input = physics_state.PhysicsState
    Output: type[O] = dataclasses.field(init=False)

    component: fw.Component[I, O]
    # the component's `Input` from the physics driver's physics state
    # NOTE: temporary, replaced in part 02 (collect): the process collects its component's
    # `Input` by quantity (`fw.collect`), and `collect_input` goes.
    collect_input: Callable[[physics_state.PhysicsState], I]
    time_control: process_time_control.ProcessTimeControl
    grid: base_grid.Grid
    allocator: gtx_typing.Allocator | None
    # the last computed output, of which only the `Tendency` leaves are used
    # NOTE: temporary, replaced in part 08 (updates): the cache holds the process's `Update`
    # state, the tendencies it contributes, instead of its whole `Output`.
    _cache: O | None = dataclasses.field(default=None, init=False)

    def __post_init__(self) -> None:
        self.Output = self.component.Output

    def run(self, input: physics_state.PhysicsState, output: O) -> O:
        # NOTE: temporary, replaced in parts 07 (tendencies) and 08 (updates): the tendencies are
        # the process's `Update` state instead of the `Tendency` leaves of its `Output`.
        if not self.time_control.is_in_window(input.step_start):
            for field in fw.fields(output, kind=quantity.Tendency).values():
                field.ndarray[...] = 0.0  # type: ignore[index]
            return output

        # NOTE: different from ICON when the first step in the window does not compute, i.e. a
        # window opening before the simulation start off the `process_dtime` grid, or a restart
        # in the window: ICON recycles its zero-initialized memory until the next compute step,
        # here the process computes on that first step.
        if self.time_control.computes_at(input.step_start) or self._cache is None:
            # NOTE: temporary, replaced in part 02 (collect), see `collect_input`.
            self.component.run(self.collect_input(input), output)
            if self.time_control.skips_steps:
                # The cache is needed iff the process does not compute at
                # every step of the driver
                if self._cache is None:
                    self._cache = fw.allocate(self.component.Output, self.grid, self.allocator)
                fw.copy(output, self._cache, kind=quantity.Tendency)
        else:
            fw.copy(self._cache, output, kind=quantity.Tendency)
        return output
