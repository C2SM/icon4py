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

    The process's input is the physics driver's entry state, its output the wrapped component's.
    It keeps no output buffers: the caller owns them, per the `fw.Component` contract. It does
    cache the last computed output, in a state of its own it allocates on its `grid` with its
    `allocator`, lazily on its first computing step; the cache's details stay private. On a
    non-computing step in its window it populates the outputs it is handed from that cache,
    whatever buffer the caller passes, fresh or reused, and a caller mutating a returned buffer
    leaves the cache untouched. Outside its time-control window the process contributes no
    tendencies: it zeroes the `Tendency` leaves of the outputs it is handed and returns them,
    the other leaves as given. Inside the window it runs its wrapped component on the steps
    its time control fires and on its first step in the window. Typed over the wrapped
    component's `Input` and `Output`: a `collect_input` that builds another component's
    `Input` is a type error.
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
    cache: O | None = dataclasses.field(default=None, init=False)

    def run(self, inputs: physics_state.EntryState, outputs: O) -> O:
        """This step's output: computed, populated from the cache, or zeroed outside the window."""
        # Outside its window the process contributes no tendencies: the `Tendency` leaves of the
        # outputs it is handed are zeroed, the other leaves left as given.
        if not self.time_control.is_in_window(inputs.step_start):
            for field in fw.fields(outputs).values():
                if issubclass(field.quantity, quantity.Tendency):
                    # NDArrayObject declares no __setitem__
                    field.ndarray[...] = 0.0  # type: ignore[index]
            return outputs
        # Run on a computing step, and also on the first in-window step, when there is nothing
        # cached yet: the cache is allocated lazily, on the first computing step, and refreshed
        # from the outputs the component just wrote, so the values survive to the steps that
        # do not compute.
        if self.time_control.computes_at(inputs.step_start) or self.cache is None:
            self.component.run(self.collect_input(inputs), outputs)
            if self.cache is None:
                self.cache = fw.allocate(self.component.Output, self.grid, self.allocator)
            fw.copy(outputs, self.cache)
        else:
            fw.copy(self.cache, outputs)
        return outputs
