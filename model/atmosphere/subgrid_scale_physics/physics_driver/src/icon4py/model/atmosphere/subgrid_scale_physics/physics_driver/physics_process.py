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
from collections.abc import Callable
from typing import ClassVar

from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver import physics_state
from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver.process_time_control import (
    ProcessTimeControl,
)
from icon4py.model.common import framework as fw


__all__ = ["PhysicsProcess"]


@dataclasses.dataclass
class PhysicsProcess[I: fw.State, O: fw.State](fw.Component[physics_state.EntryState, O]):
    """
    A component that gates its wrapped component by time control.

    The process's input is the physics driver's entry state, its output the wrapped component's,
    which it keeps in the buffer the driver hands back to `run` each step so the values of the
    last run survive to the next. Outside its time-control window the process contributes no
    tendencies: it zeroes the `Tendency` leaves of the outputs it is handed and returns them,
    the other leaves (the muphys fluxes) as given. Inside the window it runs its wrapped
    component on the steps its time control fires and on its first step in the window; on the
    other steps it returns the outputs unchanged, caching the last run's values. Typed over
    the wrapped component's `Input` and `Output`: a `collect_input` that builds another
    component's `Input` is a type error.
    """

    Input = physics_state.EntryState
    Output: ClassVar[type[O]]

    component: fw.Component[I, O]
    # the component's `Input` from the physics driver's entry state
    # NOTE: temporary, components MWE part 02 (collect): the process collects its component's
    # `Input` by quantity (`fw.collect`), and `collect_input` goes.
    collect_input: Callable[[physics_state.EntryState], I]
    time_control: ProcessTimeControl
    outputs: O
    _has_run: bool = dataclasses.field(default=False, init=False)

    def run(self, inputs: physics_state.EntryState, outputs: O) -> O:
        """This step's output: computed, cached from the last run, or zeroed outside the window."""
        # Outside its window the process contributes no tendencies: the `Tendency` leaves of the
        # outputs it is handed are zeroed, the other leaves (the muphys fluxes) left as given.
        # It must not reach the caching branch below, which would run once on the first step.
        if not self.time_control.is_in_window(inputs.step_start):
            for field in fw.fields(outputs).values():
                if issubclass(field.quantity, fw.Tendency):
                    # NDArrayObject declares no __setitem__
                    field.ndarray[...] = 0.0  # type: ignore[index]
            return outputs
        # Run on a computing step, and also on the first in-window step, when there is nothing
        # cached yet. Otherwise the values of the last run are still in the outputs.
        if self.time_control.computes_at(inputs.step_start) or not self._has_run:
            self.component.run(self.collect_input(inputs), outputs)
            self._has_run = True
        return outputs
