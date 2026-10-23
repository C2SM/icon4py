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
import datetime
from collections.abc import Callable
from typing import Protocol

from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver import physics_state
from icon4py.model.common import framework as fw


__all__ = ["PhysicsProcess", "ProcessTimeControl"]


@dataclasses.dataclass(frozen=True)
class ProcessTimeControl:
    """icon4py analogue of the per-process time fields in AES `aes_phy_tc`.

    - `interval`   (`dt_xxx`): compute interval, > 0
    - `start_date` (`sd_xxx`), `end_date` (`ed_xxx`): half-open
      `[start, end)` window during which the process exists.
    """

    interval: datetime.timedelta
    start_date: datetime.datetime
    end_date: datetime.datetime

    def __post_init__(self) -> None:
        if self.interval <= datetime.timedelta(0):
            raise ValueError(f"time-control interval must be positive, got {self.interval}")

    def is_in_window(self, simulation_current_datetime: datetime.datetime) -> bool:
        return self.start_date <= simulation_current_datetime < self.end_date

    def computes_at(self, simulation_current_datetime: datetime.datetime) -> bool:
        """True on a whole number of intervals after `start_date` (AES `isCurrentEventActive`)."""
        if simulation_current_datetime < self.start_date:
            return False
        elapsed = simulation_current_datetime - self.start_date
        return elapsed % self.interval == datetime.timedelta(0)


class _Runs[I, O](Protocol):
    """What a process needs of its component: `run(inputs, outputs) -> outputs`."""

    def run(self, inputs: I, outputs: O) -> O: ...


@dataclasses.dataclass
class PhysicsProcess[I: fw.State, O: fw.State]:
    """
    A component under a time control, with the output buffers it keeps between two runs.

    Inside its window the process runs its component on the steps its time control fires and
    on its first step in the window; on the other steps it returns the output of its last
    run, which stays in `outputs` until the next. Typed over the component's `Input` and
    `Output`: a `collect_input` that builds another component's `Input` is a type error.
    """

    component: _Runs[I, O]
    # the component's `Input` from the physics driver's entry state
    # NOTE: temporary, components MWE part 02 (collect): the process collects its component's
    # `Input` by quantity (`fw.collect`), and `collect_input` goes.
    collect_input: Callable[[physics_state.EntryState], I]
    time_control: ProcessTimeControl
    outputs: O
    _has_run: bool = dataclasses.field(default=False, init=False)

    def step(self, entry: physics_state.EntryState, step_start: datetime.datetime) -> O | None:
        """This step's output: computed, the last one again, or None outside the window."""
        # Outside its window the process contributes nothing, and must not reach the reuse
        # branch below, which would run once on the first step.
        if not self.time_control.is_in_window(step_start):
            return None
        # Run on a computing step, and also on the first in-window step, when there is nothing
        # to reuse yet. Otherwise the last output is still in `outputs`.
        if self.time_control.computes_at(step_start) or not self._has_run:
            self.component.run(self.collect_input(entry), self.outputs)
            self._has_run = True
        return self.outputs
