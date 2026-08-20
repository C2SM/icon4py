# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The ``PhysicsDriver`` and its process / time-control types."""

from __future__ import annotations

import dataclasses
import datetime
from typing import TYPE_CHECKING, Any

from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver.process_time_control import (
    ProcessTimeControl,
)
from icon4py.model.common.components.components import Component
from icon4py.model.common.components.physics_state import PhysicsState


if TYPE_CHECKING:
    from icon4py.model.common.states import prognostic_state, tracer_states


@dataclasses.dataclass
class PhysicsProcess:
    """A registered physics process: a component, its state adapter, and its time control.

    The component is the per-process adapter (e.g. ``MuphysComponent``); it
    implements the generic ``Component`` protocol, which is how the driver types it.
    The state adapter is process-specific (it translates the prognostic state to/from
    *this* component's contract), so it is bundled per process rather than shared.
    """

    name: str
    component: Component
    state: PhysicsState
    time_control: ProcessTimeControl


class PhysicsDriver:
    """The physics driver: runs each registered physics process in order."""

    def __init__(
        self,
        processes: list[PhysicsProcess],
    ) -> None:
        self._processes = processes
        self._recycle_cache: dict[str, dict[str, Any]] = {}

    def run(
        self,
        prognostic: prognostic_state.PrognosticState,
        tracers: tracer_states.TracerState,
        dtime: datetime.timedelta,
        simulation_current_datetime: datetime.datetime,
    ) -> None:
        # <<<<<<< HEAD
        #         # TODO (Yilu): where do we
        # =======
        #         # 'simulation_current_datetime' is the end of the step being integrated (ICON's 'datetime_new');
        #         # processes are scheduled on the step-start date, per 'datetime = datetime_new - dt'
        #         # in mo_interface_iconam_aes.f90.
        #         step_start_datetime = simulation_current_datetime - dtime
        # >>>>>>> main
        step_start_datetime = simulation_current_datetime - dtime
        for process in self._processes:
            tc = process.time_control
            tc.validate_interval(dtime)
            state = process.state
            state.gather_from_prognostic(prognostic, tracers)
            if not tc.enable_process:
                continue
            if not tc.is_in_window(step_start_datetime):
                # outside the process window: no forcing
                continue
            # Compute on a firing (active) step, and also on the first in-window step -- when
            # there is nothing cached to recycle yet. Otherwise reuse the last computed forcing.
            if tc.is_active(step_start_datetime) or process.name not in self._recycle_cache:
                # compute
                outputs = process.component(state.as_component_input(), step_start_datetime)
                self._recycle_cache[process.name] = outputs
            else:
                # recycle
                outputs = self._recycle_cache[process.name]
            state.scatter_to_prognostic(prognostic, outputs, dtime)
