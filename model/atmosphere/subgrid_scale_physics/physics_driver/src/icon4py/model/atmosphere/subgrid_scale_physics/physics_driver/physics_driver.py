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
from typing import TYPE_CHECKING, Any, Protocol

from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver import physics_state
from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver.process_time_control import (
    ProcessTimeControl,
)
from icon4py.model.common.components.component_state import ComponentState
from icon4py.model.common.components.components import Component


if TYPE_CHECKING:
    from icon4py.model.common.states import prognostic_state, tracer_states


class PhysicsComponent(Component[Any, Any], Protocol):
    """A ``Component`` that can adopt diagnostic output buffers owned by its caller.

    The generic ``Component`` protocol is deliberately left untouched: writing
    results into memory the caller owns is a physics-coupling concern, not a
    property every model component should have to satisfy. Declaring the
    extension here keeps that decision while still letting the driver call
    ``bind_output_buffers`` under type checking rather than by duck typing.
    """

    def bind_output_buffers(self, buffers: dict[str, Any]) -> None:
        """Adopt the layer-owned buffers for this component's diagnostic outputs.

        Called once per process at driver construction. Implementations keep their
        own allocations as the standalone default (e.g. the granule datatests) and
        replace them with these, so the granule writes into the layer's memory in
        place and the driver never copies a diagnostic.
        """
        ...


@dataclasses.dataclass
class PhysicsProcess:
    """A registered physics process: a component, its state adapter, and its time control.

    The component is the per-process adapter (e.g. ``MuphysComponent``), typed as
    a ``PhysicsComponent``: the generic ``Component`` contract plus the
    ``bind_output_buffers`` hook the driver needs to hand it the layer-owned
    diagnostic buffers. The state adapter is process-specific (it maps the frozen
    entry state to *this* component's contract), so it is bundled per process
    rather than shared.
    """

    name: str
    component: PhysicsComponent
    state: ComponentState
    time_control: ProcessTimeControl


class PhysicsDriver:
    """Runs the physics processes under parallel coupling.

    One timestep (``run``): the ``EntryState`` binds the model state and
    diagnoses the physics fields once (dyn2phy); every enabled process reads that
    same frozen state and computes; outputs tagged ``kind == "tendency"`` are
    accumulated, while the diagnostic outputs are written by the granules directly
    into the layer-owned ``diagnostics`` store buffers bound at construction; the
    accumulated tendencies are applied to the model state exactly once at the end.
    Processes never see the raw PrognosticState/TracerState and never
    write it — the PhysicsState layer owns both conversion boundaries.
    """

    def __init__(
        self,
        processes: list[PhysicsProcess],
        entry_state: physics_state.EntryState,
        accumulators: physics_state.TendencyAccumulators,
        apply_to_prognostic: physics_state.ApplyToPrognostic,
        diagnostics: physics_state.DiagnosticsStore,
    ) -> None:
        self._processes = processes
        self._entry = entry_state
        self._accumulators = accumulators
        self._apply = apply_to_prognostic
        self._recycle_cache: dict[str, dict[str, Any]] = {}
        self.diagnostics = diagnostics
        for process in processes:
            process.component.bind_output_buffers(
                diagnostics.allocate(process.name, process.component.outputs_properties)
            )

    def run(
        self,
        prognostic: prognostic_state.PrognosticState,
        tracers: tracer_states.TracerState,
        dtime: datetime.timedelta,
        simulation_current_datetime: datetime.datetime,
    ) -> None:
        # 'simulation_current_datetime' is the end of the step being integrated (ICON's 'datetime_new');
        # processes are scheduled on the step-start date, per 'datetime = datetime_new - dt'
        # in mo_interface_iconam_aes.f90.
        step_start_datetime = simulation_current_datetime - dtime
        self._entry.diagnose_from(prognostic, tracers)
        self._accumulators.zero()
        dt_seconds = dtime.total_seconds()
        for process in self._processes:
            tc = process.time_control
            tc.validate_interval(dtime)
            if not tc.enable_process or not tc.is_in_window(step_start_datetime):
                continue
            # Compute on a firing (active) step, and also on the first in-window step -- when
            # there is nothing cached to recycle yet. Otherwise reuse the last computed forcing.
            if tc.is_active(step_start_datetime) or process.name not in self._recycle_cache:
                outputs = process.component(
                    process.state.as_component_input(self._entry), step_start_datetime
                )
                self._recycle_cache[process.name] = outputs
            else:
                outputs = self._recycle_cache[process.name]
            self._accumulators.accumulate(outputs, process.component.outputs_properties)
        self._apply(self._entry, self._accumulators, dt_seconds)
