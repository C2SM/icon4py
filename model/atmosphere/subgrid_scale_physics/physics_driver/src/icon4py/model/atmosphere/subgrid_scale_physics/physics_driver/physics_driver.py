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
import logging
from typing import TYPE_CHECKING, Any, Protocol

from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver import (
    config as physics_config,
    physics_state,
)
from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver.process_time_control import (
    ProcessTimeControl,
)
from icon4py.model.common.components.component_state import ComponentState
from icon4py.model.common.components.components import Component
from icon4py.model.common.decomposition import definitions as decomposition


if TYPE_CHECKING:
    from collections.abc import Iterator

    import gt4py.next.typing as gtx_typing

    from icon4py.model.common import field_type_aliases as fa, type_alias as ta
    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.common.states import (
        diagnostic_state,
        factory,
        prognostic_state,
        tracer_states,
    )


log = logging.getLogger(__name__)


class PhysicsComponent(Component[Any, Any], Protocol):
    """A ``Component`` that can adopt diagnostic output buffers owned by its caller.

    The generic ``Component`` protocol is deliberately left untouched: writing
    results into memory the caller owns is a physics-coupling concern, not a
    property every model component should have to satisfy. Declaring the
    extension here keeps that decision while still letting the driver call
    ``bind_output_buffers`` under type checking rather than by duck typing.
    """

    def bind_output_buffers(self, buffers: dict[str, Any]) -> None:
        """Adopt caller-owned buffers for this component's diagnostic outputs.

        Implementations keep their own allocations as the default and replace
        them with these, so the results are written in place, without a copy.
        """
        ...


@dataclasses.dataclass
class PhysicsProcess:
    """One physics process: its component, its state adapter and its time control.

    The state adapter belongs to the process rather than being shared, because it
    maps the PhysicsState to the input names of this one component. The process also
    decides whether it runs on a given step and keeps its last computed outputs, so
    a step between two firings can reuse them.
    """

    name: str
    component: PhysicsComponent
    state: ComponentState
    time_control: ProcessTimeControl
    _cached_output: dict[str, Any] | None = dataclasses.field(default=None, init=False, repr=False)

    def run(
        self,
        state: physics_state.PhysicsState,
        step_start_datetime: datetime.datetime,
        dtime: datetime.timedelta,
    ) -> dict[str, Any]:
        """This step's outputs: freshly computed, recycled, or empty if it does not run.

        Takes the PhysicsState rather than the component's input mapping: building
        that mapping may derive inputs of its own, which must not happen on a step
        where the component does not compute.
        """
        tc = self.time_control
        tc.validate_interval(dtime)
        # Outside its window the process contributes nothing -- and must not reach the
        # recycling branch below, which would compute once on the first step.
        if not tc.is_in_window(step_start_datetime):
            return {}
        # Compute on a firing (active) step, and also on the first in-window step -- when
        # there is nothing cached to recycle yet. Otherwise reuse the last computed forcing.
        if tc.is_active(step_start_datetime) or self._cached_output is None:
            self._cached_output = self.component(
                self.state.as_component_input(state), step_start_datetime
            )
        return self._cached_output


class PhysicsDriver:
    """Runs the physics processes under sequential or parallel coupling.

    In one timestep (``run``) the PhysicsState is diagnosed once from the model
    state, then the processes run in list order and their tendencies are summed
    (``Tendencies.accumulate``). The ``coupling`` decides when the state is advanced
    in place by the tendencies (``PhysicsState.update``):

    - sequential (ICON AES): by each process's tendencies, before the next one runs;
    - parallel: by the sums, once, after all processes have read the step-entry state.

    At the end of the step ``PhysicsState.update_prognostics`` writes exner, theta_v and
    vn back and syncs the halos of what the physics wrote.
    """

    def __init__(
        self,
        processes: list[PhysicsProcess],
        state: physics_state.PhysicsState,
        tendencies: physics_state.Tendencies,
        coupling: physics_config.CouplingType = physics_config.CouplingType.SEQUENTIAL,
    ) -> None:
        self._processes = processes
        self._state = state
        self._tendencies = tendencies
        self._coupling = coupling
        for process in processes:
            process.component.bind_output_buffers(
                state.allocate_process_diagnostics(
                    process.name, process.component.outputs_properties
                )
            )

    @property
    def coupling(self) -> physics_config.CouplingType:
        return self._coupling

    def diagnostic_outputs(self) -> Iterator[physics_state.DiagnosticOutput]:
        """The processes' diagnostic output buffers with their metadata, for the model output."""
        return self._state.diagnostic_outputs()

    @classmethod
    def from_sources(
        cls,
        processes: list[PhysicsProcess],
        *,
        grid: base_grid.Grid,
        geometry: factory.FieldSource,
        interpolation: factory.FieldSource,
        metrics: factory.FieldSource,
        backend: gtx_typing.Backend | None = None,
        process_props: decomposition.ProcessProperties,
        decomposition_info: decomposition.DecompositionInfo,
        coupling: physics_config.CouplingType = physics_config.CouplingType.SEQUENTIAL,
    ) -> PhysicsDriver:
        """Build the driver together with the coupling pieces it owns.

        The pieces stay constructor arguments so a caller -- a test, above all --
        can substitute them; this is the assembly every real caller wants.
        """
        return cls(
            processes,
            state=physics_state.PhysicsState(
                grid=grid,
                geometry=geometry,
                interpolation=interpolation,
                metrics=metrics,
                process_props=process_props,
                decomposition_info=decomposition_info,
                backend=backend,
            ),
            tendencies=physics_state.Tendencies(backend=backend),
            coupling=coupling,
        )

    def run(
        self,
        *,
        prognostic: prognostic_state.PrognosticState,
        tracers: tracer_states.TracerState,
        diagnostics: diagnostic_state.DiagnosticState,
        dtime: datetime.timedelta,
        simulation_current_datetime: datetime.datetime,
        air_mass: fa.CellKField[ta.wpfloat] | None = None,
        perturbed_exner: fa.CellKField[ta.wpfloat] | None = None,
    ) -> None:
        step_start_datetime = simulation_current_datetime - dtime
        self._state.compute_diagnostics(prognostic, tracers, diagnostics, air_mass)
        self._tendencies.zero()
        dt_seconds = dtime.total_seconds()
        sequential = self._coupling is physics_config.CouplingType.SEQUENTIAL
        for process in self._processes:
            log.info(f"Running {process.component.__class__}")
            outputs = process.run(self._state, step_start_datetime, dtime)
            tendencies = physics_state.tendency_outputs(
                outputs, process.component.outputs_properties
            )
            self._tendencies.accumulate(tendencies)
            if sequential:
                # the next process reads the state advanced by this one's tendencies
                self._state.update(tendencies, dt_seconds)
        if not sequential:
            # every process read the step-entry state: advance it once by the sums
            self._state.update(self._tendencies.sums, dt_seconds)
        self._state.update_prognostics(self._tendencies.sums, dt_seconds, perturbed_exner)
