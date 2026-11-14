# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""A physics driver on the simple grid, with stub processes, for the unit tests."""

from __future__ import annotations

import datetime
from collections.abc import Sequence

import gt4py.next as gtx
import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver import (
    physics_driver,
    physics_process,
    physics_state,
    process_time_control,
    static_fields,
)
from icon4py.model.common import dimension as dims, field, framework as fw
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.framework import quantities as qty, states
from icon4py.model.common.grid import base as base_grid, simple
from icon4py.model.common.utils import data_allocation as data_alloc


T0 = datetime.datetime(2024, 1, 1, 0, 0, 0)
DT = datetime.timedelta(seconds=300)


def time_control(
    process_dtime: datetime.timedelta = DT,
    start: datetime.datetime = T0,
    end: datetime.datetime = T0 + datetime.timedelta(days=1),
    driver_dtime: datetime.timedelta = DT,
    simulation_start: datetime.datetime = T0,
) -> process_time_control.ProcessTimeControl:
    return process_time_control.ProcessTimeControl(
        driver_dtime=driver_dtime,
        process_dtime=process_dtime,
        window_start=start,
        window_end=end,
        start_of_simulation=simulation_start,
    )


def driver(
    grid: base_grid.Grid,
    processes: Sequence[physics_process.PhysicsProcess] = (),
    exchange: decomposition.ExchangeRuntime | None = None,
) -> physics_driver.PhysicsDriver:
    # neutral geometry: primal_normal_cell_x = 1, primal_normal_cell_y = 0, c_lin_e = 0.5, so
    # the two-neighbour projection of a uniform u-tendency onto the edges is the identity
    return physics_driver.PhysicsDriver(
        processes,
        dtime=DT,
        grid=grid,
        static_state=static_fields.PhysicsDriverStaticState(
            ddqz_z_full=data_alloc.constant_field(grid, 100.0, dims.CellDim, dims.KDim),
            rbf_vec_coeff_c1=data_alloc.zero_field(grid, dims.CellDim, dims.C2E2C2EDim),
            rbf_vec_coeff_c2=data_alloc.zero_field(grid, dims.CellDim, dims.C2E2C2EDim),
            primal_normal_cell_x=data_alloc.constant_field(grid, 1.0, dims.EdgeDim, dims.E2CDim),
            primal_normal_cell_y=data_alloc.zero_field(grid, dims.EdgeDim, dims.E2CDim),
            c_lin_e=data_alloc.constant_field(grid, 0.5, dims.EdgeDim, dims.E2CDim),
        ),
        exchange=exchange if exchange is not None else decomposition.SingleNodeExchange(),
        backend=None,
    )


def prognostics(
    grid: base_grid.Grid, *, rho: float = 1.2, exner: float = 0.95, theta_v: float = 300.0
) -> states.PrognosticState:
    """Uniform prognostics, zero winds."""
    values = {"rho": rho, "exner": exner, "theta_v": theta_v}
    return fw.allocate(
        states.PrognosticState, grid, None, fill=lambda name, _: values.get(name, 0.0)
    )


def tracers(grid: base_grid.Grid, *, qv: float = 1e-3) -> states.TracerState:
    """All six tracers, qv uniform, the others zero."""
    return fw.allocate(
        states.TracerState, grid, None, fill=lambda name, _: qv if name == "qv" else 0.0
    )


def make_input(
    prognostic: states.PrognosticState,
    tracer_state: states.TracerState,
    *,
    simulation_current_datetime: datetime.datetime = T0 + DT,
    air_mass: field.Field[qty.AirMassOnCellK] | None = None,
) -> physics_driver.PhysicsDriver.Input:
    return physics_driver.PhysicsDriver.Input(
        vn=prognostic.vn,
        w=prognostic.w,
        exner=prognostic.exner,
        theta_v=prognostic.theta_v,
        rho=prognostic.rho,
        qv=tracer_state.qv,
        qc=tracer_state.qc,
        qi=tracer_state.qi,
        qr=tracer_state.qr,
        qs=tracer_state.qs,
        qg=tracer_state.qg,
        air_mass=air_mass,
        simulation_current_datetime=simulation_current_datetime,
    )


def make_output(
    prognostic: states.PrognosticState,
    tracer_state: states.TracerState,
    *,
    diagnostics: states.Diagnostics | None = None,
    perturbed_exner: field.Field[qty.PerturbedExnerOnCellK] | None = None,
) -> physics_driver.PhysicsDriver.Output:
    """The output continuing the given states, with new diagnostics on the simple grid if none."""
    if diagnostics is None:
        diagnostics = fw.allocate(states.Diagnostics, simple.simple_grid(), None)
    return physics_driver.PhysicsDriver.Output(
        vn=prognostic.vn,
        w=prognostic.w,
        exner=prognostic.exner,
        theta_v=prognostic.theta_v,
        qv=tracer_state.qv,
        qc=tracer_state.qc,
        qi=tracer_state.qi,
        qr=tracer_state.qr,
        qs=tracer_state.qs,
        qg=tracer_state.qg,
        temperature=diagnostics.temperature,
        virtual_temperature=diagnostics.virtual_temperature,
        pressure=diagnostics.pressure,
        pressure_ifc=diagnostics.pressure_ifc,
        u=diagnostics.u,
        v=diagnostics.v,
        perturbed_exner=perturbed_exner,
    )


def filled[S: fw.State](cls: type[S], grid: base_grid.Grid, **values: float) -> S:
    """A state of `cls` with the given uniform leaves, zero elsewhere."""
    return fw.allocate(cls, grid, None, fill=lambda name, _: values.get(name, 0.0))


def zero_physics_state(
    grid: base_grid.Grid, *, step_start: datetime.datetime
) -> physics_state.PhysicsState:
    """A physics state of zero fields with the given step start."""
    return fw.allocate(physics_state.PhysicsState, grid, None, step_start=step_start)


class Recorder(fw.Component[physics_state.PhysicsState, fw.State]):
    """
    A component that writes fixed values into the leaves of its output and records the physics
    states it ran on, with a copy of the qv each one read.

    The driver allocates the process's output buffer from `Output`: the concrete state the
    recorder writes, so `recording` declares a subclass per output shape.

    """

    Input = physics_state.PhysicsState
    Output = fw.State

    def __init__(self, values: dict[str, float]) -> None:
        self.values = values
        self.states: list[physics_state.PhysicsState] = []
        self.qv_read: list[np.ndarray] = []

    def run(self, input: physics_state.PhysicsState, output: fw.State) -> fw.State:
        self.states.append(input)
        self.qv_read.append(input.qv.asnumpy().copy())
        for name, value in self.values.items():
            getattr(output, name).ndarray[...] = value
        return output


def recording(
    output: type[fw.State],
    values: dict[str, float],
    time_control: process_time_control.ProcessTimeControl,
    grid: base_grid.Grid,
) -> tuple[physics_process.PhysicsProcess[physics_state.PhysicsState, fw.State], Recorder]:
    """A process running a `Recorder` that writes `values` into its `output`-shaped outputs,
    on the whole physics state. The process's cache is allocated on `grid`."""

    class _Recorder(Recorder):
        """A `Recorder` whose outputs are `output`-shaped, so the driver can allocate them."""

        Output = output

    recorder = _Recorder(values)
    process = physics_process.PhysicsProcess(
        component=recorder,
        collect_input=lambda state: state,
        time_control=time_control,
        grid=grid,
        allocator=None,
    )
    return process, recorder


class RecordingExchange(decomposition.SingleNodeExchange):
    """Records the halo exchanges as (dim, field identities), so their order and targets are assertable."""

    def __init__(self) -> None:
        self.events: list[tuple[gtx.Dimension, list[int]]] = []

    def start(
        self,
        dim: gtx.Dimension,
        *fields: gtx.Field | data_alloc.NDArray,
        stream: decomposition.StreamLike = decomposition.DEFAULT_STREAM,
    ) -> decomposition.ExchangeResult:
        self.events.append((dim, [id(f) for f in fields]))
        return super().start(dim, *fields, stream=stream)
