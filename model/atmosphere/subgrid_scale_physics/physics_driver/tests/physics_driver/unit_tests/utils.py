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
import functools
from collections.abc import Sequence

import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver import (
    physics_driver,
    physics_state,
    static_fields,
)
from icon4py.model.common import dimension as dims, framework as fw
from icon4py.model.common.framework import states
from icon4py.model.common.grid import base as base_grid
from icon4py.model.common.utils import data_allocation as data_alloc


T0 = datetime.datetime(2024, 1, 1, 0, 0, 0)
DT = datetime.timedelta(seconds=300)


def time_control(
    interval: datetime.timedelta = DT,
    start: datetime.datetime = T0,
    end: datetime.datetime = T0 + datetime.timedelta(days=1),
) -> fw.ProcessTimeControl:
    return fw.ProcessTimeControl(interval=interval, start_date=start, end_date=end)


def driver(
    grid: base_grid.Grid, processes: Sequence[fw.PhysicsComponent] = ()
) -> physics_driver.PhysicsDriver:
    # neutral geometry: primal_normal_cell_x = 1, primal_normal_cell_y = 0, c_lin_e = 0.5, so
    # the two-neighbour projection of a uniform u-tendency onto the edges is the identity
    return physics_driver.PhysicsDriver(
        processes,
        grid=grid,
        static_state=static_fields.PhysicsDriverStaticState(
            ddqz_z_full=data_alloc.constant_field(grid, 100.0, dims.CellDim, dims.KDim),
            rbf_vec_coeff_c1=data_alloc.zero_field(grid, dims.CellDim, dims.C2E2C2EDim),
            rbf_vec_coeff_c2=data_alloc.zero_field(grid, dims.CellDim, dims.C2E2C2EDim),
            primal_normal_cell_x=data_alloc.constant_field(grid, 1.0, dims.EdgeDim, dims.E2CDim),
            primal_normal_cell_y=data_alloc.zero_field(grid, dims.EdgeDim, dims.E2CDim),
            c_lin_e=data_alloc.constant_field(grid, 0.5, dims.EdgeDim, dims.E2CDim),
        ),
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


def inputs(
    prognostic: states.PrognosticState,
    tracer_state: states.TracerState,
    *,
    simulation_current_datetime: datetime.datetime = T0 + DT,
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
        dtime=DT,
        simulation_current_datetime=simulation_current_datetime,
    )


def output(
    prognostic: states.PrognosticState, tracer_state: states.TracerState
) -> physics_driver.PhysicsDriver.Output:
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
    )


def filled[S: fw.State](cls: type[S], grid: base_grid.Grid, **values: float) -> S:
    """A state of `cls` with the given uniform leaves, zero elsewhere."""
    return fw.allocate(cls, grid, None, fill=lambda name, _: values.get(name, 0.0))


class RecordingProcess(fw.PhysicsComponent):
    """A physics process computing a fixed output and recording the entry states it collected
    from, with a copy of the qv each one read."""

    Input = physics_state.EntryState
    Output = fw.State

    def __init__(
        self, grid: base_grid.Grid, output: fw.State, time_control: fw.ProcessTimeControl
    ) -> None:
        super().__init__(grid=grid, allocator=None, time_control=time_control)
        self.output = output
        self.entries: list[physics_state.EntryState] = []
        self.qv_read: list[np.ndarray] = []

    @functools.cached_property
    def cached_output(self) -> fw.State:
        return self.output

    def collect_input(self, entry: physics_state.EntryState) -> physics_state.EntryState:
        self.entries.append(entry)
        self.qv_read.append(entry.qv.asnumpy().copy())
        return entry

    def run(self, inputs: physics_state.EntryState, outputs: fw.State) -> fw.State:
        return outputs
