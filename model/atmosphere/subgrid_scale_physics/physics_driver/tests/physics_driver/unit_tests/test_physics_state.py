# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of the physics state the physics driver diagnoses and hands to its processes."""

import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver.physics_driver import (
    PhysicsProcess,
)
from icon4py.model.common.components import framework as fw, quantities as qty, states
from icon4py.model.common.grid import base as base_grid, simple

from .utils import (
    DT,
    T0,
    RecordingStep,
    driver,
    filled,
    inputs,
    output,
    prognostics,
    time_control,
    tracers,
)


class QvTendencyOutput(fw.State):
    tend_qv: fw.Field[qty.TendencyOfQvOnCellK]


@pytest.fixture
def grid() -> base_grid.Grid:
    return simple.simple_grid()


def test_physics_state_views_the_leaves_the_driver_does_not_advance(
    grid: base_grid.Grid,
) -> None:
    step = RecordingStep(fw.Empty())
    physics = driver(grid, [PhysicsProcess(name="p", step=step, time_control=time_control())])
    prognostic, tracer_state = prognostics(grid), tracers(grid)
    driver_inputs = inputs(prognostic, tracer_state)

    physics.run(driver_inputs, out=output(prognostic, tracer_state))

    (state,) = step.states
    assert isinstance(physics.diagnostics, states.Diagnostics)
    # no copies: the prognostics as given, the diagnostics in the driver's buffers
    for name in ("vn", "exner", "theta_v", "rho"):
        assert getattr(state, name) is getattr(driver_inputs, name), name
    for name in ("virtual_temperature", "pressure", "pressure_ifc"):
        assert getattr(state, name) is getattr(physics.diagnostics, name), name


def test_physics_state_starts_the_advanced_leaves_from_the_entry_values_in_own_buffers(
    grid: base_grid.Grid,
) -> None:
    step = RecordingStep(fw.Empty())
    physics = driver(grid, [PhysicsProcess(name="p", step=step, time_control=time_control())])
    # vn = 1 so that the reconstructed u is not the zero of a fresh buffer
    prognostic, tracer_state = prognostics(grid, vn=1.0, w=0.5), tracers(grid, qv=1e-3)
    first_inputs = inputs(prognostic, tracer_state)

    physics.run(first_inputs, out=output(prognostic, tracer_state))
    physics.run(
        inputs(prognostic, tracer_state, simulation_current_datetime=T0 + 2 * DT),
        out=output(prognostic, tracer_state),
    )

    first, second = step.states
    assert step.read[0]["u"].any()
    # the driver advances copies: of the prognostics and tracers as given ...
    for name in ("w", *states.TRACERS):
        assert getattr(first, name) is not getattr(first_inputs, name), name
        np.testing.assert_array_equal(
            step.read[0][name], getattr(first_inputs, name).data.asnumpy(), err_msg=name
        )
    # ... and of the diagnosed temperature and (u, v)
    for name in ("temperature", "u", "v"):
        assert getattr(first, name) is not getattr(physics.diagnostics, name), name
        np.testing.assert_array_equal(
            step.read[0][name], getattr(physics.diagnostics, name).data.asnumpy(), err_msg=name
        )
    # allocated once, not per step
    for name in ("temperature", "u", "v", "w", *states.TRACERS):
        assert getattr(second, name) is getattr(first, name), name


def test_physics_state_restarts_the_advanced_leaves_from_each_step_s_entry_values(
    grid: base_grid.Grid,
) -> None:
    # A process advances qv on step 1; the dynamics (here: by hand) then change the
    # prognostics, and on step 2 the process must read those, not the stale buffers.
    step = RecordingStep(filled(QvTendencyOutput, grid, tend_qv=1e-7))
    physics = driver(grid, [PhysicsProcess(name="p", step=step, time_control=time_control())])
    prognostic, tracer_state = prognostics(grid, vn=1.0, w=0.5), tracers(grid, qv=1e-3)
    assert tracer_state.qv is not None

    physics.run(inputs(prognostic, tracer_state), out=output(prognostic, tracer_state))
    changed = {"qv": 5e-4, "w": -0.25, "vn": 2.0, "theta_v": 290.0}
    for name, value in changed.items():
        leaf = getattr(tracer_state if name == "qv" else prognostic, name)
        leaf.data.ndarray[...] = value
    physics.run(
        inputs(prognostic, tracer_state, simulation_current_datetime=T0 + 2 * DT),
        out=output(prognostic, tracer_state),
    )

    first, second = step.read
    np.testing.assert_array_equal(second["qv"], 5e-4)
    np.testing.assert_array_equal(second["w"], -0.25)
    for name in ("temperature", "u"):
        assert not np.array_equal(second[name], first[name]), name
        np.testing.assert_array_equal(
            second[name], getattr(physics.diagnostics, name).data.asnumpy(), err_msg=name
        )


def test_diagnose_fills_the_diagnostics_and_leaves_the_inputs_untouched(
    grid: base_grid.Grid,
) -> None:
    """
    The diagnosis fills plausible fields and is strictly read only.

    Read-only-ness is what the final update relies on: it starts from the entry values (ICON's
    phy2dyn), so the prognostics and tracers stay bitwise identical until the driver's single
    apply step.
    """
    step = RecordingStep(fw.Empty())
    physics = driver(grid, [PhysicsProcess(name="p", step=step, time_control=time_control())])
    prognostic, tracer_state = prognostics(grid, exner=0.95, theta_v=300.0), tracers(grid)
    before = {d.name: field.data.asnumpy().copy() for d, field in prognostic.leaves()}

    physics.run(inputs(prognostic, tracer_state), out=output(prognostic, tracer_state))

    diagnostics = physics.diagnostics
    assert 200.0 < diagnostics.temperature.data.asnumpy().mean() < 320.0
    assert (diagnostics.pressure.data.asnumpy() > 0).all()
    # pressure grows downward: the surface interface above the top full level
    assert (
        diagnostics.pressure_ifc.data.asnumpy()[:, -1] > diagnostics.pressure.data.asnumpy()[:, 0]
    ).all()
    for declaration, field in prognostic.leaves():
        np.testing.assert_array_equal(
            field.data.asnumpy(), before[declaration.name], err_msg=declaration.name
        )
    assert tracer_state.qv is not None
    np.testing.assert_array_equal(tracer_state.qv.data.asnumpy(), 1e-3)
