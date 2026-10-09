# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of the entry state the physics driver diagnoses and hands to its processes."""

import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver.physics_driver import (
    PhysicsProcess,
)
from icon4py.model.common.components import framework as fw, states
from icon4py.model.common.grid import base as base_grid, simple

from .utils import RecordingStep, driver, inputs, output, prognostics, time_control, tracers


@pytest.fixture
def grid() -> base_grid.Grid:
    return simple.simple_grid()


def test_entry_state_views_the_inputs_and_the_driver_diagnostics(grid: base_grid.Grid) -> None:
    step = RecordingStep(fw.Empty())
    physics = driver(grid, [PhysicsProcess(name="p", step=step, time_control=time_control())])
    prognostic, tracer_state = prognostics(grid), tracers(grid)
    driver_inputs = inputs(prognostic, tracer_state)

    physics.run(driver_inputs, out=output(prognostic, tracer_state))

    (entry,) = step.entries
    assert isinstance(physics.diagnostics, states.Diagnostics)
    # no copies: the prognostics and tracers as given, the diagnostics in the driver's buffers
    for name in ("vn", "w", "exner", "theta_v", "rho", *states.TRACERS):
        assert getattr(entry, name) is getattr(driver_inputs, name), name
    for declaration, field in physics.diagnostics.leaves():
        assert getattr(entry, declaration.name) is field, declaration.name


def test_diagnose_fills_the_diagnostics_and_leaves_the_inputs_untouched(
    grid: base_grid.Grid,
) -> None:
    """
    The diagnosis fills plausible fields and is strictly read only.

    Read-only-ness is the load-bearing invariant of parallel coupling: the prognostics and
    tracers stay bitwise identical until the driver's single apply step.
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
