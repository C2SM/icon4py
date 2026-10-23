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

from icon4py.model.common import framework as fw
from icon4py.model.common.framework import states
from icon4py.model.common.grid import base as base_grid, simple

from .utils import driver, inputs, output, prognostics, recording, time_control, tracers


@pytest.fixture
def grid() -> base_grid.Grid:
    return simple.simple_grid()


def test_entry_state_views_the_inputs_and_the_driver_diagnostics(grid: base_grid.Grid) -> None:
    process, recorder = recording(fw.Empty(), time_control())
    physics = driver(grid, [process])
    prognostic, tracer_state = prognostics(grid), tracers(grid)
    driver_inputs = inputs(prognostic, tracer_state)

    physics.run(driver_inputs, outputs=output(prognostic, tracer_state))

    (entry,) = recorder.entries
    assert isinstance(physics.diagnostics, states.Diagnostics)
    # no copies: the prognostics and tracers as given, the diagnostics in the driver's buffers
    for name in ("vn", "w", "exner", "theta_v", "rho", *states.TRACERS):
        assert getattr(entry, name) is getattr(driver_inputs, name), name
    for name, field in fw.fields(physics.diagnostics).items():
        assert getattr(entry, name) is field, name


def test_diagnose_fills_the_diagnostics_and_leaves_the_inputs_untouched(
    grid: base_grid.Grid,
) -> None:
    """
    The diagnosis fills plausible fields and is strictly read only.

    Read-only-ness is the load-bearing invariant of parallel coupling: the prognostics and
    tracers stay bitwise identical until the driver's single apply step.
    """
    process, _ = recording(fw.Empty(), time_control())
    physics = driver(grid, [process])
    prognostic, tracer_state = prognostics(grid, exner=0.95, theta_v=300.0), tracers(grid)
    before = {name: field.asnumpy().copy() for name, field in fw.fields(prognostic).items()}

    physics.run(inputs(prognostic, tracer_state), outputs=output(prognostic, tracer_state))

    diagnostics = physics.diagnostics
    assert 200.0 < diagnostics.temperature.asnumpy().mean() < 320.0
    assert (diagnostics.pressure.asnumpy() > 0).all()
    # pressure grows downward: the surface interface above the top full level
    assert (diagnostics.pressure_ifc.asnumpy()[:, -1] > diagnostics.pressure.asnumpy()[:, 0]).all()
    for name, field in fw.fields(prognostic).items():
        np.testing.assert_array_equal(field.asnumpy(), before[name], err_msg=name)
    assert tracer_state.qv is not None
    np.testing.assert_array_equal(tracer_state.qv.asnumpy(), 1e-3)
