# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of the muphys component around a stand-in for the granule."""

import dataclasses
import datetime
from typing import Any

import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys import component as muphys_component
from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver import physics_state
from icon4py.model.common import dimension as dims
from icon4py.model.common.components import framework as fw
from icon4py.model.common.grid import base as base_grid, simple
from icon4py.model.common.utils import data_allocation as data_alloc


DTIME = datetime.timedelta(seconds=10)


@dataclasses.dataclass
class GranuleStandIn:
    """Overwrites what it is given, as the granule does: te + 1, qv + 2, pflx = 3."""

    calls: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    def __call__(
        self, *, te: Any, q_in: Any, t_out: Any, q_out: Any, pflx: Any, **kwargs: Any
    ) -> None:
        self.calls.append({"te": te, "q_in": q_in, "t_out": t_out, "q_out": q_out, **kwargs})
        t_out.ndarray[...] = te.ndarray + 1.0
        q_out.v.ndarray[...] = q_in.v.ndarray + 2.0
        pflx.ndarray[...] = 3.0


@pytest.fixture
def grid() -> base_grid.Grid:
    return simple.simple_grid()


def _component(
    grid: base_grid.Grid, step: GranuleStandIn
) -> tuple[muphys_component.MuphysComponent, Any]:
    dz = data_alloc.constant_field(grid, 100.0, dims.CellDim, dims.KDim)
    component = muphys_component.MuphysComponent(
        grid=grid, dtime=DTIME, qnc=100.0, dz=dz, backend=None, step=step
    )
    return component, dz


def _inputs(grid: base_grid.Grid) -> muphys_component.MuphysComponent.Input:
    values = {"te": 280.0, "p": 9e4, "rho": 1.1, "qv": 1e-3}
    return fw.allocate(
        muphys_component.MuphysComponent.Input,
        grid,
        None,
        fill=lambda name, _: values.get(name, 0.0),
    )


def test_collect_input_views_the_physics_state(grid: base_grid.Grid) -> None:
    state = fw.allocate(physics_state.PhysicsState, grid, None)

    inputs = muphys_component.collect_input(state)

    assert inputs.te is state.temperature
    assert inputs.p is state.pressure
    assert inputs.rho is state.rho
    for name in ("qv", "qc", "qi", "qr", "qs", "qg"):
        assert getattr(inputs, name) is getattr(state, name), name


def test_run_reports_tendencies_and_leaves_the_inputs_untouched(grid: base_grid.Grid) -> None:
    step = GranuleStandIn()
    component, dz = _component(grid, step)
    inputs = _inputs(grid)

    out = component.run(inputs)

    dt = DTIME.total_seconds()
    np.testing.assert_allclose(out.tend_temperature.data.asnumpy(), 1.0 / dt, rtol=1e-12)
    np.testing.assert_allclose(out.tend_qv.data.asnumpy(), 2.0 / dt, rtol=1e-12)
    np.testing.assert_array_equal(out.tend_qc.data.asnumpy(), 0.0)
    np.testing.assert_array_equal(out.pflx.data.asnumpy(), 3.0)
    # the granule ran on private copies; the static and read-only fields went in as given
    np.testing.assert_array_equal(inputs.te.data.asnumpy(), 280.0)
    np.testing.assert_array_equal(inputs.qv.data.asnumpy(), 1e-3)
    (call,) = step.calls
    assert call["te"] is not inputs.te.data
    assert call["dz"] is dz
    assert call["p"] is inputs.p.data
    assert call["rho"] is inputs.rho.data


def test_run_writes_where_the_caller_says(grid: base_grid.Grid) -> None:
    component, _ = _component(grid, GranuleStandIn())
    view = fw.allocate(muphys_component.MuphysComponent.Output, grid, None)

    out = component.run(_inputs(grid), out=view)

    assert out is view
    np.testing.assert_array_equal(view.pflx.data.asnumpy(), 3.0)
