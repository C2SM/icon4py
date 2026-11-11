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
from icon4py.model.common import dimension as dims, field_type_aliases as fa, framework as fw
from icon4py.model.common.grid import base as base_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import component_tests


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


class TestMuphys(component_tests.ComponentTest, component_tests.CollectInputTest):
    collect_input = staticmethod(muphys_component.collect_input)

    @pytest.fixture
    def compute(self) -> GranuleStandIn:
        return GranuleStandIn()

    @pytest.fixture
    def dz(self, grid: base_grid.Grid) -> fa.CellKField[float]:
        return data_alloc.constant_field(grid, 100.0, dims.CellDim, dims.KDim)

    @pytest.fixture
    def component(
        self, grid: base_grid.Grid, compute: GranuleStandIn, dz: fa.CellKField[float]
    ) -> muphys_component.Muphys:
        return muphys_component.Muphys(
            grid=grid, dtime=DTIME, qnc=100.0, dz=dz, backend_like=None, compute=compute
        )

    @pytest.fixture
    def state(self, grid: base_grid.Grid) -> physics_state.PhysicsState:
        return fw.allocate(
            physics_state.PhysicsState, grid, None, step_start=datetime.datetime(2024, 1, 1)
        )

    def test_run_hands_the_granule_copies_and_reports_tendencies(
        self,
        component: muphys_component.Muphys,
        compute: GranuleStandIn,
        dz: fa.CellKField[float],
        input: muphys_component.Muphys.Input,
        output: muphys_component.Muphys.Output,
    ) -> None:
        component.run(input, output)

        dt = DTIME.total_seconds()
        np.testing.assert_allclose(output.tend_temperature.asnumpy(), 1.0 / dt, rtol=1e-12)
        np.testing.assert_allclose(output.tend_qv.asnumpy(), 2.0 / dt, rtol=1e-12)
        np.testing.assert_array_equal(output.tend_qc.asnumpy(), 0.0)
        np.testing.assert_array_equal(output.pflx.asnumpy(), 3.0)
        # what the granule overwrites goes in as a copy, what it only reads as given
        (call,) = compute.calls
        assert call["te"] is not input.te
        assert call["dz"] is dz
        assert call["p"] is input.p
        assert call["rho"] is input.rho
