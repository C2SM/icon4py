# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import datetime

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import (
    component as tmx_component,
    data as tmx_data,
    tmx_states,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import simple
from icon4py.model.common.utils import data_allocation as data_alloc


class _FakeGranule:
    def __init__(self):
        self.calls = []

    def run(
        self, *, input_state, surface_flux_state, diagnostic_state, tendency_state, new_state, dtime
    ):
        self.calls.append(dtime)


def _input_dict(grid):
    ck = lambda: data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
    half = lambda: data_alloc.zero_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
    cell = lambda: data_alloc.zero_field(grid, dims.CellDim)
    d = {k: ck() for k in tmx_data.INPUTS_PROPERTIES}
    d["w"] = half()
    d["pressure_ifc"] = half()
    for k in ("evapotranspiration", "sensible_heat_flux", "u_stress", "v_stress", "q_snocpymlt"):
        d[k] = cell()
    return d


def test_call_runs_granule_and_returns_output_contract():
    grid = simple.simple_grid()
    fake = _FakeGranule()
    comp = tmx_component.TmxComponent(
        grid=grid,
        config=None,
        metric_state=None,
        interpolation_state=None,
        edge_params=None,
        cell_params=None,
        dtime=datetime.timedelta(seconds=300),
        backend=None,
        granule=fake,
    )
    out = comp(_input_dict(grid), datetime.datetime(2008, 9, 1))
    assert fake.calls == [300.0]
    assert set(out) == set(tmx_data.OUTPUTS_PROPERTIES)
    assert out["ddt_qv"] is comp._tendency_state.ddt_qv
