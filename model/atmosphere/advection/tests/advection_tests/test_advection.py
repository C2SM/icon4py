# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.atmosphere.advection import advection, advection_states
from icon4py.model.atmosphere.dycore.state_utils import states as solve_nh_states
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim
from icon4py.model.common.grid import simple as simple_grid
from icon4py.model.common.test_utils.helpers import constant_field, random_field
from icon4py.model.common.utils import gt4py_field_allocation as field_alloc


def test_run_horizontal_advection_single_step():
    config = advection.AdvectionConfig()
    grid = simple_grid.SimpleGrid()
    advection_granule = advection.Advection(grid=grid, config=config)

    diagnostic_state = advection_states.AdvectionDiagnosticState(
        airmass_now=constant_field(grid, 1.0, CellDim, KDim),
        airmass_new=constant_field(grid, 1.0, CellDim, KDim),
        hfl_tracer=field_alloc.allocate_zero_field(EdgeDim, KDim, grid=grid),
        vfl_tracer=field_alloc.allocate_zero_field(CellDim, KDim, grid=grid),
        ddt_tracer_adv=field_alloc.allocate_zero_field(CellDim, KDim, grid=grid),
    )

    metric_state = advection_states.AdvectionMetricState(
        deepatmo_divzl=constant_field(grid, 1.0, KDim),
        deepatmo_divzu=constant_field(grid, 1.0, KDim),
    )

    prep_adv = solve_nh_states.PrepAdvection(
        vn_traj=field_alloc.allocate_zero_field(EdgeDim, KDim, grid=grid),
        mass_flx_me=constant_field(grid, 1.0, CellDim, KDim),
        mass_flx_ic=constant_field(grid, 1.0, CellDim, KDim),
        vol_flx_ic=field_alloc.allocate_zero_field(CellDim, KDim, grid=grid),
    )

    p_tracer_now = random_field(grid, CellDim, KDim)
    p_tracer_new = field_alloc.allocate_zero_field(CellDim, KDim, grid=grid)

    advection_granule.run(
        diagnostic_state=diagnostic_state,
        metric_state=metric_state,
        prep_adv=prep_adv,
        p_tracer_now=p_tracer_now,
        p_tracer_new=p_tracer_new,
        dtime=10.0,
    )

    assert 0.125 == pytest.approx(0.125, abs=1e-12)
