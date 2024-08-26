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
from icon4py.model.common.dimension import C2E2CDim, CECDim, CEDim, CellDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import as_1D_sparse_field, constant_field, random_field
from icon4py.model.common.utils import gt4py_field_allocation as field_alloc


@pytest.mark.datatest
def test_run_horizontal_advection_single_step(grid_savepoint, icon_grid, interpolation_savepoint):
    edge_geometry = grid_savepoint.construct_edge_geometry()
    cell_geometry = grid_savepoint.construct_cell_geometry()
    config = advection.AdvectionConfig()

    interpolation_state = advection_states.AdvectionInterpolationState(
        geofac_div=as_1D_sparse_field(interpolation_savepoint.geofac_div(), CEDim),
        rbf_vec_coeff_e=interpolation_savepoint.rbf_vec_coeff_e(),
        pos_on_tplane_e_1=interpolation_savepoint.pos_on_tplane_e_x(),
        pos_on_tplane_e_2=interpolation_savepoint.pos_on_tplane_e_y(),
    )

    least_squares_state = advection_states.AdvectionLeastSquaresState(
        lsq_pseudoinv_1=as_1D_sparse_field(
            random_field(icon_grid, CellDim, C2E2CDim), CECDim
        ),  # TODO (dastrm): need to serialize this
        lsq_pseudoinv_2=as_1D_sparse_field(
            random_field(icon_grid, CellDim, C2E2CDim), CECDim
        ),  # TODO (dastrm): need to serialize this
    )

    metric_state = advection_states.AdvectionMetricState(
        deepatmo_divh=constant_field(icon_grid, 1.0, KDim),
        deepatmo_divzl=constant_field(icon_grid, 1.0, KDim),
        deepatmo_divzu=constant_field(icon_grid, 1.0, KDim),
    )

    advection_granule = advection.Advection(
        grid=icon_grid,
        config=config,
        interpolation_state=interpolation_state,
        least_squares_state=least_squares_state,
        metric_state=metric_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
    )

    diagnostic_state = advection_states.AdvectionDiagnosticState(
        airmass_now=constant_field(icon_grid, 1.0, CellDim, KDim),
        airmass_new=constant_field(icon_grid, 1.0, CellDim, KDim),
        grf_tend_tracer=field_alloc.allocate_zero_field(CellDim, KDim, grid=icon_grid),
        hfl_tracer=field_alloc.allocate_zero_field(EdgeDim, KDim, grid=icon_grid),
        vfl_tracer=field_alloc.allocate_zero_field(CellDim, KDim, grid=icon_grid),
    )

    prep_adv = solve_nh_states.PrepAdvection(
        vn_traj=field_alloc.allocate_zero_field(EdgeDim, KDim, grid=icon_grid),
        mass_flx_me=constant_field(icon_grid, 1.0, EdgeDim, KDim),
        mass_flx_ic=field_alloc.allocate_zero_field(
            CellDim, KDim, is_halfdim=True, grid=icon_grid
        ),  # TODO (dastrm): should be KHalfDim
        vol_flx_ic=field_alloc.allocate_zero_field(CellDim, KDim, grid=icon_grid),
    )

    p_tracer_now = random_field(icon_grid, CellDim, KDim)
    p_tracer_new = field_alloc.allocate_zero_field(CellDim, KDim, grid=icon_grid)

    advection_granule.run(
        diagnostic_state=diagnostic_state,
        prep_adv=prep_adv,
        p_tracer_now=p_tracer_now,
        p_tracer_new=p_tracer_new,
        dtime=10.0,
    )

    assert 0.125 == pytest.approx(0.125, abs=1e-12)
