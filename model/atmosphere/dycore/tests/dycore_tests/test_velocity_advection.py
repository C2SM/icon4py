# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import logging

import pytest

from icon4py.model.atmosphere.dycore import dycore_states, velocity_advection as advection
from icon4py.model.common import dimension as dims, utils as common_utils
from icon4py.model.common.grid import (
    horizontal as h_grid,
    states as grid_states,
    vertical as v_grid,
)
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.testing import datatest_utils as dt_utils, helpers

from . import utils


log = logging.getLogger(__name__)


def create_vertical_params(vertical_config, grid_savepoint):
    return v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
        _min_index_flat_horizontal_grad_pressure=grid_savepoint.nflat_gradp(),
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment, step_date_init",
    [
        ("mch_ch_r04b09_dsl", "2021-06-20T12:00:10.000"),
        ("exclaim_ape_R02B04", "2000-01-01T00:00:02.000"),
    ],
)
def test_verify_velocity_init_against_savepoint(
    interpolation_savepoint,
    step_date_init,
    grid_savepoint,
    icon_grid,
    metrics_savepoint,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    experiment,
    backend,
):
    interpolation_state = utils.construct_interpolation_state(interpolation_savepoint)
    metric_state_nonhydro = utils.construct_metric_state(metrics_savepoint, icon_grid.num_levels)
    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    vertical_params = create_vertical_params(vertical_config, grid_savepoint)

    velocity_advection = advection.VelocityAdvection(
        grid=icon_grid,
        metric_state=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        edge_params=grid_savepoint.construct_edge_geometry(),
        owner_mask=grid_savepoint.c_owner_mask(),
        backend=backend,
    )
    assert velocity_advection.cfl_w_limit == 0.65
    assert velocity_advection.scalfac_exdiff == 0.05
    assert helpers.dallclose(velocity_advection.cfl_clipping.asnumpy(), 0.0)
    assert helpers.dallclose(velocity_advection.levmask.asnumpy(), False)
    assert helpers.dallclose(velocity_advection.vcfl_dsl.asnumpy(), 0.0)


@pytest.mark.datatest
@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment, step_date_init",
    [
        ("mch_ch_r04b09_dsl", "2021-06-20T12:00:10.000"),
        ("exclaim_ape_R02B04", "2000-01-01T00:00:02.000"),
    ],
)
def test_scale_factors_by_dtime(savepoint_velocity_init, icon_grid, backend):
    dtime = savepoint_velocity_init.get_metadata("dtime").get("dtime")
    velocity_advection = advection.VelocityAdvection(
        grid=icon_grid,
        metric_state=None,
        interpolation_state=None,
        vertical_params=None,
        edge_params=None,
        owner_mask=None,
        backend=backend,
    )
    (cfl_w_limit, scalfac_exdiff) = velocity_advection._scale_factors_by_dtime(dtime)
    assert cfl_w_limit == savepoint_velocity_init.cfl_w_limit()
    assert scalfac_exdiff == savepoint_velocity_init.scalfac_exdiff()


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize("istep_init, substep_init, istep_exit, substep_exit ", [(1, 1, 1, 1)])
@pytest.mark.parametrize(
    "experiment, step_date_init, step_date_exit",
    [
        (dt_utils.REGIONAL_EXPERIMENT, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000"),
        (dt_utils.GLOBAL_EXPERIMENT, "2000-01-01T00:00:02.000", "2000-01-01T00:00:02.000"),
    ],
)
def test_velocity_predictor_step(
    experiment,
    istep_init,
    istep_exit,
    step_date_init,
    step_date_exit,
    substep_init,
    substep_exit,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    icon_grid,
    grid_savepoint,
    savepoint_velocity_init,
    metrics_savepoint,
    interpolation_savepoint,
    savepoint_velocity_exit,
    backend,
    caplog,
):
    caplog.set_level(logging.WARN)
    init_savepoint = savepoint_velocity_init
    vn_only = init_savepoint.vn_only()
    dtime = init_savepoint.get_metadata("dtime").get("dtime")

    diagnostic_state = dycore_states.DiagnosticStateNonHydro(
        vt=init_savepoint.vt(),
        vn_ie=init_savepoint.vn_ie(),
        w_concorr_c=init_savepoint.w_concorr_c(),
        theta_v_ic=None,
        exner_pr=None,
        rho_ic=None,
        ddt_exner_phy=None,
        grf_tend_rho=None,
        grf_tend_thv=None,
        grf_tend_w=None,
        mass_fl_e=None,
        ddt_vn_phy=None,
        grf_tend_vn=None,
        ddt_vn_apc_pc=common_utils.PredictorCorrectorPair(
            init_savepoint.ddt_vn_apc_pc(0), init_savepoint.ddt_vn_apc_pc(1)
        ),
        ddt_w_adv_pc=common_utils.PredictorCorrectorPair(
            init_savepoint.ddt_w_adv_pc(0), init_savepoint.ddt_w_adv_pc(1)
        ),
        rho_incr=None,
        vn_incr=None,
        exner_incr=None,
        exner_dyn_incr=None,
    )
    prognostic_state = prognostics.PrognosticState(
        w=init_savepoint.w(),
        vn=init_savepoint.vn(),
        theta_v=None,
        rho=None,
        exner=None,
    )
    interpolation_state = utils.construct_interpolation_state(interpolation_savepoint)
    metric_state_nonhydro = utils.construct_metric_state(metrics_savepoint, icon_grid.num_levels)

    cell_geometry: grid_states.CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: grid_states.EdgeParams = grid_savepoint.construct_edge_geometry()

    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    vertical_params = create_vertical_params(vertical_config, grid_savepoint)

    velocity_advection = advection.VelocityAdvection(
        grid=icon_grid,
        metric_state=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        edge_params=edge_geometry,
        owner_mask=grid_savepoint.c_owner_mask(),
        backend=backend,
    )

    velocity_advection.run_predictor_step(
        vn_only=vn_only,
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        z_w_concorr_me=init_savepoint.z_w_concorr_me(),
        z_kin_hor_e=init_savepoint.z_kin_hor_e(),
        z_vt_ie=init_savepoint.z_vt_ie(),
        dtime=dtime,
        cell_areas=cell_geometry.area,
    )

    icon_result_ddt_vn_apc_pc = savepoint_velocity_exit.ddt_vn_apc_pc(0).asnumpy()
    icon_result_ddt_w_adv_pc = savepoint_velocity_exit.ddt_w_adv_pc(0).asnumpy()
    icon_result_vn_ie = savepoint_velocity_exit.vn_ie().asnumpy()
    icon_result_vt = savepoint_velocity_exit.vt().asnumpy()
    icon_result_w_concorr_c = savepoint_velocity_exit.w_concorr_c().asnumpy()
    icon_result_z_w_concorr_mc = savepoint_velocity_exit.z_w_concorr_mc().asnumpy()
    icon_result_z_v_grad_w = savepoint_velocity_exit.z_v_grad_w().asnumpy()

    # FIX
    # stencil 01
    assert helpers.dallclose(diagnostic_state.vt.asnumpy(), icon_result_vt, atol=1.0e-14)
    # stencil 02,05
    assert helpers.dallclose(diagnostic_state.vn_ie.asnumpy(), icon_result_vn_ie, atol=1.0e-14)

    start_edge_lateral_boundary_6 = icon_grid.start_index(
        h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7)
    )
    # stencil 07
    if not vn_only:
        assert helpers.dallclose(
            icon_result_z_v_grad_w[start_edge_lateral_boundary_6:, :],
            velocity_advection.z_v_grad_w.asnumpy()[start_edge_lateral_boundary_6:, :],
            atol=1.0e-16,
        )

    # stencil 08
    start_cell_nudging = icon_grid.start_index(h_grid.domain(dims.CellDim)(h_grid.Zone.NUDGING))
    assert helpers.dallclose(
        savepoint_velocity_exit.z_ekinh().asnumpy()[start_cell_nudging:, :],
        velocity_advection.z_ekinh.asnumpy()[start_cell_nudging:, :],
    )
    # stencil 09
    assert helpers.dallclose(
        velocity_advection.z_w_concorr_mc.asnumpy()[
            start_cell_nudging:, vertical_params.nflatlev : icon_grid.num_levels
        ],
        icon_result_z_w_concorr_mc[
            start_cell_nudging:, vertical_params.nflatlev : icon_grid.num_levels
        ],
        atol=1.0e-15,
    )
    # stencil 10
    assert helpers.dallclose(
        diagnostic_state.w_concorr_c.asnumpy()[
            start_cell_nudging:, vertical_params.nflatlev + 1 : icon_grid.num_levels
        ],
        icon_result_w_concorr_c[
            start_cell_nudging:, vertical_params.nflatlev + 1 : icon_grid.num_levels
        ],
        atol=1.0e-15,
    )
    # stencil 11,12,13,14
    assert helpers.dallclose(
        velocity_advection.z_w_con_c.asnumpy()[start_cell_nudging:, :],
        savepoint_velocity_exit.z_w_con_c().asnumpy()[start_cell_nudging:, :],
        atol=1.0e-15,
    )
    # stencil 16
    assert helpers.dallclose(
        diagnostic_state.ddt_w_adv_pc.predictor.asnumpy()[start_cell_nudging:, :],
        icon_result_ddt_w_adv_pc[start_cell_nudging:, :],
        atol=5.0e-16,
        rtol=1.0e-10,
    )
    # stencil 19
    assert helpers.dallclose(
        diagnostic_state.ddt_vn_apc_pc.predictor.asnumpy(),
        icon_result_ddt_vn_apc_pc,
        atol=1.0e-15,
    )


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize("istep_init, istep_exit, substep_init", [(2, 2, 1)])
@pytest.mark.parametrize(
    "experiment, step_date_init, step_date_exit",
    [
        (dt_utils.REGIONAL_EXPERIMENT, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000"),
        (dt_utils.GLOBAL_EXPERIMENT, "2000-01-01T00:00:02.000", "2000-01-01T00:00:02.000"),
    ],
)
def test_velocity_corrector_step(
    istep_init,
    istep_exit,
    substep_init,
    step_date_init,
    step_date_exit,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    icon_grid,
    grid_savepoint,
    savepoint_velocity_init,
    savepoint_velocity_exit,
    interpolation_savepoint,
    metrics_savepoint,
    backend,
):
    init_savepoint = savepoint_velocity_init
    vn_only = init_savepoint.vn_only()
    dtime = init_savepoint.get_metadata("dtime").get("dtime")

    assert not vn_only

    diagnostic_state = dycore_states.DiagnosticStateNonHydro(
        vt=init_savepoint.vt(),
        vn_ie=init_savepoint.vn_ie(),
        w_concorr_c=init_savepoint.w_concorr_c(),
        theta_v_ic=None,
        exner_pr=None,
        rho_ic=None,
        ddt_exner_phy=None,
        grf_tend_rho=None,
        grf_tend_thv=None,
        grf_tend_w=None,
        mass_fl_e=None,
        ddt_vn_phy=None,
        grf_tend_vn=None,
        ddt_vn_apc_pc=common_utils.PredictorCorrectorPair(
            init_savepoint.ddt_vn_apc_pc(0), init_savepoint.ddt_vn_apc_pc(1)
        ),
        ddt_w_adv_pc=common_utils.PredictorCorrectorPair(
            init_savepoint.ddt_w_adv_pc(0), init_savepoint.ddt_w_adv_pc(1)
        ),
        rho_incr=None,  # sp.rho_incr(),
        vn_incr=None,  # sp.vn_incr(),
        exner_incr=None,  # sp.exner_incr(),
        exner_dyn_incr=None,
    )
    prognostic_state = prognostics.PrognosticState(
        w=init_savepoint.w(),
        vn=init_savepoint.vn(),
        theta_v=None,
        rho=None,
        exner=None,
    )

    interpolation_state = utils.construct_interpolation_state(interpolation_savepoint)

    metric_state_nonhydro = utils.construct_metric_state(metrics_savepoint, icon_grid.num_levels)

    cell_geometry: grid_states.CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: grid_states.EdgeParams = grid_savepoint.construct_edge_geometry()

    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    vertical_params = create_vertical_params(vertical_config, grid_savepoint)

    velocity_advection = advection.VelocityAdvection(
        grid=icon_grid,
        metric_state=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        edge_params=edge_geometry,
        owner_mask=grid_savepoint.c_owner_mask(),
        backend=backend,
    )

    velocity_advection.run_corrector_step(
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        z_kin_hor_e=init_savepoint.z_kin_hor_e(),
        z_vt_ie=init_savepoint.z_vt_ie(),
        dtime=dtime,
        cell_areas=cell_geometry.area,
    )

    icon_result_ddt_vn_apc_pc = savepoint_velocity_exit.ddt_vn_apc_pc(1).asnumpy()
    icon_result_ddt_w_adv_pc = savepoint_velocity_exit.ddt_w_adv_pc(1).asnumpy()
    icon_result_z_v_grad_w = savepoint_velocity_exit.z_v_grad_w().asnumpy()

    # stencil 07
    start_cell_lateral_boundary_level_7 = icon_grid.start_index(
        h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7)
    )
    assert helpers.dallclose(
        velocity_advection.z_v_grad_w.asnumpy()[start_cell_lateral_boundary_level_7:, :],
        icon_result_z_v_grad_w[start_cell_lateral_boundary_level_7:, :],
        atol=1e-16,
    )
    # stencil 08
    start_cell_nudging = icon_grid.start_index(h_grid.domain(dims.CellDim)(h_grid.Zone.NUDGING))
    assert helpers.dallclose(
        velocity_advection.z_ekinh.asnumpy()[start_cell_nudging:, :],
        savepoint_velocity_exit.z_ekinh().asnumpy()[start_cell_nudging:, :],
    )

    # stencil 11,12,13,14
    assert helpers.dallclose(
        velocity_advection.z_w_con_c.asnumpy()[start_cell_nudging:, :],
        savepoint_velocity_exit.z_w_con_c().asnumpy()[start_cell_nudging:, :],
    )
    # stencil 16
    assert helpers.dallclose(
        diagnostic_state.ddt_w_adv_pc.corrector.asnumpy()[start_cell_nudging:, :],
        icon_result_ddt_w_adv_pc[start_cell_nudging:, :],
        atol=5.0e-16,
    )
    # stencil 19
    assert helpers.dallclose(
        diagnostic_state.ddt_vn_apc_pc.corrector.asnumpy(),
        icon_result_ddt_vn_apc_pc,
        atol=5.0e-16,
    )
