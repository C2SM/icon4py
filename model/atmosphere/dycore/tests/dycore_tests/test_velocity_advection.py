# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import logging

import gt4py.next as gtx
import pytest

from icon4py.model.atmosphere.dycore import (
    compute_cell_diagnostics_for_velocity_advection,
    dycore_states,
    velocity_advection as advection,
)
from icon4py.model.atmosphere.dycore.stencils import (
    compute_advection_in_horizontal_momentum_equation,
    compute_advection_in_vertical_momentum_equation,
    compute_edge_diagnostics_for_velocity_advection,
)
from icon4py.model.common import dimension as dims, utils as common_utils
from icon4py.model.common.grid import (
    horizontal as h_grid,
    states as grid_states,
    vertical as v_grid,
)
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.common.utils import data_allocation as data_alloc
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
        (dt_utils.REGIONAL_EXPERIMENT, "2021-06-20T12:00:10.000"),
        (dt_utils.GLOBAL_EXPERIMENT, "2000-01-01T00:00:02.000"),
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
        (dt_utils.REGIONAL_EXPERIMENT, "2021-06-20T12:00:10.000"),
        (dt_utils.GLOBAL_EXPERIMENT, "2000-01-01T00:00:02.000"),
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
@pytest.mark.parametrize("istep_init, istep_exit", [(1, 1)])
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
        tangential_wind=init_savepoint.vt(),
        vn_on_half_levels=init_savepoint.vn_ie(),
        contravariant_correction_at_cells_on_half_levels=init_savepoint.w_concorr_c(),
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
        normal_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            init_savepoint.ddt_vn_apc_pc(0), init_savepoint.ddt_vn_apc_pc(1)
        ),
        vertical_wind_advective_tendency=common_utils.PredictorCorrectorPair(
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
        skip_compute_predictor_vertical_advection=vn_only,
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        contravariant_correction_at_edges_on_model_levels=init_savepoint.z_w_concorr_me(),
        horizontal_kinetic_energy_at_edges_on_model_levels=init_savepoint.z_kin_hor_e(),
        tangential_wind_on_half_levels=init_savepoint.z_vt_ie(),
        dtime=dtime,
        cell_areas=cell_geometry.area,
    )

    icon_result_ddt_vn_apc_pc = savepoint_velocity_exit.ddt_vn_apc_pc(0).asnumpy()
    icon_result_ddt_w_adv_pc = savepoint_velocity_exit.ddt_w_adv_pc(0).asnumpy()
    icon_result_vn_ie = savepoint_velocity_exit.vn_ie().asnumpy()
    icon_result_vt = savepoint_velocity_exit.vt().asnumpy()
    icon_result_w_concorr_c = savepoint_velocity_exit.w_concorr_c().asnumpy()
    icon_result_z_v_grad_w = savepoint_velocity_exit.z_v_grad_w().asnumpy()

    # stencil 01
    assert helpers.dallclose(
        diagnostic_state.tangential_wind.asnumpy(), icon_result_vt, atol=1.0e-14
    )
    # stencil 02,05
    assert helpers.dallclose(
        diagnostic_state.vn_on_half_levels.asnumpy(), icon_result_vn_ie, atol=1.0e-14
    )

    start_edge_lateral_boundary_6 = icon_grid.start_index(
        h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7)
    )
    # stencil 07
    if not vn_only:
        assert helpers.dallclose(
            icon_result_z_v_grad_w[start_edge_lateral_boundary_6:, :],
            velocity_advection._horizontal_advection_of_w_at_edges_on_half_levels.asnumpy()[
                start_edge_lateral_boundary_6:, :
            ],
            atol=1.0e-16,
        )

    # stencil 08
    start_cell_nudging = icon_grid.start_index(h_grid.domain(dims.CellDim)(h_grid.Zone.NUDGING))
    assert helpers.dallclose(
        savepoint_velocity_exit.z_ekinh().asnumpy()[start_cell_nudging:, :],
        velocity_advection._horizontal_kinetic_energy_at_cells_on_model_levels.asnumpy()[
            start_cell_nudging:, :
        ],
    )
    # stencil 10
    assert helpers.dallclose(
        diagnostic_state.contravariant_correction_at_cells_on_half_levels.asnumpy()[
            start_cell_nudging:, vertical_params.nflatlev + 1 : icon_grid.num_levels
        ],
        icon_result_w_concorr_c[
            start_cell_nudging:, vertical_params.nflatlev + 1 : icon_grid.num_levels
        ],
        atol=1.0e-15,
    )
    # stencil 11,12,13,14
    assert helpers.dallclose(
        velocity_advection._contravariant_corrected_w_at_cells_on_half_levels.asnumpy()[
            start_cell_nudging:, :
        ],
        savepoint_velocity_exit.z_w_con_c().asnumpy()[start_cell_nudging:, :],
        atol=1.0e-15,
    )

    # stencil 16
    assert helpers.dallclose(
        diagnostic_state.vertical_wind_advective_tendency.predictor.asnumpy()[
            start_cell_nudging:, :
        ],
        icon_result_ddt_w_adv_pc[start_cell_nudging:, :],
        atol=5.0e-16,
        rtol=1.0e-10,
    )
    # stencil 19
    assert helpers.dallclose(
        diagnostic_state.normal_wind_advective_tendency.predictor.asnumpy(),
        icon_result_ddt_vn_apc_pc,
        atol=1.0e-15,
    )


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize("istep_init, istep_exit", [(2, 2)])
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
        tangential_wind=init_savepoint.vt(),
        vn_on_half_levels=init_savepoint.vn_ie(),
        contravariant_correction_at_cells_on_half_levels=init_savepoint.w_concorr_c(),
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
        normal_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            init_savepoint.ddt_vn_apc_pc(0), init_savepoint.ddt_vn_apc_pc(1)
        ),
        vertical_wind_advective_tendency=common_utils.PredictorCorrectorPair(
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
        horizontal_kinetic_energy_at_edges_on_model_levels=init_savepoint.z_kin_hor_e(),
        tangential_wind_on_half_levels=init_savepoint.z_vt_ie(),
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
        velocity_advection._horizontal_advection_of_w_at_edges_on_half_levels.asnumpy()[
            start_cell_lateral_boundary_level_7:, :
        ],
        icon_result_z_v_grad_w[start_cell_lateral_boundary_level_7:, :],
        atol=1e-16,
    )
    # stencil 08
    start_cell_nudging = icon_grid.start_index(h_grid.domain(dims.CellDim)(h_grid.Zone.NUDGING))
    assert helpers.dallclose(
        velocity_advection._horizontal_kinetic_energy_at_cells_on_model_levels.asnumpy()[
            start_cell_nudging:, :
        ],
        savepoint_velocity_exit.z_ekinh().asnumpy()[start_cell_nudging:, :],
    )

    # stencil 11,12,13,14
    assert helpers.dallclose(
        velocity_advection._contravariant_corrected_w_at_cells_on_half_levels.asnumpy()[
            start_cell_nudging:, :
        ],
        savepoint_velocity_exit.z_w_con_c().asnumpy()[start_cell_nudging:, :],
    )
    # stencil 16
    assert helpers.dallclose(
        diagnostic_state.vertical_wind_advective_tendency.corrector.asnumpy()[
            start_cell_nudging:, :
        ],
        icon_result_ddt_w_adv_pc[start_cell_nudging:, :],
        atol=5.0e-16,
    )
    # stencil 19
    assert helpers.dallclose(
        diagnostic_state.normal_wind_advective_tendency.corrector.asnumpy(),
        icon_result_ddt_vn_apc_pc,
        atol=5.0e-16,
    )


@pytest.mark.dataset
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "experiment, step_date_init, step_date_exit",
    [
        (dt_utils.REGIONAL_EXPERIMENT, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000"),
        (dt_utils.GLOBAL_EXPERIMENT, "2000-01-01T00:00:02.000", "2000-01-01T00:00:02.000"),
    ],
)
@pytest.mark.parametrize("istep_init", [1])
def test_compute_edge_diagnostics_for_velocity_advection_in_predictor_step(
    icon_grid,
    grid_savepoint,
    savepoint_compute_edge_diagnostics_for_velocity_advection_init,
    savepoint_compute_edge_diagnostics_for_velocity_advection_exit,
    interpolation_savepoint,
    metrics_savepoint,
    savepoint_velocity_init,
    savepoint_velocity_exit,
    step_date_init,
    step_date_exit,
    substep_init,
    istep_init,
    backend,
):
    edge_domain = h_grid.domain(dims.EdgeDim)

    tangential_wind_on_half_levels = (
        savepoint_compute_edge_diagnostics_for_velocity_advection_init.z_vt_ie()
    )
    tangential_wind = savepoint_compute_edge_diagnostics_for_velocity_advection_init.vt()
    vn_on_half_levels = savepoint_compute_edge_diagnostics_for_velocity_advection_init.vn_ie()
    horizontal_kinetic_energy_at_edges_on_model_levels = (
        savepoint_compute_edge_diagnostics_for_velocity_advection_init.z_kin_hor_e()
    )
    horizontal_advection_of_w_at_edges_on_half_levels = (
        savepoint_compute_edge_diagnostics_for_velocity_advection_init.z_v_grad_w()
    )
    vn = savepoint_compute_edge_diagnostics_for_velocity_advection_init.vn()
    w = savepoint_compute_edge_diagnostics_for_velocity_advection_init.w()

    rbf_vec_coeff_e = interpolation_savepoint.rbf_vec_coeff_e()
    wgtfac_e = metrics_savepoint.wgtfac_e()
    ddxn_z_full = metrics_savepoint.ddxn_z_full()
    ddxt_z_full = metrics_savepoint.ddxt_z_full()
    contravariant_correction_at_edges_on_model_levels = (
        savepoint_compute_edge_diagnostics_for_velocity_advection_init.z_w_concorr_me()
    )
    wgtfacq_e = metrics_savepoint.wgtfacq_e_dsl(icon_grid.num_levels)
    nflatlev = grid_savepoint.nflatlev()
    c_intp = interpolation_savepoint.c_intp()
    inv_dual_edge_length = grid_savepoint.inv_dual_edge_length()
    inv_primal_edge_length = grid_savepoint.inverse_primal_edge_lengths()
    tangent_orientation = grid_savepoint.tangent_orientation()
    k = data_alloc.index_field(
        dim=dims.KDim, grid=icon_grid, extend={dims.KDim: 1}, backend=backend
    )

    skip_compute_predictor_vertical_advection = (
        savepoint_compute_edge_diagnostics_for_velocity_advection_init.lvn_only()
    )
    edge = data_alloc.index_field(dim=dims.EdgeDim, grid=icon_grid, backend=backend)
    lateral_boundary_7 = icon_grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7))
    halo_1 = icon_grid.end_index(edge_domain(h_grid.Zone.HALO))

    horizontal_start = icon_grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5))
    horizontal_end = icon_grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_2))

    vt_ref = savepoint_compute_edge_diagnostics_for_velocity_advection_exit.vt()
    z_vt_ie_ref = savepoint_velocity_exit.z_vt_ie()
    vn_ie_ref = savepoint_compute_edge_diagnostics_for_velocity_advection_exit.vn_ie()
    z_kin_hor_e_ref = savepoint_compute_edge_diagnostics_for_velocity_advection_exit.z_kin_hor_e()
    z_w_concorr_me_ref = (
        savepoint_compute_edge_diagnostics_for_velocity_advection_exit.z_w_concorr_me()
    )
    z_v_grad_w_ref = savepoint_compute_edge_diagnostics_for_velocity_advection_exit.z_v_grad_w()

    compute_edge_diagnostics_for_velocity_advection.compute_derived_horizontal_winds_and_ke_and_horizontal_advection_of_w_and_contravariant_correction.with_backend(
        backend
    )(
        tangential_wind=tangential_wind,
        tangential_wind_on_half_levels=tangential_wind_on_half_levels,
        vn_on_half_levels=vn_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
        contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
        horizontal_advection_of_w_at_edges_on_half_levels=horizontal_advection_of_w_at_edges_on_half_levels,
        vn=vn,
        w=w,
        rbf_vec_coeff_e=rbf_vec_coeff_e,
        wgtfac_e=wgtfac_e,
        ddxn_z_full=ddxn_z_full,
        ddxt_z_full=ddxt_z_full,
        wgtfacq_e=wgtfacq_e,
        c_intp=c_intp,
        inv_dual_edge_length=inv_dual_edge_length,
        inv_primal_edge_length=inv_primal_edge_length,
        tangent_orientation=tangent_orientation,
        skip_compute_predictor_vertical_advection=skip_compute_predictor_vertical_advection,
        k=k,
        edge=edge,
        nflatlev=gtx.int32(nflatlev),
        nlev=gtx.int32(icon_grid.num_levels),
        lateral_boundary_7=lateral_boundary_7,
        halo_1=halo_1,
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(icon_grid.num_levels + 1),
        offset_provider={
            "E2C": icon_grid.get_offset_provider("E2C"),
            "E2V": icon_grid.get_offset_provider("E2V"),
            "V2C": icon_grid.get_offset_provider("V2C"),
            "E2C2E": icon_grid.get_offset_provider("E2C2E"),
            "Koff": dims.KDim,
        },
    )

    assert helpers.dallclose(
        vt_ref.asnumpy(), tangential_wind.asnumpy(), rtol=1.0e-14, atol=1.0e-14
    )
    assert helpers.dallclose(
        z_vt_ie_ref.asnumpy(), tangential_wind_on_half_levels.asnumpy(), rtol=1.0e-14, atol=1.0e-14
    )
    assert helpers.dallclose(
        vn_ie_ref.asnumpy(), vn_on_half_levels.asnumpy(), rtol=1.0e-15, atol=1.0e-15
    )
    assert helpers.dallclose(
        z_kin_hor_e_ref.asnumpy(),
        horizontal_kinetic_energy_at_edges_on_model_levels.asnumpy(),
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    assert helpers.dallclose(
        z_w_concorr_me_ref.asnumpy(),
        contravariant_correction_at_edges_on_model_levels.asnumpy(),
        rtol=1.0e-15,
        atol=1.0e-15,
    )
    assert helpers.dallclose(
        z_v_grad_w_ref.asnumpy(),
        horizontal_advection_of_w_at_edges_on_half_levels.asnumpy(),
        rtol=1.0e-15,
        atol=1.0e-15,
    )


@pytest.mark.dataset
@pytest.mark.parametrize(
    "experiment, step_date_init, step_date_exit",
    [
        (dt_utils.REGIONAL_EXPERIMENT, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000"),
        (dt_utils.GLOBAL_EXPERIMENT, "2000-01-01T00:00:02.000", "2000-01-01T00:00:02.000"),
    ],
)
@pytest.mark.parametrize("istep_init", [2])
def test_compute_edge_diagnostics_for_velocity_advection_in_corrector_step(
    icon_grid,
    grid_savepoint,
    savepoint_compute_edge_diagnostics_for_velocity_advection_init,
    savepoint_compute_edge_diagnostics_for_velocity_advection_exit,
    interpolation_savepoint,
    metrics_savepoint,
    savepoint_velocity_init,
    savepoint_velocity_exit,
    step_date_init,
    step_date_exit,
    substep_init,
    istep_init,
    backend,
):
    edge_domain = h_grid.domain(dims.EdgeDim)
    vertex_domain = h_grid.domain(dims.VertexDim)

    tangential_wind_on_half_levels = (
        savepoint_compute_edge_diagnostics_for_velocity_advection_init.z_vt_ie()
    )
    vn_on_half_levels = savepoint_compute_edge_diagnostics_for_velocity_advection_init.vn_ie()
    horizontal_advection_of_w_at_edges_on_half_levels = (
        savepoint_compute_edge_diagnostics_for_velocity_advection_init.z_v_grad_w()
    )
    w = savepoint_compute_edge_diagnostics_for_velocity_advection_init.w()

    c_intp = interpolation_savepoint.c_intp()
    inv_dual_edge_length = grid_savepoint.inv_dual_edge_length()
    inv_primal_edge_length = grid_savepoint.inverse_primal_edge_lengths()
    tangent_orientation = grid_savepoint.tangent_orientation()

    edge = data_alloc.index_field(dim=dims.EdgeDim, grid=icon_grid, backend=backend)
    vertex = data_alloc.index_field(dim=dims.VertexDim, grid=icon_grid, backend=backend)
    lateral_boundary_7 = icon_grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7))
    halo_1 = icon_grid.end_index(edge_domain(h_grid.Zone.HALO))

    horizontal_start = icon_grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7))
    horizontal_end = icon_grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_2))

    z_v_grad_w_ref = savepoint_compute_edge_diagnostics_for_velocity_advection_exit.z_v_grad_w()
    start_vertex_lateral_boundary_level_2 = icon_grid.start_index(
        vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    end_vertex_halo = icon_grid.end_index(vertex_domain(h_grid.Zone.HALO))

    compute_edge_diagnostics_for_velocity_advection.compute_horizontal_advection_of_w.with_backend(
        backend
    )(
        horizontal_advection_of_w_at_edges_on_half_levels=horizontal_advection_of_w_at_edges_on_half_levels,
        w=w,
        tangential_wind_on_half_levels=tangential_wind_on_half_levels,
        vn_on_half_levels=vn_on_half_levels,
        c_intp=c_intp,
        inv_dual_edge_length=inv_dual_edge_length,
        inv_primal_edge_length=inv_primal_edge_length,
        tangent_orientation=tangent_orientation,
        edge=edge,
        vertex=vertex,
        lateral_boundary_7=lateral_boundary_7,
        halo_1=halo_1,
        start_vertex_lateral_boundary_level_2=start_vertex_lateral_boundary_level_2,
        end_vertex_halo=end_vertex_halo,
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(icon_grid.num_levels),
        offset_provider={
            "E2C": icon_grid.get_offset_provider("E2C"),
            "E2V": icon_grid.get_offset_provider("E2V"),
            "V2C": icon_grid.get_offset_provider("V2C"),
            "E2C2E": icon_grid.get_offset_provider("E2C2E"),
            "Koff": dims.KDim,
        },
    )

    assert helpers.dallclose(
        z_v_grad_w_ref.asnumpy()[horizontal_start:horizontal_end, :],
        horizontal_advection_of_w_at_edges_on_half_levels.asnumpy()[
            horizontal_start:horizontal_end, :
        ],
        rtol=1.0e-15,
        atol=1.0e-15,
    )


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "experiment, step_date_init, step_date_exit",
    [
        (dt_utils.REGIONAL_EXPERIMENT, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000"),
        (dt_utils.GLOBAL_EXPERIMENT, "2000-01-01T00:00:02.000", "2000-01-01T00:00:02.000"),
    ],
)
@pytest.mark.parametrize("istep_init", [1])
def test_compute_cell_diagnostics_for_velocity_advection_predictor(
    icon_grid,
    grid_savepoint,
    savepoint_compute_cell_diagnostics_for_velocity_advection_init,
    savepoint_compute_cell_diagnostics_for_velocity_advection_exit,
    metrics_savepoint,
    interpolation_savepoint,
    istep_init,
    substep_init,
    substep_exit,
    step_date_init,
    step_date_exit,
    backend,
):
    cell_domain = h_grid.domain(dims.CellDim)
    z_ekinh_ref = savepoint_compute_cell_diagnostics_for_velocity_advection_exit.z_ekinh()
    w_concorr_c_ref = savepoint_compute_cell_diagnostics_for_velocity_advection_exit.w_concorr_c()
    z_w_con_c_ref = savepoint_compute_cell_diagnostics_for_velocity_advection_exit.z_w_con_c()

    horizontal_kinetic_energy_at_edges_on_model_levels = (
        savepoint_compute_cell_diagnostics_for_velocity_advection_init.z_kin_hor_e()
    )
    contravariant_correction_at_edges_on_model_levels = (
        savepoint_compute_cell_diagnostics_for_velocity_advection_init.z_w_concorr_me()
    )
    w = savepoint_compute_cell_diagnostics_for_velocity_advection_init.w()
    contravariant_correction_at_cells_on_half_levels = (
        savepoint_compute_cell_diagnostics_for_velocity_advection_init.w_concorr_c()
    )
    horizontal_kinetic_energy_at_cells_on_model_levels = (
        savepoint_compute_cell_diagnostics_for_velocity_advection_init.z_ekinh()
    )
    contravariant_corrected_w_at_cells_on_half_levels = (
        savepoint_compute_cell_diagnostics_for_velocity_advection_init.z_w_con_c()
    )

    e_bln_c_s = data_alloc.flatten_first_two_dims(
        dims.CEDim, field=interpolation_savepoint.e_bln_c_s()
    )
    wgtfac_c = metrics_savepoint.wgtfac_c()
    k = data_alloc.index_field(
        dim=dims.KDim, grid=icon_grid, extend={dims.KDim: 1}, backend=backend
    )
    nflatlev = grid_savepoint.nflatlev()
    lateral_boundary_4 = icon_grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4))
    end_halo = icon_grid.end_index(cell_domain(h_grid.Zone.HALO))

    compute_cell_diagnostics_for_velocity_advection.interpolate_horizontal_kinetic_energy_to_cells_and_compute_contravariant_terms.with_backend(
        backend
    )(
        horizontal_kinetic_energy_at_cells_on_model_levels=horizontal_kinetic_energy_at_cells_on_model_levels,
        contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
        contravariant_corrected_w_at_cells_on_half_levels=contravariant_corrected_w_at_cells_on_half_levels,
        w=w,
        horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
        contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
        e_bln_c_s=e_bln_c_s,
        wgtfac_c=wgtfac_c,
        k=k,
        nflatlev=nflatlev,
        nlev=icon_grid.num_levels,
        horizontal_start=lateral_boundary_4,
        horizontal_end=end_halo,
        vertical_start=0,
        vertical_end=icon_grid.num_levels + 1,
        offset_provider={
            "C2E": icon_grid.get_offset_provider("C2E"),
            "C2CE": icon_grid.get_offset_provider("C2CE"),
            "Koff": dims.KDim,
        },
    )

    assert helpers.dallclose(
        z_ekinh_ref.asnumpy(),
        horizontal_kinetic_energy_at_cells_on_model_levels.asnumpy(),
        rtol=1.0e-15,
        atol=1.0e-15,
    )
    assert helpers.dallclose(
        w_concorr_c_ref.asnumpy(),
        contravariant_correction_at_cells_on_half_levels.asnumpy(),
        rtol=1.0e-15,
        atol=1.0e-15,
    )
    assert helpers.dallclose(
        z_w_con_c_ref.asnumpy(),
        contravariant_corrected_w_at_cells_on_half_levels.asnumpy(),
        rtol=1.0e-15,
        atol=1.0e-15,
    )


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "experiment, step_date_init, step_date_exit",
    [
        (dt_utils.REGIONAL_EXPERIMENT, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000"),
        (dt_utils.GLOBAL_EXPERIMENT, "2000-01-01T00:00:02.000", "2000-01-01T00:00:02.000"),
    ],
)
@pytest.mark.parametrize("istep_init", [2])
def test_compute_cell_diagnostics_for_velocity_advection_corrector(
    icon_grid,
    grid_savepoint,
    savepoint_compute_cell_diagnostics_for_velocity_advection_init,
    savepoint_compute_cell_diagnostics_for_velocity_advection_exit,
    metrics_savepoint,
    interpolation_savepoint,
    istep_init,
    substep_init,
    substep_exit,
    step_date_init,
    step_date_exit,
    backend,
):
    cell_domain = h_grid.domain(dims.CellDim)
    z_ekinh_ref = savepoint_compute_cell_diagnostics_for_velocity_advection_exit.z_ekinh()
    w_concorr_c_ref = savepoint_compute_cell_diagnostics_for_velocity_advection_exit.w_concorr_c()
    z_w_con_c_ref = savepoint_compute_cell_diagnostics_for_velocity_advection_exit.z_w_con_c()

    horizontal_kinetic_energy_at_edges_on_model_levels = (
        savepoint_compute_cell_diagnostics_for_velocity_advection_init.z_kin_hor_e()
    )
    w = savepoint_compute_cell_diagnostics_for_velocity_advection_init.w()
    contravariant_correction_at_cells_on_half_levels = (
        savepoint_compute_cell_diagnostics_for_velocity_advection_init.w_concorr_c()
    )
    horizontal_kinetic_energy_at_cells_on_model_levels = (
        savepoint_compute_cell_diagnostics_for_velocity_advection_init.z_ekinh()
    )
    contravariant_corrected_w_at_cells_on_half_levels = (
        savepoint_compute_cell_diagnostics_for_velocity_advection_init.z_w_con_c()
    )

    e_bln_c_s = data_alloc.flatten_first_two_dims(
        dims.CEDim, field=interpolation_savepoint.e_bln_c_s()
    )
    k = data_alloc.index_field(
        dim=dims.KDim, grid=icon_grid, extend={dims.KDim: 1}, backend=backend
    )
    nflatlev = grid_savepoint.nflatlev()
    lateral_boundary_4 = icon_grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4))
    end_halo = icon_grid.end_index(cell_domain(h_grid.Zone.HALO))

    compute_cell_diagnostics_for_velocity_advection.interpolate_horizontal_kinetic_energy_to_cells_and_compute_contravariant_corrected_w.with_backend(
        backend
    )(
        horizontal_kinetic_energy_at_cells_on_model_levels=horizontal_kinetic_energy_at_cells_on_model_levels,
        contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
        contravariant_corrected_w_at_cells_on_half_levels=contravariant_corrected_w_at_cells_on_half_levels,
        w=w,
        horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
        e_bln_c_s=e_bln_c_s,
        k=k,
        nflatlev=nflatlev,
        nlev=icon_grid.num_levels,
        # TODO: serialization test works for lateral_boundary_4 but not on lateral_boundary_3, but it should be in lateral_boundary_3 in driver code
        horizontal_start=lateral_boundary_4,
        horizontal_end=end_halo,
        vertical_start=0,
        vertical_end=icon_grid.num_levels + 1,
        offset_provider={
            "C2E": icon_grid.get_offset_provider("C2E"),
            "C2CE": icon_grid.get_offset_provider("C2CE"),
            "Koff": dims.KDim,
        },
    )

    assert helpers.dallclose(
        z_ekinh_ref.asnumpy(),
        horizontal_kinetic_energy_at_cells_on_model_levels.asnumpy(),
        rtol=1.0e-15,
        atol=1.0e-15,
    )
    assert helpers.dallclose(
        w_concorr_c_ref.asnumpy(),
        contravariant_correction_at_cells_on_half_levels.asnumpy(),
        rtol=1.0e-15,
        atol=1.0e-15,
    )
    assert helpers.dallclose(
        z_w_con_c_ref.asnumpy(),
        contravariant_corrected_w_at_cells_on_half_levels.asnumpy(),
        rtol=1.0e-15,
        atol=1.0e-15,
    )


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "experiment, step_date_init, step_date_exit",
    [
        (dt_utils.REGIONAL_EXPERIMENT, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000"),
        (dt_utils.GLOBAL_EXPERIMENT, "2000-01-01T00:00:02.000", "2000-01-01T00:00:02.000"),
    ],
)
@pytest.mark.parametrize("istep_init", [1, 2])
def test_compute_advection_in_vertical_momentum_equation(
    icon_grid,
    grid_savepoint,
    savepoint_compute_advection_in_vertical_momentum_equation_init,
    savepoint_compute_advection_in_vertical_momentum_equation_exit,
    savepoint_compute_cell_diagnostics_for_velocity_advection_exit,
    interpolation_savepoint,
    metrics_savepoint,
    savepoint_velocity_exit,
    backend,
    savepoint_velocity_init,
    step_date_init,
    step_date_exit,
    substep_init,
    substep_exit,
    istep_init,
):
    scalfac_exdiff = savepoint_velocity_init.scalfac_exdiff()
    cfl_w_limit = savepoint_velocity_init.cfl_w_limit()
    ddqz_z_half = metrics_savepoint.ddqz_z_half()
    # calculate cfl_clipping
    z_w_con_c_8_13 = (
        savepoint_compute_cell_diagnostics_for_velocity_advection_exit.z_w_con_c().asnumpy()
    )
    cfl_clipping_np = z_w_con_c_8_13 > (cfl_w_limit * ddqz_z_half.asnumpy())
    cfl_clipping = gtx.as_field((dims.CellDim, dims.KDim), cfl_clipping_np)
    contravariant_corrected_w_at_cells_on_half_levels = (
        savepoint_compute_advection_in_vertical_momentum_equation_init.z_w_con_c()
    )
    w = savepoint_compute_advection_in_vertical_momentum_equation_init.w()
    vertical_wind_advective_tendency = (
        savepoint_compute_advection_in_vertical_momentum_equation_init.ddt_w_adv()
    )
    horizontal_advection_of_w_at_edges_on_half_levels = (
        savepoint_compute_advection_in_vertical_momentum_equation_init.z_v_grad_w()
    )
    levmask = savepoint_compute_advection_in_vertical_momentum_equation_init.levmask()
    contravariant_corrected_w_at_cells_on_model_levels = (
        savepoint_compute_advection_in_vertical_momentum_equation_init.z_w_con_c_full()
    )
    skip_compute_predictor_vertical_advection = (
        savepoint_compute_advection_in_vertical_momentum_equation_init.lvn_only()
    )

    coeff1_dwdz = metrics_savepoint.coeff1_dwdz()
    coeff2_dwdz = metrics_savepoint.coeff2_dwdz()
    e_bln_c_s = data_alloc.flatten_first_two_dims(
        dims.CEDim, field=interpolation_savepoint.e_bln_c_s()
    )
    owner_mask = grid_savepoint.c_owner_mask()
    area = grid_savepoint.cell_areas()
    geofac_n2s = interpolation_savepoint.geofac_n2s()

    z_w_con_c_full_ref = (
        savepoint_compute_advection_in_vertical_momentum_equation_exit.z_w_con_c_full()
    )
    ddt_w_adv_ref = savepoint_compute_advection_in_vertical_momentum_equation_exit.ddt_w_adv()

    k = data_alloc.index_field(dim=dims.KDim, grid=icon_grid, backend=backend)
    cell = data_alloc.index_field(dim=dims.CellDim, grid=icon_grid, backend=backend)

    nrdmax = grid_savepoint.nrdmax()[0]

    cell_domain = h_grid.domain(dims.CellDim)
    cell_lower_bound = icon_grid.start_index(cell_domain(h_grid.Zone.NUDGING))
    cell_upper_bound = icon_grid.end_index(cell_domain(h_grid.Zone.LOCAL))

    dtime = 5.0
    start_cell_lateral_boundary = (
        icon_grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4))
        if istep_init == 1
        else icon_grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3))
    )
    end_cell_halo = icon_grid.end_index(cell_domain(h_grid.Zone.HALO))
    horizontal_start = icon_grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4))
    horizontal_end = icon_grid.end_index(cell_domain(h_grid.Zone.HALO))
    vertical_start = 0
    vertical_end = icon_grid.num_levels
    compute_advection_in_vertical_momentum_equation.compute_advection_in_vertical_momentum_equation.with_backend(
        backend
    )(
        contravariant_corrected_w_at_cells_on_model_levels=contravariant_corrected_w_at_cells_on_model_levels,
        vertical_wind_advective_tendency=vertical_wind_advective_tendency,
        w=w,
        contravariant_corrected_w_at_cells_on_half_levels=contravariant_corrected_w_at_cells_on_half_levels,
        horizontal_advection_of_w_at_edges_on_half_levels=horizontal_advection_of_w_at_edges_on_half_levels,
        coeff1_dwdz=coeff1_dwdz,
        coeff2_dwdz=coeff2_dwdz,
        e_bln_c_s=e_bln_c_s,
        ddqz_z_half=ddqz_z_half,
        area=area,
        geofac_n2s=geofac_n2s,
        levelmask=levmask,
        cfl_clipping=cfl_clipping,
        owner_mask=owner_mask,
        scalfac_exdiff=scalfac_exdiff,
        cfl_w_limit=cfl_w_limit,
        dtime=dtime,
        skip_compute_predictor_vertical_advection=skip_compute_predictor_vertical_advection,
        cell=cell,
        k=k,
        cell_lower_bound=cell_lower_bound,
        cell_upper_bound=cell_upper_bound,
        nlev=icon_grid.num_levels,
        nrdmax=nrdmax,
        start_cell_lateral_boundary=start_cell_lateral_boundary,
        end_cell_halo=end_cell_halo,
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        vertical_start=vertical_start,
        vertical_end=vertical_end,
        offset_provider={
            "C2E": icon_grid.get_offset_provider("C2E"),
            "C2CE": icon_grid.get_offset_provider("C2CE"),
            "C2E2CO": icon_grid.get_offset_provider("C2E2CO"),
            "Koff": dims.KDim,
        },
    )

    assert helpers.dallclose(
        z_w_con_c_full_ref.asnumpy(),
        contravariant_corrected_w_at_cells_on_model_levels.asnumpy(),
        rtol=1.0e-15,
        atol=1.0e-15,
    )
    assert helpers.dallclose(
        ddt_w_adv_ref.asnumpy(),
        vertical_wind_advective_tendency.asnumpy(),
        rtol=1.0e-15,
        atol=1.0e-15,
    )


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "experiment, step_date_init, step_date_exit",
    [
        (dt_utils.REGIONAL_EXPERIMENT, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000"),
        (dt_utils.GLOBAL_EXPERIMENT, "2000-01-01T00:00:02.000", "2000-01-01T00:00:02.000"),
    ],
)
@pytest.mark.parametrize("istep_init", [1, 2])
def test_compute_advection_in_horizontal_momentum_equation(
    icon_grid,
    grid_savepoint,
    savepoint_compute_advection_in_horizontal_momentum_equation_init,
    savepoint_compute_advection_in_horizontal_momentum_equation_exit,
    interpolation_savepoint,
    metrics_savepoint,
    backend,
    savepoint_velocity_init,
    istep_init,
    substep_init,
    step_date_init,
    step_date_exit,
):
    vn = savepoint_compute_advection_in_horizontal_momentum_equation_init.vn()
    horizontal_kinetic_energy_at_edges_on_model_levels = (
        savepoint_compute_advection_in_horizontal_momentum_equation_init.z_kin_hor_e()
    )
    horizontal_kinetic_energy_at_cells_on_model_levels = (
        savepoint_compute_advection_in_horizontal_momentum_equation_init.z_ekinh()
    )
    tangential_wind = savepoint_compute_advection_in_horizontal_momentum_equation_init.vt()
    contravariant_corrected_w_at_cells_on_model_levels = (
        savepoint_compute_advection_in_horizontal_momentum_equation_init.z_w_con_c_full()
    )
    vn_on_half_levels = savepoint_compute_advection_in_horizontal_momentum_equation_init.vn_ie()
    levelmask = savepoint_compute_advection_in_horizontal_momentum_equation_init.levelmask()
    normal_wind_advective_tendency = (
        savepoint_compute_advection_in_horizontal_momentum_equation_init.ddt_vn_apc()
    )

    geofac_rot = interpolation_savepoint.geofac_rot()
    coeff_gradekin = metrics_savepoint.coeff_gradekin()
    coriolis_frequency = grid_savepoint.f_e()
    c_lin_e = interpolation_savepoint.c_lin_e()
    ddqz_z_full_e = metrics_savepoint.ddqz_z_full_e()
    area_edge = grid_savepoint.edge_areas()
    tangent_orientation = grid_savepoint.tangent_orientation()
    inv_primal_edge_length = grid_savepoint.inverse_primal_edge_lengths()
    geofac_grdiv = interpolation_savepoint.geofac_grdiv()
    k = data_alloc.index_field(dim=dims.KDim, grid=icon_grid, backend=backend)
    vertex = data_alloc.index_field(dim=dims.VertexDim, grid=icon_grid, backend=backend)
    edge = data_alloc.index_field(dim=dims.EdgeDim, grid=icon_grid, backend=backend)

    edge_domain = h_grid.domain(dims.EdgeDim)
    vertex_domain = h_grid.domain(dims.VertexDim)

    start_vertex_lateral_boundary_level_2 = icon_grid.start_index(
        vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    end_vertex_halo = icon_grid.end_index(vertex_domain(h_grid.Zone.HALO))
    start_edge_nudging_level_2 = icon_grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2))
    end_edge_local = icon_grid.end_index(edge_domain(h_grid.Zone.LOCAL))

    d_time = savepoint_velocity_init.get_metadata("dtime").get("dtime")
    nrdmax = grid_savepoint.nrdmax()[0]

    ddt_vn_apc_ref = savepoint_compute_advection_in_horizontal_momentum_equation_exit.ddt_vn_apc()

    scalfac_exdiff = savepoint_velocity_init.scalfac_exdiff()
    cfl_w_limit = savepoint_velocity_init.cfl_w_limit()

    compute_advection_in_horizontal_momentum_equation.compute_advection_in_horizontal_momentum_equation.with_backend(
        backend
    )(
        normal_wind_advective_tendency=normal_wind_advective_tendency,
        vn=vn,
        horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
        horizontal_kinetic_energy_at_cells_on_model_levels=horizontal_kinetic_energy_at_cells_on_model_levels,
        tangential_wind=tangential_wind,
        coriolis_frequency=coriolis_frequency,
        contravariant_corrected_w_at_cells_on_model_levels=contravariant_corrected_w_at_cells_on_model_levels,
        vn_on_half_levels=vn_on_half_levels,
        geofac_rot=geofac_rot,
        coeff_gradekin=coeff_gradekin,
        c_lin_e=c_lin_e,
        ddqz_z_full_e=ddqz_z_full_e,
        levelmask=levelmask,
        area_edge=area_edge,
        tangent_orientation=tangent_orientation,
        inv_primal_edge_length=inv_primal_edge_length,
        geofac_grdiv=geofac_grdiv,
        k=k,
        vertex=vertex,
        edge=edge,
        cfl_w_limit=cfl_w_limit,
        scalfac_exdiff=scalfac_exdiff,
        d_time=d_time,
        nlev=icon_grid.num_levels,
        nrdmax=nrdmax,
        start_vertex_lateral_boundary_level_2=start_vertex_lateral_boundary_level_2,
        end_vertex_halo=end_vertex_halo,
        start_edge_nudging_level_2=start_edge_nudging_level_2,
        end_edge_local=end_edge_local,
        horizontal_start=0,
        horizontal_end=icon_grid.num_edges,
        vertical_start=0,
        vertical_end=icon_grid.num_levels,
        offset_provider={
            "V2E": icon_grid.get_offset_provider("V2E"),
            "E2EC": icon_grid.get_offset_provider("E2EC"),
            "E2V": icon_grid.get_offset_provider("E2V"),
            "E2C": icon_grid.get_offset_provider("E2C"),
            "E2C2EO": icon_grid.get_offset_provider("E2C2EO"),
            "Koff": dims.KDim,
        },
    )

    assert helpers.dallclose(
        ddt_vn_apc_ref.asnumpy(),
        normal_wind_advective_tendency.asnumpy(),
        rtol=1.0e-15,
        atol=1.0e-15,
    )
