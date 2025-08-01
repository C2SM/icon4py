# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import icon4py.model.common.grid.states as grid_states
import icon4py.model.common.utils as common_utils
from icon4py.model.atmosphere.diffusion import diffusion
from icon4py.model.atmosphere.dycore import dycore_states, solve_nonhydro as solve_nh
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.driver import (
    icon4py_configuration,
    icon4py_driver,
    serialbox_helpers as driver_sb,
)
from icon4py.model.testing import datatest_utils as dt_utils, helpers
from icon4py.model.testing.fixtures.datatest import backend

from ..fixtures import *  # noqa: F403
from ..utils import (
    construct_diffusion_config,
    construct_icon4pyrun_config,
    construct_nonhydrostatic_config,
)


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment, istep_init, istep_exit, substep_init, substep_exit, timeloop_date_init, timeloop_date_exit, step_date_init, step_date_exit, timeloop_diffusion_linit_init, timeloop_diffusion_linit_exit, vn_only",
    [
        (
            dt_utils.REGIONAL_EXPERIMENT,
            1,
            2,
            1,
            2,
            "2021-06-20T12:00:00.000",
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
            True,
            False,
            False,
        ),
        (
            dt_utils.REGIONAL_EXPERIMENT,
            1,
            2,
            1,
            2,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:20.000",
            "2021-06-20T12:00:20.000",
            "2021-06-20T12:00:20.000",
            False,
            False,
            True,
        ),
        (
            dt_utils.GLOBAL_EXPERIMENT,
            1,
            2,
            1,
            2,
            "2000-01-01T00:00:00.000",
            "2000-01-01T00:00:02.000",
            "2000-01-01T00:00:02.000",
            "2000-01-01T00:00:02.000",
            False,
            False,
            False,
        ),
        (
            dt_utils.GLOBAL_EXPERIMENT,
            1,
            2,
            1,
            2,
            "2000-01-01T00:00:02.000",
            "2000-01-01T00:00:04.000",
            "2000-01-01T00:00:04.000",
            "2000-01-01T00:00:04.000",
            False,
            False,
            True,
        ),
        (
            dt_utils.GAUSS3D_EXPERIMENT,
            1,
            2,
            1,
            5,
            "2001-01-01T00:00:00.000",
            "2001-01-01T00:00:04.000",
            "2001-01-01T00:00:04.000",
            "2001-01-01T00:00:04.000",
            False,
            False,
            False,
        ),
    ],
)
def test_run_timeloop_single_step(
    experiment,
    istep_init,
    istep_exit,
    substep_init,
    substep_exit,
    timeloop_date_init,
    timeloop_date_exit,
    step_date_init,
    step_date_exit,
    timeloop_diffusion_linit_init,
    timeloop_diffusion_linit_exit,
    vn_only,
    *,
    grid_savepoint,
    icon_grid,
    metrics_savepoint,
    interpolation_savepoint,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    ndyn_substeps,
    timeloop_diffusion_savepoint_init,
    timeloop_diffusion_savepoint_exit,
    savepoint_velocity_init,
    savepoint_nonhydro_init,
    savepoint_nonhydro_exit,
    backend,
):
    if experiment == dt_utils.GAUSS3D_EXPERIMENT:
        config = icon4py_configuration.read_config(
            experiment_type=experiment,
            backend=backend,
        )
        diffusion_config = config.diffusion_config
        nonhydro_config = config.solve_nonhydro_config
        icon4pyrun_config = config.run_config

    else:
        diffusion_config = construct_diffusion_config(experiment, ndyn_substeps=ndyn_substeps)
        nonhydro_config = construct_nonhydrostatic_config(experiment)
        icon4pyrun_config = construct_icon4pyrun_config(
            experiment,
            timeloop_date_init,
            timeloop_date_exit,
            timeloop_diffusion_linit_init,
            ndyn_substeps=ndyn_substeps,
            backend=backend,
        )

    edge_geometry: grid_states.EdgeParams = grid_savepoint.construct_edge_geometry()
    cell_geometry: grid_states.CellParams = grid_savepoint.construct_cell_geometry()

    diffusion_interpolation_state = driver_sb.construct_interpolation_state_for_diffusion(
        interpolation_savepoint
    )
    diffusion_metric_state = driver_sb.construct_metric_state_for_diffusion(metrics_savepoint)

    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
        _min_index_flat_horizontal_grad_pressure=grid_savepoint.nflat_gradp(),
    )
    additional_parameters = diffusion.DiffusionParams(diffusion_config)

    diffusion_granule = diffusion.Diffusion(
        grid=icon_grid,
        config=diffusion_config,
        params=additional_parameters,
        vertical_grid=vertical_params,
        metric_state=diffusion_metric_state,
        interpolation_state=diffusion_interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
        backend=backend,
    )

    sp = savepoint_nonhydro_init
    nonhydro_params = solve_nh.NonHydrostaticParams(nonhydro_config)
    sp_v = savepoint_velocity_init
    do_prep_adv = sp.get_metadata("prep_adv").get("prep_adv")

    linit = sp.get_metadata("linit").get("linit")

    grg = interpolation_savepoint.geofac_grg()
    nonhydro_interpolation_state = dycore_states.InterpolationState(
        c_lin_e=interpolation_savepoint.c_lin_e(),
        c_intp=interpolation_savepoint.c_intp(),
        e_flx_avg=interpolation_savepoint.e_flx_avg(),
        geofac_grdiv=interpolation_savepoint.geofac_grdiv(),
        geofac_rot=interpolation_savepoint.geofac_rot(),
        pos_on_tplane_e_1=interpolation_savepoint.pos_on_tplane_e_x(),
        pos_on_tplane_e_2=interpolation_savepoint.pos_on_tplane_e_y(),
        rbf_vec_coeff_e=interpolation_savepoint.rbf_vec_coeff_e(),
        e_bln_c_s=data_alloc.flatten_first_two_dims(
            dims.CEDim,
            field=interpolation_savepoint.e_bln_c_s(),
            backend=backend,
        ),
        rbf_coeff_1=interpolation_savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=interpolation_savepoint.rbf_vec_coeff_v2(),
        geofac_div=data_alloc.flatten_first_two_dims(
            dims.CEDim,
            field=interpolation_savepoint.geofac_div(),
            backend=backend,
        ),
        geofac_n2s=interpolation_savepoint.geofac_n2s(),
        geofac_grg_x=grg[0],
        geofac_grg_y=grg[1],
        nudgecoeff_e=interpolation_savepoint.nudgecoeff_e(),
    )
    nonhydro_metric_state = dycore_states.MetricStateNonHydro(
        bdy_halo_c=metrics_savepoint.bdy_halo_c(),
        mask_prog_halo_c=metrics_savepoint.mask_prog_halo_c(),
        rayleigh_w=metrics_savepoint.rayleigh_w(),
        time_extrapolation_parameter_for_exner=metrics_savepoint.exner_exfac(),
        reference_exner_at_cells_on_model_levels=metrics_savepoint.exner_ref_mc(),
        wgtfac_c=metrics_savepoint.wgtfac_c(),
        wgtfacq_c=metrics_savepoint.wgtfacq_c_dsl(),
        inv_ddqz_z_full=metrics_savepoint.inv_ddqz_z_full(),
        reference_rho_at_cells_on_model_levels=metrics_savepoint.rho_ref_mc(),
        reference_theta_at_cells_on_model_levels=metrics_savepoint.theta_ref_mc(),
        exner_w_explicit_weight_parameter=metrics_savepoint.vwind_expl_wgt(),
        ddz_of_reference_exner_at_cells_on_half_levels=metrics_savepoint.d_exner_dz_ref_ic(),
        ddqz_z_half=metrics_savepoint.ddqz_z_half(),
        reference_theta_at_cells_on_half_levels=metrics_savepoint.theta_ref_ic(),
        d2dexdz2_fac1_mc=metrics_savepoint.d2dexdz2_fac1_mc(),
        d2dexdz2_fac2_mc=metrics_savepoint.d2dexdz2_fac2_mc(),
        reference_rho_at_edges_on_model_levels=metrics_savepoint.rho_ref_me(),
        reference_theta_at_edges_on_model_levels=metrics_savepoint.theta_ref_me(),
        ddxn_z_full=metrics_savepoint.ddxn_z_full(),
        zdiff_gradp=metrics_savepoint.zdiff_gradp(),
        vertoffset_gradp=metrics_savepoint.vertoffset_gradp(),
        pg_edgeidx_dsl=metrics_savepoint.pg_edgeidx_dsl(),
        pg_exdist=metrics_savepoint.pg_exdist(),
        ddqz_z_full_e=metrics_savepoint.ddqz_z_full_e(),
        ddxt_z_full=metrics_savepoint.ddxt_z_full(),
        wgtfac_e=metrics_savepoint.wgtfac_e(),
        wgtfacq_e=metrics_savepoint.wgtfacq_e_dsl(icon_grid.num_levels),
        exner_w_implicit_weight_parameter=metrics_savepoint.vwind_impl_wgt(),
        horizontal_mask_for_3d_divdamp=metrics_savepoint.hmask_dd3d(),
        scaling_factor_for_3d_divdamp=metrics_savepoint.scalfac_dd3d(),
        coeff1_dwdz=metrics_savepoint.coeff1_dwdz(),
        coeff2_dwdz=metrics_savepoint.coeff2_dwdz(),
        coeff_gradekin=metrics_savepoint.coeff_gradekin(),
    )

    solve_nonhydro_granule = solve_nh.SolveNonhydro(
        grid=icon_grid,
        config=nonhydro_config,
        params=nonhydro_params,
        metric_state_nonhydro=nonhydro_metric_state,
        interpolation_state=nonhydro_interpolation_state,
        vertical_params=vertical_params,
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
        owner_mask=grid_savepoint.c_owner_mask(),
        backend=backend,
    )

    diffusion_diagnostic_state = driver_sb.construct_diagnostics_for_diffusion(
        timeloop_diffusion_savepoint_init,
    )

    prep_adv = dycore_states.PrepAdvection(
        vn_traj=sp.vn_traj(),
        mass_flx_me=sp.mass_flx_me(),
        dynamical_vertical_mass_flux_at_cells_on_half_levels=sp.mass_flx_ic(),
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels=data_alloc.zero_field(
            icon_grid,
            dims.CellDim,
            dims.KDim,
            backend=backend,
        ),
    )

    current_index, next_index = (1, 0) if not linit else (0, 1)
    nonhydro_diagnostic_state = dycore_states.DiagnosticStateNonHydro(
        max_vertical_cfl=0.0,
        theta_v_at_cells_on_half_levels=sp.theta_v_ic(),
        perturbed_exner_at_cells_on_model_levels=sp.exner_pr(),
        rho_at_cells_on_half_levels=sp.rho_ic(),
        exner_tendency_due_to_slow_physics=sp.ddt_exner_phy(),
        grf_tend_rho=sp.grf_tend_rho(),
        grf_tend_thv=sp.grf_tend_thv(),
        grf_tend_w=sp.grf_tend_w(),
        mass_flux_at_edges_on_model_levels=sp.mass_fl_e(),
        normal_wind_tendency_due_to_slow_physics_process=sp.ddt_vn_phy(),
        grf_tend_vn=sp.grf_tend_vn(),
        normal_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            sp_v.ddt_vn_apc_pc(0), sp_v.ddt_vn_apc_pc(1)
        ),
        vertical_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            sp_v.ddt_w_adv_pc(current_index), sp_v.ddt_w_adv_pc(next_index)
        ),
        tangential_wind=sp_v.vt(),
        vn_on_half_levels=sp_v.vn_ie(),
        contravariant_correction_at_cells_on_half_levels=sp_v.w_concorr_c(),
        rho_iau_increment=data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim, backend=backend
        ),  # sp.rho_incr(),
        normal_wind_iau_increment=data_alloc.zero_field(
            icon_grid, dims.EdgeDim, dims.KDim, backend=backend
        ),  # sp.vn_incr(),
        exner_iau_increment=data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim, backend=backend
        ),  # sp.exner_incr(),
        exner_dynamical_increment=sp.exner_dyn_incr(),
    )

    timeloop = icon4py_driver.TimeLoop(icon4pyrun_config, diffusion_granule, solve_nonhydro_granule)

    if timeloop_diffusion_linit_init:
        prognostic_state = timeloop_diffusion_savepoint_init.construct_prognostics()
    else:
        prognostic_state = prognostics.PrognosticState(
            w=sp.w_now(),
            vn=sp.vn_now(),
            theta_v=sp.theta_v_now(),
            rho=sp.rho_now(),
            exner=sp.exner_now(),
        )

    prognostic_state_new = prognostics.PrognosticState(
        w=sp.w_new(),
        vn=sp.vn_new(),
        theta_v=sp.theta_v_new(),
        rho=sp.rho_new(),
        exner=sp.exner_new(),
    )

    prognostic_states = common_utils.TimeStepPair(prognostic_state, prognostic_state_new)

    timeloop.time_integration(
        diffusion_diagnostic_state,
        nonhydro_diagnostic_state,
        prognostic_states,
        prep_adv,
        sp.divdamp_fac_o2(),
        do_prep_adv,
    )

    rho_sp = savepoint_nonhydro_exit.rho_new()
    exner_sp = timeloop_diffusion_savepoint_exit.exner()
    theta_sp = timeloop_diffusion_savepoint_exit.theta_v()
    vn_sp = timeloop_diffusion_savepoint_exit.vn()
    w_sp = timeloop_diffusion_savepoint_exit.w()

    assert helpers.dallclose(
        prognostic_states.current.vn.asnumpy(),
        vn_sp.asnumpy(),
        atol=6e-12,
    )

    assert helpers.dallclose(
        prognostic_states.current.w.asnumpy(),
        w_sp.asnumpy(),
        atol=8e-14,
    )

    assert helpers.dallclose(
        prognostic_states.current.exner.asnumpy(),
        exner_sp.asnumpy(),
    )

    assert helpers.dallclose(
        prognostic_states.current.theta_v.asnumpy(),
        theta_sp.asnumpy(),
        atol=4e-12,
    )

    assert helpers.dallclose(
        prognostic_states.current.rho.asnumpy(),
        rho_sp.asnumpy(),
    )
