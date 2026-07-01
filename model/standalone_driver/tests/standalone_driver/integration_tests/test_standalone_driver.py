# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import datetime
import pathlib

import gt4py.next.typing as gtx_typing
import pytest

from icon4py.model.atmosphere.advection import advection, advection_states
from icon4py.model.atmosphere.diffusion import diffusion, diffusion_states
from icon4py.model.atmosphere.dycore import dycore_states, solve_nonhydro as solve_nh
from icon4py.model.common import (
    dimension as dims,
    grid as common_grid,
    model_backends,
    utils as common_utils,
)
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.grid import states as grid_states, vertical as v_grid
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.standalone_driver import (
    config as driver_config,
    driver_states,
    standalone_driver,
)
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    definitions as test_defs,
    grid_utils,
    serialbox as sb,
    test_utils,
)

from ..fixtures import *  # noqa: F403


_MCH_VN_ATOL = 6e-7
_MCH_W_ATOL = 8e-9
_MCH_EXNER_ATOL = 2e-10
_MCH_THETA_V_ATOL = 1e-7
_MCH_RHO_ATOL = 9e-10


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "experiment_description, istep_exit, substep_exit, timeloop_date_init, timeloop_date_exit, step_date_exit, timeloop_diffusion_linit_init, timeloop_diffusion_linit_exit",
    [
        (
            test_defs.Experiments.JW,
            2,
            5,
            "2008-09-01T00:00:00.000",
            "2008-09-01T00:05:00.000",
            "2008-09-01T00:05:00.000",
            False,
            False,
        ),
        (
            test_defs.Experiments.GAUSS3D,
            2,
            5,
            "2001-01-01T00:00:00.000",
            "2001-01-01T00:00:04.000",
            "2001-01-01T00:00:04.000",
            False,
            False,
        ),
        (
            test_defs.Experiments.MCH_CH_R04B09,
            2,
            2,
            "2021-06-20T12:00:00.000",
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
            True,
            False,
        ),
        (
            test_defs.Experiments.MCH_CH_R04B09,
            2,
            2,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:20.000",
            "2021-06-20T12:00:20.000",
            False,
            False,
        ),
    ],
)
def test_standalone_driver(
    experiment_description: test_defs.ExperimentDescription,
    timeloop_date_init: str,
    timeloop_date_exit: str,
    timeloop_diffusion_linit_init: bool,
    *,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
    savepoint_nonhydro_exit: sb.IconNonHydroExitSavepoint,
    substep_exit: int,
    savepoint_diffusion_exit: sb.IconDiffusionExitSavepoint,
) -> None:
    allocator = model_backends.get_allocator(backend)

    grid_file_path = grid_utils._download_grid_file(experiment_description.grid)
    config_file_path = dt_utils.get_path_for_experiment(experiment_description, process_props)

    config = driver_config.read_config(config_file_path)
    config = config.with_overrides(
        driver={
            "output_path": tmp_path / "ci_driver_output",
            "start_of_simulation": datetime.datetime.fromisoformat(timeloop_date_init).replace(
                tzinfo=datetime.timezone.utc
            ),
            "end_of_simulation": datetime.datetime.fromisoformat(timeloop_date_exit).replace(
                tzinfo=datetime.timezone.utc
            ),
        }
    )

    grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file_path,
        vertical_grid_config=config.vertical_grid,
        allocator=allocator,
        process_props=process_props,
    )
    ds, _ = standalone_driver.run_driver(
        config=config,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=backend,
    )

    rho_sp = savepoint_nonhydro_exit.rho_new()
    exner_sp = savepoint_diffusion_exit.exner()
    theta_sp = savepoint_diffusion_exit.theta_v()
    vn_sp = savepoint_diffusion_exit.vn()
    w_sp = savepoint_diffusion_exit.w()

    if (
        experiment_description == test_defs.Experiments.JW
        or experiment_description == test_defs.Experiments.GAUSS3D
    ):
        vn_atol = 6e-12
        w_atol = 1e-13
        exner_atol = 0.0
        theta_v_atol = 4e-12
        rho_atol = 0.0
    else:
        vn_atol = _MCH_VN_ATOL
        w_atol = _MCH_W_ATOL
        exner_atol = _MCH_EXNER_ATOL
        theta_v_atol = _MCH_THETA_V_ATOL
        rho_atol = _MCH_RHO_ATOL

    test_utils.assert_dallclose(
        ds.prognostics.current.vn.asnumpy(),
        vn_sp.asnumpy(),
        atol=vn_atol,
    )
    test_utils.assert_dallclose(
        ds.prognostics.current.w.asnumpy(),
        w_sp.asnumpy(),
        atol=w_atol,
    )
    test_utils.assert_dallclose(
        ds.prognostics.current.exner.asnumpy(),
        exner_sp.asnumpy(),
        atol=exner_atol,
    )
    test_utils.assert_dallclose(
        ds.prognostics.current.theta_v.asnumpy(),
        theta_sp.asnumpy(),
        atol=theta_v_atol,
    )
    test_utils.assert_dallclose(
        ds.prognostics.current.rho.asnumpy(),
        rho_sp.asnumpy(),
        atol=rho_atol,
    )


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "experiment_description, istep_init, istep_exit, substep_init, substep_exit, timeloop_date_init, timeloop_date_exit, step_date_init, step_date_exit, timeloop_diffusion_linit_init, timeloop_diffusion_linit_exit",
    [
        (
            test_defs.Experiments.MCH_CH_R04B09,
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
        ),
        (
            test_defs.Experiments.MCH_CH_R04B09,
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
        ),
        (
            test_defs.Experiments.GAUSS3D,
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
        ),
        (
            test_defs.Experiments.JW,
            1,
            2,
            1,
            5,
            "2008-09-01T00:00:00.000",
            "2008-09-01T00:05:00.000",
            "2008-09-01T00:05:00.000",
            "2008-09-01T00:05:00.000",
            False,
            False,
        ),
    ],
)
def test_standalone_driver_from_savepoints(
    experiment: test_defs.Experiment,
    timeloop_date_init: str,
    timeloop_date_exit: str,
    timeloop_diffusion_linit_init: bool,
    *,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
    icon_grid: common_grid.base.Grid,
    grid_savepoint: sb.IconGridSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    data_provider: sb.IconSerialDataProvider,
    savepoint_nonhydro_init: sb.IconNonHydroInitSavepoint,
    savepoint_velocity_init: sb.IconVelocityInitSavepoint,
    savepoint_nonhydro_exit: sb.IconNonHydroExitSavepoint,
    savepoint_diffusion_exit: sb.IconDiffusionExitSavepoint,
    step_date_init: str,
    step_date_exit: str,
    istep_init: int,
    istep_exit: int,
    substep_init: int,
    substep_exit: int,
    timeloop_diffusion_linit_exit: bool,
) -> None:
    _ = istep_init, substep_init, istep_exit, substep_exit, step_date_init, step_date_exit
    _ = timeloop_diffusion_linit_exit

    config = experiment.config.with_overrides(
        driver={
            "output_path": tmp_path / "ci_driver_output",
            "start_of_simulation": datetime.datetime.fromisoformat(timeloop_date_init).replace(
                tzinfo=datetime.timezone.utc
            ),
            "end_of_simulation": datetime.datetime.fromisoformat(timeloop_date_exit).replace(
                tzinfo=datetime.timezone.utc
            ),
            "ntracer": 0,
        }
    )

    diffusion_config = experiment.config.diffusion
    nonhydro_config = experiment.config.nonhydrostatic

    edge_geometry = grid_savepoint.construct_edge_geometry()
    cell_geometry = grid_savepoint.construct_cell_geometry()
    grg = interpolation_savepoint.geofac_grg()

    diffusion_interpolation_state = diffusion_states.DiffusionInterpolationState(
        e_bln_c_s=interpolation_savepoint.e_bln_c_s(),
        rbf_coeff_1=interpolation_savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=interpolation_savepoint.rbf_vec_coeff_v2(),
        geofac_div=interpolation_savepoint.geofac_div(),
        geofac_n2s=interpolation_savepoint.geofac_n2s(),
        geofac_grg_x=grg[0],
        geofac_grg_y=grg[1],
        nudgecoeff_e=interpolation_savepoint.nudgecoeff_e(),
    )
    diffusion_metric_state = diffusion_states.DiffusionMetricState(
        theta_ref_mc=metrics_savepoint.theta_ref_mc(),
        wgtfac_c=metrics_savepoint.wgtfac_c(),
        zd_intcoef=metrics_savepoint.zd_intcoef(),
        zd_vertoffset=metrics_savepoint.zd_vertoffset(),
        zd_diffcoef=metrics_savepoint.zd_diffcoef(),
    )

    vertical_params = v_grid.VerticalGrid(
        config=experiment.config.vertical_grid,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
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
        exchange=decomp_defs.single_node_exchange,
    )

    nonhydro_params = solve_nh.NonHydrostaticParams(nonhydro_config)

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
        e_bln_c_s=interpolation_savepoint.e_bln_c_s(),
        rbf_coeff_1=interpolation_savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=interpolation_savepoint.rbf_vec_coeff_v2(),
        geofac_div=interpolation_savepoint.geofac_div(),
        geofac_n2s=interpolation_savepoint.geofac_n2s(),
        geofac_grg_x=grg[0],
        geofac_grg_y=grg[1],
        nudgecoeff_e=interpolation_savepoint.nudgecoeff_e(),
    )
    nonhydro_metric_state = dycore_states.MetricStateNonHydro(
        mask_prog_halo_c=metrics_savepoint.mask_prog_halo_c(),
        rayleigh_w=metrics_savepoint.rayleigh_w(),
        time_extrapolation_parameter_for_exner=metrics_savepoint.exner_exfac(),
        reference_exner_at_cells_on_model_levels=metrics_savepoint.exner_ref_mc(),
        wgtfac_c=metrics_savepoint.wgtfac_c(),
        wgtfacq_c=metrics_savepoint.wgtfacq_c(),
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
        nflat_gradp=grid_savepoint.nflat_gradp(),
        pg_exdist=metrics_savepoint.pg_exdist_dsl(),
        ddqz_z_full_e=metrics_savepoint.ddqz_z_full_e(),
        ddxt_z_full=metrics_savepoint.ddxt_z_full(),
        wgtfac_e=metrics_savepoint.wgtfac_e(),
        wgtfacq_e=metrics_savepoint.wgtfacq_e(),
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
        exchange=decomp_defs.single_node_exchange,
    )

    timeloop_diffusion_savepoint_init = data_provider.from_savepoint_diffusion_init(
        linit=timeloop_diffusion_linit_init, date=step_date_init
    )
    diffusion_diagnostic_state = diffusion_states.DiffusionDiagnosticState(
        hdef_ic=timeloop_diffusion_savepoint_init.hdef_ic(),
        div_ic=timeloop_diffusion_savepoint_init.div_ic(),
        dwdx=timeloop_diffusion_savepoint_init.dwdx(),
        dwdy=timeloop_diffusion_savepoint_init.dwdy(),
    )

    allocator = model_backends.get_allocator(backend)
    sp = savepoint_nonhydro_init
    sp_v = savepoint_velocity_init
    linit = sp.get_metadata("linit").get("linit")
    do_prep_adv = sp.get_metadata("prep_adv").get("prep_adv")

    prep_adv = dycore_states.PrepAdvection(
        vn_traj=sp.vn_traj(),
        mass_flx_me=sp.mass_flx_me(),
        dynamical_vertical_mass_flux_at_cells_on_half_levels=sp.mass_flx_ic(),
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels=data_alloc.zero_field(
            icon_grid,
            dims.CellDim,
            dims.KDim,
            allocator=backend,
        ),
    )

    current_index, next_index = (1, 0) if not linit else (0, 1)
    nonhydro_diagnostic_state = dycore_states.DiagnosticStateNonHydro(
        max_vertical_cfl=data_alloc.scalar_like_array(0.0, backend),
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
            icon_grid,
            dims.CellDim,
            dims.KDim,
            allocator=backend,
        ),
        normal_wind_iau_increment=data_alloc.zero_field(
            icon_grid,
            dims.EdgeDim,
            dims.KDim,
            allocator=backend,
        ),
        exner_iau_increment=data_alloc.zero_field(
            icon_grid,
            dims.CellDim,
            dims.KDim,
            allocator=backend,
        ),
        exner_dynamical_increment=sp.exner_dyn_incr(),
    )

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

    tracer_advection_diagnostic_state = advection_states.initialize_advection_diagnostic_state(
        grid=icon_grid,
        allocator=allocator,
    )
    prep_tracer_adv = advection_states.AdvectionPrepAdvState(
        vn_traj=data_alloc.zero_field(icon_grid, dims.EdgeDim, dims.KDim, allocator=allocator),
        mass_flx_me=data_alloc.zero_field(icon_grid, dims.EdgeDim, dims.KDim, allocator=allocator),
        mass_flx_ic=data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, allocator=allocator),
    )
    from icon4py.model.common.states import diagnostic_state as diagnostics

    diagnostic_state = diagnostics.initialize_diagnostic_state(grid=icon_grid, allocator=allocator)

    ds = driver_states.DriverStates(
        prep_advection_prognostic=prep_adv,
        solve_nonhydro_diagnostic=nonhydro_diagnostic_state,
        tracer_advection_diagnostic=tracer_advection_diagnostic_state,
        prep_tracer_advection_prognostic=prep_tracer_adv,
        diffusion_diagnostic=diffusion_diagnostic_state,
        prognostics=prognostic_states,
        diagnostic=diagnostic_state,
    )

    decomposition_info = grid_savepoint.construct_decomposition_info()

    advection_config = experiment.config.advection
    tracer_advection_granule = advection.convert_config_to_advection(
        grid=icon_grid,
        backend=backend,
        config=advection_config,
        interpolation_state=advection_states.AdvectionInterpolationState(
            geofac_div=interpolation_savepoint.geofac_div(),
            rbf_vec_coeff_e=interpolation_savepoint.rbf_vec_coeff_e(),
            pos_on_tplane_e_1=interpolation_savepoint.pos_on_tplane_e_x(),
            pos_on_tplane_e_2=interpolation_savepoint.pos_on_tplane_e_y(),
        ),
        least_squares_state=advection_states.AdvectionLeastSquaresState(
            lsq_pseudoinv_1=data_alloc.zero_field(icon_grid, dims.CellDim, allocator=allocator),
            lsq_pseudoinv_2=data_alloc.zero_field(icon_grid, dims.CellDim, allocator=allocator),
        ),
        metric_state=advection_states.AdvectionMetricState(
            deepatmo_divh=data_alloc.zero_field(
                icon_grid, dims.CellDim, dims.KDim, allocator=allocator
            ),
            deepatmo_divzl=data_alloc.zero_field(
                icon_grid, dims.CellDim, dims.KDim, allocator=allocator
            ),
            deepatmo_divzu=data_alloc.zero_field(
                icon_grid, dims.CellDim, dims.KDim, allocator=allocator
            ),
            ddqz_z_full=data_alloc.zero_field(
                icon_grid, dims.CellDim, dims.KDim, allocator=allocator
            ),
        ),
        edge_params=edge_geometry,
        cell_params=cell_geometry,
        exchange=decomp_defs.single_node_exchange,
    )

    icon4py_driver = standalone_driver.Icon4pyDriver(
        config=config,
        backend=backend,
        grid=icon_grid,
        decomposition_info=decomposition_info,
        static_field_factories=driver_states.StaticFieldFactories(
            geometry_field_source=None,
            interpolation_field_source=None,
            metrics_field_source=type(
                "_FakeMetricsFactory", (), {"_vertical_grid": vertical_params}
            )(),
        ),
        diffusion_granule=diffusion_granule,
        solve_nonhydro_granule=solve_nonhydro_granule,
        vertical_grid_config=experiment.config.vertical_grid,
        tracer_advection_granule=tracer_advection_granule,
        exchange=decomp_defs.single_node_exchange,
        global_reductions=decomp_defs.create_reduction(process_props, decomposition_info),
    )

    icon4py_driver.model_time_variables.is_first_step_in_simulation = True

    if (
        diffusion_granule.config.apply_to_horizontal_wind
        and timeloop_diffusion_linit_init
        and icon4py_driver.model_time_variables.is_first_step_in_simulation
    ):
        diffusion_granule.run(
            ds.diffusion_diagnostic,
            ds.prognostics.current,
            icon4py_driver.model_time_variables.dtime_in_seconds,
            initial_run=True,
        )

    second_order_divdamp_factor = savepoint_nonhydro_init.divdamp_fac_o2()
    icon4py_driver._update_spinup_second_order_divergence_damping = (
        lambda self=icon4py_driver, _factor=second_order_divdamp_factor: _factor
    )

    icon4py_driver.time_integration(ds, do_prep_adv=do_prep_adv)

    rho_sp = savepoint_nonhydro_exit.rho_new()
    exner_sp = savepoint_diffusion_exit.exner()
    theta_sp = savepoint_diffusion_exit.theta_v()
    vn_sp = savepoint_diffusion_exit.vn()
    w_sp = savepoint_diffusion_exit.w()

    test_utils.assert_dallclose(
        ds.prognostics.current.vn.asnumpy(),
        vn_sp.asnumpy(),
        atol=6e-12,
    )
    test_utils.assert_dallclose(
        ds.prognostics.current.w.asnumpy(),
        w_sp.asnumpy(),
        atol=1e-13,
    )
    test_utils.assert_dallclose(
        ds.prognostics.current.exner.asnumpy(),
        exner_sp.asnumpy(),
    )
    test_utils.assert_dallclose(
        ds.prognostics.current.theta_v.asnumpy(),
        theta_sp.asnumpy(),
        atol=4e-12,
    )
    test_utils.assert_dallclose(
        ds.prognostics.current.rho.asnumpy(),
        rho_sp.asnumpy(),
    )
