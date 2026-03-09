# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING
import pytest

import gt4py.next as gtx

from icon4py.model.common import model_backends
from icon4py.model.standalone_driver import main
from icon4py.model.testing import definitions, grid_utils


import icon4py.model.common.grid.states as grid_states
import icon4py.model.common.utils as common_utils
from icon4py.model.atmosphere.diffusion import diffusion, diffusion_states
from icon4py.model.atmosphere.dycore import dycore_states, solve_nonhydro as solve_nh
from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import datatest_utils as dt_utils, definitions, grid_utils, test_utils
from icon4py.model.testing.fixtures.datatest import backend, backend_like
from icon4py.model.standalone_driver import driver_states, driver_utils, standalone_driver
from icon4py.model.standalone_driver.testcases import initial_condition
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.initialization import topography
from icon4py.model.common.grid import geometry_attributes as geometry_meta

from ..fixtures import *  # noqa: F403


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.testing import serialbox as sb


# @pytest.mark.embedded_remap_error
# def test_standalone_driver(
#     backend_like: model_backends.BackendLike,
#     tmp_path: pathlib.Path,
# ) -> None:
#     """
#     Currently, this is a only test to check if the driver runs from a grid file without verifying the end result.
#     TODO(anyone): Modify this test for scientific validation after IO is ready.
#     """

#     backend_name = "embedded"
#     for k, v in model_backends.BACKENDS.items():
#         if backend_like == v:
#             backend_name = k

#     grid_file_path = grid_utils._download_grid_file(definitions.Grids.R02B04_GLOBAL)

#     output_path = tmp_path / f"ci_driver_output_for_backend_{backend_name}"
#     main.main(
#         configuration_file_path=pathlib.Path("./"),
#         grid_file_path=grid_file_path,
#         icon4py_backend=backend_name,
#         output_path=output_path,
#     )



@pytest.mark.datatest
@pytest.mark.parametrize("experiment, rank", [(definitions.Experiments.JW, 0)])
def test_jabw_initial_condition(
    experiment: definitions.Experiment,
    processor_props: decomposition.ProcessProperties,
    backend: gtx_typing.Backend,
    rank: int,
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    icon_grid: base_grid.Grid,
):
    default_surface_pressure = data_alloc.constant_field(icon_grid, 1e5, dims.CellDim)

    # grid_file_path = grid_utils._download_grid_file(definitions.Grids.R02B04_GLOBAL)
    # output_path = pathlib.Path("/home/ong/PycharmProjects/icon4py")
    # icon4py_driver: standalone_driver.Icon4pyDriver = standalone_driver.initialize_driver(
    #     configuration_file_path="./",
    #     output_path=output_path,
    #     grid_file_path=grid_file_path,
    #     log_level="debug",
    #     backend_name="gtfn_cpu",
    # )
    
    parallel_props = decomposition.get_processor_properties(
        decomposition.get_runtype(with_mpi=False)
    )
    driver_utils.configure_logging(
        logging_level="debug",
        processor_procs=parallel_props,
    )

    global_reductions = decomposition.create_reduction(parallel_props)

    # backend = model_options.customize_backend(
    #     program=None, backend=driver_utils.get_backend_from_name(backend_name)
    # )
    allocator = model_backends.get_allocator(backend)

    vertical_grid_config = v_grid.VerticalGridConfig(
        num_levels=35,
        rayleigh_damping_height=45000.0,
    )

    grid_file_path = grid_utils._download_grid_file(definitions.Grids.R02B04_GLOBAL)
    grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file_path,
        vertical_grid_config=vertical_grid_config,
        allocator=allocator,
        global_reductions=global_reductions,
    )

    decomposition_info = driver_utils.create_decomposition_info(
        grid_manager=grid_manager,
        allocator=allocator,
    )
    exchange = decomposition.create_exchange(parallel_props, decomposition_info)

    vertical_grid = driver_utils.create_vertical_grid(
        vertical_grid_config=vertical_grid_config,
        allocator=allocator,
    )

    cell_topography = topography.jablonowski_williamson(
        cell_lat=grid_manager.coordinates[dims.CellDim]["lat"].ndarray,
        u0=35.0,
        array_ns=data_alloc.import_array_ns(allocator=allocator),
    )

    static_field_factories = driver_utils.create_static_field_factories(
        grid_manager=grid_manager,
        decomposition_info=decomposition_info,
        vertical_grid=vertical_grid,
        cell_topography=gtx.as_field((dims.CellDim,), data=cell_topography, allocator=allocator),  # type: ignore[arg-type] # due to array_ns opacity
        backend=backend,
    )
    
    ds: driver_states.DriverStates = initial_condition.jablonowski_williamson(
        grid=grid_manager.grid,
        geometry_field_source=static_field_factories.geometry_field_source,
        interpolation_field_source=static_field_factories.interpolation_field_source,
        metrics_field_source=static_field_factories.metrics_field_source,
        backend=backend,
        path=dt_utils.get_datapath_for_experiment(experiment, processor_props),
        edge_param=grid_savepoint.construct_edge_geometry(),
    )

    edge_geometry: grid_states.EdgeParams = grid_savepoint.construct_edge_geometry()
    fac_edge_lat = static_field_factories.geometry_field_source.get(geometry_meta.EDGE_LAT).ndarray
    fac_edge_lon = static_field_factories.geometry_field_source.get(geometry_meta.EDGE_LON).ndarray
    edge_lat = edge_geometry.edge_center[0].ndarray
    edge_lon = edge_geometry.edge_center[1].ndarray
    print()
    print(fac_edge_lat.shape, edge_lat.shape)
    import numpy as np
    for i in range(fac_edge_lat.shape[0]):
        if np.abs(fac_edge_lat[i] - edge_lat[i]) > 1.e-10:
            print("CATCH LAT: ", i, fac_edge_lat[i], edge_lat[i], " === ", fac_edge_lat[i] - edge_lat[i])
        if np.abs(fac_edge_lon[i] - edge_lon[i]) > 1.e-10:
            print("CATCH LON: ", i, fac_edge_lon[i], edge_lon[i], " === ", fac_edge_lon[i] - edge_lon[i])
    # note that w is not verified because we decided to force w to zero in python framework after discussion
    jabw_exit_savepoint = data_provider.from_savepoint_jabw_exit()

    def print_error(input_var, reference_var, var_name: str):
        import numpy as np
        r_error = np.where(
            np.abs(reference_var) > 1.e-15, 
            np.abs((input_var - reference_var)/reference_var),
            0.0,
        )
        mask = r_error > 0.01
        print(var_name, np.abs(input_var - reference_var).max(), r_error.max())
        print(np.count_nonzero(mask))
        print(input_var[mask])
        
    print()
    print_error(ds.prognostics.current.rho.asnumpy(), jabw_exit_savepoint.rho().asnumpy(), "rho")
    print_error(ds.prognostics.current.exner.asnumpy(), jabw_exit_savepoint.exner().asnumpy(), "exner")
    print_error(ds.prognostics.current.theta_v.asnumpy(), jabw_exit_savepoint.theta_v().asnumpy(), "theta_v")
    print_error(ds.prognostics.current.vn.asnumpy(), jabw_exit_savepoint.vn().asnumpy(), "vn")
    print_error(ds.diagnostic.pressure.asnumpy(), jabw_exit_savepoint.pressure().asnumpy(), "pressure")
    print_error(ds.diagnostic.temperature.asnumpy(), jabw_exit_savepoint.temperature().asnumpy(), "temperature")
    print_error(ds.diagnostic.surface_pressure.asnumpy(), default_surface_pressure.asnumpy(), "surface_pressure")
    print_error(ds.solve_nonhydro_diagnostic.perturbed_exner_at_cells_on_model_levels.asnumpy(), data_provider.from_savepoint_diagnostics_initial().exner_pr().asnumpy(), "exner_pr")

    assert test_utils.dallclose(
        ds.prognostics.current.rho.asnumpy(),
        jabw_exit_savepoint.rho().asnumpy(),
    )

    assert test_utils.dallclose(
        ds.prognostics.current.exner.asnumpy(),
        jabw_exit_savepoint.exner().asnumpy(),
    )

    assert test_utils.dallclose(
        ds.prognostics.current.theta_v.asnumpy(),
        jabw_exit_savepoint.theta_v().asnumpy(),
    )

    assert test_utils.dallclose(
        ds.prognostics.current.vn.asnumpy(),
        jabw_exit_savepoint.vn().asnumpy(),
        atol=1.0e-12,
    )

    assert test_utils.dallclose(
        ds.diagnostic.pressure.asnumpy(),
        jabw_exit_savepoint.pressure().asnumpy(),
    )

    assert test_utils.dallclose(
        ds.diagnostic.temperature.asnumpy(),
        jabw_exit_savepoint.temperature().asnumpy(),
    )

    assert test_utils.dallclose(
        ds.diagnostic.surface_pressure.asnumpy(),
        default_surface_pressure.asnumpy(),
    )

    assert test_utils.dallclose(
        ds.solve_nonhydro_diagnostic.perturbed_exner_at_cells_on_model_levels.asnumpy(),
        data_provider.from_savepoint_diagnostics_initial().exner_pr().asnumpy(),
        atol=1.0e-14,
    )



# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment, istep_init, istep_exit, substep_init, substep_exit, timeloop_date_init, timeloop_date_exit, step_date_init, step_date_exit, timeloop_diffusion_linit_init, timeloop_diffusion_linit_exit",
    [
        
        (
            definitions.Experiments.JW,
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
def test_run_timeloop_single_step(
    experiment: definitions.Experiment,
    timeloop_date_init: str,
    timeloop_date_exit: str,
    timeloop_diffusion_linit_init: bool,
    *,
    grid_savepoint: sb.IconGridSavepoint,
    icon_grid: base_grid.Grid,
    metrics_savepoint: sb.MetricSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    lowest_layer_thickness: float,
    model_top_height: float,
    stretch_factor: float,
    damping_height: float,
    ndyn_substeps: int,
    timeloop_diffusion_savepoint_init: sb.IconDiffusionInitSavepoint,
    timeloop_diffusion_savepoint_exit: sb.IconDiffusionExitSavepoint,
    savepoint_velocity_init: sb.IconVelocityInitSavepoint,
    savepoint_nonhydro_init: sb.IconNonHydroInitSavepoint,
    savepoint_nonhydro_exit: sb.IconNonHydroExitSavepoint,
    backend_like: model_backends.BackendLike,
    tmp_path: pathlib.Path,
):

    backend_name = "embedded"
    for k, v in model_backends.BACKENDS.items():
        if backend_like == v:
            backend_name = k

    grid_file_path = grid_utils._download_grid_file(definitions.Grids.R02B04_GLOBAL)

    # output_path = tmp_path / f"ci_driver_output_for_backend_{backend_name}"
    # main.main(
    #     configuration_file_path=pathlib.Path("./"),
    #     grid_file_path=grid_file_path,
    #     icon4py_backend=backend_name,
    #     output_path=output_path,
    # )
    
    output_path = pathlib.Path("./")
    icon4py_driver: standalone_driver.Icon4pyDriver = standalone_driver.initialize_driver(
        configuration_file_path=pathlib.Path("./"),
        output_path=output_path,
        grid_file_path=grid_file_path,
        log_level="debug",
        backend_name="gtfn_cpu",
    )

    ds: driver_states.DriverStates = initial_condition.jablonowski_williamson(
        grid=icon4py_driver.grid,
        geometry_field_source=icon4py_driver.static_field_factories.geometry_field_source,
        interpolation_field_source=icon4py_driver.static_field_factories.interpolation_field_source,
        metrics_field_source=icon4py_driver.static_field_factories.metrics_field_source,
        backend=icon4py_driver.backend,
    )
    allocator = model_backends.get_allocator(icon4py_driver.backend)

    sp = savepoint_nonhydro_init
    sp_v = savepoint_velocity_init
    do_prep_adv = sp.get_metadata("prep_adv").get("prep_adv")
    linit = sp.get_metadata("linit").get("linit")
    
    edge_geometry: grid_states.EdgeParams = grid_savepoint.construct_edge_geometry()
    cell_geometry: grid_states.CellParams = grid_savepoint.construct_cell_geometry()
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

    vertical_config = v_grid.VerticalGridConfig(
        icon4py_driver.grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
    )
    additional_parameters = diffusion.DiffusionParams(icon4py_driver.diffusion)

    diffusion_granule = diffusion.Diffusion(
        grid=icon4py_driver.grid,
        config=icon4py_driver.diffusion,
        params=additional_parameters,
        vertical_grid=vertical_params,
        metric_state=diffusion_metric_state,
        interpolation_state=diffusion_interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
        backend=icon4py_driver.backend,
    )

    nonhydro_params = solve_nh.NonHydrostaticParams(icon4py_driver.solve_nonhydro)

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
        nflat_gradp=grid_savepoint.nflat_gradp(),
        pg_exdist=metrics_savepoint.pg_exdist_dsl(),
        ddqz_z_full_e=metrics_savepoint.ddqz_z_full_e(),
        ddxt_z_full=metrics_savepoint.ddxt_z_full(),
        wgtfac_e=metrics_savepoint.wgtfac_e(),
        wgtfacq_e=metrics_savepoint.wgtfacq_e_dsl(),
        exner_w_implicit_weight_parameter=metrics_savepoint.vwind_impl_wgt(),
        horizontal_mask_for_3d_divdamp=metrics_savepoint.hmask_dd3d(),
        scaling_factor_for_3d_divdamp=metrics_savepoint.scalfac_dd3d(),
        coeff1_dwdz=metrics_savepoint.coeff1_dwdz(),
        coeff2_dwdz=metrics_savepoint.coeff2_dwdz(),
        coeff_gradekin=metrics_savepoint.coeff_gradekin(),
    )

    solve_nonhydro_granule = solve_nh.SolveNonhydro(
        grid=icon4py_driver.grid,
        config=icon4py_driver.solve_nonhydro,
        params=nonhydro_params,
        metric_state_nonhydro=nonhydro_metric_state,
        interpolation_state=nonhydro_interpolation_state,
        vertical_params=vertical_params,
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
        owner_mask=grid_savepoint.c_owner_mask(),
        backend=icon4py_driver.backend,
    )

    icon4py_driver.solve_nonhydro = solve_nonhydro_granule
    icon4py_driver.diffusion = diffusion_granule
    
    # print()
    # print(icon4py_driver.solve_nonhydro)

    ##############
    diffusion_diagnostic_state = diffusion_states.DiffusionDiagnosticState(
        hdef_ic=timeloop_diffusion_savepoint_init.hdef_ic(),
        div_ic=timeloop_diffusion_savepoint_init.div_ic(),
        dwdx=timeloop_diffusion_savepoint_init.dwdx(),
        dwdy=timeloop_diffusion_savepoint_init.dwdy(),
    )

    prep_adv = dycore_states.PrepAdvection(
        vn_traj=sp.vn_traj(),
        mass_flx_me=sp.mass_flx_me(),
        dynamical_vertical_mass_flux_at_cells_on_half_levels=sp.mass_flx_ic(),
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels=data_alloc.zero_field(
            icon4py_driver.grid,
            dims.CellDim,
            dims.KDim,
            allocator=allocator,
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
            icon4py_driver.grid, dims.CellDim, dims.KDim, allocator=allocator
        ),  # sp.rho_incr(),
        normal_wind_iau_increment=data_alloc.zero_field(
            icon4py_driver.grid, dims.EdgeDim, dims.KDim, allocator=allocator
        ),  # sp.vn_incr(),
        exner_iau_increment=data_alloc.zero_field(
            icon4py_driver.grid, dims.CellDim, dims.KDim, allocator=allocator
        ),  # sp.exner_incr(),
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

    ds_new = driver_states.DriverStates(
        prep_advection_prognostic=prep_adv,
        solve_nonhydro_diagnostic=nonhydro_diagnostic_state,
        prep_tracer_advection_prognostic=prep_adv,
        tracer_advection_diagnostic=None,
        diffusion_diagnostic=diffusion_diagnostic_state,
        prognostics=prognostic_states,
        diagnostic=None,
    )
    #############


    icon4py_driver.time_integration(
        ds_new,
        do_prep_adv=False,
    )

    rho_sp = savepoint_nonhydro_exit.rho_new()
    # exner_sp = savepoint_nonhydro_exit.exner_new()
    # theta_sp = savepoint_nonhydro_exit.theta_v_new()
    # vn_sp = savepoint_nonhydro_exit.vn_new()
    # w_sp = savepoint_nonhydro_exit.w_new()
    exner_sp = timeloop_diffusion_savepoint_exit.exner()
    theta_sp = timeloop_diffusion_savepoint_exit.theta_v()
    vn_sp = timeloop_diffusion_savepoint_exit.vn()
    w_sp = timeloop_diffusion_savepoint_exit.w()

    def print_error(input_var, reference_var, var_name: str):
        import numpy as np
        r_error = np.where(
            np.abs(reference_var) > 1.e-15, 
            np.abs((input_var - reference_var)/reference_var),
            0.0,
        )
        print(var_name, np.abs(input_var - reference_var).max(), r_error.max())

    print()
    print_error(ds.prognostics.current.vn.asnumpy(),vn_sp.asnumpy(),"vn")
    print_error(ds.prognostics.current.w.asnumpy(),w_sp.asnumpy(),"w")
    print_error(ds.prognostics.current.rho.asnumpy(),rho_sp.asnumpy(),"rho")
    print_error(ds.prognostics.current.theta_v.asnumpy(),theta_sp.asnumpy(),"theta_v")
    print_error(ds.prognostics.current.exner.asnumpy(),exner_sp.asnumpy(),"exner")
    
    assert test_utils.dallclose(
        ds.prognostics.current.vn.asnumpy(),
        vn_sp.asnumpy(),
        atol=6e-12,
    )

    assert test_utils.dallclose(
        ds.prognostics.current.w.asnumpy(),
        w_sp.asnumpy(),
        atol=8e-14,
    )

    assert test_utils.dallclose(
        ds.prognostics.current.exner.asnumpy(),
        exner_sp.asnumpy(),
    )

    assert test_utils.dallclose(
        ds.prognostics.current.theta_v.asnumpy(),
        theta_sp.asnumpy(),
        atol=4e-12,
    )

    assert test_utils.dallclose(
        ds.prognostics.current.rho.asnumpy(),
        rho_sp.asnumpy(),
    )
