# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import os
import pathlib
import pickle
from typing import TYPE_CHECKING

import gt4py.next as gtx
import pytest

import icon4py.model.common.grid.states as grid_states
import icon4py.model.common.utils as common_utils
from icon4py.model.atmosphere.diffusion import diffusion, diffusion_states
from icon4py.model.atmosphere.dycore import dycore_states, ibm, solve_nonhydro as solve_nh
from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.metrics.metric_fields import compute_ddqz_z_half_e
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.driver import (
    icon4py_configuration,
    icon4py_driver,
    initialization_utils as driver_init,
)
from icon4py.model.driver.testcases import channel
from icon4py.model.testing import datatest_utils as dt_utils, definitions, grid_utils, test_utils
from icon4py.model.testing.fixtures.datatest import _download_ser_data, backend

from ..fixtures import *  # noqa: F403
from ..utils import construct_icon4pyrun_config


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.testing import serialbox as sb


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment, istep_init, istep_exit, substep_init, substep_exit, timeloop_date_init, timeloop_date_exit, step_date_init, step_date_exit, timeloop_diffusion_linit_init, timeloop_diffusion_linit_exit, vn_only",
    [
        (
            definitions.Experiments.MCH_CH_R04B09,
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
            definitions.Experiments.MCH_CH_R04B09,
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
            definitions.Experiments.GAUSS3D,
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
    experiment: definitions.Experiment,
    timeloop_date_init: str,
    timeloop_date_exit: str,
    timeloop_diffusion_linit_init: bool,
    vn_only: bool,  # TODO unused?
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
    backend: gtx_typing.Backend,
):
    ranked_data_path = pathlib.Path("testdata/ser_icondata/mpitask1")
    savepoint_path = ranked_data_path / experiment.name / "ser_data"
    grid_file_path = (
        pathlib.Path("testdata/grids") / experiment.grid.name / experiment.grid.file_name
    )
    DO_CHANNEL = False
    DO_IBM = False
    if experiment == definitions.Experiments.GAUSS3D:
        os.environ["ICON4PY_NUM_LEVELS"] = "35"
        os.environ["ICON4PY_END_DATE"] = "0001-01-01T00:00:04"
        os.environ["ICON4PY_DTIME"] = "4.0"
        os.environ["ICON4PY_DIFFU_COEFF"] = "0.001"
        os.environ["ICON4PY_CHANNEL_SPONGE_LENGTH"] = "5000.0"
        os.environ["ICON4PY_CHANNEL_PERTURBATION"] = "0.0"
        DO_CHANNEL = True
        DO_IBM = True
        config = icon4py_configuration.read_config(
            experiment_type=driver_init.ExperimentType.GAUSS3D,
            backend=backend,
        )
        diffusion_config = config.diffusion_config
        nonhydro_config = config.solve_nonhydro_config
        icon4pyrun_config = config.run_config

    else:
        diffusion_config = definitions.construct_diffusion_config(
            experiment, ndyn_substeps=ndyn_substeps
        )
        nonhydro_config = definitions.construct_nonhydrostatic_config(experiment)
        icon4pyrun_config = construct_icon4pyrun_config(
            experiment,
            timeloop_date_init,
            timeloop_date_exit,
            timeloop_diffusion_linit_init,
            ndyn_substeps=ndyn_substeps,
            backend=backend,
        )

    ibm_masks = ibm.ImmersedBoundaryMethodMasks(
        grid=icon_grid,
        savepoint_path=str(savepoint_path),  # make these Paths some day
        grid_file_path=str(grid_file_path),  # make these Paths some day
        backend=backend,
        do_ibm=DO_IBM,
    )
    channel_inst = channel.ChannelFlow(
        grid=icon_grid,
        savepoint_path=str(savepoint_path),  # make these Paths some day
        grid_file_path=str(grid_file_path),  # make these Paths some day
        backend=backend,
        do_channel=DO_CHANNEL,
    )

    edge_geometry: grid_states.EdgeParams = grid_savepoint.construct_edge_geometry()
    cell_geometry: grid_states.CellParams = grid_savepoint.construct_cell_geometry()
    grg = interpolation_savepoint.geofac_grg()

    xp = data_alloc.import_array_ns(backend)
    ddqz_z_half_e_np = xp.zeros(
        (grid_savepoint.num(dims.EdgeDim), grid_savepoint.num(dims.KDim) + 1), dtype=float
    )
    ddqz_z_half_e = gtx.as_field((dims.EdgeDim, dims.KDim), ddqz_z_half_e_np, allocator=backend)
    compute_ddqz_z_half_e.with_backend(backend=backend)(
        ddqz_z_half=metrics_savepoint.ddqz_z_half(),
        c_lin_e=interpolation_savepoint.c_lin_e(),
        ddqz_z_half_e=ddqz_z_half_e,
        horizontal_start=0,
        horizontal_end=grid_savepoint.num(dims.EdgeDim),
        vertical_start=0,
        vertical_end=grid_savepoint.num(dims.KDim) + 1,
        offset_provider=icon_grid.connectivities,
    )

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
        mask_hdiff=metrics_savepoint.mask_hdiff(),
        theta_ref_mc=metrics_savepoint.theta_ref_mc(),
        wgtfac_c=metrics_savepoint.wgtfac_c(),
        zd_intcoef=metrics_savepoint.zd_intcoef(),
        zd_vertoffset=metrics_savepoint.zd_vertoffset(),
        zd_diffcoef=metrics_savepoint.zd_diffcoef(),
        ddqz_z_full=metrics_savepoint.ddqz_z_full(),
        ddqz_z_full_e=metrics_savepoint.ddqz_z_full_e(),
        ddqz_z_half=metrics_savepoint.ddqz_z_half(),
        ddqz_z_half_e=ddqz_z_half_e,
    )

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
        ibm_masks=ibm_masks,
    )

    sp = savepoint_nonhydro_init
    nonhydro_params = solve_nh.NonHydrostaticParams(nonhydro_config)
    sp_v = savepoint_velocity_init
    do_prep_adv = sp.get_metadata("prep_adv").get("prep_adv")

    linit = sp.get_metadata("linit").get("linit")

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
        nflat_gradp=grid_savepoint.nflat_gradp(),
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
        ibm_masks=ibm_masks,
        channel=channel_inst,
    )

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
            icon_grid,
            dims.CellDim,
            dims.KDim,
            allocator=backend,
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
            icon_grid, dims.CellDim, dims.KDim, allocator=backend
        ),  # sp.rho_incr(),
        normal_wind_iau_increment=data_alloc.zero_field(
            icon_grid, dims.EdgeDim, dims.KDim, allocator=backend
        ),  # sp.vn_incr(),
        exner_iau_increment=data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim, allocator=backend
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

    if experiment == definitions.Experiments.GAUSS3D:
        # I cannot create serialized data for this from fortran for now
        _download_ser_data(1, ranked_data_path, definitions.Experiments.CHANNEL_IBM)
        fname = "end_of_timestep_000000000.pkl"
        fpath = ranked_data_path / definitions.Experiments.CHANNEL_IBM.name / fname
        with fpath.open("rb") as ifile:
            state = pickle.load(ifile)
            vn_sp = state["vn"]
            w_sp = state["w"]
            theta_sp = state["theta_v"]
            rho_sp = state["rho"]
            exner_sp = state["exner"]
    else:
        rho_sp = savepoint_nonhydro_exit.rho_new().asnumpy()
        exner_sp = timeloop_diffusion_savepoint_exit.exner().asnumpy()
        theta_sp = timeloop_diffusion_savepoint_exit.theta_v().asnumpy()
        vn_sp = timeloop_diffusion_savepoint_exit.vn().asnumpy()
        w_sp = timeloop_diffusion_savepoint_exit.w().asnumpy()

    assert test_utils.dallclose(
        prognostic_states.current.vn.asnumpy(),
        vn_sp,
        atol=6e-12,
    )

    assert test_utils.dallclose(
        prognostic_states.current.w.asnumpy(),
        w_sp,
        atol=8e-14,
    )

    assert test_utils.dallclose(
        prognostic_states.current.exner.asnumpy(),
        exner_sp,
    )

    assert test_utils.dallclose(
        prognostic_states.current.theta_v.asnumpy(),
        theta_sp,
        atol=4e-12,
    )

    assert test_utils.dallclose(
        prognostic_states.current.rho.asnumpy(),
        rho_sp,
    )


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment, experiment_type",
    [
        (
            definitions.Experiments.MCH_CH_R04B09,
            driver_init.ExperimentType.ANY.value,
        ),
    ],
)
def test_driver(
    experiment,
    experiment_type,
    *,
    data_provider,
    ranked_data_path,
    backend,
):
    """
    This is a only test to check if the icon4py driver runs from serialized data without verifying the end result.
    The timeloop is verified by test_run_timeloop_single_step above.
    TODO(anyone): Remove or modify this test when it is ready to run the driver from the grid file without having to initialize static fields from serialized data.
    """
    data_path = dt_utils.get_datapath_for_experiment(
        ranked_base_path=ranked_data_path,
        experiment=experiment,
    )
    gm = grid_utils.get_grid_manager_from_experiment(
        experiment=experiment,
        keep_skip_values=True,
        backend=backend,
    )

    backend_name = None
    for key, value in model_backends.BACKENDS.items():
        if value == backend:
            backend_name = key

    assert backend_name is not None

    icon4py_driver.icon4py_driver(
        [
            str(data_path),
            "--experiment_type",
            experiment_type,
            "--grid_file",
            str(gm._file_name),
            "--icon4py_driver_backend",
            backend_name,
        ],
        standalone_mode=False,
    )
