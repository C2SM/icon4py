# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path
import os

import numpy as np
import pytest

from icon4py.model.atmosphere.diffusion.diffusion import Diffusion, DiffusionParams
from icon4py.model.atmosphere.dycore.nh_solve.solve_nonhydro import (
    NonHydrostaticConfig,
    NonHydrostaticParams,
    SolveNonhydro,
)
from icon4py.model.atmosphere.dycore.state_utils.states import (
    DiagnosticStateNonHydro,
    InterpolationState,
    MetricStateNonHydro,
    PrepAdvection,
)
from icon4py.model.atmosphere.dycore.state_utils.utils import _allocate
from icon4py.model.atmosphere.dycore.state_utils.z_fields import ZFields
from icon4py.model.common.dimension import CEDim, CellDim, EdgeDim, VertexDim, C2E2C2EDim, KDim
from icon4py.model.common.grid.horizontal import CellParams, EdgeParams, HorizontalMarkerIndex
from icon4py.model.common.grid.vertical import VerticalModelParams
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.common.states.diagnostic_state import DiagnosticState, DiagnosticMetricState
from icon4py.model.common.test_utils.helpers import as_1D_sparse_field, dallclose
from icon4py.model.driver.dycore_driver import TimeLoop
from icon4py.model.common.test_utils import serialbox_utils as sb
from icon4py.model.driver.io_utils import model_initialization_jabw
from icon4py.model.driver.serialbox_helpers import (
    construct_diagnostics_for_diffusion,
    construct_interpolation_state_for_diffusion,
    construct_metric_state_for_diffusion,
)
from icon4py.model.common.diagnostic_calculations.mo_diagnose_temperature_pressure import mo_diagnose_temperature, mo_diagnose_pressure_sfc, mo_diagnose_pressure
from icon4py.model.common.constants import CPD_O_RD, P0REF, GRAV_O_RD
from gt4py.next.program_processors.runners.gtfn import run_gtfn, run_gtfn_cached

compiler_backend = run_gtfn
compiler_cached_backend = run_gtfn_cached
backend = compiler_backend

@pytest.mark.datatest
@pytest.mark.parametrize(
"path, experiment_name, fname_prefix, rank, time_discretization_veladv_offctr, time_discretization_rhotheta_offctr, debug",
    [
        (
            Path("/home/ong/PycharmProjects/main/testdata/jw_node1_nproma50000/"),
            "jabw",
            "jabw",
            0,
            0.25,
            -0.1,
            False
        ),
    ],
)
def test_jabw_initial_condition(
    path,
    experiment_name,
    fname_prefix,
    rank,
    time_discretization_veladv_offctr,
    time_discretization_rhotheta_offctr,
    debug
):
    data_provider = sb.IconSerialDataProvider(
        fname_prefix, str(path.absolute()), False, mpi_rank=rank
    )
    sp = data_provider.from_savepoint_grid()

    icon_grid = data_provider.from_savepoint_grid().construct_icon_grid()

    edge_geometry = sp.construct_edge_geometry()
    cell_geometry = sp.construct_cell_geometry()

    (
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        z_fields,
        prep_adv,
        divdamp_fac_o2,
        diagnostic_state,
        prognostic_state_now,
        prognostic_state_next
    ) = model_initialization_jabw(
        icon_grid,
        cell_geometry,
        edge_geometry,
        time_discretization_veladv_offctr,
        time_discretization_rhotheta_offctr,
        path,
        rank
    )

    # note that w is not verified because we decided to force w to zero in python framework after discussion

    assert dallclose(
        data_provider.from_savepoint_jabw_final().rho().asnumpy(),
        prognostic_state_now.rho.asnumpy()
    )

    assert dallclose(
        data_provider.from_savepoint_jabw_final().exner().asnumpy(),
        prognostic_state_now.exner.asnumpy()
    )

    assert dallclose(
        data_provider.from_savepoint_jabw_final().theta_v().asnumpy(),
        prognostic_state_now.theta_v.asnumpy()
    )

    assert dallclose(
        data_provider.from_savepoint_jabw_final().vn().asnumpy(),
        prognostic_state_now.vn.asnumpy()
    )

    assert dallclose(
        data_provider.from_savepoint_jabw_final().pressure().asnumpy(),
        diagnostic_state.pressure.asnumpy()
    )

    assert dallclose(
        data_provider.from_savepoint_jabw_final().temperature().asnumpy(),
        diagnostic_state.temperature.asnumpy()
    )

    assert dallclose(
        data_provider.from_savepoint_jabw_init().pressure_sfc().asnumpy(),
        diagnostic_state.pressure_sfc.asnumpy()
    )

    # verify GT4Py version
    mo_diagnose_temperature.with_backend(backend)(
        prognostic_state_now.theta_v,
        prognostic_state_now.exner,
        diagnostic_state.temperature,
        icon_grid.get_start_index(CellDim, HorizontalMarkerIndex.interior(CellDim)),
        icon_grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim)),
        0,
        icon_grid.num_levels,
        offset_provider={}
    )

    exner_nlev_minus2 = prognostic_state_now.exner[:, icon_grid.num_levels - 3]
    temperature_nlev = diagnostic_state.temperature[:, icon_grid.num_levels - 1]
    temperature_nlev_minus1 = diagnostic_state.temperature[:, icon_grid.num_levels - 2]
    temperature_nlev_minus2 = diagnostic_state.temperature[:, icon_grid.num_levels - 3]
    # TODO (Chia Rui): ddqz_z_full is constant, move slicing to initialization
    ddqz_z_full_nlev = data_provider.from_metrics_savepoint().ddqz_z_full()[:, icon_grid.num_levels - 1]
    ddqz_z_full_nlev_minus1 = data_provider.from_metrics_savepoint().ddqz_z_full()[:, icon_grid.num_levels - 2]
    ddqz_z_full_nlev_minus2 = data_provider.from_metrics_savepoint().ddqz_z_full()[:, icon_grid.num_levels - 3]
    mo_diagnose_pressure_sfc.with_backend(backend)(
        exner_nlev_minus2,
        temperature_nlev,
        temperature_nlev_minus1,
        temperature_nlev_minus2,
        ddqz_z_full_nlev,
        ddqz_z_full_nlev_minus1,
        ddqz_z_full_nlev_minus2,
        diagnostic_state.pressure_sfc,
        CPD_O_RD,
        P0REF,
        GRAV_O_RD,
        icon_grid.get_start_index(CellDim, HorizontalMarkerIndex.interior(CellDim)),
        icon_grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim)),
        offset_provider={}
    )


    mo_diagnose_pressure.with_backend(backend)(
        data_provider.from_metrics_savepoint().ddqz_z_full(),
        diagnostic_state.temperature,
        diagnostic_state.pressure_sfc,
        diagnostic_state.pressure,
        diagnostic_state.pressure_ifc,
        icon_grid.get_start_index(CellDim, HorizontalMarkerIndex.interior(CellDim)),
        icon_grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim)),
        0,
        icon_grid.num_levels,
        offset_provider={}
    )


    if debug:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = script_dir + "/"

        def printing(ref, predict, title: str):
            with open(base_dir + "analysis_" + title + ".dat", "w") as f:
                if (len(ref.shape) == 1):
                    cell_size = ref.shape[0]
                    k_size = 0
                elif (len(ref.shape) == 2):
                    cell_size = ref.shape[0]
                    k_size = ref.shape[1]
                else:
                    print("Dimension is not 1 or 2, do nothing in printing for ", title)
                    return
                print(title, cell_size, k_size)
                difference = np.abs(ref - predict)
                if k_size > 0:
                    for i in range(cell_size):
                        for k in range(k_size):
                            f.write("{0:7d} {1:7d}".format(i, k))
                            f.write(
                                " {0:.20e} {1:.20e} {2:.20e} ".format(
                                    difference[i, k], ref[i, k], predict[i, k]
                                )
                            )
                            f.write("\n")
                else:
                    for i in range(cell_size):
                        f.write("{0:7d}".format(i))
                        f.write(
                            " {0:.20e} {1:.20e} {2:.20e} ".format(
                                difference[i], ref[i], predict[i]
                            )
                        )
                        f.write("\n")

        printing(
            data_provider.from_savepoint_jabw_final().rho().asnumpy(),
            prognostic_state_now.rho.asnumpy(),
            "rho",
        )

        printing(
            data_provider.from_savepoint_jabw_final().exner().asnumpy(),
            prognostic_state_now.exner.asnumpy(),
            "exner",
        )

        printing(
            data_provider.from_savepoint_jabw_final().theta_v().asnumpy(),
            prognostic_state_now.theta_v.asnumpy(),
            "theta_v",
        )

        printing(
            data_provider.from_savepoint_jabw_final().vn().asnumpy(),
            prognostic_state_now.vn.asnumpy(),
            "vn",
        )

        printing(
            data_provider.from_savepoint_jabw_final().w().asnumpy(),
            prognostic_state_now.w.asnumpy(),
            "w",
        )

        printing(
            data_provider.from_savepoint_jabw_final().pressure().asnumpy(),
            diagnostic_state.pressure.asnumpy(),
            "pressure",
        )

        printing(
            data_provider.from_savepoint_jabw_final().temperature().asnumpy(),
            diagnostic_state.temperature.asnumpy(),
            "temperature",
        )

        printing(
            data_provider.from_savepoint_jabw_init().pressure_sfc().asnumpy(),
            diagnostic_state.pressure_sfc.asnumpy(),
            "pressure_sfc",
        )

        printing(
            data_provider.from_savepoint_jabw_first_output().u().asnumpy(),
            diagnostic_state.u.asnumpy(),
            "u",
        )

        printing(
            data_provider.from_savepoint_jabw_first_output().v().asnumpy(),
            diagnostic_state.v.asnumpy(),
            "v",
        )

        printing(
            data_provider.from_savepoint_jabw_first_output().temperature().asnumpy(),
            diagnostic_state.temperature.asnumpy(),
            "temperature1",
        )

        printing(
            data_provider.from_savepoint_jabw_first_output().pressure_sfc().asnumpy(),
            diagnostic_state.pressure_sfc.asnumpy(),
            "pressure_sfc1",
        )

        printing(
            data_provider.from_savepoint_jabw_first_output().pressure().asnumpy(),
            diagnostic_state.pressure.asnumpy(),
            "pressure1",
        )

    assert dallclose(
        data_provider.from_savepoint_jabw_first_output().u().asnumpy(),
        diagnostic_state.u.asnumpy()
    )

    assert dallclose(
        data_provider.from_savepoint_jabw_first_output().v().asnumpy(),
        diagnostic_state.v.asnumpy(),
        atol=1.e-13
    )

    assert dallclose(
        data_provider.from_savepoint_jabw_first_output().temperature().asnumpy(),
        diagnostic_state.temperature.asnumpy()
    )

    assert dallclose(
        data_provider.from_savepoint_jabw_first_output().pressure_sfc().asnumpy(),
        diagnostic_state.pressure_sfc.asnumpy()
    )

    #assert dallclose(
    #    data_provider.from_savepoint_jabw_first_output().pressure().asnumpy(),
    #    diagnostic_state.pressure.asnumpy()
    #)

# testing on MCH_CH_r04b09_dsl data

@pytest.mark.datatest
@pytest.mark.parametrize(
    "debug_mode,istep_init, istep_exit, jstep_init, jstep_exit,timeloop_date_init, timeloop_date_exit, step_date_init, step_date_exit, timeloop_diffusion_linit_init, timeloop_diffusion_linit_exit, vn_only",
    [
        (
            False,
            1,
            2,
            0,
            1,
            "2021-06-20T12:00:00.000",
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
            True,
            False,
            False,
        ),
        (
            False,
            1,
            2,
            0,
            1,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:20.000",
            "2021-06-20T12:00:20.000",
            "2021-06-20T12:00:20.000",
            False,
            False,
            True,
        ),
    ],
)
def test_run_timeloop_single_step(
    debug_mode,
    timeloop_date_init,
    timeloop_date_exit,
    grid_savepoint,
    icon_grid,
    metrics_savepoint,
    interpolation_savepoint,
    r04b09_diffusion_config,
    r04b09_iconrun_config,
    damping_height,
    timeloop_diffusion_savepoint_init,
    timeloop_diffusion_savepoint_exit,
    savepoint_velocity_init,
    savepoint_nonhydro_init,
    savepoint_nonhydro_exit,
):
    diffusion_config = r04b09_diffusion_config
    diffusion_dtime = timeloop_diffusion_savepoint_init.get_metadata("dtime").get("dtime")
    edge_geometry: EdgeParams = grid_savepoint.construct_edge_geometry()
    cell_geometry: CellParams = grid_savepoint.construct_cell_geometry()
    diffusion_interpolation_state = construct_interpolation_state_for_diffusion(
        interpolation_savepoint
    )
    diffusion_metric_state = construct_metric_state_for_diffusion(metrics_savepoint)
    diffusion_diagnostic_state = construct_diagnostics_for_diffusion(
        timeloop_diffusion_savepoint_init
    )
    vertical_params = VerticalModelParams(
        vct_a=grid_savepoint.vct_a(),
        rayleigh_damping_height=damping_height,
        nflatlev=grid_savepoint.nflatlev(),
        nflat_gradp=grid_savepoint.nflat_gradp(),
    )
    additional_parameters = DiffusionParams(diffusion_config)

    diffusion = Diffusion()
    diffusion.init(
        grid=icon_grid,
        config=diffusion_config,
        params=additional_parameters,
        vertical_params=vertical_params,
        metric_state=diffusion_metric_state,
        interpolation_state=diffusion_interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
    )

    # Default construction is for the MCH_CH_r04b09_dsl run config for nonhydro
    nonhydro_config = NonHydrostaticConfig()
    sp = savepoint_nonhydro_init
    nonhydro_params = NonHydrostaticParams(nonhydro_config)
    sp_v = savepoint_velocity_init
    nonhydro_dtime = savepoint_velocity_init.get_metadata("dtime").get("dtime")
    # do_prep_adv actually depends on other factors: idiv_method == 1 .AND. (ltransport .OR. p_patch%n_childdom > 0 .AND. grf_intmethod_e >= 5)
    do_prep_adv = sp_v.get_metadata("prep_adv").get("prep_adv")
    prep_adv = PrepAdvection(
        vn_traj=sp.vn_traj(), mass_flx_me=sp.mass_flx_me(), mass_flx_ic=sp.mass_flx_ic()
    )

    assert timeloop_diffusion_savepoint_init.fac_bdydiff_v() == diffusion.fac_bdydiff_v
    assert r04b09_iconrun_config.dtime == diffusion_dtime

    z_fields = ZFields(
        z_gradh_exner=_allocate(EdgeDim, KDim, grid=icon_grid),
        z_alpha=_allocate(CellDim, KDim, is_halfdim=True, grid=icon_grid),
        z_beta=_allocate(CellDim, KDim, grid=icon_grid),
        z_w_expl=_allocate(CellDim, KDim, is_halfdim=True, grid=icon_grid),
        z_exner_expl=_allocate(CellDim, KDim, grid=icon_grid),
        z_q=_allocate(CellDim, KDim, grid=icon_grid),
        z_contr_w_fl_l=_allocate(CellDim, KDim, is_halfdim=True, grid=icon_grid),
        z_rho_e=_allocate(EdgeDim, KDim, grid=icon_grid),
        z_theta_v_e=_allocate(EdgeDim, KDim, grid=icon_grid),
        z_graddiv_vn=_allocate(EdgeDim, KDim, grid=icon_grid),
        z_rho_expl=_allocate(CellDim, KDim, grid=icon_grid),
        z_dwdz_dd=_allocate(CellDim, KDim, grid=icon_grid),
        z_kin_hor_e=_allocate(EdgeDim, KDim, grid=icon_grid),
        z_vt_ie=_allocate(EdgeDim, KDim, grid=icon_grid),
    )

    grg = interpolation_savepoint.geofac_grg()
    nonhydro_interpolation_state = InterpolationState(
        c_lin_e=interpolation_savepoint.c_lin_e(),
        c_intp=interpolation_savepoint.c_intp(),
        e_flx_avg=interpolation_savepoint.e_flx_avg(),
        geofac_grdiv=interpolation_savepoint.geofac_grdiv(),
        geofac_rot=interpolation_savepoint.geofac_rot(),
        pos_on_tplane_e_1=interpolation_savepoint.pos_on_tplane_e_x(),
        pos_on_tplane_e_2=interpolation_savepoint.pos_on_tplane_e_y(),
        rbf_vec_coeff_e=interpolation_savepoint.rbf_vec_coeff_e(),
        e_bln_c_s=as_1D_sparse_field(interpolation_savepoint.e_bln_c_s(), CEDim),
        rbf_coeff_1=interpolation_savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=interpolation_savepoint.rbf_vec_coeff_v2(),
        geofac_div=as_1D_sparse_field(interpolation_savepoint.geofac_div(), CEDim),
        geofac_n2s=interpolation_savepoint.geofac_n2s(),
        geofac_grg_x=grg[0],
        geofac_grg_y=grg[1],
        nudgecoeff_e=interpolation_savepoint.nudgecoeff_e(),
    )
    nonhydro_metric_state = MetricStateNonHydro(
        bdy_halo_c=metrics_savepoint.bdy_halo_c(),
        mask_prog_halo_c=metrics_savepoint.mask_prog_halo_c(),
        rayleigh_w=metrics_savepoint.rayleigh_w(),
        exner_exfac=metrics_savepoint.exner_exfac(),
        exner_ref_mc=metrics_savepoint.exner_ref_mc(),
        wgtfac_c=metrics_savepoint.wgtfac_c(),
        wgtfacq_c=metrics_savepoint.wgtfacq_c_dsl(),
        inv_ddqz_z_full=metrics_savepoint.inv_ddqz_z_full(),
        rho_ref_mc=metrics_savepoint.rho_ref_mc(),
        theta_ref_mc=metrics_savepoint.theta_ref_mc(),
        vwind_expl_wgt=metrics_savepoint.vwind_expl_wgt(),
        d_exner_dz_ref_ic=metrics_savepoint.d_exner_dz_ref_ic(),
        ddqz_z_half=metrics_savepoint.ddqz_z_half(),
        theta_ref_ic=metrics_savepoint.theta_ref_ic(),
        d2dexdz2_fac1_mc=metrics_savepoint.d2dexdz2_fac1_mc(),
        d2dexdz2_fac2_mc=metrics_savepoint.d2dexdz2_fac2_mc(),
        rho_ref_me=metrics_savepoint.rho_ref_me(),
        theta_ref_me=metrics_savepoint.theta_ref_me(),
        ddxn_z_full=metrics_savepoint.ddxn_z_full(),
        zdiff_gradp=metrics_savepoint.zdiff_gradp(),
        vertoffset_gradp=metrics_savepoint.vertoffset_gradp(),
        ipeidx_dsl=metrics_savepoint.ipeidx_dsl(),
        pg_exdist=metrics_savepoint.pg_exdist(),
        ddqz_z_full_e=metrics_savepoint.ddqz_z_full_e(),
        ddxt_z_full=metrics_savepoint.ddxt_z_full(),
        wgtfac_e=metrics_savepoint.wgtfac_e(),
        wgtfacq_e=metrics_savepoint.wgtfacq_e_dsl(icon_grid.num_levels),
        vwind_impl_wgt=metrics_savepoint.vwind_impl_wgt(),
        hmask_dd3d=metrics_savepoint.hmask_dd3d(),
        scalfac_dd3d=metrics_savepoint.scalfac_dd3d(),
        coeff1_dwdz=metrics_savepoint.coeff1_dwdz(),
        coeff2_dwdz=metrics_savepoint.coeff2_dwdz(),
        coeff_gradekin=metrics_savepoint.coeff_gradekin(),
    )

    cell_geometry: CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: EdgeParams = grid_savepoint.construct_edge_geometry()

    solve_nonhydro = SolveNonhydro()
    solve_nonhydro.init(
        grid=icon_grid,
        config=nonhydro_config,
        params=nonhydro_params,
        metric_state_nonhydro=nonhydro_metric_state,
        interpolation_state=nonhydro_interpolation_state,
        vertical_params=vertical_params,
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
        owner_mask=grid_savepoint.c_owner_mask(),
    )

    nonhydro_diagnostic_state = DiagnosticStateNonHydro(
        theta_v_ic=sp.theta_v_ic(),
        exner_pr=sp.exner_pr(),
        rho_ic=sp.rho_ic(),
        ddt_exner_phy=sp.ddt_exner_phy(),
        grf_tend_rho=sp.grf_tend_rho(),
        grf_tend_thv=sp.grf_tend_thv(),
        grf_tend_w=sp.grf_tend_w(),
        mass_fl_e=sp.mass_fl_e(),
        ddt_vn_phy=sp.ddt_vn_phy(),
        grf_tend_vn=sp.grf_tend_vn(),
        ddt_vn_apc_ntl1=sp_v.ddt_vn_apc_pc(1),
        ddt_vn_apc_ntl2=sp_v.ddt_vn_apc_pc(2),
        ddt_w_adv_ntl1=sp_v.ddt_w_adv_pc(1),
        ddt_w_adv_ntl2=sp_v.ddt_w_adv_pc(2),
        vt=sp_v.vt(),
        vn_ie=sp_v.vn_ie(),
        w_concorr_c=sp_v.w_concorr_c(),
        rho_incr=None,  # sp.rho_incr(),
        vn_incr=None,  # sp.vn_incr(),
        exner_incr=None,  # sp.exner_incr(),
        exner_dyn_incr=sp.exner_dyn_incr(),
    )

    timeloop = TimeLoop(r04b09_iconrun_config, icon_grid, diffusion, solve_nonhydro, is_run_from_serializedData=True)

    assert timeloop.substep_timestep == nonhydro_dtime

    if timeloop_date_exit == "2021-06-20T12:00:10.000":
        prognostic_state = timeloop_diffusion_savepoint_init.construct_prognostics()
    else:
        prognostic_state = PrognosticState(
            w=sp.w_now(),
            vn=sp.vn_now(),
            theta_v=sp.theta_v_now(),
            rho=sp.rho_now(),
            exner=sp.exner_now(),
        )
    prognostic_state_new = PrognosticState(
        w=sp.w_new(),
        vn=sp.vn_new(),
        theta_v=sp.theta_v_new(),
        rho=sp.rho_new(),
        exner=sp.exner_new(),
    )

    prognostic_state_list = [prognostic_state, prognostic_state_new]

    diagnostic_state = DiagnosticState(
        pressure=_allocate(CellDim, KDim, grid=icon_grid),
        pressure_ifc=_allocate(CellDim, KDim, grid=icon_grid),
        temperature=_allocate(CellDim, KDim, grid=icon_grid),
        pressure_sfc=_allocate(CellDim, grid=icon_grid),
        u=_allocate(CellDim, KDim, grid=icon_grid),
        v=_allocate(CellDim, KDim, grid=icon_grid),
    )

    diagnostic_metric_state = DiagnosticMetricState(
        ddqz_z_full=_allocate(CellDim, KDim, grid=icon_grid, dtype=float),
        rbf_vec_coeff_c1=_allocate(CellDim, C2E2C2EDim, grid=icon_grid, dtype=float), # TODO: change to C2E2C2EDim
        rbf_vec_coeff_c2=_allocate(CellDim, C2E2C2EDim, grid=icon_grid, dtype=float),
        v_lat=_allocate(VertexDim, grid=icon_grid),
        v_lon=_allocate(VertexDim, grid=icon_grid),
        e_lat=_allocate(EdgeDim, grid=icon_grid),
        e_lon=_allocate(EdgeDim, grid=icon_grid),
        cell_center_lat=_allocate(CellDim, grid=icon_grid),
        cell_center_lon=_allocate(CellDim, grid=icon_grid),
        vct_a=grid_savepoint.vct_a(),
    )

    timeloop.time_integration(
        diffusion_diagnostic_state,
        nonhydro_diagnostic_state,
        diagnostic_metric_state,
        diagnostic_state,
        prognostic_state_list,
        prep_adv,
        z_fields,
        sp.divdamp_fac_o2(),
        do_prep_adv,
    )

    rho_sp = savepoint_nonhydro_exit.rho_new()
    exner_sp = timeloop_diffusion_savepoint_exit.exner()
    theta_sp = timeloop_diffusion_savepoint_exit.theta_v()
    vn_sp = timeloop_diffusion_savepoint_exit.vn()
    w_sp = timeloop_diffusion_savepoint_exit.w()


    if debug_mode:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = script_dir + "/"

        def printing(ref, predict, title: str):
            with open(base_dir + "analysis_" + title + ".dat", "w") as f:
                cell_size = ref.shape[0]
                k_size = ref.shape[1]
                print(title, cell_size, k_size)
                difference = np.abs(ref - predict)
                for i in range(cell_size):
                    for k in range(k_size):
                        f.write("{0:7d} {1:7d}".format(i, k))
                        f.write(
                            " {0:.20e} {1:.20e} {2:.20e} ".format(
                                difference[i, k], ref[i, k], predict[i, k]
                            )
                        )
                        f.write("\n")

        printing(
            rho_sp.asnumpy(),
            prognostic_state_list[timeloop.prognostic_now].rho.asnumpy(),
            "rho",
        )
        printing(
            exner_sp.asnumpy(),
            prognostic_state_list[timeloop.prognostic_now].exner.asnumpy(),
            "exner",
        )
        printing(
            theta_sp.asnumpy(),
            prognostic_state_list[timeloop.prognostic_now].theta_v.asnumpy(),
            "theta_v",
        )
        printing(
            w_sp.asnumpy(),
            prognostic_state_list[timeloop.prognostic_now].w.asnumpy(),
            "w",
        )
        printing(
            vn_sp.asnumpy(),
            prognostic_state_list[timeloop.prognostic_now].vn.asnumpy(),
            "vn",
        )

    assert dallclose(
        vn_sp.asnumpy(),
        prognostic_state_list[timeloop.prognostic_now].vn.asnumpy(),
        atol=5e-13,
    )

    assert dallclose(
        w_sp.asnumpy(),
        prognostic_state_list[timeloop.prognostic_now].w.asnumpy(),
        atol=8e-14,
    )

    assert dallclose(
        exner_sp.asnumpy(),
        prognostic_state_list[timeloop.prognostic_now].exner.asnumpy(),
    )

    assert dallclose(
        theta_sp.asnumpy(),
        prognostic_state_list[timeloop.prognostic_now].theta_v.asnumpy(),
    )

    assert dallclose(
        rho_sp.asnumpy(),
        prognostic_state_list[timeloop.prognostic_now].rho.asnumpy(),
    )


