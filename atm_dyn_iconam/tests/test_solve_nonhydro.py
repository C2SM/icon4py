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
import numpy as np
import pytest

from icon4py.common.dimension import KDim
from icon4py.nh_solve.solve_nonydro import (
    NonHydrostaticConfig,
    NonHydrostaticParams,
    SolveNonhydro,
)
from icon4py.state_utils.diagnostic_state import (
    DiagnosticState,
    DiagnosticStateNonHydro,
)
from icon4py.state_utils.horizontal import CellParams, EdgeParams
from icon4py.state_utils.icon_grid import VerticalModelParams
from icon4py.state_utils.metric_state import MetricStateNonHydro
from icon4py.state_utils.prep_adv_state import PrepAdvection
from icon4py.state_utils.prognostic_state import PrognosticState

from .test_utils.helpers import random_field, zero_field
from .test_utils.simple_mesh import SimpleMesh


@pytest.mark.datatest
def test_nonhydro_params():
    config = NonHydrostaticConfig()
    nonhydro_params = NonHydrostaticParams(config)

    assert nonhydro_params.df32 == pytest.approx(
        config.divdamp_fac3 - config.divdamp_fac2, abs=1e-12
    )
    assert nonhydro_params.dz32 == pytest.approx(
        config.divdamp_z3 - config.divdamp_z2, abs=1e-12
    )
    assert nonhydro_params.df42 == pytest.approx(
        config.divdamp_fac4 - config.divdamp_fac2, abs=1e-12
    )
    assert nonhydro_params.dz42 == pytest.approx(
        config.divdamp_z4 - config.divdamp_z2, abs=1e-12
    )

    assert nonhydro_params.bqdr == pytest.approx(
        (
            nonhydro_params.df42 * nonhydro_params.dz32
            - nonhydro_params.df32 * nonhydro_params.dz42
        )
        / (
            nonhydro_params.dz32
            * nonhydro_params.dz42
            * (nonhydro_params.dz42 - nonhydro_params.dz32)
        ),
        abs=1e-12,
    )
    assert nonhydro_params.aqdr == pytest.approx(
        nonhydro_params.df32 / nonhydro_params.dz32
        - nonhydro_params.bqdr * nonhydro_params.dz32,
        abs=1e-12,
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "istep, step_date_init, step_date_exit",
    [(1, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000")],
)
def test_nonhydro_predictor_step(
    icon_grid,
    savepoint_nonhydro_init,
    data_provider,
    step_date_init,
    damping_height,
    grid_savepoint,
    savepoint_velocity_init,
    diffusion_savepoint_init,
    metrics_savepoint,
    metrics_nonhydro_savepoint,
    interpolation_savepoint,
    savepoint_nonhydro_exit,
):
    config = NonHydrostaticConfig()
    sp = savepoint_nonhydro_init
    sp_exit = savepoint_nonhydro_exit
    sp_dif = diffusion_savepoint_init
    sp_met = metrics_savepoint
    nonhydro_params = NonHydrostaticParams(config)
    vertical_params = VerticalModelParams(
        vct_a=grid_savepoint.vct_a(), rayleigh_damping_height=damping_height
    )
    sp_met_nh = metrics_nonhydro_savepoint
    sp_d = data_provider.from_savepoint_grid()
    sp_v = savepoint_velocity_init
    ntl1 = sp.get_metadata("ntl1").get("ntl1")
    ntl2 = sp.get_metadata("ntl2").get("ntl2")
    mesh = SimpleMesh()
    dtime = sp_v.get_metadata("dtime").get("dtime")
    recompute = sp_v.get_metadata("recompute").get("recompute")
    dyn_timestep = sp.get_metadata("dyn_timestep").get("dyn_timestep")
    linit = sp_v.get_metadata("linit").get("linit")
    prep_adv = PrepAdvection(vn_traj=sp.vn_traj(), mass_flx_me=sp.mass_flx_me())

    enh_smag_fac = zero_field(mesh, KDim)
    a_vec = random_field(mesh, KDim, low=1.0, high=10.0, extend={KDim: 1})
    fac = (0.67, 0.5, 1.3, 0.8)
    z = (0.1, 0.2, 0.3, 0.4)
    ntnd = sp_v.get_metadata("ntnd").get("ntnd")
    nnow = 0
    nnew = 1

    diagnostic_state = DiagnosticState(
        hdef_ic=None,
        div_ic=None,
        dwdx=None,
        dwdy=None,
        vt=sp_v.vt(),
        vn_ie=sp_v.vn_ie(),
        w_concorr_c=sp_v.w_concorr_c(),
        ddt_w_adv_pc_before=sp_v.ddt_w_adv_pc_before(ntnd),
        ddt_vn_apc_pc_before=sp_v.ddt_vn_apc_pc_before(ntnd),
        ntnd=ntnd,
    )

    diagnostic_state_nonhydro = DiagnosticStateNonHydro(
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
        ddt_vn_adv_ntl1=sp.ddt_vn_adv_ntl(1),
        ddt_vn_adv_ntl2=sp.ddt_vn_adv_ntl(2),
        ddt_w_adv_ntl1=sp.ddt_w_adv_ntl(1),
        ddt_w_adv_ntl2=sp.ddt_w_adv_ntl(2),
        ntl1=ntl1,
        ntl2=ntl2,
        rho_incr=None,  # TODO @nfarabullini: change back to this sp.rho_incr()
        vn_incr=None,  # TODO @nfarabullini: change back to this sp.vn_incr()
        exner_incr=None,  # TODO @nfarabullini: change back to this sp.exner_incr()
    )

    prognostic_state = PrognosticState(
        w=sp_v.w(),
        vn=sp_v.vn(),
        exner_pressure=None,
        theta_v=sp.theta_v_now(),
        rho=sp.rho_now(),
        exner=sp.exner_now(),
    )

    interpolation_state = interpolation_savepoint.construct_interpolation_state()
    metric_state = metrics_savepoint.construct_metric_state()

    metric_state_nonhydro = MetricStateNonHydro(
        exner_exfac=sp_met_nh.exner_exfac(),
        exner_ref_mc=sp_met_nh.exner_ref_mc(),
        wgtfacq_c=sp_met_nh.wgtfacq_c(),
        inv_ddqz_z_full=sp_met_nh.inv_ddqz_z_full(),
        rho_ref_mc=sp_met_nh.rho_ref_mc(),
        vwind_expl_wgt=sp_met_nh.vwind_expl_wgt(),
        d_exner_dz_ref_ic=sp_met_nh.d_exner_dz_ref_ic(),
        theta_ref_ic=sp_met_nh.theta_ref_ic(),
        d2dexdz2_fac1_mc=sp_met_nh.d2dexdz2_fac1_mc(),
        d2dexdz2_fac2_mc=sp_met_nh.d2dexdz2_fac2_mc(),
        vwind_impl_wgt=sp_met_nh.vwind_impl_wgt(),
        bdy_halo_c=sp_met_nh.bdy_halo_c(),
        ipeidx_dsl=sp_met_nh.ipeidx_dsl(),
        hmask_dd3d=sp_met_nh.hmask_dd3d(),
        scalfac_dd3d=sp_met_nh.scalfac_dd3d(),
        rayleigh_w=sp_met_nh.rayleigh_w(),
        rho_ref_me=sp_met_nh.rho_ref_me(),
        theta_ref_me=sp_met_nh.theta_ref_me(),
        mask_prog_halo_c=sp_met_nh.mask_prog_halo_c(),
        pg_exdist=sp_met_nh.pg_exdist(),
        wgtfacq_c_dsl=sp_met_nh.wgtfacq_c_dsl(),
        zdiff_gradp=sp_met_nh.zdiff_gradp(),
    )

    cell_geometry: CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: EdgeParams = grid_savepoint.construct_edge_geometry()

    prognostic_state_ls = [prognostic_state, prognostic_state]

    solve_nonhydro = SolveNonhydro()
    solve_nonhydro.init(
        grid=icon_grid,
        config=config,
        params=nonhydro_params,
        metric_state=metric_state,
        metric_state_nonhydro=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        a_vec=a_vec,
        enh_smag_fac=enh_smag_fac,
        cell_areas=cell_geometry.area,
        fac=fac,
        z=z,
    )

    solve_nonhydro.run_predictor_step(
        diagnostic_state=diagnostic_state,
        diagnostic_state_nonhydro=diagnostic_state_nonhydro,
        prognostic_state=prognostic_state_ls,
        config=config,
        params=nonhydro_params,
        inv_dual_edge_length=edge_geometry.inverse_dual_edge_lengths,
        primal_normal_cell=edge_geometry.primal_normal_cell,
        dual_normal_cell=edge_geometry.dual_normal_cell,
        inv_primal_edge_length=edge_geometry.inverse_primal_edge_lengths,
        tangent_orientation=edge_geometry.tangent_orientation,
        cfl_w_limit=sp_v.cfl_w_limit(),
        scalfac_exdiff=sp_v.scalfac_exdiff(),
        cell_areas=cell_geometry.area,
        owner_mask=sp_d.owner_mask(),
        f_e=sp_d.f_e(),
        area_edge=edge_geometry.edge_areas,
        dtime=dtime,
        idyn_timestep=dyn_timestep,
        l_recompute=recompute,
        l_init=linit,
        nnow=nnow,
        nnew=nnew,
    )

    icon_result_vn_new = sp_exit.vn_new()
    icon_result_vn_ie = sp_exit.vn_ie()
    icon_result_w_new = sp_exit.w_new()
    icon_result_exner_new = sp_exit.exner_new()
    icon_result_exner_now = sp_exit.exner_now()
    icon_result_theta_v_new = sp_exit.theta_v_new()
    icon_result_theta_v_ic = sp_exit.theta_v_ic()
    icon_result_rho_ic = sp_exit.rho_ic()
    icon_result_w_concorr_c = sp_exit.w_concorr_c()
    icon_result_mass_fl_e = sp_exit.mass_fl_e()

    icon_result_prep_adv_mass_flx_me = sp_exit.prep_adv_mass_flx_me()
    icon_result_prep_adv_vn_traj = sp_exit.prep_adv_vn_traj()

    # stencils 2, 3
    assert np.allclose(np.asarray(sp_exit.exner_pr())[1688:20896, :],
                       np.asarray(diagnostic_state_nonhydro.exner_pr)[1688:20896, :])
    assert np.allclose(np.asarray(sp_exit.z_exner_ex_pr())[1688:20896,:], np.asarray(solve_nonhydro.z_exner_ex_pr)[1688:20896,:])

    # stencils 4,5,6
    assert np.allclose(np.asarray(sp_exit.z_exner_ic())[1688:20896, :],
                       np.asarray(solve_nonhydro.z_exner_ic)[1688:20896, :])
    assert np.allclose(np.asarray(sp_exit.z_dexner_dz_c(1))[1688:20896, :],
                       np.asarray(solve_nonhydro.z_dexner_dz_c_1)[1688:20896, :])

    # stencils 7,8,9
    assert np.allclose(
        np.asarray(icon_result_rho_ic)[1688:20896, :], np.asarray(diagnostic_state_nonhydro.rho_ic)[1688:20896, :]
    )
    assert np.allclose(
        np.asarray(sp_exit.z_th_ddz_exner_c())[1688:20896, 1:], np.asarray(solve_nonhydro.z_th_ddz_exner_c)[1688:20896, 1:]
    )

    # stencils 7,8,9, 11
    assert np.allclose(
        np.asarray(sp_exit.z_theta_v_pr_ic())[1688:20896, :], np.asarray(solve_nonhydro.z_theta_v_pr_ic)[1688:20896, :]
    )
    assert np.allclose(
        np.asarray(sp_exit.theta_v_ic())[1688:20896, :], np.asarray(diagnostic_state_nonhydro.theta_v_ic)[1688:20896, :]
    )
    # stencils 7,8,9, 13
    assert np.allclose(  ## wrong
        np.asarray(sp_exit.z_rth_pr(1))[1688:20896, :], np.asarray(solve_nonhydro.z_rth_pr_1)[1688:20896, :]
    )
    assert np.allclose(
        np.asarray(sp_exit.z_rth_pr(2))[1688:20896, :], np.asarray(solve_nonhydro.z_rth_pr_2)[1688:20896, :]
    )

    # stencils 12
    assert np.allclose(np.asarray(sp_exit.z_dexner_dz_c(2))[1688:20896, :],
                       np.asarray(solve_nonhydro.z_dexner_dz_c_2)[1688:20896, :])

    # mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1
    assert np.allclose(
        np.asarray(sp_exit.z_rho_e())[3777:31558,:], np.asarray(solve_nonhydro.z_rho_e)[3777:31558,:]
    )
    assert np.allclose(
        np.asarray(sp_exit.z_theta_v_e())[3777:31558,:], np.asarray(solve_nonhydro.z_theta_v_e)[3777:31558,:]
    )

    # stencils 18,19, 20, 22
    assert np.allclose(
        np.asarray(sp_exit.z_gradh_exner()), np.asarray(solve_nonhydro.z_gradh_exner)
    )
    # stencil 21
    assert np.allclose(
        np.asarray(sp_exit.z_hydro_corr())[5387:31558,:], np.asarray(solve_nonhydro.z_hydro_corr)[5387:31558,:]
    )
    # stencils 24, 29,
    assert np.allclose(np.asarray(icon_result_vn_new), np.asarray(prognostic_state.vn))

    # stencil 30
    assert np.allclose(
        np.asarray(sp_exit.z_vn_avg()), np.asarray(solve_nonhydro.z_vn_avg)
    )
    # stencil 30
    assert np.allclose(
        np.asarray(sp_exit.z_graddiv_vn()), np.asarray(solve_nonhydro.z_graddiv_vn)
    )
    # stencil 30
    assert np.allclose(
        np.asarray(sp_exit.vt()), np.asarray(diagnostic_state.vt)
    )

    # stencil 32
    assert np.allclose(
        np.asarray(icon_result_mass_fl_e),
        np.asarray(diagnostic_state_nonhydro.mass_fl_e),
    )
    # stencil 32
    assert np.allclose(
        np.asarray(sp_exit.z_theta_v_fl_e()), np.asarray(solve_nonhydro.z_theta_v_fl_e)
    )

    # stencil 35,36, 37,38
    assert np.allclose(
        np.asarray(icon_result_vn_ie), np.asarray(diagnostic_state.vn_ie)
    )

    # stencil 35,36, 37,38
    assert np.allclose(
        np.asarray(sp_exit.z_vt_ie()), np.asarray(solve_nonhydro.z_vt_ie)
    )
    # stencil 35,36
    assert np.allclose(
        np.asarray(sp_exit.z_kin_hor_e()), np.asarray(solve_nonhydro.z_w_concorr_me)
    )


    # stencil 35,36, 37,38
    assert np.allclose(
        np.asarray(sp_exit.z_w_concorr_me()), np.asarray(solve_nonhydro.z_kin_hor_e)
    )
    # stencils 39,40
    assert np.allclose(
        np.asarray(icon_result_w_concorr_c), np.asarray(diagnostic_state.w_concorr_c)
    )

    assert np.allclose(np.asarray(icon_result_w_new), np.asarray(prognostic_state.w))
    assert np.allclose(
        np.asarray(icon_result_exner_new), np.asarray(prognostic_state.exner)
    )
    assert np.allclose(
        np.asarray(icon_result_theta_v_new), np.asarray(prognostic_state.theta_v)
    )



    assert np.allclose(
        np.asarray(icon_result_theta_v_ic),
        np.asarray(diagnostic_state_nonhydro.theta_v_ic),
    )


    assert np.allclose(
        np.asarray(icon_result_prep_adv_mass_flx_me), np.asarray(prep_adv.mass_flx_me)
    )
    assert np.allclose(
        np.asarray(icon_result_prep_adv_vn_traj), np.asarray(prep_adv.vn_traj)
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "istep, step_date_init, step_date_exit",
    [(2, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000")],
)
def test_nonhydro_corrector_step(
    icon_grid,
    savepoint_nonhydro_init,
    data_provider,
    step_date_init,
    damping_height,
    grid_savepoint,
    savepoint_velocity_init,
    diffusion_savepoint_init,
    metrics_savepoint,
    metrics_nonhydro_savepoint,
    interpolation_savepoint,
    savepoint_nonhydro_exit,
):
    config = NonHydrostaticConfig()
    sp = savepoint_nonhydro_init
    sp_exit = savepoint_nonhydro_exit
    sp_dif = diffusion_savepoint_init
    sp_met = metrics_savepoint
    sp_met_nh = metrics_nonhydro_savepoint
    nonhydro_params = NonHydrostaticParams(config)
    vertical_params = VerticalModelParams(
        vct_a=grid_savepoint.vct_a(), rayleigh_damping_height=damping_height
    )
    sp_d = data_provider.from_savepoint_grid()
    sp_v = savepoint_velocity_init
    ntl1 = sp.get_metadata("ntl1").get("ntl1")
    ntl2 = sp.get_metadata("ntl2").get("ntl2")
    ntnd = sp_v.get_metadata("ntnd").get("ntnd")
    mesh = SimpleMesh()
    dtime = sp_v.get_metadata("dtime").get("dtime")
    clean_mflx = sp_v.get_metadata("clean_mflx").get("clean_mflx")
    r_nsubsteps = sp_d.get_metadata("nsteps").get("nsteps")
    lprep_adv = sp_v.get_metadata("prep_adv").get("prep_adv")
    prep_adv = PrepAdvection(vn_traj=sp.vn_traj(), mass_flx_me=sp.mass_flx_me())

    enh_smag_fac = zero_field(mesh, KDim)
    a_vec = random_field(mesh, KDim, low=1.0, high=10.0, extend={KDim: 1})
    fac = (0.67, 0.5, 1.3, 0.8)
    z = (0.1, 0.2, 0.3, 0.4)
    nnow = 0
    nnew = 1

    diagnostic_state = DiagnosticState(
        hdef_ic=None,
        div_ic=None,
        dwdx=None,
        dwdy=None,
        vt=sp_v.vt(),
        vn_ie=sp_v.vn_ie(),
        w_concorr_c=sp_v.w_concorr_c(),
        ddt_w_adv_pc_before=sp_v.ddt_w_adv_pc_before(ntnd),
        ddt_vn_apc_pc_before=sp_v.ddt_vn_apc_pc_before(ntnd),
        ntnd=ntnd,
    )

    diagnostic_state_nonhydro = DiagnosticStateNonHydro(
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
        ntl1=ntl1,
        ntl2=ntl2,
        rho_incr=None,  # TODO @nfarabullini: change back to this sp.rho_incr()
        vn_incr=None,  # TODO @nfarabullini: change back to this sp.vn_incr()
        exner_incr=None,  # TODO @nfarabullini: change back to this sp.exner_incr()
    )

    prognostic_state = PrognosticState(
        w=sp_v.w(),
        vn=sp_v.vn(),
        exner_pressure=None,
        theta_v=sp_dif.theta_v(),
        rho=sp_dif.rho(),
        exner=sp_dif.exner(),
    )

    interpolation_state = interpolation_savepoint.construct_interpolation_state()
    metric_state = metrics_savepoint.construct_metric_state()

    metric_state_nonhydro = MetricStateNonHydro(
        exner_exfac=sp_met_nh.exner_exfac(),
        exner_ref_mc=sp_met_nh.exner_ref_mc(),
        wgtfacq_c=sp_met_nh.wgtfacq_c(),
        inv_ddqz_z_full=sp_met_nh.inv_ddqz_z_full(),
        rho_ref_mc=sp_met_nh.rho_ref_mc(),
        vwind_expl_wgt=sp_met_nh.vwind_expl_wgt(),
        d_exner_dz_ref_ic=sp_met_nh.d_exner_dz_ref_ic(),
        theta_ref_ic=sp_met_nh.theta_ref_ic(),
        d2dexdz2_fac1_mc=sp_met_nh.d2dexdz2_fac1_mc(),
        d2dexdz2_fac2_mc=sp_met_nh.d2dexdz2_fac2_mc(),
        vwind_impl_wgt=sp_met_nh.vwind_impl_wgt(),
        bdy_halo_c=sp_met_nh.bdy_halo_c(),
        ipeidx_dsl=sp_met_nh.ipeidx_dsl(),
        hmask_dd3d=sp_met_nh.hmask_dd3d(),
        scalfac_dd3d=sp_met_nh.scalfac_dd3d(),
        rayleigh_w=sp_met_nh.rayleigh_w(),
        rho_ref_me=sp_met_nh.rho_ref_me(),
        theta_ref_me=sp_met_nh.theta_ref_me(),
        mask_prog_halo_c=sp_met_nh.mask_prog_halo_c(),
        pg_exdist=sp_met_nh.pg_exdist(),
        wgtfacq_c_dsl=sp_met_nh.wgtfacq_c_dsl(),
        zdiff_gradp=sp_met_nh.zdiff_gradp(),
    )

    cell_geometry: CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: EdgeParams = grid_savepoint.construct_edge_geometry()

    prognostic_state_ls = [prognostic_state, prognostic_state]

    solve_nonhydro = SolveNonhydro()
    solve_nonhydro.init(
        grid=icon_grid,
        config=config,
        params=nonhydro_params,
        metric_state=metric_state,
        metric_state_nonhydro=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        a_vec=a_vec,
        enh_smag_fac=enh_smag_fac,
        cell_areas=cell_geometry.area,
        fac=fac,
        z=z,
    )

    solve_nonhydro.run_corrector_step(
        diagnostic_state=diagnostic_state,
        diagnostic_state_nonhydro=diagnostic_state_nonhydro,
        prognostic_state=prognostic_state_ls,
        config=config,
        params=nonhydro_params,
        inv_dual_edge_length=edge_geometry.inverse_dual_edge_lengths,
        inv_primal_edge_length=edge_geometry.inverse_primal_edge_lengths,
        tangent_orientation=edge_geometry.tangent_orientation,
        prep_adv=prep_adv,
        dtime=dtime,
        nnew=nnew,
        nnow=nnow,
        cfl_w_limit=sp_v.cfl_w_limit(),
        scalfac_exdiff=sp_v.scalfac_exdiff(),
        cell_areas=cell_geometry.area,
        owner_mask=sp_d.owner_mask(),
        f_e=sp_d.f_e(),
        area_edge=edge_geometry.edge_areas,
        lclean_mflx=clean_mflx,
        scal_divdamp_o2=sp.scal_divdamp_o2(),
        bdy_divdamp=sp.bdy_divdamp(),
        lprep_adv=lprep_adv,
    )

    icon_result_prep_adv_mass_flx_me = sp_exit.prep_adv_mass_flx_me()
    icon_result_prep_adv_vn_traj = sp_exit.prep_adv_vn_traj()

    # icon_result_z_graddiv_vn = sp_exit.z_graddiv_vn()
    # icon_result_exner_now = sp_exit.exner_now()

    # assert np.allclose(np.asarray(icon_result_z_graddiv_vn), np.asarray(prognostic_state_ls[nnew].exner))
    # assert np.allclose(np.asarray(icon_result_exner_now), np.asarray(prognostic_state_ls[nnow].exner))

    assert np.allclose(
        np.asarray(icon_result_prep_adv_mass_flx_me), np.asarray(prep_adv.mass_flx_me)
    )
    assert np.allclose(
        np.asarray(icon_result_prep_adv_vn_traj), np.asarray(prep_adv.vn_traj)
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "istep, step_date_init, step_date_exit",
    [(1, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000")],
)
def test_run_solve_nonhydro_multi_step(
    icon_grid,
    savepoint_nonhydro_init,
    data_provider,
    step_date_init,
    damping_height,
    grid_savepoint,
    savepoint_velocity_init,
    diffusion_savepoint_init,
    metrics_savepoint,
    metrics_nonhydro_savepoint,
    interpolation_savepoint,
    savepoint_nonhydro_exit,
):
    config = NonHydrostaticConfig()
    sp = savepoint_nonhydro_init
    sp_exit = savepoint_nonhydro_exit
    sp_dif = diffusion_savepoint_init
    sp_int = interpolation_savepoint
    sp_met = metrics_savepoint
    nonhydro_params = NonHydrostaticParams(config)
    vertical_params = VerticalModelParams(
        vct_a=grid_savepoint.vct_a(), rayleigh_damping_height=damping_height
    )
    sp_d = data_provider.from_savepoint_grid()
    sp_v = savepoint_velocity_init
    ntl1 = sp.get_metadata("ntl1").get("ntl1")
    ntl2 = sp.get_metadata("ntl2").get("ntl2")
    ntnd = sp_v.get_metadata("ntnd").get("ntnd")
    mesh = SimpleMesh()
    sp_met_nh = metrics_nonhydro_savepoint
    dtime = sp_v.get_metadata("dtime").get("dtime")
    clean_mflx = sp_v.get_metadata("clean_mflx").get("clean_mflx")
    r_nsubsteps = sp_d.get_metadata("nsteps").get("nsteps")
    lprep_adv = sp_v.get_metadata("prep_adv").get("prep_adv")
    prep_adv = PrepAdvection(vn_traj=sp.vn_traj(), mass_flx_me=sp.mass_flx_me())

    enh_smag_fac = zero_field(mesh, KDim)
    a_vec = random_field(mesh, KDim, low=1.0, high=10.0, extend={KDim: 1})
    fac = (0.67, 0.5, 1.3, 0.8)
    z = (0.1, 0.2, 0.3, 0.4)
    nnow = 0
    nnew = 1
    recompute = sp_v.get_metadata("recompute").get("recompute")
    linit = sp_v.get_metadata("linit").get("linit")
    dyn_timestep = sp_v.get_metadata("dyn_timestep").get("dyn_timestep")

    diagnostic_state = DiagnosticState(
        hdef_ic=None,
        div_ic=None,
        dwdx=None,
        dwdy=None,
        vt=sp_v.vt(),
        vn_ie=sp_v.vn_ie(),
        w_concorr_c=sp_v.w_concorr_c(),
        ddt_w_adv_pc_before=sp_v.ddt_w_adv_pc_before(ntnd),
        ddt_vn_apc_pc_before=sp_v.ddt_vn_apc_pc_before(ntnd),
        ntnd=ntnd,
    )

    diagnostic_state_nonhydro = DiagnosticStateNonHydro(
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
        ntl1=ntl1,
        ntl2=ntl2,
        rho_incr=None,  # TODO @nfarabullini: change back to this sp.rho_incr()
        vn_incr=None,  # TODO @nfarabullini: change back to this sp.vn_incr()
        exner_incr=None,  # TODO @nfarabullini: change back to this sp.exner_incr()
    )

    prognostic_state = PrognosticState(
        w=sp_v.w(),
        vn=sp_v.vn(),
        exner_pressure=None,
        theta_v=sp_dif.theta_v(),
        rho=sp_dif.rho(),
        exner=sp_dif.exner(),
    )

    interpolation_state = interpolation_savepoint.construct_interpolation_state()
    metric_state = metrics_savepoint.construct_metric_state()

    metric_state_nonhydro = MetricStateNonHydro(
        exner_exfac=sp_met_nh.exner_exfac(),
        exner_ref_mc=sp_met_nh.exner_ref_mc(),
        wgtfacq_c=sp_met_nh.wgtfacq_c(),
        inv_ddqz_z_full=sp_met_nh.inv_ddqz_z_full(),
        rho_ref_mc=sp_met_nh.rho_ref_mc(),
        vwind_expl_wgt=sp_met_nh.vwind_expl_wgt(),
        d_exner_dz_ref_ic=sp_met_nh.d_exner_dz_ref_ic(),
        theta_ref_ic=sp_met_nh.theta_ref_ic(),
        d2dexdz2_fac1_mc=sp_met_nh.d2dexdz2_fac1_mc(),
        d2dexdz2_fac2_mc=sp_met_nh.d2dexdz2_fac2_mc(),
        vwind_impl_wgt=sp_met_nh.vwind_impl_wgt(),
        bdy_halo_c=sp_met_nh.bdy_halo_c(),
        ipeidx_dsl=sp_met_nh.ipeidx_dsl(),
        hmask_dd3d=sp_met_nh.hmask_dd3d(),
        scalfac_dd3d=sp_met_nh.scalfac_dd3d(),
        rayleigh_w=sp_met_nh.rayleigh_w(),
        rho_ref_me=sp_met_nh.rho_ref_me(),
        theta_ref_me=sp_met_nh.theta_ref_me(),
        mask_prog_halo_c=sp_met_nh.mask_prog_halo_c(),
        pg_exdist=sp_met_nh.pg_exdist(),
        wgtfacq_c_dsl=sp_met_nh.wgtfacq_c_dsl(),
        zdiff_gradp=sp_met_nh.zdiff_gradp(),
    )

    cell_geometry: CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: EdgeParams = grid_savepoint.construct_edge_geometry()

    prognostic_state_ls = [prognostic_state, prognostic_state]

    solve_nonhydro = SolveNonhydro()
    solve_nonhydro.init(
        grid=icon_grid,
        config=config,
        params=nonhydro_params,
        metric_state=metric_state,
        metric_state_nonhydro=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        a_vec=a_vec,
        enh_smag_fac=enh_smag_fac,
        cell_areas=cell_geometry.area,
        fac=fac,
        z=z,
    )

    for _ in range(4):
        solve_nonhydro.time_step(
            diagnostic_state=diagnostic_state,
            diagnostic_state_nonhydro=diagnostic_state_nonhydro,
            prognostic_state=prognostic_state_ls,
            prep_adv=prep_adv,
            config=config,
            params=nonhydro_params,
            inv_dual_edge_length=edge_geometry.inverse_dual_edge_lengths,
            primal_normal_cell=edge_geometry.primal_normal_cell,
            dual_normal_cell=edge_geometry.dual_normal_cell,
            inv_primal_edge_length=edge_geometry.inverse_primal_edge_lengths,
            tangent_orientation=edge_geometry.tangent_orientation,
            cfl_w_limit=sp_v.cfl_w_limit(),
            scalfac_exdiff=sp_v.scalfac_exdiff(),
            cell_areas=cell_geometry.area,
            owner_mask=sp_d.owner_mask(),
            f_e=sp_d.f_e(),
            area_edge=sp_d.edge_areas(),
            bdy_divdamp=sp.bdy_divdamp(),
            dtime=dtime,
            idyn_timestep=dyn_timestep,
            l_recompute=recompute,
            l_init=linit,
            nnew=nnew,
            nnow=nnow,
            lclean_mflx=clean_mflx,
            lprep_adv=lprep_adv,
        )

    icon_result_exner_new = sp_exit.exner_new()
    icon_result_exner_now = sp_exit.exner_now()
    icon_result_mass_fl_e = sp_exit.mass_fl_e()
    icon_result_prep_adv_mass_flx_me = sp_exit.prep_adv_mass_flx_me()
    icon_result_prep_adv_vn_traj = sp_exit.prep_adv_vn_traj()
    icon_result_rho_ic = sp_exit.rho_ic()
    icon_result_theta_v_ic = sp_exit.theta_v_ic()
    icon_result_theta_v_new = sp_exit.theta_v_new()
    icon_result_vn_ie = sp_exit.vn_ie()
    icon_result_vn_new = sp_exit.vn_new()
    icon_result_w_concorr_c = sp_exit.w_concorr_c()
    icon_result_w_new = sp_exit.w_new()

    assert np.allclose(
        np.asarray(icon_result_exner_new), np.asarray(prognostic_state_ls[nnew].exner)
    )
    assert np.allclose(
        np.asarray(icon_result_exner_now), np.asarray(prognostic_state_ls[nnow].exner)
    )
    assert np.allclose(
        np.asarray(icon_result_prep_adv_mass_flx_me), np.asarray(prep_adv.mass_flx_me)
    )
    assert np.allclose(
        np.asarray(icon_result_mass_fl_e),
        np.asarray(diagnostic_state_nonhydro.mass_fl_e),
    )
    assert np.allclose(
        np.asarray(icon_result_prep_adv_vn_traj), np.asarray(prep_adv.vn_traj)
    )
    assert np.allclose(
        np.asarray(icon_result_rho_ic), np.asarray(diagnostic_state_nonhydro.rho_ic)
    )
    assert np.allclose(
        np.asarray(icon_result_theta_v_ic),
        np.asarray(diagnostic_state_nonhydro.theta_v_ic),
    )
    assert np.allclose(
        np.asarray(icon_result_theta_v_new),
        np.asarray(prognostic_state_ls[nnew].theta_v),
    )
    assert np.allclose(
        np.asarray(icon_result_vn_ie), np.asarray(diagnostic_state.vn_ie)
    )
    assert np.allclose(
        np.asarray(icon_result_vn_new), np.asarray(prognostic_state_ls[nnew].vn)
    )
    assert np.allclose(
        np.asarray(icon_result_w_concorr_c), np.asarray(diagnostic_state.w_concorr_c)
    )
    assert np.allclose(
        np.asarray(icon_result_w_new), np.asarray(prognostic_state_ls[nnew].w)
    )
