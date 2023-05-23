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
from icon4py.state_utils.icon_grid import VerticalModelParams
from icon4py.state_utils.interpolation_state import InterpolationState
from icon4py.state_utils.metric_state import MetricState, MetricStateNonHydro
from icon4py.state_utils.prep_adv_state import PrepAdvection
from icon4py.state_utils.prognostic_state import PrognosticState
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


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
    metric_savepoint,
    interpolation_savepoint,
    savepoint_nonhydro_exit,
):
    config = NonHydrostaticConfig()
    sp = savepoint_nonhydro_init
    sp_exit = savepoint_nonhydro_exit
    sp_dif = diffusion_savepoint_init
    sp_int = interpolation_savepoint
    sp_met = metric_savepoint
    nonhydro_params = NonHydrostaticParams(config)
    vertical_params = VerticalModelParams(
        vct_a=grid_savepoint.vct_a(), rayleigh_damping_height=damping_height
    )
    sp_d = data_provider.from_savepoint_grid()
    sp_v = savepoint_velocity_init
    ntl1 = sp.get_metadata("ntl1").get("ntl1")
    ntl2 = sp.get_metadata("ntl2").get("ntl2")
    mesh = SimpleMesh()
    dtime = sp_v.get_metadata("dtime").get("dtime")
    recompute = sp_v.get_metadata("recompute").get("recompute")
    dyn_timestep = sp.get_metadata("dyn_timestep").get("dyn_timestep")
    linit = sp_v.get_metadata("linit").get("linit")

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
        ddt_w_adv_pc_before=sp_v.ddt_w_adv_pc_before(),
        ddt_vn_apc_pc_before=sp_v.ddt_vn_apc_pc_before(),
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
        ddt_vn_adv=sp_v.ddt_vn_apc_pc_before(),
        ddt_w_adv=sp_v.ddt_w_adv_pc_before(),
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

    interpolation_state = InterpolationState(
        e_bln_c_s=sp_int.e_bln_c_s(),
        rbf_coeff_1=sp_int.rbf_vec_coeff_v1(),
        rbf_coeff_2=sp_int.rbf_vec_coeff_v1(),
        geofac_div=sp_int.geofac_div(),
        geofac_n2s=sp_int.geofac_n2s(),
        geofac_grg=(sp_int.geofac_grg(), sp_int.geofac_grg()),
        nudgecoeff_e=sp_int.nudgecoeff_e(),
        c_lin_e=sp_int.c_lin_e(),
        geofac_grdiv=sp_int.geofac_grdiv(),
        rbf_vec_coeff_e=sp_int.rbf_vec_coeff_e(),
        c_intp=sp_int.c_intp(),
        geofac_rot=sp_int.geofac_rot(),
        pos_on_tplane_e=sp_int.pos_on_tplane_e(),
        e_flx_avg=sp_int.e_flx_avg(),
    )

    metric_state = MetricState(
        mask_hdiff=sp_met.mask_hdiff(),
        theta_ref_mc=sp_met.theta_ref_mc(),
        wgtfac_c=sp_met.wgtfac_c(),
        zd_intcoef=None,  # TODO: @nfarabullini: check if this is some other value in FORTRAN
        zd_vertidx=sp_met.zd_vertidx(),
        zd_diffcoef=sp_met.zd_diffcoef(),
        coeff_gradekin=sp_met.coeff_gradekin(),
        ddqz_z_full_e=sp_met.ddqz_z_full_e(),
        wgtfac_e=sp_met.wgtfac_e(),
        wgtfacq_e=sp_met.wgtfacq_e(),
        ddxn_z_full=sp_met.ddxn_z_full(),
        ddxt_z_full=sp_met.ddxt_z_full(),
        ddqz_z_half=sp_met.ddqz_z_half(),
        coeff1_dwdz=sp_met.coeff1_dwdz(),
        coeff2_dwdz=sp_met.coeff2_dwdz(),
    )

    metric_state_nonhydro = MetricStateNonHydro(
        exner_exfac=sp_met.exner_exfac(),
        exner_ref_mc=sp_met.exner_ref_mc(),
        wgtfacq_c=sp_met.wgtfacq_c(),
        inv_ddqz_z_full=sp_met.inv_ddqz_z_full(),
        rho_ref_mc=sp_met.rho_ref_mc(),
        theta_ref_mc=sp_met.theta_ref_mc(),
        vwind_expl_wgt=sp_met.vwind_expl_wgt(),
        d_exner_dz_ref_ic=sp_met.exner_exfac(),  # TODO @nfarabullini: change back to this sp_met.d_exner_dz_ref_ic()
        ddqz_z_half=sp_met.ddqz_z_half(),
        theta_ref_ic=sp_met.theta_ref_ic(),
        d2dexdz2_fac1_mc=sp_met.d2dexdz2_fac1_mc(),
        d2dexdz2_fac2_mc=sp_met.d2dexdz2_fac2_mc(),
        vwind_impl_wgt=sp_met.vwind_expl_wgt(),  # TODO @nfarabullini: change back to this sp_met.vwind_impl_wgt()
        bdy_halo_c=sp_met.mask_prog_halo_c_dsl_low_refin(),
        ipeidx_dsl=None,  # TODO @nfarabullini: change back to this sp_met.ipeidx_dsl()
        pg_exdist=sp_met.pg_exdist_dsl(),
        hmask_dd3d=sp_met.hmask_dd3d(),
        scalfac_dd3d=sp_met.scalfac_dd3d(),
        rayleigh_w=sp_met.rayleigh_w(),
        rho_ref_me=sp_met.rho_ref_me(),
        theta_ref_me=sp_met.theta_ref_me(),
        zdiff_gradp=sp_met.zdiff_gradp_dsl(),
        mask_prog_halo_c=sp_met.mask_prog_halo_c_dsl_low_refin(),  # TODO @nfarabullini: change back to this sp_met.mask_prog_halo_c()
    )

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
        fac=fac,
        z=z,
    )

    orientation = sp_d.tangent_orientation()
    inverse_primal_edge_lengths = sp_d.inverse_primal_edge_lengths()
    inverse_dual_edge_length = sp_d.inv_dual_edge_length()
    edge_areas = sp_d.edge_areas()
    cell_areas = sp_d.cell_areas()

    solve_nonhydro.run_predictor_step(
        diagnostic_state=diagnostic_state,
        diagnostic_state_nonhydro=diagnostic_state_nonhydro,
        prognostic_state=[prognostic_state, prognostic_state],
        config=config,
        params=nonhydro_params,
        inv_dual_edge_length=inverse_dual_edge_length,
        primal_normal_cell_1=sp_d.primal_normal_cell_x(),
        dual_normal_cell_1=sp_d.dual_normal_cell_x(),
        primal_normal_cell_2=sp_d.primal_normal_cell_y(),
        dual_normal_cell_2=sp_d.dual_normal_cell_y(),
        inv_primal_edge_length=inverse_primal_edge_lengths,
        tangent_orientation=orientation,
        cfl_w_limit=sp_v.cfl_w_limit(),
        scalfac_exdiff=sp_v.scalfac_exdiff(),
        cell_areas=cell_areas,
        owner_mask=sp_d.owner_mask(),
        f_e=sp_d.f_e(),
        area_edge=edge_areas,
        dtime=dtime,
        idyn_timestep=dyn_timestep,
        l_recompute=recompute,
        l_init=linit,
        nnow=nnow,
        nnew=nnew,
    )

    icon_result_exner_new = sp_exit.exner_new()
    icon_result_exner_now = sp_exit.exner_now()
    icon_result_mass_fl_e = sp_exit.mass_fl_e()
    icon_result_prep_adv_vn_traj = sp_exit.prep_adv_vn_traj()
    icon_result_rho_ic = sp_exit.rho_ic()
    icon_result_theta_v_ic = sp_exit.theta_v_ic()
    icon_result_theta_v_new = sp_exit.theta_v_new()
    icon_result_vn_ie = sp_exit.vn_ie()
    icon_result_vn_new = sp_exit.vn_new()
    icon_result_w_concorr_c = sp_exit.w_concorr_c()
    icon_result_w_new = sp_exit.w_new()


@pytest.mark.datatest
@pytest.mark.parametrize(
    "istep, step_date_init, step_date_exit",
    [(1, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000")],
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
    metric_savepoint,
    interpolation_savepoint,
    savepoint_nonhydro_exit,
):
    config = NonHydrostaticConfig()
    sp = savepoint_nonhydro_init
    sp_exit = savepoint_nonhydro_exit
    sp_dif = diffusion_savepoint_init
    sp_int = interpolation_savepoint
    sp_met = metric_savepoint
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
        ddt_w_adv_pc_before=sp_v.ddt_w_adv_pc_before(),
        ddt_vn_apc_pc_before=sp_v.ddt_vn_apc_pc_before(),
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
        ddt_vn_adv=sp_v.ddt_vn_apc_pc_before(),
        ddt_w_adv=sp_v.ddt_w_adv_pc_before(),
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

    interpolation_state = InterpolationState(
        e_bln_c_s=sp_int.e_bln_c_s(),
        rbf_coeff_1=sp_int.rbf_vec_coeff_v1(),
        rbf_coeff_2=sp_int.rbf_vec_coeff_v1(),
        geofac_div=sp_int.geofac_div(),
        geofac_n2s=sp_int.geofac_n2s(),
        geofac_grg=(sp_int.geofac_grg(), sp_int.geofac_grg()),
        nudgecoeff_e=sp_int.nudgecoeff_e(),
        c_lin_e=sp_int.c_lin_e(),
        geofac_grdiv=sp_int.geofac_grdiv(),
        rbf_vec_coeff_e=sp_int.rbf_vec_coeff_e(),
        c_intp=sp_int.c_intp(),
        geofac_rot=sp_int.geofac_rot(),
        pos_on_tplane_e=sp_int.pos_on_tplane_e(),
        e_flx_avg=sp_int.e_flx_avg(),
    )

    metric_state = MetricState(
        mask_hdiff=sp_met.mask_hdiff(),
        theta_ref_mc=sp_met.theta_ref_mc(),
        wgtfac_c=sp_met.wgtfac_c(),
        zd_intcoef=None,  # TODO: @nfarabullini: check if this is some other value in FORTRAN
        zd_vertidx=sp_met.zd_vertidx(),
        zd_diffcoef=sp_met.zd_diffcoef(),
        coeff_gradekin=sp_met.coeff_gradekin(),
        ddqz_z_full_e=sp_met.ddqz_z_full_e(),
        wgtfac_e=sp_met.wgtfac_e(),
        wgtfacq_e=sp_met.wgtfacq_e(),
        ddxn_z_full=sp_met.ddxn_z_full(),
        ddxt_z_full=sp_met.ddxt_z_full(),
        ddqz_z_half=sp_met.ddqz_z_half(),
        coeff1_dwdz=sp_met.coeff1_dwdz(),
        coeff2_dwdz=sp_met.coeff2_dwdz(),
    )

    metric_state_nonhydro = MetricStateNonHydro(
        exner_exfac=sp_met.exner_exfac(),
        exner_ref_mc=sp_met.exner_ref_mc(),
        wgtfacq_c=sp_met.wgtfacq_c(),
        inv_ddqz_z_full=sp_met.inv_ddqz_z_full(),
        rho_ref_mc=sp_met.rho_ref_mc(),
        theta_ref_mc=sp_met.theta_ref_mc(),
        vwind_expl_wgt=sp_met.vwind_expl_wgt(),
        d_exner_dz_ref_ic=sp_met.exner_exfac(),  # TODO @nfarabullini: change back to this sp_met.d_exner_dz_ref_ic()
        ddqz_z_half=sp_met.ddqz_z_half(),
        theta_ref_ic=sp_met.theta_ref_ic(),
        d2dexdz2_fac1_mc=sp_met.d2dexdz2_fac1_mc(),
        d2dexdz2_fac2_mc=sp_met.d2dexdz2_fac2_mc(),
        vwind_impl_wgt=sp_met.vwind_expl_wgt(),  # TODO @nfarabullini: change back to this sp_met.vwind_impl_wgt()
        bdy_halo_c=sp_met.mask_prog_halo_c_dsl_low_refin(),
        ipeidx_dsl=None,  # TODO @nfarabullini: change back to this sp_met.ipeidx_dsl()
        pg_exdist=sp_met.pg_exdist_dsl(),
        hmask_dd3d=sp_met.hmask_dd3d(),
        scalfac_dd3d=sp_met.scalfac_dd3d(),
        rayleigh_w=sp_met.rayleigh_w(),
        rho_ref_me=sp_met.rho_ref_me(),
        theta_ref_me=sp_met.theta_ref_me(),
        zdiff_gradp=sp_met.zdiff_gradp_dsl(),
        mask_prog_halo_c=sp_met.mask_prog_halo_c_dsl_low_refin(),
        # TODO @nfarabullini: change back to this sp_met.mask_prog_halo_c()
    )

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
        fac=fac,
        z=z,
    )

    orientation = sp_d.tangent_orientation()
    inverse_primal_edge_lengths = sp_d.inverse_primal_edge_lengths()
    inverse_dual_edge_length = sp_d.inv_dual_edge_length()
    edge_areas = sp_d.edge_areas()
    cell_areas = sp_d.cell_areas()

    solve_nonhydro.run_corrector_step(
        diagnostic_state=diagnostic_state,
        diagnostic_state_nonhydro=diagnostic_state_nonhydro,
        prognostic_state=[prognostic_state, prognostic_state],
        config=config,
        params=nonhydro_params,
        inv_dual_edge_length=inverse_dual_edge_length,
        inv_primal_edge_length=inverse_primal_edge_lengths,
        tangent_orientation=orientation,
        prep_adv=prep_adv,
        dtime=dtime,
        nnew=nnew,
        nnow=nnow,
        cfl_w_limit=sp_v.cfl_w_limit(),
        scalfac_exdiff=sp_v.scalfac_exdiff(),
        cell_areas=cell_areas,
        owner_mask=sp_d.owner_mask(),
        f_e=sp_d.f_e(),
        area_edge=edge_areas,
        lclean_mflx=clean_mflx,
        scal_divdamp_o2=3.0,  # TODO: @nfarabullini: change this to input data entry
        bdy_divdamp=a_vec,  # TODO: @nfarabullini: change this to input data entry
        r_nsubsteps=float(r_nsubsteps),
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
        np.asarray(icon_result_prep_adv_mass_flx_me), np.asarray(prep_adv.mass_flx_me)
    )


@pytest.mark.datatest
def test_run_solve_nonhydro_multi_step(
    icon_grid,
    savepoint_nonhydro_init,
    data_provider,
    step_date_init,
    damping_height,
    grid_savepoint,
    savepoint_velocity_init,
    diffusion_savepoint_init,
    metric_savepoint,
    interpolation_savepoint,
):
    config = NonHydrostaticConfig()
    sp = savepoint_nonhydro_init
    sp_dif = diffusion_savepoint_init
    sp_int = interpolation_savepoint
    sp_met = metric_savepoint
    nonhydro_params = NonHydrostaticParams(config)
    vertical_params = VerticalModelParams(
        vct_a=grid_savepoint.vct_a(), rayleigh_damping_height=damping_height
    )
    sp_d = data_provider.from_savepoint_grid()
    sp_v = savepoint_velocity_init
    ntl1 = sp.get_metadata("ntl1").get("ntl1")
    ntl2 = sp.get_metadata("ntl2").get("ntl2")
    mesh = SimpleMesh()
    dtime = sp_v.get_metadata("dtime").get("dtime")
    clean_mflx = sp_v.get_metadata("clean_mflx").get("clean_mflx")
    r_nsubsteps = sp_d.get_metadata("nsteps").get("nsteps")
    prep_adv = PrepAdvection(vn_traj=sp.vn_traj(), mass_flx_me=sp.mass_flx_me())
    recompute = sp_v.get_metadata("recompute").get("recompute")
    dyn_timestep = sp.get_metadata("dyn_timestep").get("dyn_timestep")
    linit = sp_v.get_metadata("linit").get("linit")

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
        ddt_w_adv_pc_before=None,
        ddt_vn_apc_pc_before=sp_v.ddt_w_adv_pc_before(),
        ntnd=None,
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
        ddt_vn_adv=sp_v.ddt_w_adv_pc_before(),
        ntl1=ntl1,
        ntl2=ntl2,
        rho_incr=None,  # TODO @nfarabullini: change back to this sp.rho_incr()
        vn_incr=None,  # TODO @nfarabullini: change back to this sp.vn_incr()
        exner_incr=None,  # TODO @nfarabullini: change back to this sp.exner_incr()
    )

    prognostic_state = PrognosticState(
        w=sp_v.w(),
        vn=None,
        exner_pressure=None,
        theta_v=sp_dif.theta_v(),
        rho=sp_dif.rho(),
        exner=sp_dif.exner(),
    )

    interpolation_state = InterpolationState(
        e_bln_c_s=sp_int.e_bln_c_s(),
        rbf_coeff_1=None,
        rbf_coeff_2=None,
        geofac_div=sp_int.geofac_div(),
        geofac_n2s=None,
        geofac_grg_x=sp_int.geofac_grg_x(),
        geofac_grg_y=sp_int.geofac_grg_y(),
        nudgecoeff_e=sp_int.nudgecoeff_e(),
        c_lin_e=sp_int.c_lin_e(),
        geofac_grdiv=sp_int.geofac_grdiv(),
        rbf_vec_coeff_e=sp_int.rbf_vec_coeff_e(),
        c_intp=sp_int.c_intp(),
        geofac_rot=None,
        pos_on_tplane_e=sp_int.pos_on_tplane_e(),
        e_flx_avg=sp_int.e_flx_avg(),
    )

    metric_state = MetricState(
        theta_ref_mc=None,
        wgtfac_c=None,
        zd_intcoef=None,
        zd_vertidx=None,
        zd_diffcoef=None,
        coeff_gradekin=None,
        ddqz_z_full_e=None,
        wgtfac_e=None,
        wgtfacq_e=None,
        ddxn_z_full=None,
        ddxt_z_full=sp_met.ddxt_z_full(),
        ddqz_z_half=None,
        coeff1_dwdz=None,
        coeff2_dwdz=None,
    )

    metric_state_nonhydro = MetricStateNonHydro(
        exner_exfac=sp_met.exner_exfac(),
        exner_ref_mc=sp_met.exner_ref_mc(),
        wgtfacq_c=sp_met.wgtfacq_c(),
        inv_ddqz_z_full=sp_met.inv_ddqz_z_full(),
        rho_ref_mc=sp_met.rho_ref_mc(),
        theta_ref_mc=sp_met.theta_ref_mc(),
        vwind_expl_wgt=sp_met.vwind_expl_wgt(),
        d_exner_dz_ref_ic=None,  # TODO @nfarabullini: change back to this sp_met.d_exner_dz_ref_ic()
        ddqz_z_half=sp_met.ddqz_z_half(),
        theta_ref_ic=sp_met.theta_ref_ic(),
        d2dexdz2_fac1_mc=sp_met.d2dexdz2_fac1_mc(),
        d2dexdz2_fac2_mc=sp_met.d2dexdz2_fac2_mc(),
        vwind_impl_wgt=sp_met.vwind_expl_wgt(),  # TODO @nfarabullini: change back to this sp_met.vwind_impl_wgt()
        bdy_halo_c=sp_met.mask_prog_halo_c_dsl_low_refin(),
        ipeidx_dsl=None,  # TODO @nfarabullini: change back to this sp_met.ipeidx_dsl()
        pg_exdist=sp_met.pg_exdist_dsl(),
        hmask_dd3d=sp_met.hmask_dd3d(),
        scalfac_dd3d=sp_met.scalfac_dd3d(),
        rayleigh_w=sp_met.rayleigh_w(),
        rho_ref_me=sp_met.rho_ref_me(),
        theta_ref_me=sp_met.theta_ref_me(),
        zdiff_gradp=sp_met.zdiff_gradp_dsl(),
        mask_prog_halo_c=sp_met.mask_prog_halo_c_dsl_low_refin(),
        # TODO @nfarabullini: change back to this sp_met.mask_prog_halo_c()
        mask_hdiff=sp_met.mask_hdiff(),
    )

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
        fac=fac,
        z=z,
    )
    for _ in range(4):
        solve_nonhydro.time_step(
            diagnostic_state=diagnostic_state,
            diagnostic_state_nonhydro=diagnostic_state_nonhydro,
            prognostic_state=[prognostic_state, prognostic_state],
            prep_adv=prep_adv,
            config=config,
            params=vertical_params,
            inv_dual_edge_length=sp_d.inv_dual_edge_length(),
            primal_normal_cell_1=None,  # TODO @nfarabullini: change back to this sp_d.primal_normal_cell_1()
            dual_normal_cell_1=None,  # TODO @nfarabullini: change back to this sp_d.dual_normal_cell_1()
            primal_normal_cell_2=None,  # TODO @nfarabullini: change back to this sp_d.primal_normal_cell_2()
            dual_normal_cell_2=None,  # TODO @nfarabullini: change back to this sp_d.dual_normal_cell_2()
            inv_primal_edge_length=sp_d.inverse_primal_edge_lengths(),
            tangent_orientation=sp_d.tangent_orientation(),
            cfl_w_limit=sp_v.cfl_w_limit(),
            scalfac_exdiff=sp_v.scalfac_exdiff(),
            cell_areas=sp_d.cell_areas(),
            owner_mask=sp_d.owner_mask(),
            f_e=sp_d.f_e(),
            area_edge=sp_d.edge_areas(),
            dtime=dtime,
            idyn_timestep=dyn_timestep,
            l_recompute=recompute,
            l_init=linit,
            nnew=nnew,
            nnow=nnow,
            lclean_mflx=clean_mflx,
        )
