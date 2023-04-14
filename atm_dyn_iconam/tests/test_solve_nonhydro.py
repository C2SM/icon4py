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

from icon4py.nh_solve.solve_nonydro import NonHydrostaticParams, SolveNonhydro
from icon4py.state_utils.diagnostic_state import (
    DiagnosticState,
    DiagnosticStateNonHydro,
)
from icon4py.state_utils.icon_grid import VerticalModelParams
from icon4py.state_utils.interpolation_state import InterpolationState
from icon4py.state_utils.metric_state import MetricStateNonHydro
from icon4py.state_utils.prep_adv_state import PrepAdvection
from icon4py.state_utils.prognostic_state import PrognosticState
from icon4py.state_utils.z_fields import ZFields


@pytest.mark.datatest
def test_nonhydro_params():
    nonhydro_params = NonHydrostaticParams(config)

    assert nonhydro_params.df32 == pytest.approx(nonhydro_params.divdamp_fac3 - nonhydro_params.divdamp_fac2, abs=1e-12)
    assert nonhydro_params.dz32 == pytest.approxnonhydro_params.divdamp_z3 - nonhydro_params.divdamp_z2, abs=1e-12)
    assert nonhydro_params.df42 == pytest.approxnonhydro_params.divdamp_fac4 - nonhydro_params.divdamp_fac2, abs=1e-12)
    assert nonhydro_params.dz42 == pytest.approxnonhydro_params.divdamp_z4 - nonhydro_params.divdamp_z2, abs=1e-12)

    assert nonhydro_params.bqdr == pytest.approx(df42 * dz32 - df32 * dz42) / (dz32 * dz42 * (dz42 - dz32)), abs=1e-12)
    assert nonhydro_params.aqdr == pytest.approxdf32 / dz32 - bqdr * dz32, abs=1e-12)


@pytest.mark.datatest
@pytest.mark.parametrize(
    "istep, step_date_init, step_date_exit",
    [(1, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000")]
)
def test_nonhydro_predictor_step(
    icon_grid,
):
    nonhydro_params = NonHydrostaticParams(config)
    vertical_params = VerticalModelParams()
    sp_d = data_provider.from_savepoint_grid()
    dtime = sp.get_metadata("dtime").get("dtime")
    ntl1 = sp.get_metadata("ntl1").get("ntl1")
    ntl2 = sp.get_metadata("ntl2").get("ntl2")
    cfl_w_limit = sp.cfl_w_limit()
    scalfac_exdiff = sp.scalfac_exdiff()


    diagnostic_state = DiagnosticState(
        hdef_ic=None,
        div_ic=None,
        dwdx=None,
        dwdy=None,
        vt=sp.vt(),
        vn_ie=sp.vn_ie(),
        w_concorr_c=sp.w_concorr_c(),
        ddt_w_adv_pc_before=None,
        ddt_vn_apc_pc_before=sp.ddt_vn_adv_pc_before(),
        ntnd=None
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
        ddt_vn_adv=sp.ddt_vn_adv(),
        ntl1=ntl1,
        ntl2=ntl2,
        rho_incr=sp.rho_incr(),
        vn_incr=sp.vn_incr(),
        exner_incr=sp.exner_incr()
    )
    prognostic_state = PrognosticState(
        w=sp.w(),
        vn=None,
        exner_pressure=None,
        theta_v=sp.theta_v(),
        rho=sp.rho(),
        exner=sp.exner()
    )

    interpolation_state = InterpolationState(
        e_bln_c_s=savepoint.e_bln_c_s(),
        rbf_coeff_1=None,
        rbf_coeff_2=None,
        geofac_div=savepoint.geofac_div(),
        geofac_n2s=None,
        geofac_grg_x=savepoint.geofac_grg_x(),
        geofac_grg_y=savepoint.geofac_grg_y(),
        nudgecoeff_e=savepoint.nudgecoeff_e(),
        c_lin_e=savepoint.c_lin_e(),
        geofac_grdiv=savepoint.geofac_grdiv(),
        rbf_vec_coeff_e=savepoint.rbf_vec_coeff_e(),
        c_intp=savepoint.c_intp(),
        geofac_rot=None,
        pos_on_tplane_e=savepoint.pos_on_tplane_e(),
        e_flx_avg=savepoint.e_flx_avg()
    )

    metric_state = MetricState(
        mask_hdiff=None,
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
        ddxt_z_full=sp.ddxt_z_full(),
        ddqz_z_half=None,
        coeff1_dwdz=None,
        coeff2_dwdz=None,
    )

    metric_state_nonhydro = MetricStateNonHydro(
        exner_exfac=savepoint.exner_exfac,
        exner_ref_mc=savepoint.exner_ref_mc,
        wgtfacq_c=savepoint.wgtfacq_c,
        inv_ddqz_z_full=savepoint.inv_ddqz_z_full,
        rho_ref_mc=savepoint.rho_ref_mc,
        theta_ref_mc=savepoint.theta_ref_mc,
        vwind_expl_wgt=savepoint.vwind_expl_wgt,
        d_exner_dz_ref_ic=savepoint.d_exner_dz_ref_ic,
        ddqz_z_half=savepoint.ddqz_z_half,
        theta_ref_ic=savepoint.theta_ref_ic,
        d2dexdz2_fac1_mc=savepoint.d2dexdz2_fac1_mc,
        d2dexdz2_fac2_mc=savepoint.d2dexdz2_fac2_mc,
        vwind_impl_wgt=savepoint.vwind_impl_wgt,
        bdy_halo_c=savepoint.bdy_halo_c,
        ipeidx_dsl=savepoint.ipeidx_dsl,
        pg_exdist=savepoint.pg_exdist,
        hmask_dd3d=savepoint.hmask_dd3d,
        scalfac_dd3d=savepoint.scalfac_dd3d,
        rayleigh_w=savepoint.rayleigh_w,
        rho_ref_me=savepoint.rho_ref_me,
        theta_ref_me=savepoint.theta_ref_me,
        zdiff_gradp=savepoint.zdiff_gradp,
        mask_prog_halo_c=savepoint.mask_prog_halo_c,
        mask_hdiff=savepoint.mask_hdiff
    )

    z_fields = ZFields(
        z_w_concorr_me=sp.z_w_concorr_me(),
        z_kin_hor_e=sp.z_kin_hor_e(),
        z_vt_ie=sp.z_vt_ie(),
    )

    solve_nonhydro = SolveNonhydro()
    solve_nonhydro.init(
        grid=icon_grid,
        config=config,
        params=nonhydro_params,
        metric_state=metric_state,
        metric_state_nonhydro=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params = vertical_params
    )

    orientation = sp_d.tangent_orientation()
    inverse_primal_edge_lengths = sp_d.inverse_primal_edge_lengths()
    inverse_dual_edge_length = sp_d.inv_dual_edge_length()
    edge_areas = sp_d.edge_areas()
    cell_areas = sp_d.cell_areas()

    solve_nonhydro.run_predictor_step(
        diagnostic_state=diagnostic_state,
        diagnostic_state_nonhydro=diagnostic_state_nonhydro,
        prognostic_state=prognostic_state,
        config=config,
        z_fields=z_fields,
        inv_dual_edge_length=inverse_dual_edge_length,
        primal_normal_cell_1=primal_normal_cell_1,
        dual_normal_cell_1=dual_normal_cell_1,
        primal_normal_cell_2=primal_normal_cell_2,
        dual_normal_cell_2=dual_normal_cell_2,
        inv_primal_edge_length=inverse_primal_edge_lengths,
        tangent_orientation=orientation,
        cfl_w_limit=cfl_w_limit,
        scalfac_exdiff=scalfac_exdiff,
        cell_areas=cell_areas,
        owner_mask=sp_d.owner_mask(),
        f_e=sp_d.f_e(),
        area_edge=area_edge,
        dtime=dtime,
        idyn_timestep=idyn_timestep,
        l_recompute=l_recompute,
        l_init=l_init,
        nnow=nnow,
        nnew=nnew
    )

@pytest.mark.datatest
@pytest.mark.parametrize(
    "istep, step_date_init, step_date_exit",
    [(2, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000")]
)
def test_nonhydro_corrector_step(
    icon_grid,
):
    nonhydro_params = NonHydrostaticParams(config)
    vertical_params = VerticalModelParams()
    sp_d = data_provider.from_savepoint_grid()
    dtime = sp.get_metadata("dtime").get("dtime")
    ntl1 = sp.get_metadata("ntl1").get("ntl1")
    ntl2 = sp.get_metadata("ntl2").get("ntl2")
    cfl_w_limit = sp.cfl_w_limit()
    scalfac_exdiff = sp.scalfac_exdiff()

    prep_adv=PrepAdvection(
        vn_traj=sp.vn_traj(),
        mass_flx_me=sp.mass_flx_me()
    )

    diagnostic_state = DiagnosticState(
        hdef_ic=None,
        div_ic=None,
        dwdx=None,
        dwdy=None,
        vt=sp.vt(),
        vn_ie=sp.vn_ie(),
        w_concorr_c=sp.w_concorr_c(),
        ddt_w_adv_pc_before=None,
        ddt_vn_apc_pc_before=sp.ddt_vn_adv_pc_before(),
        ntnd=None
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
        ddt_vn_adv=sp.ddt_vn_adv(),
        ntl1=ntl1,
        ntl2=ntl2,
        rho_incr=sp.rho_incr(),
        vn_incr=sp.vn_incr(),
        exner_incr=sp.exner_incr()
    )
    prognostic_state = PrognosticState(
        w=sp.w(),
        vn=None,
        exner_pressure=None,
        theta_v=sp.theta_v(),
        rho=sp.rho(),
        exner=sp.exner()
    )

    interpolation_state = InterpolationState(
        e_bln_c_s=savepoint.e_bln_c_s(),
        rbf_coeff_1=None,
        rbf_coeff_2=None,
        geofac_div=savepoint.geofac_div(),
        geofac_n2s=None,
        geofac_grg_x=savepoint.geofac_grg_x(),
        geofac_grg_y=savepoint.geofac_grg_y(),
        nudgecoeff_e=savepoint.nudgecoeff_e(),
        c_lin_e=savepoint.c_lin_e(),
        geofac_grdiv=savepoint.geofac_grdiv(),
        rbf_vec_coeff_e=savepoint.rbf_vec_coeff_e(),
        c_intp=savepoint.c_intp(),
        geofac_rot=None,
        pos_on_tplane_e=savepoint.pos_on_tplane_e(),
        e_flx_avg=savepoint.e_flx_avg()
    )

    metric_state = MetricState(
        mask_hdiff=None,
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
        ddxt_z_full=sp.ddxt_z_full(),
        ddqz_z_half=None,
        coeff1_dwdz=None,
        coeff2_dwdz=None,
    )

    metric_state_nonhydro = MetricStateNonHydro(
        exner_exfac=savepoint.exner_exfac,
        exner_ref_mc=savepoint.exner_ref_mc,
        wgtfacq_c=savepoint.wgtfacq_c,
        inv_ddqz_z_full=savepoint.inv_ddqz_z_full,
        rho_ref_mc=savepoint.rho_ref_mc,
        theta_ref_mc=savepoint.theta_ref_mc,
        vwind_expl_wgt=savepoint.vwind_expl_wgt,
        d_exner_dz_ref_ic=savepoint.d_exner_dz_ref_ic,
        ddqz_z_half=savepoint.ddqz_z_half,
        theta_ref_ic=savepoint.theta_ref_ic,
        d2dexdz2_fac1_mc=savepoint.d2dexdz2_fac1_mc,
        d2dexdz2_fac2_mc=savepoint.d2dexdz2_fac2_mc,
        vwind_impl_wgt=savepoint.vwind_impl_wgt,
        bdy_halo_c=savepoint.bdy_halo_c,
        ipeidx_dsl=savepoint.ipeidx_dsl,
        pg_exdist=savepoint.pg_exdist,
        hmask_dd3d=savepoint.hmask_dd3d,
        scalfac_dd3d=savepoint.scalfac_dd3d,
        rayleigh_w=savepoint.rayleigh_w,
        rho_ref_me=savepoint.rho_ref_me,
        theta_ref_me=savepoint.theta_ref_me,
        zdiff_gradp=savepoint.zdiff_gradp,
        mask_prog_halo_c=savepoint.mask_prog_halo_c,
        mask_hdiff=savepoint.mask_hdiff
    )

    z_fields = ZFields(
        z_w_concorr_me=sp.z_w_concorr_me(),
        z_kin_hor_e=sp.z_kin_hor_e(),
        z_vt_ie=sp.z_vt_ie(),
    )

    solve_nonhydro = SolveNonhydro()
    solve_nonhydro.init(
        grid=icon_grid,
        config=config,
        params=nonhydro_params,
        metric_state=metric_state,
        metric_state_nonhydro=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params
    )

    orientation = sp_d.tangent_orientation()
    inverse_primal_edge_lengths = sp_d.inverse_primal_edge_lengths()
    inverse_dual_edge_length = sp_d.inv_dual_edge_length()
    edge_areas = sp_d.edge_areas()
    cell_areas = sp_d.cell_areas()

    solve_nonhydro.run_corrector_step(
        diagnostic_state=diagnostic_state,
        diagnostic_state_nonhydro=diagnostic_state_nonhydro,
        prognostic_state=prognostic_state,
        config=config,
        z_fields=z_fields,
        inv_dual_edge_length=inverse_dual_edge_length,
        inv_primal_edge_length=inverse_primal_edge_lengths,
        tangent_orientation=orientation,
        prep_adv=prep_adv,
        dtime=dtime,
        nnew=nnew,
        nnow=nnow,
        cfl_w_limit=cfl_w_limit,
        scalfac_exdiff=scalfac_exdiff,
        cell_areas=cell_areas,
        owner_mask=sp_d.owner_mask(),
        f_e=sp_d.f_e(),
        area_edge=edge_areas,
        lprep_adv=lprep_adv,
        lclean_mflx=lprep_adv
    )

