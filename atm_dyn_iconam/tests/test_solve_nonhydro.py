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

from icon4py.common.dimension import KDim, EdgeDim, CellDim
from icon4py.grid.horizontal import CellParams, EdgeParams
from icon4py.grid.vertical import VerticalModelParams
from icon4py.nh_solve.solve_nonydro import (
    NonHydrostaticConfig,
    NonHydrostaticParams,
    SolveNonhydro,
)
from icon4py.state_utils.diagnostic_state import DiagnosticStateNonHydro
from icon4py.state_utils.metric_state import MetricStateNonHydro
from icon4py.state_utils.nh_constants import NHConstants
from icon4py.state_utils.prep_adv_state import PrepAdvection
from icon4py.state_utils.prognostic_state import PrognosticState
from icon4py.state_utils.z_fields import ZFields

from .test_utils.helpers import dallclose, random_field, zero_field
from icon4py.state_utils.utils import _allocate
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
    savepoint_velocity_exit,
    metrics_savepoint,
    metrics_nonhydro_savepoint,
    interpolation_savepoint,
    savepoint_nonhydro_exit,
):
    config = NonHydrostaticConfig()
    sp = savepoint_nonhydro_init
    sp_exit = savepoint_nonhydro_exit
    sp_v_exit = savepoint_velocity_exit
    sp_dif = diffusion_savepoint_init
    sp_met = metrics_savepoint
    nonhydro_params = NonHydrostaticParams(config)
    vertical_params = VerticalModelParams(
        vct_a=grid_savepoint.vct_a(),
        rayleigh_damping_height=damping_height,
        nflat_gradp=grid_savepoint.nflat_gradp(),
        nflatlev=grid_savepoint.nflatlev(),
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
    prep_adv = PrepAdvection(vn_traj=sp.vn_traj(), mass_flx_me=sp.mass_flx_me(), mass_flx_ic=sp.mass_flx_ic())

    enh_smag_fac = zero_field(mesh, KDim)
    a_vec = random_field(mesh, KDim, low=1.0, high=10.0, extend={KDim: 1})
    fac = (0.67, 0.5, 1.3, 0.8)
    z = (0.1, 0.2, 0.3, 0.4)
    ntnd = sp_v.get_metadata("ntnd").get("ntnd")
    nnow = 0
    nnew = 1

    diagnostic_state_nh = DiagnosticStateNonHydro(
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
        ddt_vn_adv_ntl1=sp_v_exit.ddt_vn_apc_pc(1),
        ddt_vn_adv_ntl2=sp_v_exit.ddt_vn_apc_pc(2),
        ddt_w_adv_ntl1=sp_v_exit.ddt_w_adv_pc(1),
        ddt_w_adv_ntl2=sp_v_exit.ddt_w_adv_pc(2), # TODO: @abishekg7 change later
        ntl1=ntl1,
        ntl2=ntl2,
        vt=sp_v_exit.vt(), #sp_v.vt(), #TODO: @abishekg7 change back to sp_v
        vn_ie=sp_v_exit.vn_ie(),
        w_concorr_c=sp_v.w_concorr_c(),
        ddt_w_adv_pc=sp_v.ddt_w_adv_pc_before(ntnd),
        ddt_vn_apc_pc=sp_v.ddt_vn_apc_pc_before(ntnd),
        ntnd=ntnd,
        rho_incr=None,  # sp.rho_incr(),
        vn_incr=None,  # sp.vn_incr(),
        exner_incr=None,  # sp.exner_incr(),
    )

    prognostic_state_nnow = PrognosticState(
        w=sp_v_exit.w(),  # sp_v.w(), #TODO: @abishekg7 change back
        vn=sp_v_exit.vn(),  # sp_v.vn(),
        exner_pressure=None,
        theta_v=sp.theta_v_now(),
        rho=sp.rho_now(),
        exner=sp.exner_now(),
    )

    prognostic_state_nnew = PrognosticState(
        w=sp.w_new(),
        vn=sp.vn_new(),
        exner_pressure=None,
        theta_v=sp.theta_v_new(),
        rho=sp.rho_new(),
        exner=sp.exner_new(),
    )

    z_fields = ZFields(
        z_gradh_exner = _allocate(EdgeDim, KDim, mesh=icon_grid),
        z_alpha = _allocate(CellDim, KDim, is_halfdim=True, mesh=icon_grid),
        z_beta = _allocate(CellDim, KDim, mesh=icon_grid),
        z_w_expl = _allocate(CellDim, KDim, is_halfdim=True, mesh=icon_grid),
        z_exner_expl = _allocate(CellDim, KDim, mesh=icon_grid),
        z_q = _allocate(CellDim, KDim, mesh=icon_grid),
        z_contr_w_fl_l = _allocate(CellDim, KDim, is_halfdim=True, mesh=icon_grid),
        z_rho_e = _allocate(EdgeDim, KDim, mesh=icon_grid),
        z_theta_v_e = _allocate(EdgeDim, KDim, mesh=icon_grid),
        z_graddiv_vn = _allocate(EdgeDim, KDim, mesh=icon_grid),
        z_rho_expl = _allocate(CellDim, KDim, mesh=icon_grid),
        z_dwdz_dd = _allocate(CellDim, KDim, mesh=icon_grid),
    )

    nh_constants = NHConstants(
        wgt_nnow_rth=sp.wgt_nnow_rth(),
        wgt_nnew_rth=sp.wgt_nnew_rth(),
        wgt_nnow_vel=sp.wgt_nnow_vel(),
        wgt_nnew_vel=sp.wgt_nnew_vel(),
        scal_divdamp=sp.scal_divdamp(),
        scal_divdamp_o2=sp.scal_divdamp_o2(),
    )

    interpolation_state = interpolation_savepoint.construct_interpolation_state()
    metric_state_nonhydro = metrics_nonhydro_savepoint.construct_nh_metric_state(
        icon_grid.n_lev()
    )

    cell_geometry: CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: EdgeParams = grid_savepoint.construct_edge_geometry()

    solve_nonhydro = SolveNonhydro()
    solve_nonhydro.init(
        grid=icon_grid,
        config=config,
        params=nonhydro_params,
        metric_state_nonhydro=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        a_vec=a_vec,
        enh_smag_fac=enh_smag_fac,
        cell_areas=cell_geometry.area,
        fac=fac,
        z=z,
    )

    prognostic_state_ls = [prognostic_state_nnow, prognostic_state_nnew]
    solve_nonhydro.run_predictor_step(
        diagnostic_state_nh=diagnostic_state_nh,
        prognostic_state=prognostic_state_ls,
        config=config,
        params=nonhydro_params,
        edge_geometry=edge_geometry,
        z_fields=z_fields,
        nh_constants=nh_constants,
        # inv_dual_edge_length=edge_geometry.inverse_dual_edge_lengths,
        # primal_normal_cell=edge_geometry.primal_normal_cell,
        # dual_normal_cell=edge_geometry.dual_normal_cell,
        # inv_primal_edge_length=edge_geometry.inverse_primal_edge_lengths,
        # tangent_orientation=edge_geometry.tangent_orientation,
        cfl_w_limit=sp_v.cfl_w_limit(),
        scalfac_exdiff=sp_v.scalfac_exdiff(),
        cell_areas=cell_geometry.area,
        owner_mask=sp_d.c_owner_mask(),
        f_e=sp_d.f_e(),
        area_edge=edge_geometry.edge_areas,
        z_rho_e2=sp_exit.z_rho_e_01(),
        z_theta_v_e2=sp_exit.z_theta_v_e_01(),
        vn_tmp=sp_v_exit.vn(),
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

    icon_result_prep_adv_mass_flx_me = sp_exit.mass_flx_me()
    icon_result_prep_adv_vn_traj = sp_exit.vn_traj()

    # stencils 2, 3
    assert dallclose(
        np.asarray(sp_exit.exner_pr())[1688:20896, :],
        np.asarray(diagnostic_state_nh.exner_pr)[1688:20896, :],
    )
    assert dallclose(
        np.asarray(sp_exit.z_exner_ex_pr())[1688:20896, :],
        np.asarray(solve_nonhydro.z_exner_ex_pr)[1688:20896, :],
    )

    # stencils 4,5
    assert dallclose(
        np.asarray(sp_exit.z_exner_ic())[1688:20896, 64],
        np.asarray(solve_nonhydro.z_exner_ic)[1688:20896, 64],
    )
    assert dallclose(
        np.asarray(sp_exit.z_exner_ic())[1688:20896, 4:64],
        np.asarray(solve_nonhydro.z_exner_ic)[1688:20896, 4:64],
        rtol=1.0e-9,
    )
    # stencil 6
    assert dallclose(
        np.asarray(sp_exit.z_dexner_dz_c(1))[1688:20896, :],
        np.asarray(solve_nonhydro.z_dexner_dz_c_1)[1688:20896, :],
        atol=5e-18,
    )

    # stencils 7,8,9
    assert dallclose(
        np.asarray(icon_result_rho_ic)[1688:20896, :],
        np.asarray(diagnostic_state_nh.rho_ic)[1688:20896, :],
    )
    assert dallclose(
        np.asarray(sp_exit.z_th_ddz_exner_c())[1688:20896, 1:],
        np.asarray(solve_nonhydro.z_th_ddz_exner_c)[1688:20896, 1:],
    )

    # stencils 7,8,9, 11
    assert dallclose(
        np.asarray(sp_exit.z_theta_v_pr_ic())[1688:20896, :],
        np.asarray(solve_nonhydro.z_theta_v_pr_ic)[1688:20896, :],
    )
    assert dallclose(
        np.asarray(sp_exit.theta_v_ic())[1688:20896, :],
        np.asarray(diagnostic_state_nh.theta_v_ic)[1688:20896, :],
    )
    # stencils 7,8,9, 13
    assert dallclose(
        np.asarray(sp_exit.z_rth_pr(1))[1688:20896, :],
        np.asarray(solve_nonhydro.z_rth_pr_1)[1688:20896, :],
    )
    assert dallclose(
        np.asarray(sp_exit.z_rth_pr(2))[1688:20896, :],
        np.asarray(solve_nonhydro.z_rth_pr_2)[1688:20896, :],
    )

    # stencils 12
    assert dallclose(
        np.asarray(sp_exit.z_dexner_dz_c(2))[1688:20896, :],
        np.asarray(solve_nonhydro.z_dexner_dz_c_2)[1688:20896, :],
        atol=1e-22,
    )

    # grad_green_gauss_cell_dsl
    assert dallclose(
        np.asarray(sp_exit.z_grad_rth(1))[1688:20896, :],
        np.asarray(solve_nonhydro.z_grad_rth_1)[1688:20896, :],
        rtol=1e-6,
    )
    assert dallclose(
        np.asarray(sp_exit.z_grad_rth(2))[1688:20896, :],
        np.asarray(solve_nonhydro.z_grad_rth_2)[1688:20896, :],
        rtol=1e-6,
    )
    assert dallclose(
        np.asarray(sp_exit.z_grad_rth(3))[1688:20896, :],
        np.asarray(solve_nonhydro.z_grad_rth_3)[1688:20896, :],
        rtol=5e-6,
    )
    assert dallclose(
        np.asarray(sp_exit.z_grad_rth(4))[1688:20896, :],
        np.asarray(solve_nonhydro.z_grad_rth_4)[1688:20896, :],
        rtol=1e-5,
    )

    # assert dallclose(
    #     np.asarray(sp_exit.z_rho_e())[3777:31558, :], np.asarray(solve_nonhydro.z_rho_e)[3777:31558, :]
    # )
    # assert dallclose(
    #     np.asarray(sp_exit.z_theta_v_e())[3777:31558, :], np.asarray(solve_nonhydro.z_theta_v_e)[3777:31558, :]
    # )

    # mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1
    # fields not available due to fusion
    # assert dallclose(
    #     np.asarray(sp_v.p_distv_bary(1))[3777:31558, :], np.asarray(solve_nonhydro.p_distv_bary_1)[3777:31558, :]
    # )
    # assert dallclose(
    #     np.asarray(sp_v.p_distv_bary(2))[3777:31558, :], np.asarray(solve_nonhydro.p_distv_bary_2)[3777:31558, :]
    # )
    # assert dallclose(
    #     np.asarray(sp_exit.z_rho_e_01())[3777:31558, :],
    #     np.asarray(z_fields.z_rho_e)[3777:31558, :],
    # )
    # assert dallclose(
    #     np.asarray(sp_exit.z_theta_v_e_01())[3777:31558, :],
    #     np.asarray(z_fields.z_theta_v_e)[3777:31558, :],
    # )

    # stencils 18,19, 20, 22
    assert dallclose(
        np.asarray(sp_exit.z_gradh_exner())[5387:31558, :],
        np.asarray(z_fields.z_gradh_exner)[5387:31558, :],
        atol=1e-20,
    )
    # stencil 21
    assert dallclose(
        np.asarray(sp_exit.z_hydro_corr())[5387:31558, 64],
        np.asarray(solve_nonhydro.z_hydro_corr)[5387:31558, 64],
        atol=1e-20,
    )
    # stencils 24
    assert dallclose(np.asarray(icon_result_vn_new)[5387:31558,:], np.asarray(prognostic_state_nnew.vn)[5387:31558,:], atol=6e-15)
    # stencil 29
    assert dallclose(np.asarray(icon_result_vn_new)[0:5387,:], np.asarray(prognostic_state_nnew.vn)[0:5387,:])

    # stencil 30
    assert dallclose(
        np.asarray(sp_exit.z_vn_avg()), np.asarray(solve_nonhydro.z_vn_avg), atol=5e-14
    )
    # stencil 30
    assert dallclose(
        np.asarray(sp_exit.z_graddiv_vn()[2538:31558,:]), np.asarray(z_fields.z_graddiv_vn)[2538:31558,:], atol=5e-20
    )
    # stencil 30
    assert dallclose(
        np.asarray(sp_exit.vt()), np.asarray(diagnostic_state_nh.vt), atol=5e-14
    )

    # stencil 32
    assert dallclose(
        np.asarray(icon_result_mass_fl_e),
        np.asarray(diagnostic_state_nh.mass_fl_e), atol=4e-12
    )
    # stencil 32
    # TODO: @abishekg7 higher tol.
    assert dallclose(
        np.asarray(sp_exit.z_theta_v_fl_e()), np.asarray(solve_nonhydro.z_theta_v_fl_e), atol = 1e-10
    )

    # stencil 35,36, 37,38
    assert dallclose(
        np.asarray(icon_result_vn_ie)[2538:31558,:], np.asarray(diagnostic_state_nh.vn_ie)[2538:31558,:], atol=2e-14
    )

    # stencil 35,36, 37,38
    assert dallclose(
        np.asarray(sp_exit.z_vt_ie()), np.asarray(solve_nonhydro.z_vt_ie), atol=2e-14
    )
    # stencil 35,36
    assert dallclose(
        np.asarray(sp_exit.z_kin_hor_e())[2538:31558,:], np.asarray(solve_nonhydro.z_kin_hor_e)[2538:31558,:], atol=10e-13
    )

    # stencil 35
    assert dallclose(
        np.asarray(sp_exit.z_w_concorr_me()), np.asarray(solve_nonhydro.z_w_concorr_me), atol=2e-15
    )
    # stencils 39,40
    assert dallclose(
        np.asarray(icon_result_w_concorr_c), np.asarray(diagnostic_state_nh.w_concorr_c), atol=1e-15
    )

    # stencil 41
    assert dallclose(
                np.asarray(sp_exit.z_flxdiv_mass()),
                np.asarray(solve_nonhydro.z_flxdiv_mass), atol=5e-15
            )
    # TODO: @abishekg7 higher tol.
    assert dallclose(
                np.asarray(sp_exit.z_flxdiv_theta()),
                np.asarray(solve_nonhydro.z_flxdiv_theta), atol=5e-12
                )
    # stencils 43, 46, 47
    assert dallclose(
                np.asarray(sp_exit.z_contr_w_fl_l())[3316:20896, :],
                np.asarray(z_fields.z_contr_w_fl_l)[3316:20896, :], atol=2e-15
                )
    # stencil 43
    assert dallclose(
                np.asarray(sp_exit.z_w_expl())[3316:20896, 1:65],
                np.asarray(z_fields.z_w_expl)[3316:20896, 1:65], atol=1e-14
                )
    # stencil 44, 45
    assert dallclose(
                np.asarray(sp_exit.z_alpha())[3316:20896, :],
                np.asarray(z_fields.z_alpha)[3316:20896, :], atol=5e-13
                )
    # stencil 44
    assert dallclose(
                np.asarray(sp_exit.z_beta())[3316:20896, :],
                np.asarray(z_fields.z_beta)[3316:20896, :], atol=2e-15
               )
    # stencil 45_b, 52
    assert dallclose(
                np.asarray(sp_exit.z_q())[3316:20896, :],
                np.asarray(z_fields.z_q)[3316:20896, :], atol=2e-15
                )
    # stencil 48, 49  #level 0 wrong
    assert dallclose(
                np.asarray(sp_exit.z_rho_expl())[3316:20896, :],
                np.asarray(z_fields.z_rho_expl)[3316:20896, :], atol=2e-15
                )
    assert dallclose(
                np.asarray(sp_exit.z_exner_expl())[3316:20896, :],
                np.asarray(z_fields.z_exner_expl)[3316:20896, :], atol=2e-15
                )
    # stencils 46, 47 (ok). 52, 53, 54
    assert dallclose(np.asarray(icon_result_w_new), np.asarray(prognostic_state_nnew.w))

    # stencil 52
    np.max(np.abs(np.asarray(sp_exit.w_new_52())[:, :] - np.asarray(prognostic_state_nnew.w)[:, :]))

    # stencil 61
    assert dallclose(
                np.asarray(icon_result_rho_new), np.asarray(prognostic_state_nnew.rho)
                    )
    assert dallclose(
                np.asarray(icon_result_exner_new), np.asarray(prognostic_state_nnew.exner))


    assert dallclose(np.asarray(icon_result_w_new), np.asarray(prognostic_state_nnew.w))
    assert dallclose(
        np.asarray(icon_result_exner_new), np.asarray(prognostic_state_nnew.exner)
    )
    assert dallclose(
        np.asarray(icon_result_theta_v_new), np.asarray(prognostic_state_nnew.theta_v)
    )

    assert dallclose(
        np.asarray(icon_result_theta_v_ic),
        np.asarray(diagnostic_state_nh.theta_v_ic),
    )

    assert dallclose(
        np.asarray(icon_result_prep_adv_mass_flx_me), np.asarray(prep_adv.mass_flx_me)
    )
    assert dallclose(
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
    savepoint_velocity_exit,
):
    config = NonHydrostaticConfig()
    sp = savepoint_nonhydro_init
    sp_exit = savepoint_nonhydro_exit
    sp_dif = diffusion_savepoint_init
    sp_met = metrics_savepoint
    sp_v_exit = savepoint_velocity_exit
    nonhydro_params = NonHydrostaticParams(config)
    vertical_params = VerticalModelParams(
        vct_a=grid_savepoint.vct_a(),
        rayleigh_damping_height=damping_height,
        nflatlev=grid_savepoint.nflatlev(),
        nflat_gradp=grid_savepoint.nflat_gradp(),
    )
    sp_met_nh = metrics_nonhydro_savepoint
    sp_d = data_provider.from_savepoint_grid()
    sp_v = savepoint_velocity_init
    ntl1 = sp.get_metadata("ntl1").get("ntl1")
    ntl2 = sp.get_metadata("ntl2").get("ntl2")
    ntnd = sp_v.get_metadata("ntnd").get("ntnd")
    mesh = SimpleMesh()
    dtime = sp_v.get_metadata("dtime").get("dtime")
    dyn_timestep = sp.get_metadata("dyn_timestep").get("dyn_timestep")
    recompute = sp_v.get_metadata("recompute").get("recompute")
    linit = sp_v.get_metadata("linit").get("linit")
    clean_mflx = sp_v.get_metadata("clean_mflx").get("clean_mflx")
    r_nsubsteps = sp_d.get_metadata("nsteps").get("nsteps")
    lprep_adv = sp_v.get_metadata("prep_adv").get("prep_adv")
    prep_adv = PrepAdvection(
        vn_traj=sp.vn_traj(), mass_flx_me=sp.mass_flx_me(), mass_flx_ic=sp.mass_flx_ic()
    )

    enh_smag_fac = zero_field(mesh, KDim)
    a_vec = random_field(mesh, KDim, low=1.0, high=10.0, extend={KDim: 1})
    fac = (0.67, 0.5, 1.3, 0.8)
    z = (0.1, 0.2, 0.3, 0.4)
    ntnd = sp_v.get_metadata("ntnd").get("ntnd")
    nnow = 0
    nnew = 1

    diagnostic_state_nh = DiagnosticStateNonHydro(
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
        ddt_vn_adv_ntl1=sp_v_exit.ddt_vn_apc_pc(1),
        ddt_vn_adv_ntl2=sp_v_exit.ddt_vn_apc_pc(2),
        ddt_w_adv_ntl1=sp_v_exit.ddt_w_adv_pc(1),
        ddt_w_adv_ntl2=sp_v_exit.ddt_w_adv_pc(2),  # TODO: @abishekg7 change later
        ntl1=ntl1,
        ntl2=ntl2,
        vt=sp_v_exit.vt(),  # sp_v.vt(), #TODO: @abishekg7 change back to sp_v
        vn_ie=sp_v_exit.vn_ie(),
        w_concorr_c=sp_v.w_concorr_c(),
        ddt_w_adv_pc=sp_v.ddt_w_adv_pc_before(ntnd),
        ddt_vn_apc_pc=sp_v.ddt_vn_apc_pc_before(ntnd),
        ntnd=ntnd,
        rho_incr=None,  # sp.rho_incr(),
        vn_incr=None,  # sp.vn_incr(),
        exner_incr=None,  # sp.exner_incr(),
    )

    prognostic_state_nnow = PrognosticState(
        w=sp.w_now(),
        vn=sp.vn_now(),
        exner_pressure=None,
        theta_v=sp.theta_v_now(),
        rho=sp.rho_now(),
        exner=sp.exner_now(),
    )

    prognostic_state_nnew = PrognosticState(
        w=sp.w_new(),
        vn=sp.vn_new(),
        exner_pressure=None,
        theta_v=sp.theta_v_new(),
        rho=sp.rho_new(),
        exner=sp.exner_new(),
    )

    z_fields = ZFields(
        z_gradh_exner=sp.z_gradh_exner(),
        z_alpha=sp.z_alpha(),
        z_beta=sp.z_beta(),
        z_w_expl=sp.z_w_expl(),
        z_exner_expl=sp.z_exner_expl(),
        z_q=sp.z_q(),
        z_contr_w_fl_l=sp.z_contr_w_fl_l(),
        z_rho_e=sp.z_rho_e(),
        z_theta_v_e=sp.z_theta_v_e(),
        z_graddiv_vn=sp.z_graddiv_vn(),
        z_rho_expl=sp.z_rho_expl(),
        z_dwdz_dd=sp.z_dwdz_dd(),
    )

    nh_constants = NHConstants(
        wgt_nnow_rth=sp.wgt_nnow_rth(),
        wgt_nnew_rth=sp.wgt_nnew_rth(),
        wgt_nnow_vel=sp.wgt_nnow_vel(),
        wgt_nnew_vel=sp.wgt_nnew_vel(),
        scal_divdamp=sp.scal_divdamp(),
        scal_divdamp_o2=sp.scal_divdamp_o2(),
    )

    interpolation_state = interpolation_savepoint.construct_interpolation_state()
    # metric_state = metrics_savepoint.construct_metric_state()
    metric_state_nonhydro = metrics_nonhydro_savepoint.construct_nh_metric_state(
        icon_grid.n_lev()
    )
    metric_state = metrics_savepoint.construct_metric_state()

    cell_geometry: CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: EdgeParams = grid_savepoint.construct_edge_geometry()

    solve_nonhydro = SolveNonhydro()
    solve_nonhydro.init(
        grid=icon_grid,
        config=config,
        params=nonhydro_params,
        metric_state_nonhydro=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        a_vec=a_vec,
        enh_smag_fac=enh_smag_fac,
        cell_areas=cell_geometry.area,
        fac=fac,
        z=z,
    )

    prognostic_state_ls = [prognostic_state_nnow, prognostic_state_nnew]
    solve_nonhydro.run_corrector_step(
        diagnostic_state_nh=diagnostic_state_nh,
        prognostic_state=prognostic_state_ls,
        config=config,
        params=nonhydro_params,
        edge_geometry=edge_geometry,
        z_fields=z_fields,
        # inv_dual_edge_length=edge_geometry.inverse_dual_edge_lengths,
        # inv_primal_edge_length=edge_geometry.inverse_primal_edge_lengths,
        # tangent_orientation=edge_geometry.tangent_orientation,
        prep_adv=prep_adv,
        dtime=dtime,
        nnew=nnew,
        nnow=nnow,
        cfl_w_limit=sp_v.cfl_w_limit(),
        scalfac_exdiff=sp_v.scalfac_exdiff(),
        cell_areas=cell_geometry.area,
        owner_mask=sp_d.c_owner_mask(),
        f_e=sp_d.f_e(),
        area_edge=edge_geometry.edge_areas,
        lclean_mflx=clean_mflx,
        nh_constants=nh_constants,
        bdy_divdamp=sp.bdy_divdamp(),
        lprep_adv=lprep_adv,
    )

    # icon_result_z_graddiv_vn = sp_exit.z_graddiv_vn()
    # icon_result_exner_now = sp_exit.exner_now()

    # assert dallclose(np.asarray(icon_result_z_graddiv_vn), np.asarray(prognostic_state_ls[nnew].exner))
    # assert dallclose(np.asarray(icon_result_exner_now), np.asarray(prognostic_state_ls[nnow].exner))

    # assert dallclose(
    #     np.asarray(icon_result_prep_adv_mass_flx_me), np.asarray(prep_adv.mass_flx_me)
    # )
    # assert dallclose(
    #     np.asarray(icon_result_prep_adv_vn_traj), np.asarray(prep_adv.vn_traj)
    # )


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

    diagnostic_state_nh = DiagnosticStateNonHydro(
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
            diagnostic_state_nh=diagnostic_state_nh,
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
            owner_mask=sp_d.c_owner_mask(),
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

    assert dallclose(
        np.asarray(icon_result_exner_new), np.asarray(prognostic_state_ls[nnew].exner)
    )
    assert dallclose(
        np.asarray(icon_result_exner_now), np.asarray(prognostic_state_ls[nnow].exner)
    )
    assert dallclose(
        np.asarray(icon_result_prep_adv_mass_flx_me), np.asarray(prep_adv.mass_flx_me)
    )
    assert dallclose(
        np.asarray(icon_result_mass_fl_e),
        np.asarray(diagnostic_state_nh.mass_fl_e),
    )
    assert dallclose(
        np.asarray(icon_result_prep_adv_vn_traj), np.asarray(prep_adv.vn_traj)
    )
    assert dallclose(
        np.asarray(icon_result_rho_ic), np.asarray(diagnostic_state_nh.rho_ic)
    )
    assert dallclose(
        np.asarray(icon_result_theta_v_ic),
        np.asarray(diagnostic_state_nh.theta_v_ic),
    )
    assert dallclose(
        np.asarray(icon_result_theta_v_new),
        np.asarray(prognostic_state_ls[nnew].theta_v),
    )
    assert dallclose(
        np.asarray(icon_result_vn_ie), np.asarray(diagnostic_state_nh.vn_ie)
    )
    assert dallclose(
        np.asarray(icon_result_vn_new), np.asarray(prognostic_state_ls[nnew].vn)
    )
    assert dallclose(
        np.asarray(icon_result_w_concorr_c), np.asarray(diagnostic_state_nh.w_concorr_c)
    )
    assert dallclose(
        np.asarray(icon_result_w_new), np.asarray(prognostic_state_ls[nnew].w)
    )
