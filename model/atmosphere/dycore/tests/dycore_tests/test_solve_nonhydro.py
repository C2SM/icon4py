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
import logging

import numpy as np
import pytest

from icon4py.model.atmosphere.dycore.nh_solve.solve_nonydro import (
    NonHydrostaticConfig,
    NonHydrostaticParams,
    SolveNonhydro,
)
from icon4py.model.atmosphere.dycore.state_utils.diagnostic_state import DiagnosticStateNonHydro
from icon4py.model.atmosphere.dycore.state_utils.nh_constants import NHConstants
from icon4py.model.atmosphere.dycore.state_utils.prep_adv_state import PrepAdvection
from icon4py.model.atmosphere.dycore.state_utils.utils import _allocate
from icon4py.model.atmosphere.dycore.state_utils.z_fields import ZFields
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim
from icon4py.model.common.grid.horizontal import CellParams, EdgeParams
from icon4py.model.common.grid.vertical import VerticalModelParams
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.common.test_utils.helpers import dallclose


@pytest.mark.datatest
def test_nonhydro_params():
    config = NonHydrostaticConfig()
    nonhydro_params = NonHydrostaticParams(config)

    assert nonhydro_params.df32 == pytest.approx(
        config.divdamp_fac3 - config.divdamp_fac2, abs=1e-12
    )
    assert nonhydro_params.dz32 == pytest.approx(config.divdamp_z3 - config.divdamp_z2, abs=1e-12)
    assert nonhydro_params.df42 == pytest.approx(
        config.divdamp_fac4 - config.divdamp_fac2, abs=1e-12
    )
    assert nonhydro_params.dz42 == pytest.approx(config.divdamp_z4 - config.divdamp_z2, abs=1e-12)

    assert nonhydro_params.bqdr == pytest.approx(
        (nonhydro_params.df42 * nonhydro_params.dz32 - nonhydro_params.df32 * nonhydro_params.dz42)
        / (
            nonhydro_params.dz32
            * nonhydro_params.dz42
            * (nonhydro_params.dz42 - nonhydro_params.dz32)
        ),
        abs=1e-12,
    )
    assert nonhydro_params.aqdr == pytest.approx(
        nonhydro_params.df32 / nonhydro_params.dz32 - nonhydro_params.bqdr * nonhydro_params.dz32,
        abs=1e-12,
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "istep, step_date_init, step_date_exit",
    [(1, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000")],
)
def test_nonhydro_predictor_step(
    istep,
    step_date_init,
    step_date_exit,
    icon_grid,
    savepoint_nonhydro_init,
    damping_height,
    grid_savepoint,
    savepoint_velocity_init,
    metrics_savepoint,
    interpolation_savepoint,
    savepoint_nonhydro_exit,
):
    config = NonHydrostaticConfig()
    sp = savepoint_nonhydro_init
    sp_exit = savepoint_nonhydro_exit
    nonhydro_params = NonHydrostaticParams(config)
    vertical_params = create_vertical_params(damping_height, grid_savepoint)
    sp_v = savepoint_velocity_init
    dtime = sp_v.get_metadata("dtime").get("dtime")
    recompute = sp_v.get_metadata("recompute").get("recompute")
    dyn_timestep = sp.get_metadata("dyn_timestep").get("dyn_timestep")
    linit = sp_v.get_metadata("linit").get("linit")

    nnow = 0
    nnew = 1

    diagnostic_state_nh = construct_diagnostics(sp, sp_v)

    z_fields = allocate_z_fields(icon_grid)

    interpolation_state = interpolation_savepoint.construct_interpolation_state_for_nonhydro()
    metric_state_nonhydro = metrics_savepoint.construct_nh_metric_state(icon_grid.n_lev())

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
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
        owner_mask=grid_savepoint.c_owner_mask(),
    )

    prognostic_state_ls = create_prognostic_states(sp)
    solve_nonhydro.set_timelevels(nnow, nnew)
    solve_nonhydro.run_predictor_step(
        diagnostic_state_nh=diagnostic_state_nh,
        prognostic_state=prognostic_state_ls,
        z_fields=z_fields,
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
    icon_result_theta_v_new = sp_exit.theta_v_new()
    icon_result_rho_ic = sp_exit.rho_ic()
    icon_result_w_concorr_c = sp_exit.w_concorr_c()
    icon_result_mass_fl_e = sp_exit.mass_fl_e()

    prognostic_state_nnew = prognostic_state_ls[1]
    # TODO: @abishekg7 remove bounds from asserts?
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

    # mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1
    assert dallclose(
        np.asarray(sp_exit.z_rho_e())[3777:31558, :],
        np.asarray(z_fields.z_rho_e)[3777:31558, :],
    )
    assert dallclose(
        np.asarray(sp_exit.z_theta_v_e())[3777:31558, :],
        np.asarray(z_fields.z_theta_v_e)[3777:31558, :],
    )

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
    assert dallclose(
        np.asarray(icon_result_vn_new)[5387:31558, :],
        np.asarray(prognostic_state_nnew.vn)[5387:31558, :],
        atol=6e-15,
    )
    # stencil 29
    assert dallclose(
        np.asarray(icon_result_vn_new)[0:5387, :],
        np.asarray(prognostic_state_nnew.vn)[0:5387, :],
    )

    # stencil 30
    assert dallclose(
        np.asarray(sp_exit.z_vn_avg()), np.asarray(solve_nonhydro.z_vn_avg), atol=5e-14
    )
    # stencil 30
    assert dallclose(
        np.asarray(sp_exit.z_graddiv_vn()[2538:31558, :]),
        np.asarray(z_fields.z_graddiv_vn)[2538:31558, :],
        atol=5e-20,
    )
    # stencil 30
    assert dallclose(np.asarray(sp_exit.vt()), np.asarray(diagnostic_state_nh.vt), atol=5e-14)

    # stencil 32
    assert dallclose(
        np.asarray(icon_result_mass_fl_e),
        np.asarray(diagnostic_state_nh.mass_fl_e),
        atol=4e-12,
    )
    # stencil 32
    # TODO: @abishekg7 higher tol.
    assert dallclose(
        np.asarray(sp_exit.z_theta_v_fl_e()), np.asarray(solve_nonhydro.z_theta_v_fl_e), atol=1e-9
    )

    # stencil 35,36, 37,38
    assert dallclose(
        np.asarray(icon_result_vn_ie)[2538:31558, :],
        np.asarray(diagnostic_state_nh.vn_ie)[2538:31558, :],
        atol=2e-14,
    )

    # stencil 35,36, 37,38
    assert dallclose(np.asarray(sp_exit.z_vt_ie()), np.asarray(z_fields.z_vt_ie), atol=2e-14)
    # stencil 35,36
    assert dallclose(
        np.asarray(sp_exit.z_kin_hor_e())[2538:31558, :],
        np.asarray(z_fields.z_kin_hor_e)[2538:31558, :],
        atol=10e-13,
    )

    # stencil 35
    assert dallclose(
        np.asarray(sp_exit.z_w_concorr_me()),
        np.asarray(solve_nonhydro.z_w_concorr_me),
        atol=2e-15,
    )
    # stencils 39,40
    assert dallclose(
        np.asarray(icon_result_w_concorr_c),
        np.asarray(diagnostic_state_nh.w_concorr_c),
        atol=1e-15,
    )

    # stencil 41
    assert dallclose(
        np.asarray(sp_exit.z_flxdiv_mass()),
        np.asarray(solve_nonhydro.z_flxdiv_mass),
        atol=5e-15,
    )
    # TODO: @abishekg7 higher tol.
    assert dallclose(
        np.asarray(sp_exit.z_flxdiv_theta()),
        np.asarray(solve_nonhydro.z_flxdiv_theta),
        atol=5e-12,
    )
    # stencils 43, 46, 47
    assert dallclose(
        np.asarray(sp_exit.z_contr_w_fl_l())[3316:20896, :],
        np.asarray(z_fields.z_contr_w_fl_l)[3316:20896, :],
        atol=2e-15,
    )
    # stencil 43
    assert dallclose(
        np.asarray(sp_exit.z_w_expl())[3316:20896, 1:65],
        np.asarray(z_fields.z_w_expl)[3316:20896, 1:65],
        atol=1e-14,
    )
    # stencil 44, 45
    assert dallclose(
        np.asarray(sp_exit.z_alpha())[3316:20896, :],
        np.asarray(z_fields.z_alpha)[3316:20896, :],
        atol=5e-13,
    )
    # stencil 44
    assert dallclose(
        np.asarray(sp_exit.z_beta())[3316:20896, :],
        np.asarray(z_fields.z_beta)[3316:20896, :],
        atol=2e-15,
    )
    # stencil 45_b, 52
    assert dallclose(
        np.asarray(sp_exit.z_q())[3316:20896, :],
        np.asarray(z_fields.z_q)[3316:20896, :],
        atol=2e-15,
    )
    # stencil 48, 49  #level 0 wrong
    assert dallclose(
        np.asarray(sp_exit.z_rho_expl())[3316:20896, :],
        np.asarray(z_fields.z_rho_expl)[3316:20896, :],
        atol=2e-15,
    )
    assert dallclose(
        np.asarray(sp_exit.z_exner_expl())[3316:20896, :],
        np.asarray(z_fields.z_exner_expl)[3316:20896, :],
        atol=2e-15,
    )

    # end
    assert dallclose(np.asarray(sp_exit.rho()), np.asarray(prognostic_state_nnew.rho))
    assert dallclose(np.asarray(icon_result_w_new), np.asarray(prognostic_state_nnew.w), atol=7e-14)

    # not tested
    assert dallclose(np.asarray(icon_result_exner_new), np.asarray(prognostic_state_nnew.exner))

    assert dallclose(np.asarray(icon_result_theta_v_new), np.asarray(prognostic_state_nnew.theta_v))


def construct_diagnostics(sp, sp_v):
    return DiagnosticStateNonHydro(
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
    )


def allocate_z_fields(icon_grid):
    return ZFields(
        z_gradh_exner=_allocate(EdgeDim, KDim, mesh=icon_grid),
        z_alpha=_allocate(CellDim, KDim, is_halfdim=True, mesh=icon_grid),
        z_beta=_allocate(CellDim, KDim, mesh=icon_grid),
        z_w_expl=_allocate(CellDim, KDim, is_halfdim=True, mesh=icon_grid),
        z_exner_expl=_allocate(CellDim, KDim, mesh=icon_grid),
        z_q=_allocate(CellDim, KDim, mesh=icon_grid),
        z_contr_w_fl_l=_allocate(CellDim, KDim, is_halfdim=True, mesh=icon_grid),
        z_rho_e=_allocate(EdgeDim, KDim, mesh=icon_grid),
        z_theta_v_e=_allocate(EdgeDim, KDim, mesh=icon_grid),
        z_graddiv_vn=_allocate(EdgeDim, KDim, mesh=icon_grid),
        z_rho_expl=_allocate(CellDim, KDim, mesh=icon_grid),
        z_dwdz_dd=_allocate(CellDim, KDim, mesh=icon_grid),
        z_kin_hor_e=_allocate(EdgeDim, KDim, mesh=icon_grid),
        z_vt_ie=_allocate(EdgeDim, KDim, mesh=icon_grid),
    )


def create_vertical_params(damping_height, grid_savepoint):
    return VerticalModelParams(
        vct_a=grid_savepoint.vct_a(),
        rayleigh_damping_height=damping_height,
        nflat_gradp=grid_savepoint.nflat_gradp(),
        nflatlev=grid_savepoint.nflatlev(),
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "istep, step_date_init, step_date_exit",
    [(2, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000")],
)
def test_nonhydro_corrector_step(
    istep,
    step_date_init,
    step_date_exit,
    icon_grid,
    savepoint_nonhydro_init,
    damping_height,
    grid_savepoint,
    savepoint_velocity_init,
    metrics_savepoint,
    interpolation_savepoint,
    savepoint_nonhydro_exit,
    caplog,
):
    caplog.set_level(logging.DEBUG)
    config = NonHydrostaticConfig()
    sp = savepoint_nonhydro_init
    nonhydro_params = NonHydrostaticParams(config)
    vertical_params = VerticalModelParams(
        vct_a=grid_savepoint.vct_a(),
        rayleigh_damping_height=damping_height,
        nflatlev=grid_savepoint.nflatlev(),
        nflat_gradp=grid_savepoint.nflat_gradp(),
    )
    sp_v = savepoint_velocity_init
    dtime = sp_v.get_metadata("dtime").get("dtime")
    clean_mflx = sp_v.get_metadata("clean_mflx").get("clean_mflx")
    lprep_adv = sp_v.get_metadata("prep_adv").get("prep_adv")
    prep_adv = PrepAdvection(
        vn_traj=sp.vn_traj(), mass_flx_me=sp.mass_flx_me(), mass_flx_ic=sp.mass_flx_ic()
    )

    nnow = 0  # TODO: @abishekg7 read from serialized data?
    nnew = 1

    diagnostic_state_nh = construct_diagnostics(sp, sp_v)

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
        z_kin_hor_e=sp_v.z_kin_hor_e(),
        z_vt_ie=sp_v.z_vt_ie(),
    )

    nh_constants = create_nh_constants(sp)
    print(f"nh_constants.scal_divdamp = {sp.scal_divdamp()}")
    print(f"nh_constants.scal_divdamp_field = {sp.scal_divdamp_field()}")

    print(f"scal_divdamp_o2 = {sp.scal_divdamp_o2()}")
    scal_divdamp_o2 = sp.scal_divdamp_o2()  # only used in stencil_26

    interpolation_state = interpolation_savepoint.construct_interpolation_state_for_nonhydro()
    metric_state_nonhydro = metrics_savepoint.construct_nh_metric_state(icon_grid.n_lev())

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
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
        owner_mask=grid_savepoint.c_owner_mask(),
    )

    prognostic_state_ls = create_prognostic_states(sp)
    solve_nonhydro.set_timelevels(nnow, nnew)

    solve_nonhydro.run_corrector_step(
        diagnostic_state_nh=diagnostic_state_nh,
        prognostic_state=prognostic_state_ls,
        z_fields=z_fields,
        prep_adv=prep_adv,
        scal_divdamp_o2=scal_divdamp_o2,
        dtime=dtime,
        nnew=nnew,
        nnow=nnow,
        lclean_mflx=clean_mflx,
        nh_constants=nh_constants,
        bdy_divdamp=sp.bdy_divdamp(),
        lprep_adv=lprep_adv,
    )

    assert dallclose(
        np.asarray(savepoint_nonhydro_exit.rho_ic()),
        np.asarray(diagnostic_state_nh.rho_ic),
    )

    assert dallclose(
        np.asarray(savepoint_nonhydro_exit.theta_v_ic()),
        np.asarray(diagnostic_state_nh.theta_v_ic),
    )

    assert dallclose(
        np.asarray(savepoint_nonhydro_exit.z_graddiv_vn()),
        np.asarray(z_fields.z_graddiv_vn),
        atol=1e-12,
    )
    assert dallclose(
        np.asarray(savepoint_nonhydro_exit.exner_new()),
        np.asarray(prognostic_state_ls[nnew].exner),
    )

    assert dallclose(
        np.asarray(savepoint_nonhydro_exit.rho()),
        np.asarray(np.asarray(prognostic_state_ls[nnew].rho)),
    )

    assert dallclose(
        np.asarray(savepoint_nonhydro_exit.w_new()),
        np.asarray(np.asarray(prognostic_state_ls[nnew].w)),
        atol=8e-14,
    )

    assert dallclose(
        np.asarray(savepoint_nonhydro_exit.vn_new()),
        np.asarray(np.asarray(prognostic_state_ls[nnew].vn)),
        rtol=1e-10,
    )

    assert dallclose(
        np.asarray(savepoint_nonhydro_exit.theta_v_new()),
        np.asarray(np.asarray(prognostic_state_ls[nnew].theta_v)),
    )

    assert dallclose(
        np.asarray(savepoint_nonhydro_exit.mass_fl_e()),
        np.asarray(diagnostic_state_nh.mass_fl_e),
        rtol=1e-10,
    )

    assert dallclose(
        np.asarray(savepoint_nonhydro_exit.mass_flx_me()),
        np.asarray(prep_adv.mass_flx_me),
        rtol=1e-10,
    )
    assert dallclose(
        np.asarray(savepoint_nonhydro_exit.vn_traj()),
        np.asarray(prep_adv.vn_traj),
        rtol=1e-10,
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "istep, step_date_init, step_date_exit",
    [(1, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000")],
)
def test_run_solve_nonhydro_single_step(
    istep,
    step_date_init,
    step_date_exit,
    icon_grid,
    savepoint_nonhydro_init,
    damping_height,
    grid_savepoint,
    savepoint_velocity_init,
    metrics_savepoint,
    interpolation_savepoint,
    savepoint_nonhydro_exit,
    savepoint_nonhydro_step_exit,
):
    config = NonHydrostaticConfig()
    sp = savepoint_nonhydro_init
    sp_step_exit = savepoint_nonhydro_step_exit
    nonhydro_params = NonHydrostaticParams(config)
    vertical_params = VerticalModelParams(
        vct_a=grid_savepoint.vct_a(),
        rayleigh_damping_height=damping_height,
        nflat_gradp=grid_savepoint.nflat_gradp(),
        nflatlev=grid_savepoint.nflatlev(),
    )
    sp_v = savepoint_velocity_init
    dtime = sp_v.get_metadata("dtime").get("dtime")
    lprep_adv = sp_v.get_metadata("prep_adv").get("prep_adv")
    clean_mflx = sp_v.get_metadata("clean_mflx").get("clean_mflx")
    prep_adv = PrepAdvection(
        vn_traj=sp.vn_traj(), mass_flx_me=sp.mass_flx_me(), mass_flx_ic=sp.mass_flx_ic()
    )

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
    )

    z_fields = allocate_z_fields(icon_grid)

    nh_constants = create_nh_constants(sp)

    interpolation_state = interpolation_savepoint.construct_interpolation_state_for_nonhydro()
    metric_state_nonhydro = metrics_savepoint.construct_nh_metric_state(icon_grid.n_lev())

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
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
        owner_mask=grid_savepoint.c_owner_mask(),
    )

    prognostic_state_ls = create_prognostic_states(sp)
    divdamp_fac_o2 = 0.032  # TODO (magdalena) get from somewhere??
    solve_nonhydro.time_step(
        diagnostic_state_nh=diagnostic_state_nh,
        prognostic_state_ls=prognostic_state_ls,
        prep_adv=prep_adv,
        z_fields=z_fields,
        nh_constants=nh_constants,
        bdy_divdamp=sp.bdy_divdamp(),  # TODO (magdalena) local calculation in solve non-hydro based on nudge_coeff_e and scal_divdamp (also locally calculated)
        divdamp_fac_o2=divdamp_fac_o2,
        dtime=dtime,
        idyn_timestep=dyn_timestep,
        l_recompute=recompute,
        l_init=linit,
        nnew=nnew,
        nnow=nnow,
        lclean_mflx=clean_mflx,
        lprep_adv=lprep_adv,
    )
    prognostic_state_nnew = prognostic_state_ls[1]
    assert dallclose(
        np.asarray(sp_step_exit.theta_v_new()),
        np.asarray(prognostic_state_nnew.theta_v),
    )

    assert dallclose(np.asarray(sp_step_exit.exner_new()), np.asarray(prognostic_state_nnew.exner))


@pytest.mark.skip
@pytest.mark.datatest
@pytest.mark.parametrize(
    "istep, step_date_init, step_date_exit",
    [(1, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000")],
)
def test_run_solve_nonhydro_multi_step(
    istep,
    step_date_init,
    step_date_exit,
    icon_grid,
    savepoint_nonhydro_init,
    damping_height,
    grid_savepoint,
    savepoint_velocity_init,
    metrics_savepoint,
    interpolation_savepoint,
    savepoint_nonhydro_exit,
    savepoint_nonhydro_step_exit,
):
    config = NonHydrostaticConfig()
    sp = savepoint_nonhydro_init
    sp_step_exit = savepoint_nonhydro_step_exit
    nonhydro_params = NonHydrostaticParams(config)
    vertical_params = VerticalModelParams(
        vct_a=grid_savepoint.vct_a(),
        rayleigh_damping_height=damping_height,
        nflat_gradp=grid_savepoint.nflat_gradp(),
        nflatlev=grid_savepoint.nflatlev(),
    )
    sp_v = savepoint_velocity_init
    dtime = sp_v.get_metadata("dtime").get("dtime")
    r_nsubsteps = grid_savepoint.get_metadata("nsteps").get("nsteps")
    lprep_adv = sp_v.get_metadata("prep_adv").get("prep_adv")
    clean_mflx = sp_v.get_metadata("clean_mflx").get("clean_mflx")
    prep_adv = PrepAdvection(
        vn_traj=sp.vn_traj(), mass_flx_me=sp.mass_flx_me(), mass_flx_ic=sp.mass_flx_ic()
    )

    nnow = 0
    nnew = 1
    recompute = sp_v.get_metadata("recompute").get("recompute")
    linit = sp_v.get_metadata("linit").get("linit")
    dyn_timestep = sp_v.get_metadata("dyn_timestep").get("dyn_timestep")

    diagnostic_state_nh = construct_diagnostics(sp, sp_v)

    prognostic_state_ls, prognostic_state_nnew = create_prognostic_states(sp)

    z_fields = allocate_z_fields(icon_grid)

    nh_constants = create_nh_constants(sp)

    interpolation_state = interpolation_savepoint.construct_interpolation_state_for_nonhydro()
    metric_state_nonhydro = metrics_savepoint.construct_nh_metric_state(icon_grid.n_lev())

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
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
        owner_mask=grid_savepoint.c_owner_mask(),
    )

    for _ in range(r_nsubsteps):
        solve_nonhydro.time_step(
            diagnostic_state_nh=diagnostic_state_nh,
            prognostic_state_ls=prognostic_state_ls,
            prep_adv=prep_adv,
            config=config,
            params=nonhydro_params,
            z_fields=z_fields,
            nh_constants=nh_constants,
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

    assert dallclose(
        np.asarray(savepoint_nonhydro_exit.rho_ic()),
        np.asarray(diagnostic_state_nh.rho_ic),
    )

    assert dallclose(
        np.asarray(savepoint_nonhydro_exit.theta_v_ic()),
        np.asarray(diagnostic_state_nh.theta_v_ic),
    )

    assert dallclose(
        np.asarray(savepoint_nonhydro_exit.z_graddiv_vn()),
        np.asarray(z_fields.z_graddiv_vn),
    )
    assert dallclose(
        np.asarray(savepoint_nonhydro_exit.exner_new()),
        np.asarray(prognostic_state_ls[nnew].exner),
    )

    assert dallclose(
        np.asarray(savepoint_nonhydro_exit.rho()),
        np.asarray(np.asarray(prognostic_state_ls[nnew].rho)),
    )

    assert dallclose(
        np.asarray(savepoint_nonhydro_exit.w_new()),
        np.asarray(np.asarray(prognostic_state_ls[nnew].w)),
        atol=8e-14,
    )

    assert dallclose(
        np.asarray(savepoint_nonhydro_exit.vn_new()),
        np.asarray(np.asarray(prognostic_state_ls[nnew].vn)),
        rtol=1e-10,
    )

    assert dallclose(
        np.asarray(savepoint_nonhydro_exit.theta_v_new()),
        np.asarray(np.asarray(prognostic_state_ls[nnew].theta_v)),
    )

    assert dallclose(
        np.asarray(savepoint_nonhydro_exit.mass_fl_e()),
        np.asarray(diagnostic_state_nh.mass_fl_e),
        rtol=1e-10,
    )

    assert dallclose(
        np.asarray(savepoint_nonhydro_exit.mass_flx_me()),
        np.asarray(prep_adv.mass_flx_me),
        rtol=1e-10,
    )
    assert dallclose(
        np.asarray(savepoint_nonhydro_exit.vn_traj()),
        np.asarray(prep_adv.vn_traj),
        rtol=1e-10,
    )

    assert dallclose(
        np.asarray(sp_step_exit.theta_v_new()),
        np.asarray(prognostic_state_nnew.theta_v),
    )

    assert dallclose(np.asarray(sp_step_exit.exner_new()), np.asarray(prognostic_state_nnew.exner))


def create_nh_constants(sp):
    return NHConstants(
        wgt_nnow_rth=sp.wgt_nnow_rth(),
        wgt_nnew_rth=sp.wgt_nnew_rth(),
        wgt_nnow_vel=sp.wgt_nnow_vel(),
        wgt_nnew_vel=sp.wgt_nnew_vel(),
    )


def create_prognostic_states(sp):
    prognostic_state_nnow = PrognosticState(
        w=sp.w_now(),
        vn=sp.vn_now(),
        theta_v=sp.theta_v_now(),
        rho=sp.rho_now(),
        exner=sp.exner_now(),
    )
    prognostic_state_nnew = PrognosticState(
        w=sp.w_new(),
        vn=sp.vn_new(),
        theta_v=sp.theta_v_new(),
        rho=sp.rho_new(),
        exner=sp.exner_new(),
    )
    prognostic_state_ls = [prognostic_state_nnow, prognostic_state_nnew]
    return prognostic_state_ls
