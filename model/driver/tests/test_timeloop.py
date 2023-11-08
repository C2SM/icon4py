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

import os

import numpy as np
import pytest

from icon4py.model.atmosphere.diffusion.diffusion import Diffusion, DiffusionParams
from icon4py.model.atmosphere.dycore.nh_solve.solve_nonhydro import (
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
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.grid.vertical import VerticalModelParams
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.common.test_utils.helpers import random_field, zero_field
from icon4py.model.driver.dycore_driver import TimeLoop


# testing on MCH_CH_r04b09_dsl data
@pytest.mark.datatest
@pytest.mark.parametrize(
    "debug_mode,timeloop_istep_init,timeloop_istep_exit,timeloop_jstep_init,timeloop_jstep_exit,timeloop_date_init, timeloop_date, timeloop_diffusion_linit_init, timeloop_diffusion_linit_exit, vn_only_init",
    [
        (
            False,
            1,
            2,
            0,
            1,
            "2021-06-20T12:00:00.000",
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
            False,
            True,
            False,
        ),
    ],
)
def test_run_timeloop_single_step(
    debug_mode,
    timeloop_date_init,
    timeloop_date,
    grid_savepoint,
    icon_grid,
    metrics_savepoint,
    interpolation_savepoint,
    r04b09_diffusion_config,
    r04b09_iconrun_config,
    damping_height,
    timeloop_diffusion_linit_init,
    timeloop_diffusion_savepoint_init,
    timeloop_diffusion_savepoint_exit,
    timeloop_nonhydro_savepoint_init,
    timeloop_velocity_savepoint_init,
    timeloop_nonhydro_savepoint_exit,
    timeloop_nonhydro_step_savepoint_exit,
):
    diffusion_config = r04b09_diffusion_config
    diffusion_dtime = timeloop_diffusion_savepoint_init.get_metadata("dtime").get("dtime")
    edge_geometry: EdgeParams = grid_savepoint.construct_edge_geometry()
    cell_geometry: CellParams = grid_savepoint.construct_cell_geometry()
    diffusion_interpolation_state = (
        interpolation_savepoint.construct_interpolation_state_for_diffusion()
    )
    diffusion_metric_state = metrics_savepoint.construct_metric_state_for_diffusion()
    diffusion_diagnostic_state = (
        timeloop_diffusion_savepoint_init.construct_diagnostics_for_diffusion()
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
    sp = timeloop_nonhydro_savepoint_init
    nonhydro_params = NonHydrostaticParams(nonhydro_config)
    sp_v = timeloop_velocity_savepoint_init
    grid = SimpleGrid()
    nonhydro_dtime = timeloop_velocity_savepoint_init.get_metadata("dtime").get("dtime")
    # lprep_adv actually depends on other factors: idiv_method == 1 .AND. (ltransport .OR. p_patch%n_childdom > 0 .AND. grf_intmethod_e >= 5)
    lprep_adv = sp_v.get_metadata("prep_adv").get("prep_adv")
    prep_adv = PrepAdvection(
        vn_traj=sp.vn_traj(), mass_flx_me=sp.mass_flx_me(), mass_flx_ic=sp.mass_flx_ic()
    )

    enh_smag_fac = zero_field(grid, KDim)
    a_vec = random_field(grid, KDim, low=1.0, high=10.0, extend={KDim: 1})
    fac = (0.67, 0.5, 1.3, 0.8)
    z = (0.1, 0.2, 0.3, 0.4)

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

    nh_constants = NHConstants(
        wgt_nnow_rth=sp.wgt_nnow_rth(),
        wgt_nnew_rth=sp.wgt_nnew_rth(),
        wgt_nnow_vel=sp.wgt_nnow_vel(),
        wgt_nnew_vel=sp.wgt_nnew_vel(),
        scal_divdamp=sp.scal_divdamp(),
        scal_divdamp_o2=sp.scal_divdamp_o2(),
    )

    nonhydro_interpolation_state = (
        interpolation_savepoint.construct_interpolation_state_for_nonhydro()
    )
    nonhydro_metric_state = metrics_savepoint.construct_nh_metric_state(icon_grid.num_levels)

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
        cell_areas=cell_geometry.area,
        owner_mask=grid_savepoint.c_owner_mask(),
        a_vec=a_vec,
        enh_smag_fac=enh_smag_fac,
        fac=fac,
        z=z,
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
    )

    timeloop = TimeLoop(
        r04b09_iconrun_config, diffusion, solve_nonhydro, timeloop_diffusion_linit_init
    )

    assert timeloop.substep_timestep == nonhydro_dtime

    if timeloop_date == "2021-06-20T12:00:10.000":
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

    timeloop.time_integration(
        diffusion_diagnostic_state,
        nonhydro_diagnostic_state,
        prognostic_state_list,
        prep_adv,
        z_fields,
        nh_constants,
        sp.bdy_divdamp(),
        lprep_adv,
    )

    rho_sp = timeloop_nonhydro_savepoint_exit
    exner_sp = timeloop_diffusion_savepoint_exit.exner()
    theta_sp = timeloop_diffusion_savepoint_exit.theta_v()
    vn_sp = timeloop_diffusion_savepoint_exit.vn()
    w_sp = timeloop_diffusion_savepoint_exit.w()

    assert np.allclose(
        np.asarray(vn_sp),
        np.asarray(prognostic_state_list[timeloop.prognostic_now].vn),
        atol=5e-13,
    )

    assert np.allclose(
        np.asarray(w_sp),
        np.asarray(prognostic_state_list[timeloop.prognostic_now].w),
        atol=8e-14,
    )

    assert np.allclose(
        np.asarray(exner_sp), np.asarray(prognostic_state_list[timeloop.prognostic_now].exner)
    )

    assert np.allclose(
        np.asarray(theta_sp), np.asarray(prognostic_state_list[timeloop.prognostic_now].theta_v)
    )

    assert np.allclose(
        np.asarray(rho_sp.rho()), np.asarray(prognostic_state_list[timeloop.prognostic_now].rho)
    )

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
            np.asarray(rho_sp.rho()),
            np.asarray(prognostic_state_list[timeloop.prognostic_now].rho),
            "rho",
        )
        printing(
            np.asarray(exner_sp),
            np.asarray(prognostic_state_list[timeloop.prognostic_now].exner),
            "exner",
        )
        printing(
            np.asarray(theta_sp),
            np.asarray(prognostic_state_list[timeloop.prognostic_now].theta_v),
            "theta_v",
        )
        printing(
            np.asarray(w_sp), np.asarray(prognostic_state_list[timeloop.prognostic_now].w), "w"
        )
        printing(
            np.asarray(vn_sp), np.asarray(prognostic_state_list[timeloop.prognostic_now].vn), "vn"
        )
