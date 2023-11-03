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
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.dycore.state_utils.diagnostic_state import DiagnosticStateNonHydro
from icon4py.model.atmosphere.dycore.velocity.velocity_advection import VelocityAdvection
from icon4py.model.common.grid.horizontal import CellParams, EdgeParams
from icon4py.model.common.grid.vertical import VerticalModelParams
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.common.test_utils.helpers import dallclose


@pytest.mark.datatest
def test_scalfactors(savepoint_velocity_init, icon_grid):
    dtime = savepoint_velocity_init.get_metadata("dtime").get("dtime")
    velocity_advection = VelocityAdvection(
        grid=icon_grid,
        metric_state=None,
        interpolation_state=None,
        vertical_params=None,
        edge_params=None,
        owner_mask=None,
    )
    (cfl_w_limit, scalfac_exdiff) = velocity_advection._scale_factors_by_dtime(dtime)
    assert cfl_w_limit == savepoint_velocity_init.cfl_w_limit()
    assert scalfac_exdiff == savepoint_velocity_init.scalfac_exdiff()


@pytest.mark.datatest
def test_velocity_init(
    savepoint_velocity_init,
    interpolation_savepoint,
    grid_savepoint,
    icon_grid,
    metrics_savepoint,
    step_date_init,
    damping_height,
):
    interpolation_state = interpolation_savepoint.construct_interpolation_state_for_nonhydro()
    metric_state_nonhydro = metrics_savepoint.construct_nh_metric_state(icon_grid.n_lev())

    vertical_params = VerticalModelParams(
        vct_a=grid_savepoint.vct_a(),
        rayleigh_damping_height=damping_height,
        nflat_gradp=int32(grid_savepoint.nflat_gradp()),
        nflatlev=int32(grid_savepoint.nflatlev()),
    )

    velocity_advection = VelocityAdvection(
        grid=icon_grid,
        metric_state=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        edge_params=grid_savepoint.construct_edge_geometry(),
        owner_mask=grid_savepoint.c_owner_mask(),
    )

    assert dallclose(0.0, np.asarray(velocity_advection.cfl_clipping))
    assert dallclose(False, np.asarray(velocity_advection.levmask))
    assert dallclose(0.0, np.asarray(velocity_advection.vcfl_dsl))

    assert velocity_advection.cfl_w_limit == 0.65
    assert velocity_advection.scalfac_exdiff == 0.05


@pytest.mark.datatest
def test_verify_velocity_init_against_regular_savepoint(
    savepoint_velocity_init,
    interpolation_savepoint,
    grid_savepoint,
    icon_grid,
    metrics_savepoint,
    damping_height,
):
    savepoint = savepoint_velocity_init
    dtime = savepoint.get_metadata("dtime").get("dtime")

    interpolation_state = interpolation_savepoint.construct_interpolation_state_for_nonhydro()
    metric_state_nonhydro = metrics_savepoint.construct_nh_metric_state(icon_grid.n_lev())
    vertical_params = VerticalModelParams(
        vct_a=grid_savepoint.vct_a(),
        rayleigh_damping_height=damping_height,
        nflat_gradp=int32(grid_savepoint.nflat_gradp()),
        nflatlev=int32(grid_savepoint.nflatlev()),
    )

    velocity_advection = VelocityAdvection(
        grid=icon_grid,
        metric_state=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        edge_params=grid_savepoint.construct_edge_geometry(),
        owner_mask=grid_savepoint.c_owner_mask(),
    )

    assert savepoint.cfl_w_limit() == velocity_advection.cfl_w_limit / dtime
    assert savepoint.scalfac_exdiff() == velocity_advection.scalfac_exdiff / (
        dtime * (0.85 - savepoint.cfl_w_limit() * dtime)
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "istep, step_date_init, step_date_exit",
    [(1, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000")],
)
def test_velocity_predictor_step(
    istep,
    step_date_init,
    step_date_exit,
    damping_height,
    icon_grid,
    grid_savepoint,
    savepoint_velocity_init,
    metrics_savepoint,
    interpolation_savepoint,
    savepoint_velocity_exit,
):
    sp_v = savepoint_velocity_init
    vn_only = sp_v.get_metadata("vn_only").get("vn_only")
    ntnd = sp_v.get_metadata("ntnd").get("ntnd")
    dtime = sp_v.get_metadata("dtime").get("dtime")

    diagnostic_state = DiagnosticStateNonHydro(
        vt=sp_v.vt(),
        vn_ie=sp_v.vn_ie(),
        w_concorr_c=sp_v.w_concorr_c(),
        theta_v_ic=None,
        exner_pr=None,
        rho_ic=None,
        ddt_exner_phy=None,
        grf_tend_rho=None,
        grf_tend_thv=None,
        grf_tend_w=None,
        mass_fl_e=None,
        ddt_vn_phy=None,
        grf_tend_vn=None,
        ddt_vn_apc_ntl1=sp_v.ddt_vn_apc_pc(1),
        ddt_vn_apc_ntl2=sp_v.ddt_vn_apc_pc(2),
        ddt_w_adv_ntl1=sp_v.ddt_w_adv_pc(1),
        ddt_w_adv_ntl2=sp_v.ddt_w_adv_pc(2),
        rho_incr=None,  # sp.rho_incr(),
        vn_incr=None,  # sp.vn_incr(),
        exner_incr=None,  # sp.exner_incr(),
    )
    prognostic_state = PrognosticState(
        w=sp_v.w(),
        vn=sp_v.vn(),
        theta_v=None,
        rho=None,
        exner=None,
    )
    interpolation_state = interpolation_savepoint.construct_interpolation_state_for_nonhydro()

    metric_state_nonhydro = metrics_savepoint.construct_nh_metric_state(icon_grid.n_lev())

    cell_geometry: CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: EdgeParams = grid_savepoint.construct_edge_geometry()

    vertical_params = VerticalModelParams(
        vct_a=grid_savepoint.vct_a(),
        rayleigh_damping_height=damping_height,
        nflatlev=int32(grid_savepoint.nflatlev()),
        nflat_gradp=int32(grid_savepoint.nflat_gradp()),
    )

    velocity_advection = VelocityAdvection(
        grid=icon_grid,
        metric_state=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        edge_params=edge_geometry,
        owner_mask=grid_savepoint.c_owner_mask(),
    )

    velocity_advection.run_predictor_step(
        vn_only=vn_only,
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        z_w_concorr_me=sp_v.z_w_concorr_me(),
        z_kin_hor_e=sp_v.z_kin_hor_e(),
        z_vt_ie=sp_v.z_vt_ie(),
        dtime=dtime,
        ntnd=ntnd - 1,
        cell_areas=cell_geometry.area,
    )

    icon_result_ddt_vn_apc_pc = savepoint_velocity_exit.ddt_vn_apc_pc(ntnd)
    icon_result_ddt_w_adv_pc = savepoint_velocity_exit.ddt_w_adv_pc(ntnd)
    icon_result_vn_ie = savepoint_velocity_exit.vn_ie()
    icon_result_vt = savepoint_velocity_exit.vt()
    icon_result_w_concorr_c = savepoint_velocity_exit.w_concorr_c()
    icon_result_z_w_concorr_mc = savepoint_velocity_exit.z_w_concorr_mc()
    icon_result_z_v_grad_w = savepoint_velocity_exit.z_v_grad_w()

    # stencil 01
    assert dallclose(np.asarray(icon_result_vt), np.asarray(diagnostic_state.vt), atol=1.0e-14)
    # stencil 02,05
    assert dallclose(
        np.asarray(icon_result_vn_ie), np.asarray(diagnostic_state.vn_ie), atol=1.0e-14
    )
    # stencil 07
    assert dallclose(
        np.asarray(icon_result_z_v_grad_w)[3777:31558, :],
        np.asarray(velocity_advection.z_v_grad_w)[3777:31558, :],
        atol=1.0e-16,
    )
    # stencil 08
    assert dallclose(
        np.asarray(savepoint_velocity_exit.z_ekinh())[3316:20896, :],
        np.asarray(velocity_advection.z_ekinh)[3316:20896, :],
    )
    # stencil 09
    assert dallclose(
        np.asarray(icon_result_z_w_concorr_mc)[
            3316:20896, vertical_params.nflatlev : icon_grid.n_lev()
        ],
        np.asarray(velocity_advection.z_w_concorr_mc)[
            3316:20896, vertical_params.nflatlev : icon_grid.n_lev()
        ],
        atol=1.0e-15,
    )
    # stencil 10
    assert dallclose(
        np.asarray(icon_result_w_concorr_c)[
            3316:20896, vertical_params.nflatlev + 1 : icon_grid.n_lev()
        ],
        np.asarray(diagnostic_state.w_concorr_c)[
            3316:20896, vertical_params.nflatlev + 1 : icon_grid.n_lev()
        ],
        atol=1.0e-15,
    )
    # stencil 11,12,13,14
    assert dallclose(
        np.asarray(savepoint_velocity_exit.z_w_con_c())[3316:20896, :],
        np.asarray(velocity_advection.z_w_con_c)[3316:20896, :],
        atol=1.0e-15,
    )
    # stencil 16
    assert dallclose(
        np.asarray(icon_result_ddt_w_adv_pc)[3316:20896, :],
        np.asarray(diagnostic_state.ddt_w_adv_pc[ntnd - 1])[3316:20896, :],
        atol=5.0e-16,
        rtol=1.0e-10,
    )
    # stencil 19 level 0 not verifying
    assert dallclose(
        np.asarray(icon_result_ddt_vn_apc_pc)[5387:31558, 0:65],
        np.asarray(diagnostic_state.ddt_vn_apc_pc[ntnd - 1])[5387:31558, 0:65],
        atol=1.0e-15,
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "istep, step_date_init, step_date_exit",
    [(2, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000")],
)
def test_velocity_corrector_step(
    istep,
    step_date_init,
    step_date_exit,
    damping_height,
    icon_grid,
    grid_savepoint,
    savepoint_velocity_init,
    savepoint_velocity_exit,
    interpolation_savepoint,
    metrics_savepoint,
):
    sp_v = savepoint_velocity_init
    vn_only = sp_v.get_metadata("vn_only").get("vn_only")
    ntnd = sp_v.get_metadata("ntnd").get("ntnd")
    dtime = sp_v.get_metadata("dtime").get("dtime")

    diagnostic_state = DiagnosticStateNonHydro(
        vt=sp_v.vt(),
        vn_ie=sp_v.vn_ie(),
        w_concorr_c=sp_v.w_concorr_c(),
        theta_v_ic=None,
        exner_pr=None,
        rho_ic=None,
        ddt_exner_phy=None,
        grf_tend_rho=None,
        grf_tend_thv=None,
        grf_tend_w=None,
        mass_fl_e=None,
        ddt_vn_phy=None,
        grf_tend_vn=None,
        ddt_vn_apc_ntl1=sp_v.ddt_vn_apc_pc(1),
        ddt_vn_apc_ntl2=sp_v.ddt_vn_apc_pc(2),
        ddt_w_adv_ntl1=sp_v.ddt_w_adv_pc(1),
        ddt_w_adv_ntl2=sp_v.ddt_w_adv_pc(2),
        rho_incr=None,  # sp.rho_incr(),
        vn_incr=None,  # sp.vn_incr(),
        exner_incr=None,  # sp.exner_incr(),
    )
    prognostic_state = PrognosticState(
        w=sp_v.w(),
        vn=sp_v.vn(),
        theta_v=None,
        rho=None,
        exner=None,
    )

    interpolation_state = interpolation_savepoint.construct_interpolation_state_for_nonhydro()

    metric_state_nonhydro = metrics_savepoint.construct_nh_metric_state(icon_grid.n_lev())

    cell_geometry: CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: EdgeParams = grid_savepoint.construct_edge_geometry()

    vertical_params = VerticalModelParams(
        vct_a=grid_savepoint.vct_a(),
        rayleigh_damping_height=damping_height,
        nflatlev=int32(grid_savepoint.nflatlev()),
        nflat_gradp=int32(grid_savepoint.nflat_gradp()),
    )

    velocity_advection = VelocityAdvection(
        grid=icon_grid,
        metric_state=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        edge_params=edge_geometry,
        owner_mask=grid_savepoint.c_owner_mask(),
    )

    velocity_advection.run_corrector_step(
        vn_only=vn_only,
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        z_kin_hor_e=sp_v.z_kin_hor_e(),
        z_vt_ie=sp_v.z_vt_ie(),
        dtime=dtime,
        ntnd=ntnd - 1,
        cell_areas=cell_geometry.area,
    )

    icon_result_ddt_vn_apc_pc = savepoint_velocity_exit.ddt_vn_apc_pc(ntnd)
    icon_result_ddt_w_adv_pc = savepoint_velocity_exit.ddt_w_adv_pc(ntnd)
    icon_result_z_v_grad_w = savepoint_velocity_exit.z_v_grad_w()

    # stencil 07
    assert dallclose(
        np.asarray(icon_result_z_v_grad_w)[3777:31558, :],
        np.asarray(velocity_advection.z_v_grad_w)[3777:31558, :],
        atol=1e-16,
    )
    # stencil 08
    assert dallclose(
        np.asarray(savepoint_velocity_exit.z_ekinh())[3316:20896, :],
        np.asarray(velocity_advection.z_ekinh)[3316:20896, :],
    )

    # stencil 11,12,13,14
    assert dallclose(
        np.asarray(savepoint_velocity_exit.z_w_con_c())[3316:20896, :],
        np.asarray(velocity_advection.z_w_con_c)[3316:20896, :],
    )
    # stencil 16
    assert dallclose(
        np.asarray(icon_result_ddt_w_adv_pc)[3316:20896, :],
        np.asarray(diagnostic_state.ddt_w_adv_pc[ntnd - 1])[3316:20896, :],
        atol=5.0e-16,
    )
    # stencil 19 level 0 not verifying
    assert dallclose(
        np.asarray(icon_result_ddt_vn_apc_pc)[5387:31558, 0:65],
        np.asarray(diagnostic_state.ddt_vn_apc_pc[ntnd - 1])[5387:31558, 0:65],
        atol=5.0e-16,
    )
