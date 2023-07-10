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

from atm_dyn_iconam.tests.test_utils.helpers import as_1D_sparse_field
from icon4py.common.dimension import CEDim
from icon4py.state_utils.diagnostic_state import DiagnosticState
from icon4py.state_utils.horizontal import CellParams, EdgeParams
from icon4py.state_utils.icon_grid import VerticalModelParams
from icon4py.state_utils.interpolation_state import InterpolationState
from icon4py.state_utils.metric_state import MetricState
from icon4py.state_utils.prognostic_state import PrognosticState
from icon4py.velocity.velocity_advection import VelocityAdvection


@pytest.mark.datatest
def test_velocity_init(
    savepoint_velocity_init,
    interpolation_savepoint,
    metrics_savepoint,
    grid_savepoint,
    icon_grid,
    step_date_init,
    damping_height,
):

    interpolation_state = interpolation_savepoint.construct_interpolation_state()
    metric_state = metrics_savepoint.construct_metric_state()

    vertical_params = VerticalModelParams(
        vct_a=grid_savepoint.vct_a(), rayleigh_damping_height=damping_height
    )

    velocity_advection = VelocityAdvection(
        grid=icon_grid,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
    )

    assert np.allclose(0.0, np.asarray(velocity_advection.cfl_clipping))
    assert np.allclose(0.0, np.asarray(velocity_advection.pre_levelmask))
    assert np.allclose(False, np.asarray(velocity_advection.levelmask))
    assert np.allclose(0.0, np.asarray(velocity_advection.vcfl))

    assert velocity_advection.cfl_w_limit == 0.65
    assert velocity_advection.scalfac_exdiff == 0.05


@pytest.mark.datatest
def test_verify_velocity_init_against_regular_savepoint(
    savepoint_velocity_init,
    interpolation_savepoint,
    grid_savepoint,
    metrics_savepoint,
    icon_grid,
    damping_height,
):
    savepoint = savepoint_velocity_init
    dtime = savepoint.get_metadata("dtime").get("dtime")

    interpolation_state = interpolation_savepoint.construct_interpolation_state()
    metric_state = metrics_savepoint.construct_metric_state()
    vertical_params = VerticalModelParams(
        vct_a=grid_savepoint.vct_a(), rayleigh_damping_height=damping_height
    )

    velocity_advection = VelocityAdvection(
        grid=icon_grid,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
    )
    velocity_advection.init(
        grid=icon_grid,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
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
    damping_height,
    icon_grid,
    grid_savepoint,
    savepoint_velocity_init,
    data_provider,
    metrics_savepoint,
    interpolation_savepoint,
    savepoint_velocity_exit,
):
    sp_v = savepoint_velocity_init
    sp_d = data_provider.from_savepoint_grid()
    vn_only = sp_v.get_metadata("vn_only").get("vn_only")
    ntnd = sp_v.get_metadata("ntnd").get("ntnd")
    dtime = sp_v.get_metadata("dtime").get("dtime")
    scalfac_exdiff = sp_v.scalfac_exdiff()

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
    prognostic_state = PrognosticState(
        w=sp_v.w(),
        vn=sp_v.vn(),
        exner_pressure=None,
        theta_v=None,
        rho=None,
        exner=None,
    )

    interpolation_state = interpolation_savepoint.construct_interpolation_state()
    metric_state = metrics_savepoint.construct_metric_state()

    cell_geometry: CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: EdgeParams = grid_savepoint.construct_edge_geometry()

    vertical_params = VerticalModelParams(
        vct_a=grid_savepoint.vct_a(), rayleigh_damping_height=damping_height
    )

    velocity_advection = VelocityAdvection(
        grid=icon_grid,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
    )

    velocity_advection.run_predictor_step(
        vn_only=vn_only,
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        z_w_concorr_me=sp_v.z_w_concorr_me(),
        z_kin_hor_e=sp_v.z_vt_ie(),
        z_vt_ie=sp_v.z_kin_hor_e(),
        inv_dual_edge_length=edge_geometry.inverse_dual_edge_lengths,
        inv_primal_edge_length=edge_geometry.inverse_primal_edge_lengths,
        dtime=dtime,
        tangent_orientation=edge_geometry.tangent_orientation,
        cfl_w_limit=sp_v.cfl_w_limit(),
        scalfac_exdiff=scalfac_exdiff,
        cell_areas=cell_geometry.area,
        owner_mask=sp_d.owner_mask(),
        f_e=sp_d.f_e(),
        area_edge=edge_geometry.edge_areas,
    )

    # icon_result_ddt_vn_apc_pc = savepoint_velocity_exit.ddt_vn_apc_pc()
    # icon_result_ddt_w_adv_pc = savepoint_velocity_exit.ddt_w_adv_pc()
    # icon_result_max_vcfl_dyn = savepoint_velocity_exit.max_vcfl_dyn()
    icon_result_vn_ie = savepoint_velocity_exit.vn_ie()
    icon_result_vt = savepoint_velocity_exit.vt()
    icon_result_w_concorr_c = savepoint_velocity_exit.w_concorr_c()

    # assert np.allclose(
    #     np.asarray(icon_result_ddt_vn_apc_pc)[ntnd:, :],
    #     np.asarray(diagnostic_state.ddt_vn_apc_pc),
    # )
    # assert np.allclose(
    #     np.asarray(icon_result_ddt_w_adv_pc)[ntnd:, :],
    #     np.asarray(diagnostic_state.ddt_w_adv_pc),
    # )

    assert np.allclose(np.asarray(icon_result_vt), np.asarray(diagnostic_state.vt))

    # for i in range(icon_result_vn_ie.array().shape[0]):
    #     for j in range(icon_result_vn_ie.array().shape[1]):
    #         if not (icon_result_vn_ie.array()[i][j] - diagnostic_state.vn_ie.array()[i][j]) == 0e-5:
    #             print(str(i) + ", " + str(j))
    #             print("")

    assert np.allclose(
        np.asarray(icon_result_vn_ie), np.asarray(diagnostic_state.vn_ie)
    )

    assert np.allclose(
        np.asarray(icon_result_w_concorr_c), np.asarray(diagnostic_state.w_concorr_c)
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "istep, step_date_init, step_date_exit",
    [(2, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000")],
)
def test_velocity_corrector_step(
    damping_height,
    icon_grid,
    grid_savepoint,
    savepoint_velocity_init,
    data_provider,
    savepoint_velocity_exit,
    interpolation_savepoint,
    metrics_savepoint,
):
    sp_v = savepoint_velocity_init
    sp_d = data_provider.from_savepoint_grid()
    vn_only = sp_v.get_metadata("vn_only").get("vn_only")
    ntnd = sp_v.get_metadata("ntnd").get("ntnd")
    dtime = sp_v.get_metadata("dtime").get("dtime")
    cfl_w_limit = sp_v.cfl_w_limit()
    scalfac_exdiff = sp_v.scalfac_exdiff()

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
    prognostic_state = PrognosticState(
        w=sp_v.w(),
        vn=sp_v.vn(),
        exner_pressure=None,
        theta_v=None,
        rho=None,
        exner=None,
    )

    interpolation_state = interpolation_savepoint.construct_interpolation_state()
    metric_state = metrics_savepoint.construct_metric_state()

    cell_geometry: CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: EdgeParams = grid_savepoint.construct_edge_geometry()

    vertical_params = VerticalModelParams(
        vct_a=grid_savepoint.vct_a(), rayleigh_damping_height=damping_height
    )

    velocity_advection = VelocityAdvection(
        grid=icon_grid,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
    )

    velocity_advection.run_corrector_step(
        vn_only,
        diagnostic_state,
        prognostic_state,
        sp_v.z_kin_hor_e(),
        sp_v.z_vt_ie(),
        edge_geometry.inverse_dual_edge_lengths,
        edge_geometry.inverse_primal_edge_lengths,
        dtime,
        edge_geometry.tangent_orientation,
        cfl_w_limit,
        scalfac_exdiff,
        cell_geometry.area,
        sp_d.owner_mask(),
        sp_d.f_e(),
        edge_geometry.edge_areas,
    )
