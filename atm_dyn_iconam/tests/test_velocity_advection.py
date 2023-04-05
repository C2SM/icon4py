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

from icon4py.state_utils.diagnostic_state import DiagnosticState
from icon4py.state_utils.icon_grid import VerticalModelParams
from icon4py.state_utils.interpolation_state import InterpolationState
from icon4py.state_utils.metric_state import MetricState
from icon4py.state_utils.prognostic_state import PrognosticState
from icon4py.velocity.velocity_advection import VelocityAdvection
from icon4py.state_utils.z_fields import ZFields


@pytest.mark.datatest
def test_velocity_init(
    savepoint_velocity_init,
    icon_grid,
    step_date_init,
    damping_height,
):
    savepoint = savepoint_velocity_init

    interpolation_state = InterpolationState(
        e_bln_c_s=None,
        rbf_coeff_1=None,
        rbf_coeff_2=None,
        geofac_div=None,
        geofac_n2s=None,
        geofac_grg_x=None,
        geofac_grg_y=None,
        nudgecoeff_e=None,
        c_lin_e=savepoint.c_lin_e(),
        geofac_grdiv=savepoint.geofac_grdiv(),
        rbf_vec_coeff_e=savepoint.rbf_vec_coeff_e(),
        c_intp=savepoint.c_intp(),
        geofac_rot=savepoint.geofac_rot(),
    )
    metric_state = MetricState(
        mask_hdiff=None,
        theta_ref_mc=None,
        wgtfac_c=None,
        zd_intcoef=None,
        zd_vertidx=None,
        zd_diffcoef=None,
        coeff_gradekin=savepoint.coeff_gradekin(),
        ddqz_z_full_e=savepoint.ddqz_z_full_e(),
        wgtfac_e=savepoint.wgtfac_e(),
        wgtfacq_e=savepoint.wgtfacq_e(),
        ddxn_z_full=savepoint.ddxn_z_full(),
        ddxt_z_full=savepoint.ddxt_z_full(),
        ddqz_z_half=savepoint.ddqz_z_half(),
        coeff1_dwdz=savepoint.coeff1_dwdz(),
        coeff2_dwdz=savepoint.coeff2_dwdz(),
    )

    velocity_advection = VelocityAdvection()
    velocity_advection.init(
        grid=icon_grid,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        vertical_params=VerticalModelParams,
    )

    assert np.allclose(0.0, np.asarray(velocity_advection.cfl_clipping))
    assert np.allclose(0.0, np.asarray(velocity_advection.pre_levelmask))
    assert np.allclose(False, np.asarray(velocity_advection.levelmask))
    assert np.allclose(0.0, np.asarray(velocity_advection.vcfl))

    assert velocity_advection.cfl_w_limit == 0.65
    assert velocity_advection.scalfac_exdiff == 0.05


@pytest.mark.datatest
def test_verify_velocity_init_against_regular_savepoint(
    savepoint_velocity_init, icon_grid, damping_height
):
    savepoint = savepoint_velocity_init
    dtime = savepoint.get_metadata("dtime").get("dtime")

    interpolation_state = InterpolationState(
        e_bln_c_s=None,
        rbf_coeff_1=None,
        rbf_coeff_2=None,
        geofac_div=None,
        geofac_n2s=None,
        geofac_grg_x=None,
        geofac_grg_y=None,
        nudgecoeff_e=None,
        c_lin_e=savepoint.c_lin_e(),
        geofac_grdiv=savepoint.geofac_grdiv(),
        rbf_vec_coeff_e=savepoint.rbf_vec_coeff_e(),
        c_intp=savepoint.c_intp(),
        geofac_rot=savepoint.geofac_rot(),
    )
    metric_state = MetricState(
        mask_hdiff=None,
        theta_ref_mc=None,
        wgtfac_c=None,
        zd_intcoef=None,
        zd_vertidx=None,
        zd_diffcoef=None,
        coeff_gradekin=savepoint.coeff_gradekin(),
        ddqz_z_full_e=savepoint.ddqz_z_full_e(),
        wgtfac_e=savepoint.wgtfac_e(),
        wgtfacq_e=savepoint.wgtfacq_e(),
        ddxn_z_full=savepoint.ddxn_z_full(),
        ddxt_z_full=savepoint.ddxt_z_full(),
        ddqz_z_half=savepoint.ddqz_z_half(),
        coeff1_dwdz=savepoint.coeff1_dwdz(),
        coeff2_dwdz=savepoint.coeff2_dwdz(),
    )

    velocity_advection = VelocityAdvection()
    velocity_advection.init(
        grid=icon_grid,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        vertical_params=VerticalModelParams,
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
    savepoint_velocity_exit,
):
    sp = savepoint_velocity_init
    sp_d = data_provider.from_savepoint_grid()
    vn_only = sp.get_metadata("vn_only").get("vn_only")
    ntnd = sp.get_metadata("ntnd").get("ntnd")
    dtime = sp.get_metadata("dtime").get("dtime")
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
        ddt_w_adv_pc_before=sp.ddt_w_adv_pc_before(),
        ddt_vn_apc_pc_before=sp.ddt_vn_apc_pc_before(),
        ntnd=ntnd,
    )
    prognostic_state = PrognosticState(
        w=sp.w(),
        vn=sp.vn(),
        exner_pressure=None,
        theta_v=None,
    )

    z_fields = ZFields(
        z_w_concorr_me=sp.z_w_concorr_me(),
        z_kin_hor_e=sp.z_kin_hor_e(),
        z_vt_ie=sp.z_vt_ie(),
    )

    interpolation_state = InterpolationState(
        e_bln_c_s=sp.e_bln_c_s(),
        rbf_coeff_1=None,
        rbf_coeff_2=None,
        geofac_div=None,
        geofac_n2s=sp.geofac_n2s(),
        geofac_grg_x=None,
        geofac_grg_y=None,
        nudgecoeff_e=None,
        c_lin_e=sp.c_lin_e(),
        geofac_grdiv=sp.geofac_grdiv(),
        rbf_vec_coeff_e=sp.rbf_vec_coeff_e(),
        c_intp=sp.c_intp(),
        geofac_rot=sp.geofac_rot(),
    )

    metric_state = MetricState(
        mask_hdiff=None,
        theta_ref_mc=None,
        wgtfac_c=sp.wgtfac_c(),
        zd_intcoef=None,
        zd_vertidx=None,
        zd_diffcoef=None,
        coeff_gradekin=sp.coeff_gradekin(),
        ddqz_z_full_e=sp.ddqz_z_full_e(),
        wgtfac_e=sp.wgtfac_e(),
        wgtfacq_e=sp.wgtfacq_e(),
        ddxn_z_full=sp.ddxn_z_full(),
        ddxt_z_full=sp.ddxt_z_full(),
        ddqz_z_half=sp.ddqz_z_half(),
        coeff1_dwdz=sp.coeff1_dwdz(),
        coeff2_dwdz=sp.coeff2_dwdz(),
    )

    orientation = sp_d.tangent_orientation()
    inverse_primal_edge_lengths = sp_d.inverse_primal_edge_lengths()
    inverse_dual_edge_length = sp_d.inv_dual_edge_length()
    edge_areas = sp_d.edge_areas()
    cell_areas = sp_d.cell_areas()

    vertical_params = VerticalModelParams(
        vct_a=grid_savepoint.vct_a(), rayleigh_damping_height=damping_height
    )

    velocity_advection = VelocityAdvection()
    velocity_advection.init(
        grid=icon_grid,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
    )

    velocity_advection.run_predictor_step(
        vn_only=vn_only,
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        z_fields=z_fields,
        inv_dual_edge_length=inverse_dual_edge_length,
        inv_primal_edge_length=inverse_primal_edge_lengths,
        dtime=dtime,
        tangent_orientation=orientation,
        cfl_w_limit=cfl_w_limit,
        scalfac_exdiff=scalfac_exdiff,
        cell_areas=cell_areas,
        owner_mask=sp_d.owner_mask(),
        f_e=sp_d.f_e(),
        area_edge=edge_areas,
    )

    icon_result_ddt_vn_apc_pc = savepoint_velocity_exit.ddt_vn_apc_pc()
    icon_result_ddt_w_adv_pc = savepoint_velocity_exit.ddt_w_adv_pc()
    icon_result_vt = savepoint_velocity_exit.vt()
    icon_result_z_kin_hor_e = savepoint_velocity_exit.z_kin_hor_e()
    icon_result_z_vt_ie = savepoint_velocity_exit.z_vt_ie()
    icon_result_z_w_concorr_me = savepoint_velocity_exit.z_w_concorr_me()

    assert np.allclose(
        np.asarray(icon_result_ddt_vn_apc_pc)[ntnd:, :],
        np.asarray(diagnostic_state.ddt_vn_apc_pc),
    )
    assert np.allclose(
        np.asarray(icon_result_ddt_w_adv_pc)[ntnd:, :],
        np.asarray(diagnostic_state.ddt_w_adv_pc),
    )
    assert np.allclose(np.asarray(icon_result_vt), np.asarray(diagnostic_state.vt))
    assert np.allclose(
        np.asarray(icon_result_z_kin_hor_e), np.asarray(z_fields.z_kin_hor_e)
    )
    assert np.allclose(np.asarray(icon_result_z_vt_ie), np.asarray(z_fields.z_vt_ie))
    assert np.allclose(
        np.asarray(icon_result_z_w_concorr_me), np.asarray(z_fields.z_w_concorr_me)
    )


@pytest.mark.datatest
@pytest.mark.parametrize("istep, jstep", [(2, 0)])
def test_velocity_advector_step(
    damping_height,
    icon_grid,
    grid_savepoint,
    savepoint_velocity_init,
    data_provider,
    savepoint_velocity_exit,
):
    sp = savepoint_velocity_init
    sp_d = data_provider.from_savepoint_grid()
    vn_only = sp.get_metadata("vn_only").get("vn_only")
    ntnd = sp.get_metadata("ntnd").get("ntnd")
    dtime = sp.get_metadata("dtime").get("dtime")
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
        ddt_w_adv_pc_before=sp.ddt_w_adv_pc_before(),
        ddt_vn_apc_pc_before=sp.ddt_vn_apc_pc_before(),
        ntnd=ntnd,
    )
    prognostic_state = PrognosticState(
        w=sp.w(),
        vn=sp.vn(),
        exner_pressure=None,
        theta_v=None,
    )

    z_fields = ZFields(
        z_w_concorr_me=sp.z_w_concorr_me(),
        z_kin_hor_e=sp.z_kin_hor_e(),
        z_vt_ie=sp.z_vt_ie(),
    )

    interpolation_state = InterpolationState(
        e_bln_c_s=sp.e_bln_c_s(),
        rbf_coeff_1=None,
        rbf_coeff_2=None,
        geofac_div=None,
        geofac_n2s=sp.geofac_n2s(),
        geofac_grg_x=None,
        geofac_grg_y=None,
        nudgecoeff_e=None,
        c_lin_e=sp.c_lin_e(),
        geofac_grdiv=sp.geofac_grdiv(),
        rbf_vec_coeff_e=sp.rbf_vec_coeff_e(),
        c_intp=sp.c_intp(),
        geofac_rot=sp.geofac_rot(),
    )

    metric_state = MetricState(
        mask_hdiff=None,
        theta_ref_mc=None,
        wgtfac_c=sp.wgtfac_c(),
        zd_intcoef=None,
        zd_vertidx=None,
        zd_diffcoef=None,
        coeff_gradekin=sp.coeff_gradekin(),
        ddqz_z_full_e=sp.ddqz_z_full_e(),
        wgtfac_e=sp.wgtfac_e(),
        wgtfacq_e=sp.wgtfacq_e(),
        ddxn_z_full=sp.ddxn_z_full(),
        ddxt_z_full=sp.ddxt_z_full(),
        ddqz_z_half=sp.ddqz_z_half(),
        coeff1_dwdz=sp.coeff1_dwdz(),
        coeff2_dwdz=sp.coeff2_dwdz(),
    )

    orientation = sp_d.tangent_orientation()
    inverse_primal_edge_lengths = sp_d.inverse_primal_edge_lengths()
    inverse_dual_edge_length = sp_d.inv_dual_edge_length()
    edge_areas = sp_d.edge_areas()
    cell_areas = sp_d.cell_areas()

    vertical_params = VerticalModelParams(
        vct_a=grid_savepoint.vct_a(), rayleigh_damping_height=damping_height
    )

    velocity_advection = VelocityAdvection()
    velocity_advection.init(
        grid=icon_grid,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
    )

    velocity_advection.run_advector_step(
        vn_only=vn_only,
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        z_fields=z_fields,
        inv_dual_edge_length=inverse_dual_edge_length,
        inv_primal_edge_length=inverse_primal_edge_lengths,
        dtime=dtime,
        tangent_orientation=orientation,
        cfl_w_limit=cfl_w_limit,
        scalfac_exdiff=scalfac_exdiff,
        cell_areas=cell_areas,
        owner_mask=sp_d.owner_mask(),
        f_e=sp_d.f_e(),
        area_edge=edge_areas,
    )

    icon_result_ddt_vn_apc_pc = savepoint_velocity_exit.ddt_vn_apc_pc()
    icon_result_ddt_w_adv_pc = savepoint_velocity_exit.ddt_w_adv_pc()
    icon_result_vn_ie = savepoint_velocity_exit.vn_ie()
    icon_result_vt = savepoint_velocity_exit.vt()
    icon_result_z_kin_hor_e = savepoint_velocity_exit.z_kin_hor_e()
    icon_result_z_vt_ie = savepoint_velocity_exit.z_vt_ie()
    icon_result_z_w_concorr_me = savepoint_velocity_exit.z_w_concorr_me()

    assert np.allclose(
        np.asarray(icon_result_ddt_vn_apc_pc)[ntnd:, :],
        np.asarray(diagnostic_state.ddt_vn_apc_pc),
    )
    assert np.allclose(
        np.asarray(icon_result_ddt_w_adv_pc)[ntnd:, :],
        np.asarray(diagnostic_state.ddt_w_adv_pc),
    )
    assert np.allclose(
        np.asarray(icon_result_vn_ie), np.asarray(diagnostic_state.vn_ie)
    )
    assert np.allclose(np.asarray(icon_result_vt), np.asarray(diagnostic_state.vt))
    assert np.allclose(
        np.asarray(icon_result_z_kin_hor_e), np.asarray(z_fields.z_kin_hor_e), rtol=1e1
    )
    assert np.allclose(np.asarray(icon_result_z_vt_ie), np.asarray(z_fields.z_vt_ie))
    assert np.allclose(
        np.asarray(icon_result_z_w_concorr_me), np.asarray(z_fields.z_w_concorr_me)
    )
