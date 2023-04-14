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

from icon4py.common.dimension import ECVDim, KDim, VertexDim
from icon4py.diffusion.diffusion import Diffusion, DiffusionParams, VectorTuple
from icon4py.state_utils.diagnostic_state import DiagnosticState
from icon4py.state_utils.icon_grid import VerticalModelParams
from icon4py.state_utils.interpolation_state import InterpolationState
from icon4py.state_utils.metric_state import MetricState
from icon4py.state_utils.prognostic_state import PrognosticState
from icon4py.state_utils.utils import (
    _en_smag_fac_for_zero_nshift,
    _setup_runtime_diff_multfac_vn,
    _setup_smag_limit,
    scale_k,
    set_zero_v_k,
    setup_fields_for_initial_step,
)
from icon4py.testutils.serialbox_utils import IconDiffusionInitSavepoint
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import as_1D_sparse_field, random_field, zero_field


def test_scale_k():
    mesh = SimpleMesh()
    field = random_field(mesh, KDim)
    scaled_field = zero_field(mesh, KDim)
    factor = 2.0
    scale_k(field, factor, scaled_field, offset_provider={})
    assert np.allclose(factor * np.asarray(field), scaled_field)


def initial_diff_multfac_vn_numpy(shape, k4, hdiff_efdt_ratio):
    return k4 * hdiff_efdt_ratio / 3.0 * np.ones(shape)


def smag_limit_numpy(func, *args):
    return 0.125 - 4.0 * func(*args)


def test_diff_multfac_vn_and_smag_limit_for_initial_step():
    mesh = SimpleMesh()
    diff_multfac_vn_init = zero_field(mesh, KDim)
    smag_limit_init = zero_field(mesh, KDim)
    k4 = 1.0
    efdt_ratio = 24.0
    shape = np.asarray(diff_multfac_vn_init).shape

    expected_diff_multfac_vn_init = initial_diff_multfac_vn_numpy(shape, k4, efdt_ratio)
    expected_smag_limit_init = smag_limit_numpy(
        initial_diff_multfac_vn_numpy, shape, k4, efdt_ratio
    )

    setup_fields_for_initial_step(
        k4, efdt_ratio, diff_multfac_vn_init, smag_limit_init, offset_provider={}
    )

    assert np.allclose(expected_diff_multfac_vn_init, diff_multfac_vn_init)
    assert np.allclose(expected_smag_limit_init, smag_limit_init)


def test_diff_multfac_vn_smag_limit_for_time_step_with_const_value():
    mesh = SimpleMesh()
    diff_multfac_vn = zero_field(mesh, KDim)
    smag_limit = zero_field(mesh, KDim)
    k4 = 1.0
    substeps = 5.0
    efdt_ratio = 24.0
    shape = np.asarray(diff_multfac_vn).shape

    expected_diff_multfac_vn = diff_multfac_vn_numpy(shape, k4, substeps)
    expected_smag_limit = smag_limit_numpy(diff_multfac_vn_numpy, shape, k4, substeps)

    _setup_runtime_diff_multfac_vn(
        k4, efdt_ratio, out=diff_multfac_vn, offset_provider={}
    )
    _setup_smag_limit(diff_multfac_vn, out=smag_limit, offset_provider={})

    assert np.allclose(expected_diff_multfac_vn, diff_multfac_vn)
    assert np.allclose(expected_smag_limit, smag_limit)


def test_diff_multfac_vn_smag_limit_for_loop_run_with_k4_substeps():
    mesh = SimpleMesh()
    diff_multfac_vn = zero_field(mesh, KDim)
    smag_limit = zero_field(mesh, KDim)
    k4 = 0.003
    substeps = 1.0

    shape = np.asarray(diff_multfac_vn).shape
    expected_diff_multfac_vn = diff_multfac_vn_numpy(shape, k4, substeps)
    expected_smag_limit = smag_limit_numpy(diff_multfac_vn_numpy, shape, k4, substeps)
    _setup_runtime_diff_multfac_vn(
        k4, substeps, out=diff_multfac_vn, offset_provider={}
    )
    _setup_smag_limit(diff_multfac_vn, out=smag_limit, offset_provider={})

    assert np.allclose(expected_diff_multfac_vn, diff_multfac_vn)
    assert np.allclose(expected_smag_limit, smag_limit)


def test_init_enh_smag_fac():
    mesh = SimpleMesh()
    enh_smag_fac = zero_field(mesh, KDim)
    a_vec = random_field(mesh, KDim, low=1.0, high=10.0, extend={KDim: 1})
    fac = (0.67, 0.5, 1.3, 0.8)
    z = (0.1, 0.2, 0.3, 0.4)

    enhanced_smag_fac_np = enhanced_smagorinski_factor_numpy(fac, z, np.asarray(a_vec))

    _en_smag_fac_for_zero_nshift(
        a_vec, *fac, *z, out=enh_smag_fac, offset_provider={"Koff": KDim}
    )
    assert np.allclose(enhanced_smag_fac_np, np.asarray(enh_smag_fac))


def diff_multfac_vn_numpy(shape, k4, substeps):
    factor = min(1.0 / 128.0, k4 * substeps / 3.0)
    return factor * np.ones(shape)


def enhanced_smagorinski_factor_numpy(factor_in, heigths_in, a_vec):
    alin = (factor_in[1] - factor_in[0]) / (heigths_in[1] - heigths_in[0])
    df32 = factor_in[2] - factor_in[1]
    df42 = factor_in[3] - factor_in[1]
    dz32 = heigths_in[2] - heigths_in[1]
    dz42 = heigths_in[3] - heigths_in[1]
    bqdr = (df42 * dz32 - df32 * dz42) / (dz32 * dz42 * (dz42 - dz32))
    aqdr = df32 / dz32 - bqdr * dz32
    zf = 0.5 * (a_vec[:-1] + a_vec[1:])
    max0 = np.maximum(0.0, zf - heigths_in[0])
    dzlin = np.minimum(heigths_in[1] - heigths_in[0], max0)
    max1 = np.maximum(0.0, zf - heigths_in[1])
    dzqdr = np.minimum(heigths_in[3] - heigths_in[1], max1)
    return factor_in[0] + dzlin * alin + dzqdr * (aqdr + dzqdr * bqdr)


def test_set_zero_vertex_k():
    mesh = SimpleMesh()
    f = random_field(mesh, VertexDim, KDim)
    set_zero_v_k(f, offset_provider={})
    assert np.allclose(0.0, f)


@pytest.mark.datatest
def test_diffusion_coefficients_with_hdiff_efdt_ratio(r04b09_diffusion_config):
    config = r04b09_diffusion_config
    config.hdiff_efdt_ratio = 1.0
    config.hdiff_w_efdt_ratio = 2.0

    params = DiffusionParams(config)

    assert params.K2 == pytest.approx(0.125, abs=1e-12)
    assert params.K4 == pytest.approx(0.125 / 8.0, abs=1e-12)
    assert params.K6 == pytest.approx(0.125 / 64.0, abs=1e-12)
    assert params.K4W == pytest.approx(1.0 / 72.0, abs=1e-12)


@pytest.mark.datatest
def test_diffusion_coefficients_without_hdiff_efdt_ratio(r04b09_diffusion_config):
    config = r04b09_diffusion_config
    config.hdiff_efdt_ratio = 0.0
    config.hdiff_w_efdt_ratio = 0.0

    params = DiffusionParams(config)

    assert params.K2 == 0.0
    assert params.K4 == 0.0
    assert params.K6 == 0.0
    assert params.K4W == 0.0


@pytest.mark.datatest
def test_smagorinski_factor_for_diffusion_type_4(r04b09_diffusion_config):
    config = r04b09_diffusion_config
    config.smagorinski_scaling_factor = 0.15
    config.diffusion_type = 4

    params = DiffusionParams(config)
    assert len(params.smagorinski_factor) == 1
    assert params.smagorinski_factor[0] == pytest.approx(0.15, abs=1e-16)
    assert params.smagorinski_height is None


def test_smagorinski_heights_diffusion_type_5_are_consistent(
    r04b09_diffusion_config,
):
    config = r04b09_diffusion_config
    config.smagorinski_scaling_factor = 0.15
    config.diffusion_type = 5

    params = DiffusionParams(config)
    assert len(params.smagorinski_height) == 4
    assert min(params.smagorinski_height) == params.smagorinski_height[0]
    assert max(params.smagorinski_height) == params.smagorinski_height[-1]
    assert params.smagorinski_height[0] < params.smagorinski_height[1]
    assert params.smagorinski_height[1] < params.smagorinski_height[3]
    assert params.smagorinski_height[2] != params.smagorinski_height[1]
    assert params.smagorinski_height[2] != params.smagorinski_height[3]


def test_smagorinski_factor_diffusion_type_5(r04b09_diffusion_config):
    config = r04b09_diffusion_config
    params = DiffusionParams(config)
    assert len(params.smagorinski_factor) == len(params.smagorinski_height)
    assert len(params.smagorinski_factor) == 4
    assert np.all(params.smagorinski_factor >= np.zeros(len(params.smagorinski_factor)))


@pytest.mark.datatest
def test_diffusion_init(
    diffusion_savepoint_init,
    grid_savepoint,
    icon_grid,
    r04b09_diffusion_config,
    step_date_init,
    damping_height,
):
    savepoint = diffusion_savepoint_init
    config = r04b09_diffusion_config
    additional_parameters = DiffusionParams(config)
    vertical_params = VerticalModelParams(grid_savepoint.vct_a(), damping_height)

    meta = savepoint.get_metadata("nlev", "linit", "date")

    assert meta["nlev"] == 65
    assert meta["linit"] is False
    assert meta["date"] == step_date_init

    grg = savepoint.geofac_grg()
    interpolation_state = InterpolationState(
        e_bln_c_s=savepoint.e_bln_c_s(),
        rbf_coeff_1=savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=savepoint.rbf_vec_coeff_v2(),
        geofac_div=savepoint.geofac_div(),
        geofac_n2s=savepoint.geofac_n2s(),
        geofac_grg_x=grg[0],
        geofac_grg_y=grg[1],
        nudgecoeff_e=savepoint.nudgecoeff_e(),
        c_lin_e=None,
        geofac_grdiv=None,
        rbf_vec_coeff_e=None,
        c_intp=None,
        geofac_rot=None,
        pos_on_tplane_e=None,
        e_flx_avg=None,
    )
    metric_state = MetricState(
        mask_hdiff=savepoint.mask_diff(),
        theta_ref_mc=savepoint.theta_ref_mc(),
        wgtfac_c=savepoint.wgtfac_c(),
        zd_intcoef=savepoint.zd_intcoef(),
        zd_vertidx=savepoint.zd_vertidx(),
        zd_diffcoef=savepoint.zd_diffcoef(),
        coeff_gradekin=None,
        ddqz_z_full_e=None,
        wgtfac_e=None,
        wgtfacq_e=None,
        ddxn_z_full=None,
        ddxt_z_full=None,
        ddqz_z_half=None,
        coeff1_dwdz=None,
        coeff2_dwdz=None,
    )

    diffusion = Diffusion()
    diffusion.init(
        grid=icon_grid,
        vertical_params=vertical_params,
        config=config,
        params=additional_parameters,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
    )

    assert diffusion.diff_multfac_w == min(
        1.0 / 48.0, additional_parameters.K4W * config.substep_as_float()
    )

    assert np.allclose(0.0, np.asarray(diffusion.v_vert))
    assert np.allclose(0.0, np.asarray(diffusion.u_vert))
    assert np.allclose(0.0, np.asarray(diffusion.kh_smag_ec))
    assert np.allclose(0.0, np.asarray(diffusion.kh_smag_e))

    shape_k = (icon_grid.n_lev(),)
    expected_smag_limit = smag_limit_numpy(
        diff_multfac_vn_numpy,
        shape_k,
        additional_parameters.K4,
        config.substep_as_float(),
    )

    assert (
        diffusion.smag_offset
        == 0.25 * additional_parameters.K4 * config.substep_as_float()
    )
    assert np.allclose(expected_smag_limit, diffusion.smag_limit)

    expected_diff_multfac_vn = diff_multfac_vn_numpy(
        shape_k, additional_parameters.K4, config.substep_as_float()
    )
    assert np.allclose(expected_diff_multfac_vn, diffusion.diff_multfac_vn)
    expected_enh_smag_fac = enhanced_smagorinski_factor_numpy(
        additional_parameters.smagorinski_factor,
        additional_parameters.smagorinski_height,
        grid_savepoint.vct_a(),
    )
    assert np.allclose(expected_enh_smag_fac, np.asarray(diffusion.enh_smag_fac))


def _verify_init_values_against_savepoint(
    savepoint: IconDiffusionInitSavepoint, diffusion: Diffusion
):
    dtime = savepoint.get_metadata("dtime")["dtime"]

    assert savepoint.nudgezone_diff() == diffusion.nudgezone_diff
    assert savepoint.bdy_diff() == diffusion.bdy_diff
    assert savepoint.fac_bdydiff_v() == diffusion.fac_bdydiff_v
    assert savepoint.smag_offset() == diffusion.smag_offset
    assert savepoint.diff_multfac_w() == diffusion.diff_multfac_w

    # this is done in diffusion.run(...) because it depends on the dtime
    scale_k(
        diffusion.enh_smag_fac, dtime, diffusion.diff_multfac_smag, offset_provider={}
    )
    assert np.allclose(savepoint.diff_multfac_smag(), diffusion.diff_multfac_smag)

    assert np.allclose(savepoint.smag_limit(), diffusion.smag_limit)
    assert np.allclose(
        savepoint.diff_multfac_n2w(), np.asarray(diffusion.diff_multfac_n2w)
    )
    assert np.allclose(savepoint.diff_multfac_vn(), diffusion.diff_multfac_vn)


@pytest.mark.datatest
@pytest.mark.parametrize("linit", [True])
def test_verify_special_diffusion_inital_step_values_against_initial_savepoint(
    diffusion_savepoint_init, r04b09_diffusion_config, icon_grid
):
    savepoint = diffusion_savepoint_init
    config = r04b09_diffusion_config

    params = DiffusionParams(config)
    expected_diff_multfac_vn = savepoint.diff_multfac_vn()
    expected_smag_limit = savepoint.smag_limit()
    exptected_smag_offset = savepoint.smag_offset()

    diff_multfac_vn = zero_field(icon_grid, KDim)
    smag_limit = zero_field(icon_grid, KDim)
    setup_fields_for_initial_step(
        params.K4,
        config.hdiff_efdt_ratio,
        diff_multfac_vn,
        smag_limit,
        offset_provider={},
    )
    assert np.allclose(expected_smag_limit, smag_limit)
    assert np.allclose(expected_diff_multfac_vn, diff_multfac_vn)
    assert exptected_smag_offset == 0.0


@pytest.mark.datatest
def test_verify_diffusion_init_against_first_regular_savepoint(
    diffusion_savepoint_init,
    grid_savepoint,
    r04b09_diffusion_config,
    icon_grid,
    damping_height,
):
    config = r04b09_diffusion_config
    additional_parameters = DiffusionParams(config)

    savepoint = diffusion_savepoint_init
    vct_a = grid_savepoint.vct_a()

    grg = savepoint.geofac_grg()
    interpolation_state = InterpolationState(
        e_bln_c_s=savepoint.e_bln_c_s(),
        rbf_coeff_1=savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=savepoint.rbf_vec_coeff_v2(),
        geofac_div=savepoint.geofac_div(),
        geofac_n2s=savepoint.geofac_n2s(),
        geofac_grg_x=grg[0],
        geofac_grg_y=grg[1],
        nudgecoeff_e=savepoint.nudgecoeff_e(),
        c_lin_e=None,
        geofac_grdiv=None,
        rbf_vec_coeff_e=None,
        c_intp=None,
        geofac_rot=None,
        pos_on_tplane_e=None,
        e_flx_avg=None,
    )
    metric_state = MetricState(
        mask_hdiff=savepoint.mask_diff(),
        theta_ref_mc=savepoint.theta_ref_mc(),
        wgtfac_c=savepoint.wgtfac_c(),
        zd_intcoef=savepoint.zd_intcoef(),
        zd_vertidx=savepoint.zd_vertoffset(),
        zd_diffcoef=savepoint.zd_diffcoef(),
        coeff_gradekin=None,
        ddqz_z_full_e=None,
        wgtfac_e=None,
        wgtfacq_e=None,
        ddxn_z_full=None,
        ddxt_z_full=None,
        ddqz_z_half=None,
        coeff1_dwdz=None,
        coeff2_dwdz=None,
    )
    diffusion = Diffusion()
    diffusion.init(
        config=config,
        grid=icon_grid,
        params=additional_parameters,
        vertical_params=VerticalModelParams(vct_a, damping_height),
        metric_state=metric_state,
        interpolation_state=interpolation_state,
    )

    _verify_init_values_against_savepoint(savepoint, diffusion)


@pytest.mark.datatest
@pytest.mark.parametrize("step_date_init", ["2021-06-20T12:00:50.000"])
def test_verify_diffusion_init_against_other_regular_savepoint(
    r04b09_diffusion_config,
    grid_savepoint,
    icon_grid,
    diffusion_savepoint_init,
    damping_height,
):
    config = r04b09_diffusion_config
    additional_parameters = DiffusionParams(config)

    savepoint = diffusion_savepoint_init
    vertical_params = VerticalModelParams(grid_savepoint.vct_a(), damping_height)
    grg = diffusion_savepoint_init.geofac_grg()
    interpolation_state = InterpolationState(
        e_bln_c_s=savepoint.e_bln_c_s(),
        rbf_coeff_1=savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=savepoint.rbf_vec_coeff_v2(),
        geofac_div=savepoint.geofac_div(),
        geofac_n2s=savepoint.geofac_n2s(),
        geofac_grg_x=grg[0],
        geofac_grg_y=grg[1],
        nudgecoeff_e=savepoint.nudgecoeff_e(),
        c_lin_e=None,
        geofac_grdiv=None,
        rbf_vec_coeff_e=None,
        c_intp=None,
        geofac_rot=None,
        pos_on_tplane_e=None,
        e_flx_avg=None,
    )
    metric_state = MetricState(
        mask_hdiff=savepoint.mask_diff(),
        theta_ref_mc=savepoint.theta_ref_mc(),
        wgtfac_c=savepoint.wgtfac_c(),
        zd_intcoef=savepoint.zd_intcoef(),
        zd_vertidx=savepoint.zd_vertidx(),
        zd_diffcoef=savepoint.zd_diffcoef(),
        coeff_gradekin=None,
        ddqz_z_full_e=None,
        wgtfac_e=None,
        wgtfacq_e=None,
        ddxn_z_full=None,
        ddxt_z_full=None,
        ddqz_z_half=None,
        coeff1_dwdz=None,
        coeff2_dwdz=None,
    )

    diffusion = Diffusion()
    diffusion.init(
        icon_grid,
        config,
        additional_parameters,
        vertical_params,
        metric_state,
        interpolation_state,
    )

    _verify_init_values_against_savepoint(diffusion_savepoint_init, diffusion)


@pytest.mark.skip("fix: diffusion_stencil_15")
@pytest.mark.parametrize("run_with_program", [True, False])
@pytest.mark.datatest
def test_run_diffusion_single_step(
    run_with_program,
    diffusion_savepoint_init,
    diffusion_savepoint_exit,
    grid_savepoint,
    icon_grid,
    r04b09_diffusion_config,
    damping_height,
):
    (
        cell_areas,
        diagnostic_state,
        dtime,
        dual_normal_vert,
        edge_areas,
        interpolation_state,
        inverse_dual_edge_length,
        inverse_primal_edge_lengths,
        inverse_vertical_vertex_lengths,
        metric_state,
        orientation,
        primal_normal_vert,
        prognostic_state,
    ) = _read_fields(diffusion_savepoint_init, grid_savepoint)

    vct_a = grid_savepoint.vct_a()
    vertical_params = VerticalModelParams(
        vct_a=vct_a, rayleigh_damping_height=damping_height
    )
    additional_parameters = DiffusionParams(r04b09_diffusion_config)

    diffusion = Diffusion(run_program=run_with_program)

    diffusion.init(
        grid=icon_grid,
        config=r04b09_diffusion_config,
        params=additional_parameters,
        vertical_params=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
    )
    diffusion.time_step(
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        dtime=dtime,
        tangent_orientation=orientation,
        inverse_primal_edge_lengths=inverse_primal_edge_lengths,
        inverse_dual_edge_length=inverse_dual_edge_length,
        inverse_vert_vert_lengths=inverse_vertical_vertex_lengths,
        primal_normal_vert=primal_normal_vert,
        dual_normal_vert=dual_normal_vert,
        edge_areas=edge_areas,
        cell_areas=cell_areas,
    )

    icon_result_exner = diffusion_savepoint_exit.exner()
    icon_result_vn = diffusion_savepoint_exit.vn()
    icon_result_w = diffusion_savepoint_exit.w()
    icon_result_theta_w = diffusion_savepoint_exit.theta_v()

    assert np.allclose(icon_result_w, np.asarray(prognostic_state.w))
    assert np.allclose(np.asarray(icon_result_vn), np.asarray(prognostic_state.vn))
    assert np.allclose(
        np.asarray(icon_result_theta_w), np.asarray(prognostic_state.theta_v)
    )
    assert np.allclose(
        np.asarray(icon_result_exner), np.asarray(prognostic_state.exner_pressure)
    )


def _read_fields(diffusion_savepoint_init, grid_savepoint):

    grg = diffusion_savepoint_init.geofac_grg()
    interpolation_state = InterpolationState(
        e_bln_c_s=diffusion_savepoint_init.e_bln_c_s(),
        rbf_coeff_1=diffusion_savepoint_init.rbf_vec_coeff_v1(),
        rbf_coeff_2=diffusion_savepoint_init.rbf_vec_coeff_v2(),
        geofac_div=diffusion_savepoint_init.geofac_div(),
        geofac_n2s=diffusion_savepoint_init.geofac_n2s(),
        geofac_grg_x=grg[0],
        geofac_grg_y=grg[1],
        nudgecoeff_e=diffusion_savepoint_init.nudgecoeff_e(),
        c_lin_e=None,
        geofac_grdiv=None,
        rbf_vec_coeff_e=None,
        c_intp=None,
        geofac_rot=None,
        pos_on_tplane_e=None,
        e_flx_avg=None,
    )
    metric_state = MetricState(
        mask_hdiff=diffusion_savepoint_init.mask_diff(),
        theta_ref_mc=diffusion_savepoint_init.theta_ref_mc(),
        wgtfac_c=diffusion_savepoint_init.wgtfac_c(),
        zd_intcoef=diffusion_savepoint_init.zd_intcoef(),
        zd_vertidx=diffusion_savepoint_init.zd_vertoffset(),
        zd_diffcoef=diffusion_savepoint_init.zd_diffcoef(),
        coeff_gradekin=None,
        ddqz_z_full_e=None,
        wgtfac_e=None,
        wgtfacq_e=None,
        ddxn_z_full=None,
        ddxt_z_full=None,
        ddqz_z_half=None,
        coeff1_dwdz=None,
        coeff2_dwdz=None,
    )
    diagnostic_state = DiagnosticState(
        hdef_ic=diffusion_savepoint_init.hdef_ic(),
        div_ic=diffusion_savepoint_init.div_ic(),
        dwdx=diffusion_savepoint_init.dwdx(),
        dwdy=diffusion_savepoint_init.dwdy(),
        vt=None,
        vn_ie=None,
        w_concorr_c=None,
        ddt_w_adv_pc_before=None,
        ddt_vn_apc_pc_before=None,
        ntnd=None,
    )
    prognostic_state = PrognosticState(
        w=diffusion_savepoint_init.w(),
        vn=diffusion_savepoint_init.vn(),
        exner_pressure=diffusion_savepoint_init.exner(),
        theta_v=diffusion_savepoint_init.theta_v(),
        rho=None,
        exner=None,
    )
    dtime = diffusion_savepoint_init.get_metadata("dtime").get("dtime")
    orientation = grid_savepoint.tangent_orientation()
    inverse_primal_edge_lengths = grid_savepoint.inverse_primal_edge_lengths()
    inverse_vert_vert_lengths = grid_savepoint.inv_vert_vert_length()
    inverse_dual_edge_length = grid_savepoint.inv_dual_edge_length()
    primal_normal_vert: VectorTuple = (
        as_1D_sparse_field(grid_savepoint.primal_normal_vert_x(), ECVDim),
        as_1D_sparse_field(grid_savepoint.primal_normal_vert_y(), ECVDim),
    )
    dual_normal_vert: VectorTuple = (
        as_1D_sparse_field(grid_savepoint.dual_normal_vert_x(), ECVDim),
        as_1D_sparse_field(grid_savepoint.dual_normal_vert_y(), ECVDim),
    )
    edge_areas = grid_savepoint.edge_areas()
    cell_areas = grid_savepoint.cell_areas()
    return (
        cell_areas,
        diagnostic_state,
        dtime,
        dual_normal_vert,
        edge_areas,
        interpolation_state,
        inverse_dual_edge_length,
        inverse_primal_edge_lengths,
        inverse_vert_vert_lengths,
        metric_state,
        orientation,
        primal_normal_vert,
        prognostic_state,
    )


@pytest.mark.skip("fix: diffusion_stencil_15")
@pytest.mark.datatest
def test_diffusion_five_steps(
    damping_height,
    r04b09_diffusion_config,
    icon_grid,
    grid_savepoint,
    diffusion_savepoint_init,
    diffusion_savepoint_exit,
    linit=True,
    step_date_exit="2021-06-20T12:01:00.000",
):
    (
        cell_areas,
        diagnostic_state,
        dtime,
        dual_normal_vert,
        edge_areas,
        interpolation_state,
        inverse_dual_edge_length,
        inverse_primal_edge_lengths,
        inverse_vert_vert_lengths,
        metric_state,
        orientation,
        primal_normal_vert,
        prognostic_state,
    ) = _read_fields(diffusion_savepoint_init, grid_savepoint)

    vertical_params = VerticalModelParams(
        vct_a=grid_savepoint.vct_a(), rayleigh_damping_height=damping_height
    )

    additional_parameters = DiffusionParams(r04b09_diffusion_config)

    diffusion = Diffusion()
    diffusion.init(
        grid=icon_grid,
        config=r04b09_diffusion_config,
        params=additional_parameters,
        vertical_params=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
    )
    diffusion.initial_step(
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        dtime=dtime,
        tangent_orientation=orientation,
        inverse_primal_edge_lengths=inverse_primal_edge_lengths,
        inverse_dual_edge_length=inverse_dual_edge_length,
        inverse_vert_vert_lengths=inverse_vert_vert_lengths,
        primal_normal_vert=primal_normal_vert,
        dual_normal_vert=dual_normal_vert,
        edge_areas=edge_areas,
        cell_areas=cell_areas,
    )
    for _ in range(4):
        diffusion.time_step(
            diagnostic_state=diagnostic_state,
            prognostic_state=prognostic_state,
            dtime=dtime,
            tangent_orientation=orientation,
            inverse_primal_edge_lengths=inverse_primal_edge_lengths,
            inverse_dual_edge_length=inverse_dual_edge_length,
            inverse_vert_vert_lengths=inverse_vert_vert_lengths,
            primal_normal_vert=primal_normal_vert,
            dual_normal_vert=dual_normal_vert,
            edge_areas=edge_areas,
            cell_areas=cell_areas,
        )

    icon_result_exner = diffusion_savepoint_exit.exner()
    icon_result_vn = diffusion_savepoint_exit.vn()
    icon_result_w = diffusion_savepoint_exit.w()
    icon_result_theta_w = diffusion_savepoint_exit.theta_v()
    assert np.allclose(icon_result_w, np.asarray(prognostic_state.w))
    assert np.allclose(np.asarray(icon_result_vn), np.asarray(prognostic_state.vn))
    assert np.allclose(
        np.asarray(icon_result_theta_w), np.asarray(prognostic_state.theta_v)
    )
    assert np.allclose(
        np.asarray(icon_result_exner), np.asarray(prognostic_state.exner_pressure)
    )
