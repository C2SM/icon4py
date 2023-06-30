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

from atm_dyn_iconam.tests.test_utils.serialbox_utils import (
    IconDiffusionInitSavepoint,
)
from icon4py.common.dimension import KDim, VertexDim
from icon4py.diffusion.diffusion import Diffusion, DiffusionParams
from icon4py.diffusion.horizontal import CellParams, EdgeParams
from icon4py.diffusion.icon_grid import VerticalModelParams
from icon4py.diffusion.utils import (
    _en_smag_fac_for_zero_nshift,
    _setup_runtime_diff_multfac_vn,
    _setup_smag_limit,
    scale_k,
    set_zero_v_k,
    setup_fields_for_initial_step,
)

from .test_utils.helpers import random_field, zero_field
from .test_utils.simple_mesh import SimpleMesh


datarun_reduced_substeps = 2


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


def test_diffusion_coefficients_with_hdiff_efdt_ratio(r04b09_diffusion_config):
    config = r04b09_diffusion_config
    config.hdiff_efdt_ratio = 1.0
    config.hdiff_w_efdt_ratio = 2.0

    params = DiffusionParams(config)

    assert params.K2 == pytest.approx(0.125, abs=1e-12)
    assert params.K4 == pytest.approx(0.125 / 8.0, abs=1e-12)
    assert params.K6 == pytest.approx(0.125 / 64.0, abs=1e-12)
    assert params.K4W == pytest.approx(1.0 / 72.0, abs=1e-12)


def test_diffusion_coefficients_without_hdiff_efdt_ratio(r04b09_diffusion_config):
    config = r04b09_diffusion_config
    config.hdiff_efdt_ratio = 0.0
    config.hdiff_w_efdt_ratio = 0.0

    params = DiffusionParams(config)

    assert params.K2 == 0.0
    assert params.K4 == 0.0
    assert params.K6 == 0.0
    assert params.K4W == 0.0


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
    params = DiffusionParams(r04b09_diffusion_config)
    assert len(params.smagorinski_factor) == len(params.smagorinski_height)
    assert len(params.smagorinski_factor) == 4
    assert np.all(params.smagorinski_factor >= np.zeros(len(params.smagorinski_factor)))


@pytest.mark.datatest
def test_diffusion_init(
    diffusion_savepoint_init,
    interpolation_savepoint,
    metrics_savepoint,
    grid_savepoint,
    icon_grid,
    r04b09_diffusion_config,
    step_date_init,
    damping_height,
):
    config = r04b09_diffusion_config
    config.ndyn_substeps = datarun_reduced_substeps
    additional_parameters = DiffusionParams(config)
    vertical_params = VerticalModelParams(grid_savepoint.vct_a(), damping_height)

    meta = diffusion_savepoint_init.get_metadata("nlev", "linit", "date")

    assert meta["nlev"] == 65
    assert meta["linit"] is False
    assert meta["date"] == step_date_init

    interpolation_state = interpolation_savepoint.construct_interpolation_state()
    metric_state = metrics_savepoint.construct_metric_state()
    edge_params = grid_savepoint.construct_edge_geometry()
    cell_params = grid_savepoint.construct_cell_geometry()

    diffusion = Diffusion()
    diffusion.init(
        grid=icon_grid,
        config=config,
        params=additional_parameters,
        vertical_params=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_params,
        cell_params=cell_params,
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
    config.ndyn_substeps = datarun_reduced_substeps

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
    interpolation_savepoint,
    metrics_savepoint,
    grid_savepoint,
    r04b09_diffusion_config,
    icon_grid,
    damping_height,
):
    config = r04b09_diffusion_config
    config.ndyn_substeps = datarun_reduced_substeps
    additional_parameters = DiffusionParams(config)
    vct_a = grid_savepoint.vct_a()
    cell_geometry = grid_savepoint.construct_cell_geometry()
    edge_geometry = grid_savepoint.construct_edge_geometry()

    interpolation_state = interpolation_savepoint.construct_interpolation_state()
    metric_state = metrics_savepoint.construct_metric_state()

    diffusion = Diffusion()
    diffusion.init(
        grid=icon_grid,
        config=config,
        params=additional_parameters,
        vertical_params=VerticalModelParams(vct_a, damping_height),
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
    )

    _verify_init_values_against_savepoint(diffusion_savepoint_init, diffusion)


@pytest.mark.datatest
@pytest.mark.parametrize("step_date_init", ["2021-06-20T12:00:50.000"])
def test_verify_diffusion_init_against_other_regular_savepoint(
    r04b09_diffusion_config,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    metrics_savepoint,
    diffusion_savepoint_init,
    damping_height,
):
    config = r04b09_diffusion_config
    config.ndyn_substeps = datarun_reduced_substeps
    additional_parameters = DiffusionParams(config)

    vertical_params = VerticalModelParams(grid_savepoint.vct_a(), damping_height)
    interpolation_state = interpolation_savepoint.construct_interpolation_state()
    metric_state = metrics_savepoint.construct_metric_state()
    edge_params = grid_savepoint.construct_edge_geometry()
    cell_params = grid_savepoint.construct_cell_geometry()

    diffusion = Diffusion()
    diffusion.init(
        icon_grid,
        config,
        additional_parameters,
        vertical_params,
        metric_state,
        interpolation_state,
        edge_params,
        cell_params,
    )

    _verify_init_values_against_savepoint(diffusion_savepoint_init, diffusion)


@pytest.mark.datatest
@pytest.mark.parametrize(
    "step_date_init, step_date_exit",
    [
        ("2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000"),
        ("2021-06-20T12:00:20.000", "2021-06-20T12:00:20.000"),
        ("2021-06-20T12:01:00.000", "2021-06-20T12:01:00.000"),
    ],
)
def test_run_diffusion_single_step(
    diffusion_savepoint_init,
    diffusion_savepoint_exit,
    interpolation_savepoint,
    metrics_savepoint,
    grid_savepoint,
    icon_grid,
    r04b09_diffusion_config,
    damping_height,
):
    dtime = diffusion_savepoint_init.get_metadata("dtime").get("dtime")
    edge_geometry: EdgeParams = grid_savepoint.construct_edge_geometry()
    cell_geometry: CellParams = grid_savepoint.construct_cell_geometry()
    interpolation_state = interpolation_savepoint.construct_interpolation_state()
    metric_state = metrics_savepoint.construct_metric_state()
    diagnostic_state = diffusion_savepoint_init.construct_diagnostics()
    prognostic_state = diffusion_savepoint_init.construct_prognostics()
    vct_a = grid_savepoint.vct_a()
    vertical_params = VerticalModelParams(
        vct_a=vct_a, rayleigh_damping_height=damping_height
    )
    config = r04b09_diffusion_config
    config.ndyn_substeps = datarun_reduced_substeps
    additional_parameters = DiffusionParams(config)

    diffusion = Diffusion()
    diffusion.init(
        grid=icon_grid,
        config=config,
        params=additional_parameters,
        vertical_params=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
    )
    _verify_diffusion_fields(
        diagnostic_state, prognostic_state, diffusion_savepoint_init
    )
    assert diffusion_savepoint_init.fac_bdydiff_v() == diffusion.fac_bdydiff_v
    diffusion.run(
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        dtime=dtime,
    )
    _verify_diffusion_fields(
        diagnostic_state, prognostic_state, diffusion_savepoint_exit
    )


def _verify_diffusion_fields(
    diagnostic_state, prognostic_state, diffusion_savepoint_exit
):
    ref_div_ic = np.asarray(diffusion_savepoint_exit.div_ic())
    val_div_ic = np.asarray(diagnostic_state.div_ic)
    ref_hdef_ic = np.asarray(diffusion_savepoint_exit.hdef_ic())
    val_hdef_ic = np.asarray(diagnostic_state.hdef_ic)
    assert np.allclose(ref_div_ic, val_div_ic)
    assert np.allclose(ref_hdef_ic, val_hdef_ic)
    ref_w = np.asarray(diffusion_savepoint_exit.w())
    val_w = np.asarray(prognostic_state.w)
    ref_dwdx = np.asarray(diffusion_savepoint_exit.dwdx())
    val_dwdx = np.asarray(diagnostic_state.dwdx)
    ref_dwdy = np.asarray(diffusion_savepoint_exit.dwdy())
    val_dwdy = np.asarray(diagnostic_state.dwdy)
    ref_vn = np.asarray(diffusion_savepoint_exit.vn())
    val_vn = np.asarray(prognostic_state.vn)
    assert np.allclose(ref_vn, val_vn)
    assert np.allclose(ref_dwdx, val_dwdx)
    assert np.allclose(ref_dwdy, val_dwdy)
    assert np.allclose(ref_w, val_w)
    ref_exner = np.asarray(diffusion_savepoint_exit.exner())
    ref_theta_v = np.asarray(diffusion_savepoint_exit.theta_v())
    val_theta_v = np.asarray(prognostic_state.theta_v)
    val_exner = np.asarray(prognostic_state.exner_pressure)
    assert np.allclose(ref_theta_v, val_theta_v)
    assert np.allclose(ref_exner, val_exner)


@pytest.mark.datatest
@pytest.mark.parametrize("linit", [True])
def test_run_diffusion_initial_step(
    diffusion_savepoint_init,
    diffusion_savepoint_exit,
    interpolation_savepoint,
    metrics_savepoint,
    grid_savepoint,
    icon_grid,
    r04b09_diffusion_config,
    damping_height,
):
    dtime = diffusion_savepoint_init.get_metadata("dtime").get("dtime")
    edge_geometry: EdgeParams = grid_savepoint.construct_edge_geometry()
    cell_geometry: CellParams = grid_savepoint.construct_cell_geometry()
    interpolation_state = interpolation_savepoint.construct_interpolation_state()
    metric_state = metrics_savepoint.construct_metric_state()
    diagnostic_state = diffusion_savepoint_init.construct_diagnostics()
    prognostic_state = diffusion_savepoint_init.construct_prognostics()
    vct_a = grid_savepoint.vct_a()
    vertical_params = VerticalModelParams(
        vct_a=vct_a, rayleigh_damping_height=damping_height
    )
    config = r04b09_diffusion_config
    config.ndyn_substeps = datarun_reduced_substeps
    additional_parameters = DiffusionParams(config)

    diffusion = Diffusion()
    diffusion.init(
        grid=icon_grid,
        config=config,
        params=additional_parameters,
        vertical_params=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
    )
    assert diffusion_savepoint_init.fac_bdydiff_v() == diffusion.fac_bdydiff_v

    diffusion.initial_run(
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        dtime=dtime,
    )

    _verify_diffusion_fields(
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        diffusion_savepoint_exit=diffusion_savepoint_exit,
    )


def test_verify_stencil15_field_manipulation(interpolation_savepoint, icon_grid):
    geofac_n2s = np.asarray(interpolation_savepoint.geofac_n2s())
    int_state = interpolation_savepoint.construct_interpolation_state()
    geofac_c = np.asarray(int_state.geofac_n2s_c)
    geofac_nbh = np.asarray(int_state.geofac_n2s_nbh)
    cec_table = icon_grid.get_c2cec_connectivity().table
    assert np.allclose(geofac_c, geofac_n2s[:, 0])
    assert np.allclose(geofac_nbh[cec_table], geofac_n2s[:, 1:])
