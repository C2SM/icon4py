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
from icon_grid_test_utils import with_grid

from icon4py.atm_dyn_iconam.diagnostic import DiagnosticState
from icon4py.atm_dyn_iconam.diffusion import (
    Diffusion,
    DiffusionConfig,
    DiffusionParams,
    VectorTuple,
    _en_smag_fac_for_zero_nshift,
    init_diffusion_local_fields,
    scale_k,
    set_zero_v_k,
)
from icon4py.atm_dyn_iconam.icon_grid import VerticalModelParams
from icon4py.atm_dyn_iconam.interpolation_state import InterpolationState
from icon4py.atm_dyn_iconam.metric_state import MetricState
from icon4py.atm_dyn_iconam.prognostic import PrognosticState
from icon4py.common.dimension import KDim, VertexDim
from icon4py.testutils.serialbox_utils import IconSerialDataProvider
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


def test_scale_k():
    mesh = SimpleMesh()
    field = random_field(mesh, KDim)
    scaled_field = zero_field(mesh, KDim)
    factor = 2.0
    scale_k(field, factor, scaled_field, offset_provider={})
    assert np.allclose(factor * np.asarray(field), scaled_field)


def smag_limit_numpy(shape, k4, substeps):
    return 0.125 - 4.0 * diff_multfac_vn_numpy(shape, k4, substeps)


def test_init_diff_multifac_vn_const():
    mesh = SimpleMesh()
    diff_multfac_vn = zero_field(mesh, KDim)
    smag_offset = zero_field(mesh, KDim)
    k4 = 1.0
    substeps = 5.0
    shape = np.asarray(diff_multfac_vn).shape
    expected_diff_multfac_vn = diff_multfac_vn_numpy(shape, k4, substeps)
    expected_smag_limit = smag_limit_numpy(shape, k4, substeps)

    init_diffusion_local_fields(
        k4, substeps, diff_multfac_vn, smag_offset, offset_provider={}
    )
    assert np.allclose(expected_diff_multfac_vn, diff_multfac_vn)
    assert np.allclose(expected_smag_limit, smag_offset)


def test_init_diff_multifac_vn_k4_substeps():
    mesh = SimpleMesh()
    diff_multfac_vn = zero_field(mesh, KDim)
    smag_limit = zero_field(mesh, KDim)
    k4 = 0.003
    substeps = 1.0
    shape = np.asarray(diff_multfac_vn).shape
    expected_diff_multfac_vn = diff_multfac_vn_numpy(shape, k4, substeps)

    expected_smag_limit = smag_limit_numpy(shape, k4, substeps)

    init_diffusion_local_fields(
        k4, substeps, diff_multfac_vn, smag_limit, offset_provider={}
    )
    assert np.allclose(expected_diff_multfac_vn, diff_multfac_vn)
    assert np.allclose(expected_smag_limit, smag_limit)


def diff_multfac_vn_numpy(shape, k4, substeps):
    factor = min(1.0 / 128.0, k4 * substeps / 3.0)
    return factor * np.ones(shape)


def enhanced_smagorinski_factor_np(factor_in, heigths_in, a_vec):
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


def test_enhanced_smagorinski_factor():
    mesh = SimpleMesh()
    a_vec = random_field(mesh, KDim, low=1.0, high=10.0)
    result = zero_field(mesh, KDim)
    fac = (0.67, 0.5, 1.3, 0.8)
    z = (0.1, 0.2, 0.3, 0.4)
    _en_smag_fac_for_zero_nshift(
        *fac, *z, a_vec, out=result, offset_provider={"Koff": KDim}
    )
    enhanced_smag_fac_np = enhanced_smagorinski_factor_np(fac, z, np.asarray(a_vec))
    assert np.allclose(enhanced_smag_fac_np, np.asarray(result[:-1]))


def test_set_zero_vertex_k():
    mesh = SimpleMesh()
    f = random_field(mesh, VertexDim, KDim)
    set_zero_v_k(f, offset_provider={})
    assert np.allclose(0.0, f)


def test_diffusion_coefficients_with_hdiff_efdt_ratio():
    config: DiffusionConfig = DiffusionConfig.create_with_defaults()
    config.hdiff_efdt_ratio = 1.0

    params = DiffusionParams(config)

    assert params.K2 == pytest.approx(0.125, abs=1e-12)
    assert params.K4 == pytest.approx(0.125 / 8.0, abs=1e-12)
    assert params.K8 == pytest.approx(0.125 / 64.0, abs=1e-12)
    assert params.K4W == pytest.approx(0.125 / 4.0, abs=1e-12)


def test_diffusion_coefficients_without_hdiff_efdt_ratio():
    config: DiffusionConfig = DiffusionConfig.create_with_defaults()
    config.hdiff_efdt_ratio = 0.0

    params = DiffusionParams(config)

    assert params.K2 == 0.0
    assert params.K4 == 0.0
    assert params.K8 == 0.0
    assert params.K4W == 0.0


def test_smagorinski_factor_for_diffusion_type_4():
    config: DiffusionConfig = DiffusionConfig.create_with_defaults()
    config.hdiff_smag_fac = 0.15
    config.diffusion_type = 4

    params = DiffusionParams(config)
    assert len(params.smagorinski_factor) == 1
    assert params.smagorinski_factor[0] == pytest.approx(0.15, abs=1e-16)
    assert params.smagorinski_height is None


def test_smagorinski_heights_diffusion_type_5_are_consistent():
    config: DiffusionConfig = DiffusionConfig.create_with_defaults()
    config.hdiff_smag_fac = 0.15
    config.diffusion_type = 5

    params = DiffusionParams(config)
    assert len(params.smagorinski_height) == 4
    assert min(params.smagorinski_height) == params.smagorinski_height[0]
    assert max(params.smagorinski_height) == params.smagorinski_height[-1]
    assert params.smagorinski_height[0] < params.smagorinski_height[1]
    assert params.smagorinski_height[1] < params.smagorinski_height[3]
    assert params.smagorinski_height[2] != params.smagorinski_height[1]
    assert params.smagorinski_height[2] != params.smagorinski_height[3]


def test_smagorinski_factor_diffusion_type_5():
    config: DiffusionConfig = DiffusionConfig.create_with_defaults()
    config.hdiff_smag_fac = 0.15
    config.diffusion_type = 5

    params = DiffusionParams(config)
    assert len(params.smagorinski_factor) == len(params.smagorinski_height)
    assert len(params.smagorinski_factor) == 4
    assert np.all(params.smagorinski_factor >= np.zeros(len(params.smagorinski_factor)))


def test_diffusion_init():
    data_path = os.path.join(os.path.dirname(__file__), "ser_icondata")
    serializer = IconSerialDataProvider("icon_diffusion_init", data_path)
    serializer.print_info()
    first_run_date = "2021-06-20T12:00:10.000"
    savepoint = serializer.from_savepoint(linit=False, date=first_run_date)
    vct_a = savepoint.physical_height_field()
    meta = savepoint.get_metadata("nlev", "linit", "date")

    assert meta["nlev"] == 65
    assert meta["linit"] is False
    assert meta["date"] == first_run_date
    config = DiffusionConfig.create_with_defaults()
    additional_parameters = DiffusionParams(config)

    diffusion = Diffusion(config, additional_parameters, vct_a)
    # assert static local fields are initialized and correct:
    assert (
        diffusion.smag_offset
        == 0.25 * additional_parameters.K4 * config.substep_as_float()
    )
    assert diffusion.diff_multfac_w == min(
        1.0 / 48.0, additional_parameters.K4W * config.substep_as_float()
    )

    assert np.allclose(0.0, np.asarray(diffusion.v_vert))
    assert np.allclose(0.0, np.asarray(diffusion.u_vert))
    assert np.allclose(0.0, np.asarray(diffusion.diff_multfac_n2w))
    assert np.allclose(0.0, np.asarray(diffusion.kh_smag_ec))
    assert np.allclose(0.0, np.asarray(diffusion.kh_smag_e))

    shape_k = np.asarray(diffusion.diff_multfac_vn.shape)
    expected_smag_limit = smag_limit_numpy(
        shape_k, additional_parameters.K4, config.substep_as_float()
    )
    assert np.allclose(expected_smag_limit, np.asarray(diffusion.smag_limit))

    expected_diff_multfac_vn = diff_multfac_vn_numpy(
        shape_k, additional_parameters.K4, config.substep_as_float()
    )
    assert np.allclose(expected_diff_multfac_vn, np.asarray(diffusion.diff_multfac_vn))
    expected_enh_smag_fac = enhanced_smagorinski_factor_np(
        additional_parameters.smagorinski_factor,
        additional_parameters.smagorinski_height,
        vct_a,
    )
    assert np.allclose(expected_enh_smag_fac, np.asarray(diffusion.enh_smag_fac))


@pytest.mark.xfail
def test_diffusion_run(with_grid):
    data_path = os.path.join(os.path.dirname(__file__), "ser_icondata")
    data_provider = IconSerialDataProvider("icon_diffusion_init", data_path)
    data_provider.print_info()
    sp = data_provider.from_savepoint(linit=False, date="2021-06-20T12:00:10.000")
    vct_a = sp.physical_height_field()

    config = DiffusionConfig(
        with_grid,
        vertical_params=VerticalModelParams(
            vct_a=vct_a, rayleigh_damping_height=12500.0
        ),
    )

    additional_parameters = DiffusionParams(config)

    diffusion = Diffusion(config, additional_parameters, vct_a)

    diagnostic_state = DiagnosticState(
        hdef_ic=sp.hdef_ic(), div_ic=sp.div_ic(), dwdx=sp.dwdx(), dwdy=sp.dwdy()
    )
    prognostic_state = PrognosticState(
        vertical_wind=sp.vn(),
        normal_wind=sp.w(),
        exner_pressure=sp.exner(),
        theta_v=sp.theta_v(),
    )
    grg = sp.geofac_grg()

    rbf_coef = sp.rbf_coeff()
    interpolation_state = InterpolationState(
        e_bln_c_s=sp.e_bln_c_s(),
        rbf_coeff_1=rbf_coef[0],
        rbf_coeff_2=sp.rbf_coeff[1],
        geofac_div=sp.geofac_div(),
        geofac_n2s=sp.geofac_n2s(),
        geofac_grg_x=grg[0],
        geofac_grg_y=grg[1],
        nudgecoeff_e=sp.nudgecoeff_e(),
    )
    metric_state = MetricState(
        mask_hdiff=sp.theta_ref_mc(),
        theta_ref_mc=sp.theta_ref_mc(),
        wgtfac_c=sp.wgtfac_c(),
        zd_intcoef=sp.zd_intcoef(),
        zd_vertidx=sp.zd_vertidx(),
        zd_diffcoef=sp.zd_diffcoef(),
    )
    dtime = sp.get_metadata("dtime").get("dtime")
    orientation = sp.tangent_orientation()

    inverse_primal_edge_lengths = sp.inverse_primal_edge_lengths()
    inverse_vertical_vertex_lengths = sp.inv_vert_vert_length()
    inverse_dual_edge_length = sp.inv_dual_edge_length()
    primal_normal_vert: VectorTuple = (
        sp.primal_normal_vert_x(),
        sp.primal_normal_vert_y(),
    )
    dual_normal_vert: VectorTuple = (
        sp.dual_normal_vert_x(),
        sp.dual_normal_vert_y(),
    )
    edge_areas = sp.edge_areas()
    cell_areas = sp.cell_areas()

    diffusion.run(
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        dtime=dtime,
        tangent_orientation=orientation,
        inverse_primal_edge_lengths=inverse_primal_edge_lengths,
        inverse_dual_edge_length=inverse_dual_edge_length,
        inverse_vertical_vertex_lengths=inverse_vertical_vertex_lengths,
        primal_normal_vert=primal_normal_vert,
        dual_normal_vert=dual_normal_vert,
        edge_areas=edge_areas,
        cell_areas=cell_areas,
    )

    pytest.fail("not implemented yet")
