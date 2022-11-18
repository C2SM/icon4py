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

from icon4py.atm_dyn_iconam.diffusion import (
    DiffusionConfig,
    DiffusionParams,
    enhanced_smagorinski_factor,
    init_diffusion_local_fields,
)
from icon4py.common.dimension import KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


def _smag_limit_numpy(diff_multfac_vn: np.array):
    return 0.125 - 4.0 * diff_multfac_vn


def test_init_diff_multifac_vn_const():
    mesh = SimpleMesh()
    diff_multfac_vn = zero_field(mesh, KDim)
    smag_offset = zero_field(mesh, KDim)
    expected_diff_multfac_vn = 1.0 / 128.0 * np.ones(np.asarray(diff_multfac_vn).shape)
    expected_smag_limit = _smag_limit_numpy(expected_diff_multfac_vn)

    k4 = 1.0
    substeps = 5.0
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
    expected_diff_multfac_vn = (
        k4 * substeps / 3.0 * np.ones(np.asarray(diff_multfac_vn).shape)
    )
    expected_smag_limit = _smag_limit_numpy(expected_diff_multfac_vn)

    init_diffusion_local_fields(
        k4, substeps, diff_multfac_vn, smag_limit, offset_provider={}
    )
    assert np.allclose(expected_diff_multfac_vn, diff_multfac_vn)
    assert np.allclose(expected_smag_limit, smag_limit)


def test_enhanced_smagorinski_factor():
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

    mesh = SimpleMesh()
    a_vec = random_field(mesh, KDim, low=1.0, high=10.0)
    result = zero_field(mesh, KDim)
    fac = (0.67, 0.5, 1.3, 0.8)
    z = (0.1, 0.2, 0.3, 0.4)
    enhanced_smagorinski_factor(*fac, *z, a_vec, result, offset_provider={"Koff": KDim})
    enhanced_smag_fac_np = enhanced_smagorinski_factor_np(fac, z, np.asarray(a_vec))
    assert np.allclose(enhanced_smag_fac_np, np.asarray(result[:-1]))


@pytest.mark.xfail
def test_diffusion_init():
    pytest.fail("not implemented yet")


@pytest.mark.xfail
def test_diffusion_run():
    pytest.fail("not implemented yet")


def test_diffusion_coefficients_with_hdiff_efdt_ratio():
    config: DiffusionConfig = DiffusionConfig()
    config.hdiff_efdt_ratio = 1.0

    params = DiffusionParams(config)

    assert params.K2 == pytest.approx(0.125, abs=1e-12)
    assert params.K4 == pytest.approx(0.125 / 8.0, abs=1e-12)
    assert params.K8 == pytest.approx(0.125 / 64.0, abs=1e-12)
    assert params.K4W == pytest.approx(0.125 / 4.0, abs=1e-12)


def test_diffusion_coefficients_without_hdiff_efdt_ratio():
    config: DiffusionConfig = DiffusionConfig()
    config.hdiff_efdt_ratio = 0.0

    params = DiffusionParams(config)

    assert params.K2 == 0.0
    assert params.K4 == 0.0
    assert params.K8 == 0.0
    assert params.K4W == 0.0


def test_smagorinski_factor_for_diffusion_type_4():
    config: DiffusionConfig = DiffusionConfig()
    config.hdiff_smag_fac = 0.15
    config.diffusion_type = 4

    params = DiffusionParams(config)
    assert len(params.smagorinski_factor) == 1
    assert params.smagorinski_factor[0] == pytest.approx(0.15, abs=1e-16)
    assert params.smagorinski_height is None


def test_smagorinski_heights_diffusion_type_5_are_consistent():
    config: DiffusionConfig = DiffusionConfig()
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
    config: DiffusionConfig = DiffusionConfig()
    config.hdiff_smag_fac = 0.15
    config.diffusion_type = 5

    params = DiffusionParams(config)
    assert len(params.smagorinski_factor) == len(params.smagorinski_height)
    assert len(params.smagorinski_factor) == 4
    assert np.all(params.smagorinski_factor >= np.zeros(len(params.smagorinski_factor)))
