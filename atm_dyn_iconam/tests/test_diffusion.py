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
    init_diffusion_local_fields,
    init_nabla2_factor_in_upper_damping_zone,
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


def test_init_nabla2_factor_in_upper_damping_zone():
    mesh = SimpleMesh()
    diff_multfac_n2w = random_field(mesh, KDim)
    init_nabla2_factor_in_upper_damping_zone(diff_multfac_n2w, offset_provider={})

    assert np.allclose(0, diff_multfac_n2w)


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


def test_smagorinsik_factor_diffusion_type_5():
    config: DiffusionConfig = DiffusionConfig()
    config.hdiff_smag_fac = 0.15
    config.diffusion_type = 5

    params = DiffusionParams(config)
    assert len(params.smagorinski_factor) == 4
