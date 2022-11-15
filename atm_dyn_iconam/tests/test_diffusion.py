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

from icon4py.atm_dyn_iconam.diffusion import init_diffusion_local_fields
from icon4py.common.dimension import KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import zero_field


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


@pytest.mark.xfail
def test_diffusion_init():
    pytest.fail("not implemented yet")


@pytest.mark.xfail
def test_diffusion_run():
    pytest.fail("not implemented yet")
