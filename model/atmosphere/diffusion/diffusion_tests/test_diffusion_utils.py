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

from icon4py.model.common.test_utils.helpers import random_field, zero_field
from icon4py.model.common.test_utils.simple_mesh import SimpleMesh
from icon4py.model.common.dimension import KDim, VertexDim
from icon4py.model.atmosphere.diffusion.diffusion import DiffusionParams
from icon4py.model.atmosphere.diffusion.diffusion_utils import (
    _en_smag_fac_for_zero_nshift,
    _setup_runtime_diff_multfac_vn,
    _setup_smag_limit,
    scale_k,
    set_zero_v_k,
    setup_fields_for_initial_step,
)

from .utils import diff_multfac_vn_numpy, enhanced_smagorinski_factor_numpy, smag_limit_numpy

def initial_diff_multfac_vn_numpy(shape, k4, hdiff_efdt_ratio):
    return k4 * hdiff_efdt_ratio / 3.0 * np.ones(shape)


def test_scale_k():
    mesh = SimpleMesh()
    field = random_field(mesh, KDim)
    scaled_field = zero_field(mesh, KDim)
    factor = 2.0
    scale_k(field, factor, scaled_field, offset_provider={})
    assert np.allclose(factor * np.asarray(field), scaled_field)


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


def test_set_zero_vertex_k():
    mesh = SimpleMesh()
    f = random_field(mesh, VertexDim, KDim)
    set_zero_v_k(f, offset_provider={})
    assert np.allclose(0.0, f)


@pytest.mark.datatest
@pytest.mark.parametrize("linit", [True])
def test_verify_special_diffusion_inital_step_values_against_initial_savepoint(
    diffusion_savepoint_init, r04b09_diffusion_config, icon_grid, linit
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
