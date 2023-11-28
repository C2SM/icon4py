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

from icon4py.model.atmosphere.diffusion.diffusion import DiffusionParams
from icon4py.model.atmosphere.diffusion.diffusion_utils import (
    _setup_runtime_diff_multfac_vn,
    _setup_smag_limit,
    scale_k,
    set_zero_v_k,
    setup_fields_for_initial_step,
)
from icon4py.model.common.dimension import KDim, VertexDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.helpers import random_field, zero_field

from .utils import diff_multfac_vn_numpy, smag_limit_numpy


def initial_diff_multfac_vn_numpy(shape, k4, hdiff_efdt_ratio):
    return k4 * hdiff_efdt_ratio / 3.0 * np.ones(shape)


def test_scale_k(backend):
    grid = SimpleGrid()
    field = random_field(grid, KDim)
    scaled_field = zero_field(grid, KDim)
    factor = 2.0
    scale_k.with_backend(backend)(field, factor, scaled_field, offset_provider={})
    assert np.allclose(factor * field.asnumpy(), scaled_field.asnumpy())


def test_diff_multfac_vn_and_smag_limit_for_initial_step(backend):
    grid = SimpleGrid()
    diff_multfac_vn_init = zero_field(grid, KDim)
    smag_limit_init = zero_field(grid, KDim)
    k4 = 1.0
    efdt_ratio = 24.0
    shape = diff_multfac_vn_init.asnumpy().shape

    expected_diff_multfac_vn_init = initial_diff_multfac_vn_numpy(shape, k4, efdt_ratio)
    expected_smag_limit_init = smag_limit_numpy(
        initial_diff_multfac_vn_numpy, shape, k4, efdt_ratio
    )

    setup_fields_for_initial_step.with_backend(backend)(
        k4, efdt_ratio, diff_multfac_vn_init, smag_limit_init, offset_provider={}
    )

    assert np.allclose(expected_diff_multfac_vn_init, diff_multfac_vn_init.asnumpy())
    assert np.allclose(expected_smag_limit_init, smag_limit_init.asnumpy())


def test_diff_multfac_vn_smag_limit_for_time_step_with_const_value(backend):
    grid = SimpleGrid()
    diff_multfac_vn = zero_field(grid, KDim)
    smag_limit = zero_field(grid, KDim)
    k4 = 1.0
    substeps = 5.0
    efdt_ratio = 24.0
    shape = diff_multfac_vn.asnumpy().shape

    expected_diff_multfac_vn = diff_multfac_vn_numpy(shape, k4, substeps)
    expected_smag_limit = smag_limit_numpy(diff_multfac_vn_numpy, shape, k4, substeps)

    _setup_runtime_diff_multfac_vn.with_backend(backend)(
        k4, efdt_ratio, out=diff_multfac_vn, offset_provider={}
    )
    _setup_smag_limit.with_backend(backend)(diff_multfac_vn, out=smag_limit, offset_provider={})

    assert np.allclose(expected_diff_multfac_vn, diff_multfac_vn.asnumpy())
    assert np.allclose(expected_smag_limit, smag_limit.asnumpy())


def test_diff_multfac_vn_smag_limit_for_loop_run_with_k4_substeps(backend):
    grid = SimpleGrid()
    diff_multfac_vn = zero_field(grid, KDim)
    smag_limit = zero_field(grid, KDim)
    k4 = 0.003
    substeps = 1.0

    shape = diff_multfac_vn.asnumpy().shape
    expected_diff_multfac_vn = diff_multfac_vn_numpy(shape, k4, substeps)
    expected_smag_limit = smag_limit_numpy(diff_multfac_vn_numpy, shape, k4, substeps)
    _setup_runtime_diff_multfac_vn.with_backend(backend)(
        k4, substeps, out=diff_multfac_vn, offset_provider={}
    )
    _setup_smag_limit.with_backend(backend)(diff_multfac_vn, out=smag_limit, offset_provider={})

    assert np.allclose(expected_diff_multfac_vn, diff_multfac_vn.asnumpy())
    assert np.allclose(expected_smag_limit, smag_limit.asnumpy())


def test_set_zero_vertex_k(backend):
    grid = SimpleGrid()
    f = random_field(grid, VertexDim, KDim)
    set_zero_v_k.with_backend(backend)(f, offset_provider={})
    assert np.allclose(0.0, f.asnumpy())


@pytest.mark.datatest
@pytest.mark.parametrize("linit", [True])
def test_verify_special_diffusion_inital_step_values_against_initial_savepoint(
    diffusion_savepoint_init, r04b09_diffusion_config, icon_grid, linit, backend
):
    savepoint = diffusion_savepoint_init
    config = r04b09_diffusion_config

    params = DiffusionParams(config)
    expected_diff_multfac_vn = savepoint.diff_multfac_vn()
    expected_smag_limit = savepoint.smag_limit()
    exptected_smag_offset = savepoint.smag_offset()

    diff_multfac_vn = zero_field(icon_grid, KDim)
    smag_limit = zero_field(icon_grid, KDim)
    setup_fields_for_initial_step.with_backend(backend)(
        params.K4,
        config.hdiff_efdt_ratio,
        diff_multfac_vn,
        smag_limit,
        offset_provider={},
    )
    assert np.allclose(expected_smag_limit, smag_limit.asnumpy())
    assert np.allclose(expected_diff_multfac_vn, diff_multfac_vn.asnumpy())
    assert exptected_smag_offset == 0.0
