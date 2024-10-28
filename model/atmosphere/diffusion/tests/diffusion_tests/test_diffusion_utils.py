# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from icon4pytools.py2fgen.wrappers.settings import backend

from icon4py.model.atmosphere.diffusion import diffusion, diffusion_utils
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import simple as simple_grid
from icon4py.model.common.test_utils import helpers

from .utils import construct_diffusion_config, diff_multfac_vn_numpy, smag_limit_numpy


def initial_diff_multfac_vn_numpy(shape, k4, hdiff_efdt_ratio):
    return k4 * hdiff_efdt_ratio / 3.0 * np.ones(shape)


def test_scale_k(backend):
    grid = simple_grid.SimpleGrid()
    field = helpers.random_field(grid, dims.KDim)
    scaled_field = helpers.zero_field(grid, dims.KDim)
    factor = 2.0
    diffusion_utils.scale_k.with_backend(backend)(field, factor, scaled_field, offset_provider={})
    assert np.allclose(factor * field.asnumpy(), scaled_field.asnumpy())


def test_diff_multfac_vn_and_smag_limit_for_initial_step(backend):
    grid = simple_grid.SimpleGrid()
    diff_multfac_vn_init = helpers.zero_field(grid, dims.KDim)
    smag_limit_init = helpers.zero_field(grid, dims.KDim)
    k4 = 1.0
    efdt_ratio = 24.0
    shape = diff_multfac_vn_init.asnumpy().shape

    expected_diff_multfac_vn_init = initial_diff_multfac_vn_numpy(shape, k4, efdt_ratio)
    expected_smag_limit_init = smag_limit_numpy(
        initial_diff_multfac_vn_numpy, shape, k4, efdt_ratio
    )

    diffusion_utils.setup_fields_for_initial_step.with_backend(backend)(
        k4, efdt_ratio, diff_multfac_vn_init, smag_limit_init, offset_provider={}
    )

    assert np.allclose(expected_diff_multfac_vn_init, diff_multfac_vn_init.asnumpy())
    assert np.allclose(expected_smag_limit_init, smag_limit_init.asnumpy())


def test_diff_multfac_vn_smag_limit_for_time_step_with_const_value(backend):
    grid = simple_grid.SimpleGrid()
    diff_multfac_vn = helpers.zero_field(grid, dims.KDim)
    smag_limit = helpers.zero_field(grid, dims.KDim)
    k4 = 1.0
    substeps = 5.0
    efdt_ratio = 24.0
    shape = diff_multfac_vn.asnumpy().shape

    expected_diff_multfac_vn = diff_multfac_vn_numpy(shape, k4, substeps)
    expected_smag_limit = smag_limit_numpy(diff_multfac_vn_numpy, shape, k4, substeps)

    diffusion_utils._setup_runtime_diff_multfac_vn.with_backend(backend)(
        k4, efdt_ratio, out=diff_multfac_vn, offset_provider={}
    )
    diffusion_utils._setup_smag_limit.with_backend(backend)(
        diff_multfac_vn, out=smag_limit, offset_provider={}
    )

    assert np.allclose(expected_diff_multfac_vn, diff_multfac_vn.asnumpy())
    assert np.allclose(expected_smag_limit, smag_limit.asnumpy())


def test_diff_multfac_vn_smag_limit_for_loop_run_with_k4_substeps(backend):
    grid = simple_grid.SimpleGrid()
    diff_multfac_vn = helpers.zero_field(grid, dims.KDim)
    smag_limit = helpers.zero_field(grid, dims.KDim)
    k4 = 0.003
    substeps = 1.0

    shape = diff_multfac_vn.asnumpy().shape
    expected_diff_multfac_vn = diff_multfac_vn_numpy(shape, k4, substeps)
    expected_smag_limit = smag_limit_numpy(diff_multfac_vn_numpy, shape, k4, substeps)
    diffusion_utils._setup_runtime_diff_multfac_vn.with_backend(backend)(
        k4, substeps, out=diff_multfac_vn, offset_provider={}
    )
    diffusion_utils._setup_smag_limit.with_backend(backend)(
        diff_multfac_vn, out=smag_limit, offset_provider={}
    )

    assert np.allclose(expected_diff_multfac_vn, diff_multfac_vn.asnumpy())
    assert np.allclose(expected_smag_limit, smag_limit.asnumpy())


def test_init_zero_vertex_k(backend):
    grid = simple_grid.SimpleGrid()
    f = helpers.random_field(grid, dims.VertexDim, dims.KDim)
    diffusion_utils.init_zero_v_k.with_backend(backend)(f, offset_provider={})
    assert np.allclose(0.0, f.asnumpy())


@pytest.mark.datatest
@pytest.mark.parametrize("linit", [True])
def test_verify_special_diffusion_inital_step_values_against_initial_savepoint(
    savepoint_diffusion_init, experiment, icon_grid, linit, ndyn_substeps
):
    savepoint = savepoint_diffusion_init
    config = construct_diffusion_config(experiment, ndyn_substeps=ndyn_substeps)

    params = diffusion.DiffusionParams(config)
    expected_diff_multfac_vn = savepoint.diff_multfac_vn()
    expected_smag_limit = savepoint.smag_limit()
    exptected_smag_offset = savepoint.smag_offset()

    diff_multfac_vn = helpers.zero_field(icon_grid, dims.KDim)
    smag_limit = helpers.zero_field(icon_grid, dims.KDim)
    diffusion_utils.setup_fields_for_initial_step.with_backend(backend)(
        params.K4,
        config.hdiff_efdt_ratio,
        diff_multfac_vn,
        smag_limit,
        offset_provider={},
    )
    assert np.allclose(expected_smag_limit, smag_limit.asnumpy())
    assert np.allclose(expected_diff_multfac_vn, diff_multfac_vn.asnumpy())
    assert exptected_smag_offset == 0.0
