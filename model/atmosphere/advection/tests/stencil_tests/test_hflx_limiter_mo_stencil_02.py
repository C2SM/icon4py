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
from numpy import int32

from icon4py.model.atmosphere.advection.hflx_limiter_mo_stencil_02 import hflx_limiter_mo_stencil_02
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.helpers import constant_field, random_field, zero_field


def hflx_limiter_mo_stencil_02_numpy(
    refin_ctrl: np.ndarray,
    p_cc: np.ndarray,
    z_tracer_new_low: np.ndarray,
    z_tracer_max: np.ndarray,
    z_tracer_min: np.ndarray,
    lo_bound: float,
    hi_bound: float,
):
    refin_ctrl = np.expand_dims(refin_ctrl, axis=1)
    condition = np.logical_or(
        np.equal(refin_ctrl, lo_bound * np.ones(refin_ctrl.shape, dtype=int32)),
        np.equal(refin_ctrl, hi_bound * np.ones(refin_ctrl.shape, dtype=int32)),
    )
    z_tracer_new_out = np.where(
        condition,
        np.minimum(1.1 * p_cc, np.maximum(0.9 * p_cc, z_tracer_new_low)),
        z_tracer_new_low,
    )
    z_tracer_max_out = np.where(condition, np.maximum(p_cc, z_tracer_new_out), z_tracer_max)
    z_tracer_min_out = np.where(condition, np.minimum(p_cc, z_tracer_new_out), z_tracer_min)
    return z_tracer_new_out, z_tracer_max_out, z_tracer_min_out


def test_hflx_limiter_mo_stencil_02_some_matching_condition(backend):
    grid = SimpleGrid()

    hi_bound = np.int32(1)
    lo_bound = np.int32(5)

    refin_ctrl = constant_field(grid, hi_bound, CellDim, dtype=int32)

    refin_ctrl[0:2] = np.int32(3)

    p_cc = random_field(grid, CellDim, KDim)
    z_tracer_new_low_in = random_field(grid, CellDim, KDim)
    z_tracer_max_in = random_field(grid, CellDim, KDim)
    z_tracer_min_in = random_field(grid, CellDim, KDim)

    z_tracer_new_low = zero_field(grid, CellDim, KDim)
    z_tracer_max = zero_field(grid, CellDim, KDim)
    z_tracer_min = zero_field(grid, CellDim, KDim)

    ref_new_low, ref_max, ref_min = hflx_limiter_mo_stencil_02_numpy(
        np.asarray(refin_ctrl),
        np.asarray(p_cc),
        np.asarray(z_tracer_new_low_in),
        np.asarray(z_tracer_max_in),
        np.asarray(z_tracer_min_in),
        lo_bound,
        hi_bound,
    )

    hflx_limiter_mo_stencil_02.with_backend(backend)(
        refin_ctrl,
        p_cc,
        z_tracer_new_low_in,
        z_tracer_max_in,
        z_tracer_min_in,
        lo_bound,
        hi_bound,
        z_tracer_new_low,
        z_tracer_max,
        z_tracer_min,
        offset_provider={},
    )

    assert np.allclose(z_tracer_new_low, ref_new_low)
    assert np.allclose(z_tracer_max, ref_max)
    assert np.allclose(z_tracer_min, ref_min)


def test_hflx_limiter_mo_stencil_02_none_matching_condition(backend):
    grid = SimpleGrid()

    hi_bound = np.int32(3)
    lo_bound = np.int32(1)

    refin_ctrl = constant_field(grid, 2, CellDim, dtype=int32)

    p_cc = random_field(grid, CellDim, KDim)

    z_tracer_new_low_in = random_field(grid, CellDim, KDim)
    z_tracer_max_in = random_field(grid, CellDim, KDim)
    z_tracer_min_in = random_field(grid, CellDim, KDim)

    z_tracer_new_low = zero_field(grid, CellDim, KDim)
    z_tracer_max = zero_field(grid, CellDim, KDim)
    z_tracer_min = zero_field(grid, CellDim, KDim)

    hflx_limiter_mo_stencil_02.with_backend(backend)(
        refin_ctrl,
        p_cc,
        z_tracer_new_low_in,
        z_tracer_max_in,
        z_tracer_min_in,
        lo_bound,
        hi_bound,
        z_tracer_new_low,
        z_tracer_max,
        z_tracer_min,
        offset_provider={},
    )

    assert np.allclose(z_tracer_new_low_in, z_tracer_new_low)
    assert np.allclose(z_tracer_min_in, z_tracer_min)
    assert np.allclose(z_tracer_max_in, z_tracer_max)
