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

from icon4py.model.atmosphere.advection.hflx_limiter_pd_stencil_02 import hflx_limiter_pd_stencil_02
from icon4py.model.common.dimension import CellDim, E2CDim, EdgeDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.helpers import constant_field, random_field


def hflx_limiter_pd_stencil_02_numpy(
    e2c: np.array,
    refin_ctrl: np.array,
    r_m: np.array,
    p_mflx_tracer_h_in: np.array,
    bound,
):
    r_m_e2c = r_m[e2c]
    refin_ctrl = np.expand_dims(refin_ctrl, axis=-1)
    p_mflx_tracer_h_out = np.where(
        refin_ctrl != bound,
        np.where(
            p_mflx_tracer_h_in >= 0,
            p_mflx_tracer_h_in * r_m_e2c[:, 0],
            p_mflx_tracer_h_in * r_m_e2c[:, 1],
        ),
        p_mflx_tracer_h_in,
    )
    return p_mflx_tracer_h_out


def test_hflx_limiter_pd_stencil_02_nowhere_matching_refin_ctl():
    grid = SimpleGrid()
    bound = np.int32(7)
    refin_ctrl = constant_field(grid, 4, EdgeDim, dtype=np.int32)
    r_m = random_field(grid, CellDim, KDim)
    p_mflx_tracer_h_in = random_field(grid, EdgeDim, KDim)

    ref = hflx_limiter_pd_stencil_02_numpy(
        grid.connectivities[E2CDim],
        np.asarray(refin_ctrl),
        np.asarray(r_m),
        np.asarray(p_mflx_tracer_h_in),
        bound,
    )

    hflx_limiter_pd_stencil_02(
        refin_ctrl,
        r_m,
        p_mflx_tracer_h_in,
        bound,
        offset_provider={
            "E2C": grid.get_e2c_offset_provider(),
        },
    )
    assert np.allclose(p_mflx_tracer_h_in, ref)


def test_hflx_limiter_pd_stencil_02_everywhere_matching_refin_ctl():
    grid = SimpleGrid()
    bound = np.int32(7)
    refin_ctrl = constant_field(grid, bound, EdgeDim, dtype=np.int32)
    r_m = random_field(grid, CellDim, KDim)
    p_mflx_tracer_h_in = random_field(grid, EdgeDim, KDim)

    hflx_limiter_pd_stencil_02(
        refin_ctrl,
        r_m,
        p_mflx_tracer_h_in,
        bound,
        offset_provider={
            "E2C": grid.get_e2c_offset_provider(),
        },
    )
    assert np.allclose(p_mflx_tracer_h_in, p_mflx_tracer_h_in)


def test_hflx_limiter_pd_stencil_02_partly_matching_refin_ctl():
    grid = SimpleGrid()
    bound = np.int32(4)
    refin_ctrl = constant_field(grid, 5, EdgeDim, dtype=np.int32)
    refin_ctrl[2:6] = bound
    r_m = random_field(grid, CellDim, KDim)
    p_mflx_tracer_h_in = random_field(grid, EdgeDim, KDim)

    hflx_limiter_pd_stencil_02(
        refin_ctrl,
        r_m,
        p_mflx_tracer_h_in,
        bound,
        offset_provider={
            "E2C": grid.get_e2c_offset_provider(),
        },
    )
    assert np.allclose(p_mflx_tracer_h_in, p_mflx_tracer_h_in)


def test_hflx_limiter_pd_stencil_02_everywhere_matching_refin_ctl_does_not_change_inout_arg():
    grid = SimpleGrid()
    bound = np.int32(7)
    refin_ctrl = constant_field(grid, bound, EdgeDim, dtype=np.int32)
    r_m = random_field(grid, CellDim, KDim)
    p_mflx_tracer_h_in = random_field(grid, EdgeDim, KDim)

    hflx_limiter_pd_stencil_02(
        refin_ctrl,
        r_m,
        p_mflx_tracer_h_in,
        bound,
        offset_provider={
            "E2C": grid.get_e2c_offset_provider(),
        },
    )
    assert np.allclose(p_mflx_tracer_h_in, p_mflx_tracer_h_in)
