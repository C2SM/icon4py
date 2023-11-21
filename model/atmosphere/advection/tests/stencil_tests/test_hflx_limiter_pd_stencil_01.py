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
from gt4py.next.iterator.embedded import StridedNeighborOffsetProvider
from gt4py.next.program_processors.runners import roundtrip

from icon4py.model.atmosphere.advection.hflx_limiter_pd_stencil_01 import hflx_limiter_pd_stencil_01
from icon4py.model.common.dimension import C2EDim, CEDim, CellDim, EdgeDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.helpers import as_1D_sparse_field, random_field, zero_field


def hflx_limiter_pd_stencil_01_numpy(
    c2e: np.array,
    geofac_div: np.array,
    p_cc: np.array,
    p_rhodz_now: np.array,
    p_mflx_tracer_h: np.array,
    p_dtime,
    dbl_eps,
):
    geofac_div = np.expand_dims(geofac_div, axis=-1)
    p_m_0 = np.maximum(0.0, p_mflx_tracer_h[c2e[:, 0]] * geofac_div[:, 0] * p_dtime)
    p_m_1 = np.maximum(0.0, p_mflx_tracer_h[c2e[:, 1]] * geofac_div[:, 1] * p_dtime)
    p_m_2 = np.maximum(0.0, p_mflx_tracer_h[c2e[:, 2]] * geofac_div[:, 2] * p_dtime)

    p_m = p_m_0 + p_m_1 + p_m_2
    r_m = np.minimum(1.0, p_cc * p_rhodz_now / (p_m + dbl_eps))

    return r_m


def test_hflx_limiter_pd_stencil_01():
    grid = SimpleGrid()
    geofac_div = random_field(grid, CellDim, C2EDim)
    p_cc = random_field(grid, CellDim, KDim)
    p_rhodz_now = random_field(grid, CellDim, KDim)
    p_mflx_tracer_h = random_field(grid, EdgeDim, KDim)
    r_m = zero_field(grid, CellDim, KDim)
    p_dtime = np.float64(5)
    dbl_eps = np.float64(1e-9)

    ref = hflx_limiter_pd_stencil_01_numpy(
        grid.connectivities[C2EDim],
        np.asarray(geofac_div),
        np.asarray(p_cc),
        np.asarray(p_rhodz_now),
        np.asarray(p_mflx_tracer_h),
        p_dtime,
        dbl_eps,
    )

    hflx_limiter_pd_stencil_01.with_backend(roundtrip.backend)(
        as_1D_sparse_field(geofac_div, CEDim),
        p_cc,
        p_rhodz_now,
        p_mflx_tracer_h,
        r_m,
        p_dtime,
        dbl_eps,
        offset_provider={
            "C2CE": StridedNeighborOffsetProvider(CellDim, CEDim, grid.size[C2EDim]),
            "C2E": grid.get_offset_provider("C2E"),
        },
    )
    assert np.allclose(r_m, ref)
