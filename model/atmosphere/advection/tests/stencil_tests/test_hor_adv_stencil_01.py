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

from icon4py.model.atmosphere.advection.hor_adv_stencil_01 import hor_adv_stencil_01
from icon4py.model.common.dimension import C2EDim, CEDim, CellDim, EdgeDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.helpers import as_1D_sparse_field, random_field


def hor_adv_stencil_01_numpy(
    c2e: np.array,
    p_mflx_tracer_h: np.array,
    deepatmo_divh: np.array,
    tracer_now: np.array,
    rhodz_now: np.array,
    rhodz_new: np.array,
    geofac_div: np.array,
    p_dtime,
) -> np.array:
    geofac_div = np.expand_dims(geofac_div, axis=-1)

    tracer_new = (
        tracer_now * rhodz_now
        - p_dtime * deepatmo_divh * np.sum(p_mflx_tracer_h[c2e] * geofac_div, axis=1)
    ) / rhodz_new

    return tracer_new


def test_hor_adv_stencil_01():
    grid = SimpleGrid()

    p_mflx_tracer_h = random_field(grid, EdgeDim, KDim)
    deepatmo_divh = random_field(grid, KDim)
    tracer_now = random_field(grid, CellDim, KDim)
    rhodz_now = random_field(grid, CellDim, KDim)
    rhodz_new = random_field(grid, CellDim, KDim)
    geofac_div = random_field(grid, CellDim, C2EDim)
    geofac_div_new = as_1D_sparse_field(geofac_div, CEDim)
    tracer_new = random_field(grid, CellDim, KDim)
    p_dtime = np.float64(5.0)

    ref = hor_adv_stencil_01_numpy(
        grid.connectivities[C2EDim],
        np.asarray(p_mflx_tracer_h),
        np.asarray(deepatmo_divh),
        np.asarray(tracer_now),
        np.asarray(rhodz_now),
        np.asarray(rhodz_new),
        np.asarray(geofac_div),
        p_dtime,
    )
    hor_adv_stencil_01(
        p_mflx_tracer_h,
        deepatmo_divh,
        tracer_now,
        rhodz_now,
        rhodz_new,
        geofac_div_new,
        tracer_new,
        p_dtime,
        offset_provider={
            "C2E": grid.get_c2e_offset_provider(),
            "C2CE": StridedNeighborOffsetProvider(CellDim, CEDim, grid.size[C2EDim]),
        },
    )
    assert np.allclose(tracer_new, ref)
