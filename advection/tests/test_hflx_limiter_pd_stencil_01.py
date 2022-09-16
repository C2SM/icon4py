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
from functional.iterator.embedded import StridedNeighborOffsetProvider

from icon4py.advection.hflx_limiter_pd_stencil_01 import (
    hflx_limiter_pd_stencil_01,
)
from icon4py.common.dimension import C2EDim, CEDim, CellDim, EdgeDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import as_1D_sparse_field, random_field, zero_field


def hflx_limiter_pd_stencil_01_numpy(
    c2e: np.array,
    p_dtime: float,
    dbl_eps: float,
    geofac_div: np.array,
    p_cc: np.array,
    p_rhodz_now: np.array,
    p_mflx_tracer_h: np.array,
)-> tuple[np.ndarray]:
    geofac_div = np.expand_dims(geofac_div, axis=-1)
    p_m_0 = np.maximum(0.0, p_mflx_tracer_h[c2e[:, 0]] * geofac_div[:, 0] * p_dtime)
    p_m_1 = np.maximum(0.0, p_mflx_tracer_h[c2e[:, 1]] * geofac_div[:, 1] * p_dtime)
    p_m_2 = np.maximum(0.0, p_mflx_tracer_h[c2e[:, 2]] * geofac_div[:, 2] * p_dtime)

    p_m = p_m_0 + p_m_1 + p_m_2
    r_m = np.minimum(1.0, p_cc * p_rhodz_now / (p_m + dbl_eps))

    return r_m, p_m


def test_hflx_limiter_pd_stencil_01():
    mesh = SimpleMesh()
    geofac_div = random_field(mesh, CellDim, C2EDim)
    p_cc = random_field(mesh, CellDim, KDim)
    p_rhodz_now = random_field(mesh, CellDim, KDim)
    p_mflx_tracer_h = random_field(mesh, EdgeDim, KDim)
    r_m = zero_field(mesh, CellDim, KDim)
    p_m = zero_field(mesh, CellDim, KDim)
    p_dtime = np.float64(5)
    dbl_eps = np.float64(1e-9)

    ref_r_m, ref_p_m = hflx_limiter_pd_stencil_01_numpy(
        mesh.c2e,
        p_dtime,
        dbl_eps,
        np.asarray(geofac_div),
        np.asarray(p_cc),
        np.asarray(p_rhodz_now),
        np.asarray(p_mflx_tracer_h)
    )

    hflx_limiter_pd_stencil_01(
        p_dtime,
        dbl_eps,
        as_1D_sparse_field(geofac_div, CEDim),
        p_cc,
        p_rhodz_now,
        p_mflx_tracer_h,
        r_m,
        p_m,
        offset_provider={
            "C2CE": StridedNeighborOffsetProvider(CellDim, CEDim, mesh.n_c2e),
            "C2E": mesh.get_c2e_offset_provider(),
        },
    )
    assert np.allclose(r_m, ref_r_m)
    assert np.allclose(p_m, ref_p_m)
