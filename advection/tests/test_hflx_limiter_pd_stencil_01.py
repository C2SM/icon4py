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

from icon4py.advection.hflx_limiter_pd_stencil_01 import (
    hflx_limiter_pd_stencil_01,
)
from icon4py.common.dimension import C2EDim, CellDim, EdgeDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


def hflx_limiter_pd_stencil_01_numpy(
    c2e: np.array,
    geofac_div: np.array,
    p_cc: np.array,
    p_rhodz_now: np.array,
    p_mflx_tracer_h: np.array,
    p_dtime,
    dbl_eps,
):
    #    p_m: np.array
    p_mflx_tracer_h_c2e = p_mflx_tracer_h[c2e]
    # p_m = max(0.0, geofac_div[:, 0]*p_mflx_tracer_h_c2e[:, 0]*p_dtime) + max(0.0, geofac_div[:, 1]*p_mflx_tracer_h_c2e[:, 1]*p_dtime) + max(0.0, geofac_div[:, 2]*p_mflx_tracer_h_c2e[:, 2]*p_dtime)
    geofac_div = np.expand_dims(geofac_div, axis=-1)
    #    p_m = max(0.0, geofac_div[:, 0]*p_mflx_tracer_h_c2e[:, 0]*p_dtime)
    p_m = np.sum(
        np.maximum(
            np.zeros(p_mflx_tracer_h_c2e.shape, dtype=float),
            p_mflx_tracer_h_c2e * geofac_div * p_dtime,
        )
    )
    r_m = np.minimum(
        np.ones(p_cc.shape, dtype=float), (p_cc * p_rhodz_now) / (p_m + dbl_eps)
    )

    return r_m


def test_hflx_limiter_pd_stencil_01():
    mesh = SimpleMesh()
    geofac_div = random_field(mesh, CellDim, C2EDim)
    p_cc = random_field(mesh, CellDim, KDim)
    p_rhodz_now = random_field(mesh, CellDim, KDim)
    p_mflx_tracer_h = random_field(mesh, EdgeDim, KDim)
    r_m = zero_field(mesh, CellDim, KDim)
    p_dtime = np.float64(5)
    dbl_eps = np.float64(1e-9)

    ref = hflx_limiter_pd_stencil_01_numpy(
        mesh.c2e,
        np.asarray(geofac_div),
        np.asarray(p_cc),
        np.asarray(p_rhodz_now),
        np.asarray(p_mflx_tracer_h),
        p_dtime,
        dbl_eps,
    )

    hflx_limiter_pd_stencil_01(
        geofac_div,
        p_cc,
        p_rhodz_now,
        p_mflx_tracer_h,
        r_m,
        p_dtime,
        dbl_eps,
        offset_provider={
            "C2E": mesh.get_c2e_offset_provider(),
        },
    )
    assert np.allclose(r_m, ref)
