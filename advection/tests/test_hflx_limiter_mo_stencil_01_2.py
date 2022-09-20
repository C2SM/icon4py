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

from icon4py.advection.hflx_limiter_mo_stencil_01_2 import (
    hflx_limiter_mo_stencil_01_2,
)
from icon4py.common.dimension import C2EDim, CEDim, CellDim, EdgeDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import as_1D_sparse_field, random_field


def hflx_limiter_mo_stencil_01_2_numpy(
    e2c: np.array,
    c2e: np.array,
    geofac_div: np.array,
    p_rhodz_now: np.array,
    p_rhodz_new: np.array,
    p_cc: np.array,
    z_mflx_low: np.array,
    z_anti: np.array,
    p_dtime: float,
):
    z_mflx_anti = p_dtime * geofac_div / p_rhodz_new * z_anti(e2c)

    min_z_flx_anti = min(0.0, z_mflx_anti)
    max_z_flx_anti = max(0.0, z_mflx_anti)

    z_mflx_anti_in = -np.sum(min_z_flx_anti[c2e], axis=1)  # sum along edge dimension
    z_mflx_anti_out = np.sum(max_z_flx_anti[c2e], axis=1)  # sum along edge dimension

    geofac_div = np.expand_dims(geofac_div, axis=-1)
    z_fluxdiv_c = np.sum(z_mflx_low[c2e] * geofac_div, axis=1)

    z_tracer_new_low = (p_cc * p_rhodz_now - p_dtime * z_fluxdiv_c) / p_rhodz_new
    z_tracer_max = max(p_cc, z_tracer_new_low)
    z_tracer_min = min(p_cc, z_tracer_new_low)

    return z_mflx_anti_in, z_mflx_anti_out, z_fluxdiv_c, z_tracer_new_low, z_tracer_max, z_tracer_min


def test_hflx_limiter_mo_stencil_01_2():
    mesh = SimpleMesh()
    p_rhodz_now = random_field(mesh, CellDim, KDim)
    p_rhodz_new = random_field(mesh, CellDim, KDim)
    p_cc = random_field(mesh, CellDim, KDim)
    z_mflx_low = random_field(mesh, EdgeDim, KDim)
    z_anti = random_field(mesh, EdgeDim, KDim)
    geofac_div = random_field(mesh, CellDim, C2EDim)
    #geofac_div_new = as_1D_sparse_field(geofac_div, CEDim)
    z_mflx_anti_in = random_field(mesh, CellDim, KDim)
    z_mflx_anti_out = random_field(mesh, CellDim, KDim)
    z_fluxdiv_c = random_field(mesh, CellDim, KDim)
    z_tracer_new_low = random_field(mesh, CellDim, KDim)
    z_tracer_max = random_field(mesh, CellDim, KDim)
    z_tracer_min = random_field(mesh, CellDim, KDim)
    p_dtime = 5.0


    ref1, ref2, ref3, ref4, ref5, ref6 = hflx_limiter_mo_stencil_01_2_numpy(
        mesh.e2c,
        mesh.c2e,
        np.asarray(geofac_div),
        np.asarray(p_rhodz_now),
        np.asarray(p_rhodz_new),
        np.asarray(p_cc),
        np.asarray(z_mflx_low),
        np.asarray(z_anti),
        p_dtime,
    )

    hflx_limiter_mo_stencil_01_2(
        #geofac_div_new,
        geofac_div,
        p_rhodz_now,
        p_rhodz_new,
        p_cc,
        z_mflx_low,
        z_anti,
        z_mflx_anti_in,
        z_mflx_anti_out,
        z_fluxdiv_c,
        z_tracer_new_low,
        z_tracer_max,
        z_tracer_min,
        p_dtime,
        offset_provider={
            "E2C": mesh.get_e2c_offset_provider(),
            "C2E": mesh.get_c2e_offset_provider(),
            "C2CE": StridedNeighborOffsetProvider(CellDim, CEDim, mesh.n_c2e),
        },
    )

    assert np.allclose(ref1, z_mflx_anti_in)
    assert np.allclose(ref2, z_mflx_anti_out)
