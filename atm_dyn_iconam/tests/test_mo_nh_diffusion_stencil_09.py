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

from icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_09 import (
    mo_nh_diffusion_stencil_09,
)
from icon4py.common.dimension import C2E2CODim, CellDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field


def mo_nh_diffusion_stencil_09_numpy(
    c2e2c0: np.array,
    area: np.array,
    z_nabla2_c: np.array,
    geofac_n2s: np.array,
    w: np.array,
    diff_multfac_w,
) -> np.array:
    geofac_n2s = np.expand_dims(geofac_n2s, axis=-1)
    area = np.expand_dims(area, axis=-1)
    w = w - diff_multfac_w * area * area * np.sum(
        z_nabla2_c[c2e2c0] * geofac_n2s, axis=1
    )
    return w


def test_mo_nh_diffusion_stencil_09():
    mesh = SimpleMesh()

    area = random_field(mesh, CellDim)
    z_nabla2_c = random_field(mesh, CellDim, KDim)
    geofac_n2s = random_field(mesh, CellDim, C2E2CODim)
    w = random_field(mesh, CellDim, KDim)
    diff_multfac_w = 5.0

    ref = mo_nh_diffusion_stencil_09_numpy(
        mesh.c2e2cO,
        np.asarray(area),
        np.asarray(z_nabla2_c),
        np.asarray(geofac_n2s),
        np.asarray(w),
        diff_multfac_w,
    )

    mo_nh_diffusion_stencil_09(
        area,
        z_nabla2_c,
        geofac_n2s,
        w,
        diff_multfac_w,
        offset_provider={"C2E2CO": mesh.get_c2e2cO_offset_provider()},
    )

    assert np.allclose(w, ref)
