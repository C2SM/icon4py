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

from icon4py.atm_dyn_iconam.apply_nabla2_to_w import apply_nabla2_to_w
from icon4py.common.dimension import C2E2CODim, CellDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field


def mo_apply_nabla2_to_w(
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


def test_apply_nabla2_to_w():
    mesh = SimpleMesh()

    area = random_field(mesh, CellDim)
    z_nabla2_c = random_field(mesh, CellDim, KDim)
    geofac_n2s = random_field(mesh, CellDim, C2E2CODim)
    w = random_field(mesh, CellDim, KDim)
    diff_multfac_w = 5.0

    ref = mo_apply_nabla2_to_w(
        mesh.c2e2cO,
        np.asarray(area),
        np.asarray(z_nabla2_c),
        np.asarray(geofac_n2s),
        np.asarray(w),
        diff_multfac_w,
    )

    apply_nabla2_to_w(
        area,
        z_nabla2_c,
        geofac_n2s,
        w,
        diff_multfac_w,
        0,
        mesh.n_cells,
        0,
        mesh.k_level,
        offset_provider={
            "C2E2CO": mesh.get_c2e2cO_offset_provider(),
            "C2E2CODim": C2E2CODim,
        },
    )

    assert np.allclose(w, ref)
