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
import pytest

from icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_15 import (
    mo_nh_diffusion_stencil_15,
)
from icon4py.common.dimension import C2E2CDim, CellDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, random_mask, zero_field


def mo_nh_diffusion_stencil_15_numpy(
    c2e2c: np.array,
    mask: np.array,
    zd_vertoffset: np.array,
    zd_diffcoef: np.array,
    geofac_n2s_c: np.array,
    geofac_n2s_nbh: np.array,
    vcoef: np.array,
    theta_v: np.array,
    z_temp: np.array,
) -> np.array:
    full_shape = vcoef.shape

    geofac_n2s_nbh = np.expand_dims(geofac_n2s_nbh, axis=2)

    theta_v_at_zd_vertidx = np.zeros_like(vcoef)
    theta_v_at_zd_vertidx_p1 = np.zeros_like(vcoef)
    for ic in range(full_shape[0]):
        for isparse in range(full_shape[1]):
            for ik in range(full_shape[2]):
                theta_v_at_zd_vertidx[ic, isparse, ik] = theta_v[
                    c2e2c[ic, isparse], ik + zd_vertoffset[ic, isparse, ik]
                ]
                theta_v_at_zd_vertidx_p1[ic, isparse, ik] = theta_v[
                    c2e2c[ic, isparse], ik + zd_vertoffset[ic, isparse, ik] + 1
                ]

    sum_over = np.sum(
        geofac_n2s_nbh
        * (vcoef * theta_v_at_zd_vertidx + (1.0 - vcoef) * theta_v_at_zd_vertidx_p1),
        axis=1,
    )

    geofac_n2s_c = np.expand_dims(geofac_n2s_c, axis=1)  # add KDim
    return np.where(
        mask, z_temp + zd_diffcoef * (theta_v * geofac_n2s_c + sum_over), z_temp
    )


@pytest.mark.skip("new lowering: dims in offset provider")
def test_mo_nh_diffusion_stencil_15():
    mesh = SimpleMesh()

    mask = random_mask(mesh, CellDim, KDim)

    zd_vertoffset = zero_field(mesh, CellDim, C2E2CDim, KDim, dtype=int)
    rng = np.random.default_rng()
    for k in range(mesh.k_level):
        # construct offsets that reach all k-levels except the last (because we are using the entries of this field with `+1`)
        zd_vertoffset[:, :, k] = rng.integers(
            low=0 - k,
            high=mesh.k_level - k - 1,
            size=(zd_vertoffset.shape[0], zd_vertoffset.shape[1]),
        )

    zd_diffcoef = random_field(mesh, CellDim, KDim)
    geofac_n2s_c = random_field(mesh, CellDim)
    geofac_n2s_nbh = random_field(mesh, CellDim, C2E2CDim)
    vcoef = random_field(mesh, CellDim, C2E2CDim, KDim)
    theta_v = random_field(mesh, CellDim, KDim)
    z_temp = random_field(mesh, CellDim, KDim)

    ref = mo_nh_diffusion_stencil_15_numpy(
        mesh.c2e2c,
        np.asarray(mask),
        np.asarray(zd_vertoffset),
        np.asarray(zd_diffcoef),
        np.asarray(geofac_n2s_c),
        np.asarray(geofac_n2s_nbh),
        np.asarray(vcoef),
        np.asarray(theta_v),
        np.asarray(z_temp),
    )

    hstart = 0
    hend = mesh.n_cells
    kstart = 0
    kend = mesh.k_level

    mo_nh_diffusion_stencil_15(
        mask,
        zd_vertoffset,
        zd_diffcoef,
        geofac_n2s_c,
        geofac_n2s_nbh,
        vcoef,
        theta_v,
        z_temp,
        hstart,
        hend,
        kstart,
        kend,
        offset_provider={
            "C2E2C": mesh.get_c2e2c_offset_provider(),
            "Koff": KDim,
        },
    )

    assert np.allclose(z_temp, ref)
