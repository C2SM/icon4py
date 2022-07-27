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

from icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_13 import (
    mo_nh_diffusion_stencil_13,
)
from icon4py.common.dimension import CellDim, EdgeDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field


def mo_nh_diffusion_stencil_13_numpy(
    e2c: np.array,
    kh_smag_e: np.array,
    inv_dual_edge_length: np.array,
    theta_v: np.array,
) -> np.array:
    inv_dual_edge_length = np.expand_dims(inv_dual_edge_length, axis=-1)

    theta_v_e2c = theta_v[e2c]
    theta_v_weighted = theta_v_e2c[:, 1] - theta_v_e2c[:, 0]

    z_nabla2_e = kh_smag_e * inv_dual_edge_length * theta_v_weighted
    return z_nabla2_e


def test_mo_nh_diffusion_stencil_13():
    mesh = SimpleMesh()

    kh_smag_e = random_field(mesh, EdgeDim, KDim)
    inv_dual_edge_length = random_field(mesh, EdgeDim)
    theta_v = random_field(mesh, CellDim, KDim)
    z_nabla2_e = random_field(mesh, EdgeDim, KDim)

    ref = mo_nh_diffusion_stencil_13_numpy(
        mesh.e2c,
        np.asarray(kh_smag_e),
        np.asarray(inv_dual_edge_length),
        np.asarray(theta_v),
    )

    mo_nh_diffusion_stencil_13(
        kh_smag_e,
        inv_dual_edge_length,
        theta_v,
        z_nabla2_e,
        offset_provider={
            "E2C": mesh.get_e2c_offset_provider(),
        },
    )
    assert np.allclose(z_nabla2_e, ref)
