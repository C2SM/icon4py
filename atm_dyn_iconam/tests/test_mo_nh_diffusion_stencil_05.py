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

from icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_05 import (
    mo_nh_diffusion_stencil_05,
)
from icon4py.common.dimension import EdgeDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field


def mo_nh_diffusion_stencil_05_numpy(
    area_edge: np.array,
    kh_smag_e: np.array,
    z_nabla2_e: np.array,
    z_nabla4_e2: np.array,
    diff_multfac_vn: np.array,
    nudgecoeff_e: np.array,
    vn: np.array,
    nudgezone_diff,
):
    area_edge = np.expand_dims(area_edge, axis=-1)
    diff_multfac_vn = np.expand_dims(diff_multfac_vn, axis=0)
    nudgecoeff_e = np.expand_dims(nudgecoeff_e, axis=-1)
    vn = vn + area_edge * (
        np.maximum(nudgezone_diff * nudgecoeff_e, kh_smag_e) * z_nabla2_e
        - diff_multfac_vn * z_nabla4_e2 * area_edge
    )
    return vn


def test_mo_nh_diffusion_stencil_05():
    mesh = SimpleMesh()

    area_edge = random_field(mesh, EdgeDim)
    kh_smag_e = random_field(mesh, EdgeDim, KDim)
    z_nabla2_e = random_field(mesh, EdgeDim, KDim)
    z_nabla4_e2 = random_field(mesh, EdgeDim, KDim)
    diff_multfac_vn = random_field(mesh, KDim)
    nudgecoeff_e = random_field(mesh, EdgeDim)
    vn = random_field(mesh, EdgeDim, KDim)
    nudgezone_diff = 9.0

    vn_ref = mo_nh_diffusion_stencil_05_numpy(
        np.asarray(area_edge),
        np.asarray(kh_smag_e),
        np.asarray(z_nabla2_e),
        np.asarray(z_nabla4_e2),
        np.asarray(diff_multfac_vn),
        np.asarray(nudgecoeff_e),
        np.asarray(vn),
        nudgezone_diff,
    )

    mo_nh_diffusion_stencil_05(
        area_edge,
        kh_smag_e,
        z_nabla2_e,
        z_nabla4_e2,
        diff_multfac_vn,
        nudgecoeff_e,
        vn,
        nudgezone_diff,
        offset_provider={},
    )
    assert np.allclose(vn, vn_ref)
