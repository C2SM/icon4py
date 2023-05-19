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
from simple_mesh import SimpleMesh
from utils import random_field

from icon4py.atm_dyn_iconam.apply_nabla2_and_nabla4_global_to_vn import (
    apply_nabla2_and_nabla4_global_to_vn,
)
from icon4py.common.dimension import EdgeDim, KDim


def apply_nabla2_and_nabla4_global_to_vn_numpy(
    area_edge: np.array,
    kh_smag_e: np.array,
    z_nabla2_e: np.array,
    z_nabla4_e2: np.array,
    diff_multfac_vn: np.array,
    vn: np.array,
):
    area_edge = np.expand_dims(area_edge, axis=-1)
    diff_multfac_vn = np.expand_dims(diff_multfac_vn, axis=0)
    vn = vn + area_edge * (
        kh_smag_e * z_nabla2_e - diff_multfac_vn * z_nabla4_e2 * area_edge
    )
    return vn


def test_apply_nabla2_and_nabla4_global_to_vn():
    mesh = SimpleMesh()

    area_edge = random_field(mesh, EdgeDim)
    kh_smag_e = random_field(mesh, EdgeDim, KDim)
    z_nabla2_e = random_field(mesh, EdgeDim, KDim)
    z_nabla4_e2 = random_field(mesh, EdgeDim, KDim)
    diff_multfac_vn = random_field(mesh, KDim)
    vn = random_field(mesh, EdgeDim, KDim)

    vn_ref = apply_nabla2_and_nabla4_global_to_vn_numpy(
        np.asarray(area_edge),
        np.asarray(kh_smag_e),
        np.asarray(z_nabla2_e),
        np.asarray(z_nabla4_e2),
        np.asarray(diff_multfac_vn),
        np.asarray(vn),
    )

    apply_nabla2_and_nabla4_global_to_vn(
        area_edge,
        kh_smag_e,
        z_nabla2_e,
        z_nabla4_e2,
        diff_multfac_vn,
        vn,
        offset_provider={},
    )
    assert np.allclose(vn, vn_ref)
