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

from icon4py.atm_dyn_iconam.apply_nabla2_vn_in_lateral_boundary import (
    apply_nabla2_vn_in_lateral_boundary,
)
from icon4py.common.dimension import EdgeDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field


def apply_nabla2_vn_in_lateral_boundary_numpy(
    z_nabla2_e: np.array, area_edge: np.array, vn: np.array, fac_bdydiff_v
) -> np.array:
    area_edge = np.expand_dims(area_edge, axis=-1)
    vn = vn + (z_nabla2_e * area_edge * fac_bdydiff_v)
    return vn


def test_apply_nabla2_vn_in_lateral_boundary():
    mesh = SimpleMesh()

    fac_bdydiff_v = 5.0
    z_nabla2_e = random_field(mesh, EdgeDim, KDim)
    area_edge = random_field(mesh, EdgeDim)
    vn = random_field(mesh, EdgeDim, KDim)

    ref = apply_nabla2_vn_in_lateral_boundary_numpy(
        np.asarray(z_nabla2_e), np.asarray(area_edge), np.asarray(vn), fac_bdydiff_v
    )
    apply_nabla2_vn_in_lateral_boundary(
        z_nabla2_e, area_edge, vn, fac_bdydiff_v, offset_provider={}
    )
    assert np.allclose(vn, ref)
