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

from icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_06 import (
    mo_nh_diffusion_stencil_06,
)
from icon4py.common.dimension import EdgeDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field


fac_bdydiff_v = np.float32(5.0)


def mo_nh_diffusion_stencil_06_numpy(
    z_nabla2_e: np.array, area_edge: np.array, vn: np.array
) -> np.array:
    area_edge = np.expand_dims(area_edge, axis=-1)
    vn = vn + (z_nabla2_e * area_edge * fac_bdydiff_v)
    return vn


def test_mo_nh_diffusion_stencil_06():
    mesh = SimpleMesh()

    z_nabla2_e = random_field(mesh, EdgeDim, KDim)
    area_edge = random_field(mesh, EdgeDim)
    vn = random_field(mesh, EdgeDim, KDim)

    ref = mo_nh_diffusion_stencil_06_numpy(
        np.asarray(z_nabla2_e), np.asarray(area_edge), np.asarray(vn)
    )
    mo_nh_diffusion_stencil_06(z_nabla2_e, area_edge, vn, offset_provider={})
    assert np.allclose(vn, ref)
