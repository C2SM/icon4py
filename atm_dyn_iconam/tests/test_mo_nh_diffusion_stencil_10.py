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

from icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_10 import (
    mo_nh_diffusion_stencil_10,
)
from icon4py.common.dimension import CellDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field


def mo_nh_diffusion_stencil_10_numpy(
    w: np.array,
    diff_multfac_n2w: np.array,
    cell_area: np.array,
    z_nabla2_c: np.array,
) -> np.array:
    cell_area = np.expand_dims(cell_area, axis=-1)
    w = w + diff_multfac_n2w * cell_area * z_nabla2_c
    return w


def test_mo_nh_diffusion_stencil_10():
    mesh = SimpleMesh()

    w = random_field(mesh, CellDim, KDim)
    diff_multfac_n2w = random_field(mesh, KDim)
    cell_area = random_field(mesh, CellDim)
    z_nabla2_c = random_field(mesh, CellDim, KDim)

    ref = mo_nh_diffusion_stencil_10_numpy(
        np.asarray(w),
        np.asarray(diff_multfac_n2w),
        np.asarray(cell_area),
        np.asarray(z_nabla2_c),
    )
    mo_nh_diffusion_stencil_10(
        w,
        diff_multfac_n2w,
        cell_area,
        z_nabla2_c,
        offset_provider={},
    )
    assert np.allclose(w, ref)
