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

from icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_11 import (
    mo_nh_diffusion_stencil_11,
)
from icon4py.common.dimension import CellDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


def mo_nh_diffusion_stencil_11_numpy(
    c2e2c: np.array, theta_v: np.array, theta_ref_mc: np.array, thresh_tdiff
) -> np.array:
    tdiff = theta_v - np.sum(theta_v[c2e2c], axis=1) / 3
    trefdiff = theta_ref_mc - np.sum(theta_ref_mc[c2e2c], axis=1) / 3

    enh_diffu_3d = np.where(
        ((tdiff - trefdiff) < thresh_tdiff) & (trefdiff < 0),
        (thresh_tdiff - tdiff + trefdiff) * 5e-4,
        -1.7976931348623157e308,
    )

    return enh_diffu_3d


def test_mo_nh_diffusion_stencil_11():
    mesh = SimpleMesh()

    theta_v = random_field(mesh, CellDim, KDim)
    theta_ref_mc = random_field(mesh, CellDim, KDim)
    enh_diffu_3d = zero_field(mesh, CellDim, KDim)
    thresh_tdiff = 5.0

    ref = mo_nh_diffusion_stencil_11_numpy(
        mesh.c2e2c, np.asarray(theta_v), np.asarray(theta_ref_mc), thresh_tdiff
    )
    mo_nh_diffusion_stencil_11(
        theta_v,
        theta_ref_mc,
        enh_diffu_3d,
        thresh_tdiff,
        offset_provider={"C2E2C": mesh.get_c2e2c_offset_provider()},
    )
    assert np.allclose(enh_diffu_3d, ref)
