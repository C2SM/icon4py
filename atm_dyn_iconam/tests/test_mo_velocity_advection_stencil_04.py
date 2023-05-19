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
from utils.helpers import random_field, zero_field
from utils.simple_mesh import SimpleMesh

from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_04 import (
    mo_velocity_advection_stencil_04,
)
from icon4py.common.dimension import EdgeDim, KDim


def mo_velocity_advection_stencil_04_numpy(
    vn: np.array, ddxn_z_full: np.array, ddxt_z_full: np.array, vt: np.array
) -> np.array:
    z_w_concorr_me = vn * ddxn_z_full + vt * ddxt_z_full
    return z_w_concorr_me


def test_mo_velocity_advection_stencil_04():
    mesh = SimpleMesh()

    vn = random_field(mesh, EdgeDim, KDim)
    ddxn_z_full = random_field(mesh, EdgeDim, KDim)
    ddxt_z_full = random_field(mesh, EdgeDim, KDim)
    vt = random_field(mesh, EdgeDim, KDim)
    z_w_concorr_me = zero_field(mesh, EdgeDim, KDim)

    ref = mo_velocity_advection_stencil_04_numpy(
        np.asarray(vn),
        np.asarray(ddxn_z_full),
        np.asarray(ddxt_z_full),
        np.asarray(vt),
    )
    mo_velocity_advection_stencil_04(
        vn, ddxn_z_full, ddxt_z_full, vt, z_w_concorr_me, offset_provider={}
    )
    assert np.allclose(z_w_concorr_me, ref)
