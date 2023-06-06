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

from icon4py.model.atm_dyn_iconam.mo_solve_nonhydro_stencil_29 import (
    mo_solve_nonhydro_stencil_29,
)
from icon4py.model.common.dimension import EdgeDim, KDim

from .test_utils.helpers import random_field, zero_field
from .test_utils.simple_mesh import SimpleMesh


def mo_solve_nonhydro_stencil_29_numpy(
    grf_tend_vn: np.array, vn_now: np.array, dtime
) -> np.array:
    vn_new = vn_now + dtime * grf_tend_vn
    return vn_new


def test_mo_solve_nonhydro_stencil_29():
    mesh = SimpleMesh()

    grf_tend_vn = random_field(mesh, EdgeDim, KDim)
    vn_now = random_field(mesh, EdgeDim, KDim)
    vn_new = zero_field(mesh, EdgeDim, KDim)
    dtime = 6.0

    ref = mo_solve_nonhydro_stencil_29_numpy(
        np.asarray(grf_tend_vn), np.asarray(vn_now), dtime
    )
    mo_solve_nonhydro_stencil_29(
        grf_tend_vn,
        vn_now,
        vn_new,
        dtime,
        offset_provider={},
    )
    assert np.allclose(vn_new, ref)
