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
from utils import random_field, zero_field

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_62 import (
    mo_solve_nonhydro_stencil_62,
)
from icon4py.common.dimension import CellDim, KDim


def mo_solve_nonhydro_stencil_62_numpy(
    w_now: np.array, grf_tend_w: np.array, dtime: float
) -> np.array:
    w_new = w_now + dtime * grf_tend_w
    return w_new


def test_mo_solve_nonhydro_stencil_62():
    mesh = SimpleMesh()

    dtime = 10.0
    w_now = random_field(mesh, CellDim, KDim)
    grf_tend_w = random_field(mesh, CellDim, KDim)
    w_new = zero_field(mesh, CellDim, KDim)

    ref = mo_solve_nonhydro_stencil_62_numpy(
        np.asarray(w_now), np.asarray(grf_tend_w), dtime
    )

    mo_solve_nonhydro_stencil_62(
        w_now,
        grf_tend_w,
        w_new,
        dtime,
        offset_provider={},
    )
    assert np.allclose(w_new, ref)
