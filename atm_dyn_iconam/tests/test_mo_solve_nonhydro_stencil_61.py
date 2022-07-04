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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_61 import (
    mo_solve_nonhydro_stencil_61_exner_new,
    mo_solve_nonhydro_stencil_61_rho_new,
    mo_solve_nonhydro_stencil_61_w_new,
)
from icon4py.common.dimension import CellDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


def mo_solve_nonhydro_stencil_61_rho_new_numpy(
    rho_now: np.array, grf_tend_rho: np.array, dtime: float
):
    rho_new = rho_now + dtime * grf_tend_rho
    return rho_new


def mo_solve_nonhydro_stencil_61_exner_new_numpy(
    theta_v_now: np.array, grf_tend_thv: np.array, dtime: float
):
    exner_new = theta_v_now + dtime * grf_tend_thv
    return exner_new


def mo_solve_nonhydro_stencil_61_w_new_numpy(
    w_now: np.array, grf_tend_w: np.array, dtime: float
):
    w_new = w_now + dtime * grf_tend_w
    return w_new


def test_mo_solve_nonhydro_stencil_61():
    mesh = SimpleMesh()

    a = random_field(mesh, CellDim, KDim)
    b = random_field(mesh, CellDim, KDim)
    out = zero_field(mesh, CellDim, KDim)
    dtime = 5.0

    stencil_funcs = {
        mo_solve_nonhydro_stencil_61_rho_new_numpy: mo_solve_nonhydro_stencil_61_rho_new,
        mo_solve_nonhydro_stencil_61_exner_new_numpy: mo_solve_nonhydro_stencil_61_exner_new,
        mo_solve_nonhydro_stencil_61_w_new_numpy: mo_solve_nonhydro_stencil_61_w_new,
    }

    for ref_func, func in stencil_funcs.items():
        ref = ref_func(np.asarray(a), np.asarray(b), dtime)
        func(a, b, out, dtime, offset_provider={})
        assert np.allclose(out, ref)
