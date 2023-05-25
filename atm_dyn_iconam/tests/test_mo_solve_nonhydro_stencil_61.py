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
    mo_solve_nonhydro_stencil_61,
)
from icon4py.common.dimension import CellDim, KDim

from .test_utils.helpers import random_field, zero_field
from .test_utils.simple_mesh import SimpleMesh


def mo_solve_nonhydro_stencil_61_numpy(
    rho_now: np.array,
    grf_tend_rho: np.array,
    theta_v_now: np.array,
    grf_tend_thv: np.array,
    w_now: np.array,
    grf_tend_w: np.array,
    dtime,
) -> tuple[np.array]:
    rho_new = rho_now + dtime * grf_tend_rho
    exner_new = theta_v_now + dtime * grf_tend_thv
    w_new = w_now + dtime * grf_tend_w
    return rho_new, exner_new, w_new


def test_mo_solve_nonhydro_stencil_61():
    mesh = SimpleMesh()

    rho_now = random_field(mesh, CellDim, KDim)
    grf_tend_rho = random_field(mesh, CellDim, KDim)
    theta_v_now = random_field(mesh, CellDim, KDim)
    grf_tend_thv = random_field(mesh, CellDim, KDim)
    w_now = random_field(mesh, CellDim, KDim)
    grf_tend_w = random_field(mesh, CellDim, KDim)
    dtime = 5.0
    rho_new = zero_field(mesh, CellDim, KDim)
    exner_new = zero_field(mesh, CellDim, KDim)
    w_new = zero_field(mesh, CellDim, KDim)

    rho_new_ref, exner_new_ref, w_new_ref = mo_solve_nonhydro_stencil_61_numpy(
        np.asarray(rho_now),
        np.asarray(grf_tend_rho),
        np.asarray(theta_v_now),
        np.asarray(grf_tend_thv),
        np.asarray(w_now),
        np.asarray(grf_tend_w),
        dtime,
    )
    mo_solve_nonhydro_stencil_61(
        rho_now,
        grf_tend_rho,
        theta_v_now,
        grf_tend_thv,
        w_now,
        grf_tend_w,
        rho_new,
        exner_new,
        w_new,
        dtime,
        offset_provider={},
    )
    assert np.allclose(rho_new, rho_new_ref)
    assert np.allclose(exner_new, exner_new_ref)
    assert np.allclose(w_new, w_new_ref)
