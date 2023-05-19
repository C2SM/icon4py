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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_13 import (
    mo_solve_nonhydro_stencil_13,
)
from icon4py.common.dimension import CellDim, KDim


def mo_solve_nonhydro_stencil_13_numpy(
    rho: np.array, rho_ref_mc: np.array, theta_v: np.array, theta_ref_mc: np.array
) -> tuple[np.array]:
    z_rth_pr_1 = rho - rho_ref_mc
    z_rth_pr_2 = theta_v - theta_ref_mc
    return z_rth_pr_1, z_rth_pr_2


def test_mo_solve_nonhydro_stencil_13():
    mesh = SimpleMesh()

    rho = random_field(mesh, CellDim, KDim)
    rho_ref_mc = random_field(mesh, CellDim, KDim)
    theta_v = random_field(mesh, CellDim, KDim)
    theta_ref_mc = random_field(mesh, CellDim, KDim)
    z_rth_pr_1 = zero_field(mesh, CellDim, KDim)
    z_rth_pr_2 = zero_field(mesh, CellDim, KDim)

    z_rth_pr_1_ref, z_rth_pr_2_ref = mo_solve_nonhydro_stencil_13_numpy(
        np.asarray(rho),
        np.asarray(rho_ref_mc),
        np.asarray(theta_v),
        np.asarray(theta_ref_mc),
    )
    mo_solve_nonhydro_stencil_13(
        rho,
        rho_ref_mc,
        theta_v,
        theta_ref_mc,
        z_rth_pr_1,
        z_rth_pr_2,
        offset_provider={},
    )

    assert np.allclose(z_rth_pr_1, z_rth_pr_1_ref)
    assert np.allclose(z_rth_pr_2, z_rth_pr_2_ref)
