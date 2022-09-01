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

from icon4py.common.dimension import CellDim, E2CDim, EdgeDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


def mo_solve_nonhydro_stencil_21_numpy(
    e2c: np.array,
    theta_v: np.array,
    ikidx: np.array,
    zdiff_gradp: np.array,
    theta_v_ic: np.array,
    inv_ddqz_z_full: np.array,
    inv_dual_edge_length: np.array,
) -> tuple[np.array]:
    # todo: implement numpy version
    z_theta1, z_theta2, z_hydro_corr = None, None, None
    return z_theta1, z_theta2, z_hydro_corr


def test_mo_solve_nonhydro_stencil_21():
    mesh = SimpleMesh()

    theta_v = random_field(mesh, CellDim, KDim)
    ikidx = zero_field(mesh, CellDim, E2CDim, KDim, dtype=int)
    zdiff_grap = random_field(mesh, CellDim, E2CDim, KDim)
    theta_v_ic = random_field(mesh, CellDim, KDim)
    inv_ddqz_z_full = random_field(mesh, CellDim, KDim)
    inv_dual_edge_length = random_field(mesh, EdgeDim)

    mo_solve_nonhydro_stencil_21_numpy(
        mesh.e2c,
        np.asarray(theta_v),
        np.asarray(ikidx),
        np.asarray(zdiff_grap),
        np.asarray(theta_v_ic),
        np.asarray(inv_ddqz_z_full),
        np.asarray(inv_dual_edge_length),
    )

    # todo: call gt4py stencil
    # todo: assert equality between numpy and gt4py
