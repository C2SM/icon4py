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


def mo_solve_nonhydro_stencil_20_numpy(
    e2c: np.array,
    inv_dual_edge_length: np.array,
    z_exner_ex_pr: np.array,
    zdiff_gradp: np.array,
    ikidx: np.array,
    z_dexner_dz_c_1: np.array,
    z_dexner_dz_c_2: np.array,
) -> np.array:
    # todo: implement numpy version
    return None


def test_mo_solve_nonhydro_stencil_20():
    mesh = SimpleMesh()

    inv_dual_edge_length = random_field(mesh, EdgeDim)
    z_exner_ex_pr = random_field(mesh, CellDim, KDim)
    zdiff_gradp = random_field(mesh, CellDim, E2CDim, KDim)
    ikidx = zero_field(mesh, CellDim, E2CDim, KDim, dtype=int)
    z_dexner_dz_c_1 = random_field(mesh, CellDim, KDim)
    z_dexner_dz_c_2 = random_field(mesh, CellDim, KDim)

    mo_solve_nonhydro_stencil_20_numpy(
        mesh.e2c,
        np.asarray(inv_dual_edge_length),
        np.asarray(z_exner_ex_pr),
        np.asarray(zdiff_gradp),
        np.asarray(ikidx),
        np.asarray(z_dexner_dz_c_1),
        np.asarray(z_dexner_dz_c_2),
    )

    # todo: call gt4py stencil
    # todo: assert equality between numpy and gt4py
