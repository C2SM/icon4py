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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_01 import (
    mo_solve_nonhydro_stencil_01_z_rth_pr_1,
    mo_solve_nonhydro_stencil_01_z_rth_pr_2,
)
from icon4py.common.dimension import CellDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


def mo_solve_nonhydro_stencil_01_z_rth_pr_1_numpy(z_rth_pr_1: np.array) -> np.array:
    z_rth_pr_1 = np.zeros_like(z_rth_pr_1)
    return z_rth_pr_1


def test_mo_solve_nonhydro_stencil_01_z_rth_pr_1():
    mesh = SimpleMesh()

    z_rth_pr_1 = random_field(mesh, CellDim, KDim)
    z_rth_pr_1_out = zero_field(mesh, CellDim, KDim)

    ref = mo_solve_nonhydro_stencil_01_z_rth_pr_1_numpy(np.asarray(z_rth_pr_1))
    mo_solve_nonhydro_stencil_01_z_rth_pr_1(
        z_rth_pr_1_out,
        offset_provider={},
    )
    assert np.allclose(z_rth_pr_1_out, ref)


def mo_solve_nonhydro_stencil_01_z_rth_pr_2_numpy(z_rth_pr_2: np.array) -> np.array:
    z_rth_pr_2 = np.zeros_like(z_rth_pr_2)
    return z_rth_pr_2


def test_mo_solve_nonhydro_stencil_01_z_rth_pr_2():
    mesh = SimpleMesh()

    z_rth_pr_2 = random_field(mesh, CellDim, KDim)
    z_rth_pr_2_out = zero_field(mesh, CellDim, KDim)

    ref = mo_solve_nonhydro_stencil_01_z_rth_pr_1_numpy(np.asarray(z_rth_pr_2))
    mo_solve_nonhydro_stencil_01_z_rth_pr_2(
        z_rth_pr_2_out,
        offset_provider={},
    )
    assert np.allclose(z_rth_pr_2_out, ref)
