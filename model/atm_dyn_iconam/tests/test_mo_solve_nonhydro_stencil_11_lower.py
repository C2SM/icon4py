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

from icon4py.model.atm_dyn_iconam.mo_solve_nonhydro_stencil_11_lower import (
    mo_solve_nonhydro_stencil_11_lower,
)
from icon4py.model.common.dimension import CellDim, KDim

from .test_utils.helpers import random_field
from .test_utils.simple_mesh import SimpleMesh


def mo_solve_nonhydro_stencil_11_lower_numpy() -> np.array:
    z_theta_v_pr_ic = 0

    return z_theta_v_pr_ic


def test_mo_solve_nonhydro_stencil_11_lower():
    mesh = SimpleMesh()

    z_theta_v_pr_ic = random_field(mesh, CellDim, KDim)

    z_theta_v_pr_ic_ref = mo_solve_nonhydro_stencil_11_lower_numpy()

    mo_solve_nonhydro_stencil_11_lower(
        z_theta_v_pr_ic,
        offset_provider={},
    )

    assert np.allclose(z_theta_v_pr_ic, z_theta_v_pr_ic_ref)
