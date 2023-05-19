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
from utils import zero_field

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_45 import (
    mo_solve_nonhydro_stencil_45,
)
from icon4py.common.dimension import CellDim, KDim


def mo_solve_nonhydro_stencil_45_numpy(z_alpha: np.array) -> np.array:
    z_alpha = np.zeros_like(z_alpha)
    return z_alpha


def test_mo_solve_nonhydro_stencil_45():
    mesh = SimpleMesh()

    z_alpha = zero_field(mesh, CellDim, KDim)

    ref = mo_solve_nonhydro_stencil_45_numpy(np.asarray(z_alpha))
    mo_solve_nonhydro_stencil_45(
        z_alpha,
        offset_provider={},
    )
    assert np.allclose(z_alpha, ref)
