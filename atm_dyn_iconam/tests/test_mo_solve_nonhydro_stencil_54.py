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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_54 import (
    mo_solve_nonhydro_stencil_54,
)
from icon4py.common.dimension import CellDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field


def mo_solve_nonhydro_stencil_54_numpy(
    z_raylfac: np.array, w_1: np.array, w: np.array
) -> np.array:
    z_raylfac = np.expand_dims(z_raylfac, axis=0)
    w_1 = np.expand_dims(w_1, axis=-1)
    w = z_raylfac * w + (float(1.0) - z_raylfac) * w_1
    return w


def test_mo_solve_nonhydro_stencil_54():
    mesh = SimpleMesh()

    z_raylfac = random_field(mesh, KDim)
    w_1 = random_field(mesh, CellDim)
    w = random_field(mesh, CellDim, KDim)

    ref = mo_solve_nonhydro_stencil_54_numpy(
        np.asarray(z_raylfac), np.asarray(w_1), np.asarray(w)
    )
    mo_solve_nonhydro_stencil_54(
        z_raylfac,
        w_1,
        w,
        offset_provider={},
    )
    assert np.allclose(w, ref)
