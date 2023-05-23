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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_53 import (
    mo_solve_nonhydro_stencil_53,
)
from icon4py.common.dimension import CellDim, KDim

from .simple_mesh import SimpleMesh
from .utils import random_field


def mo_solve_nonhydro_stencil_53_numpy(
    z_q: np.array,
    w: np.array,
) -> np.array:
    w_new = np.zeros_like(w)
    last_k_level = w.shape[1] - 1

    w_new[:, last_k_level] = w[:, last_k_level]
    for k in reversed(range(1, last_k_level)):
        w_new[:, k] = w[:, k] + w_new[:, k + 1] * z_q[:, k]
    w_new[:, 0] = w[:, 0]
    return w_new


def test_mo_solve_nonhydro_stencil_53():
    mesh = SimpleMesh()
    z_q = random_field(mesh, CellDim, KDim)
    w = random_field(mesh, CellDim, KDim)

    w_ref = mo_solve_nonhydro_stencil_53_numpy(
        np.asarray(z_q),
        np.asarray(w),
    )

    mo_solve_nonhydro_stencil_53(z_q, w, offset_provider={"Koff": KDim})

    assert np.allclose(w_ref, w)
