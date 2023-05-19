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
from utils.helpers import random_field
from utils.simple_mesh import SimpleMesh

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_26 import (
    mo_solve_nonhydro_stencil_26,
)
from icon4py.common.dimension import EdgeDim, KDim


def mo_solve_nonhydro_stencil_26_numpy(
    z_graddiv_vn: np.array, vn: np.array, scal_divdamp_o2
) -> np.array:
    vn = vn + (scal_divdamp_o2 * z_graddiv_vn)
    return vn


def test_mo_solve_nonhydro_stencil_26():
    mesh = SimpleMesh()

    z_graddiv_vn = random_field(mesh, EdgeDim, KDim)
    vn = random_field(mesh, EdgeDim, KDim)
    scal_divdamp_o2 = 5.0

    ref = mo_solve_nonhydro_stencil_26_numpy(
        np.asarray(z_graddiv_vn), np.asarray(vn), scal_divdamp_o2
    )
    mo_solve_nonhydro_stencil_26(
        z_graddiv_vn,
        vn,
        scal_divdamp_o2,
        offset_provider={},
    )
    assert np.allclose(vn, ref)
