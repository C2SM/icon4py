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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_28 import (
    mo_solve_nonhydro_stencil_28,
)
from icon4py.common.dimension import EdgeDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field


def mo_solve_nonhydro_stencil_28_numpy(
    vn_incr: np.array, vn: np.array, iau_wgt_dyn
) -> np.array:
    vn = vn + (iau_wgt_dyn * vn_incr)
    return vn


def test_mo_solve_nonhydro_stencil_28():
    mesh = SimpleMesh()

    vn_incr = random_field(mesh, EdgeDim, KDim)
    vn = random_field(mesh, EdgeDim, KDim)
    iau_wgt_dyn = np.float64(5.0)

    ref = mo_solve_nonhydro_stencil_28_numpy(
        np.asarray(vn_incr), np.asarray(vn), iau_wgt_dyn
    )
    mo_solve_nonhydro_stencil_28(
        vn_incr,
        vn,
        iau_wgt_dyn,
        offset_provider={},
    )
    assert np.allclose(vn, ref)
