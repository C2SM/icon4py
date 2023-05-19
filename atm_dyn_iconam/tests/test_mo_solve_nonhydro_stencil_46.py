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
from utils.helpers import zero_field
from utils.simple_mesh import SimpleMesh

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_46 import (
    mo_solve_nonhydro_stencil_46,
)
from icon4py.common.dimension import CellDim, KDim


def mo_solve_nonhydro_stencil_46_numpy(
    w_nnew: np.array,
    z_contr_w_fl_l: np.array,
) -> tuple[np.array]:
    w_nnew = np.zeros_like(w_nnew)
    z_contr_w_fl_l = np.zeros_like(z_contr_w_fl_l)
    return w_nnew, z_contr_w_fl_l


def test_mo_solve_nonhydro_stencil_46_z_contr_w_fl_l():
    mesh = SimpleMesh()

    z_contr_w_fl_l = zero_field(mesh, CellDim, KDim)
    w_nnew = zero_field(mesh, CellDim, KDim)

    w_nnew_ref, z_contr_w_fl_l_ref = mo_solve_nonhydro_stencil_46_numpy(
        np.asarray(w_nnew), np.asarray(z_contr_w_fl_l)
    )
    mo_solve_nonhydro_stencil_46(
        w_nnew,
        z_contr_w_fl_l,
        offset_provider={},
    )
    assert np.allclose(w_nnew_ref, w_nnew)
    assert np.allclose(z_contr_w_fl_l_ref, z_contr_w_fl_l)
