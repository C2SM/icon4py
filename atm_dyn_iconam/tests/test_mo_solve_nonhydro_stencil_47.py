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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_47 import (
    mo_solve_nonhydro_stencil_47,
)
from icon4py.common.dimension import CellDim, KDim

from .test_utils.helpers import random_field, zero_field
from .test_utils.simple_mesh import SimpleMesh


def mo_solve_nonhydro_stencil_47_numpy(
    w_concorr_c: np.array,
    z_contr_w_fl_l: np.array,
) -> tuple[np.array]:
    w_nnew = w_concorr_c
    z_contr_w_fl_l = np.zeros_like(z_contr_w_fl_l)
    return w_nnew, z_contr_w_fl_l


def test_mo_solve_nonhydro_stencil_47_z_contr_w_fl_l():
    mesh = SimpleMesh()

    w_concorr_c = random_field(mesh, CellDim, KDim)
    z_contr_w_fl_l = zero_field(mesh, CellDim, KDim)
    w_nnew = zero_field(mesh, CellDim, KDim)

    w_nnew_ref, z_contr_w_fl_l_ref = mo_solve_nonhydro_stencil_47_numpy(
        np.asarray(w_concorr_c), np.asarray(z_contr_w_fl_l)
    )
    mo_solve_nonhydro_stencil_47(
        w_nnew,
        z_contr_w_fl_l,
        w_concorr_c,
        offset_provider={},
    )
    assert np.allclose(w_nnew, w_nnew_ref)
    assert np.allclose(z_contr_w_fl_l, z_contr_w_fl_l_ref)
