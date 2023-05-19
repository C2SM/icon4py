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
from utils.helpers import random_field, zero_field
from utils.simple_mesh import SimpleMesh

from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_15 import (
    mo_velocity_advection_stencil_15,
)
from icon4py.common.dimension import CellDim, KDim


def mo_velocity_advection_stencil_15_numpy(
    z_w_con_c: np.array,
):
    z_w_con_c_full = 0.5 * (z_w_con_c[:, :-1] + z_w_con_c[:, 1:])
    return z_w_con_c_full


def test_mo_velocity_advection_stencil_15():
    mesh = SimpleMesh()

    z_w_con_c = random_field(mesh, CellDim, KDim, extend={KDim: 1})

    z_w_con_c_full = zero_field(mesh, CellDim, KDim)

    z_w_con_c_full_ref = mo_velocity_advection_stencil_15_numpy(
        np.asarray(z_w_con_c),
    )

    mo_velocity_advection_stencil_15(
        z_w_con_c,
        z_w_con_c_full,
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(z_w_con_c_full, z_w_con_c_full_ref)
