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

from icon4py.model.atm_dyn_iconam.mo_velocity_advection_stencil_17 import (
    mo_velocity_advection_stencil_17,
)
from icon4py.model.common.dimension import C2EDim, CellDim, EdgeDim, KDim

from .test_utils.helpers import random_field
from .test_utils.simple_mesh import SimpleMesh


def mo_velocity_advection_stencil_17_numpy(
    c2e: np.array, e_bln_c_s: np.array, z_v_grad_w: np.array, ddt_w_adv: np.array
) -> np.array:
    e_bln_c_s = np.expand_dims(e_bln_c_s, axis=-1)
    ddt_w_adv = ddt_w_adv + np.sum(z_v_grad_w[c2e] * e_bln_c_s, axis=1)
    return ddt_w_adv


def test_mo_velocity_advection_stencil_17():
    mesh = SimpleMesh()

    z_v_grad_w = random_field(mesh, EdgeDim, KDim)
    e_bln_c_s = random_field(mesh, CellDim, C2EDim)
    ddt_w_adv = random_field(mesh, CellDim, KDim)

    ref = mo_velocity_advection_stencil_17_numpy(
        mesh.c2e, np.asarray(e_bln_c_s), np.asarray(z_v_grad_w), np.asarray(ddt_w_adv)
    )
    mo_velocity_advection_stencil_17(
        e_bln_c_s,
        z_v_grad_w,
        ddt_w_adv,
        offset_provider={"C2E": mesh.get_c2e_offset_provider()},
    )
    assert np.allclose(ddt_w_adv, ref)
