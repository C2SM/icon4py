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
from icon4py.model.common.dimension import C2EDim, CellDim, EdgeDim, KDim

from icon4py.model.atm_dyn_iconam.mo_velocity_advection_stencil_09 import (
    mo_velocity_advection_stencil_09,
)

from .test_utils.helpers import random_field, zero_field
from .test_utils.simple_mesh import SimpleMesh


def mo_velocity_advection_stencil_09_numpy(
    c2e: np.array, z_w_concorr_me: np.array, e_bln_c_s: np.array
) -> np.array:
    e_bln_c_s = np.expand_dims(e_bln_c_s, axis=-1)
    z_w_concorr_mc = np.sum(z_w_concorr_me[c2e] * e_bln_c_s, axis=1)
    return z_w_concorr_mc


def test_mo_velocity_advection_stencil_09():
    mesh = SimpleMesh()

    z_w_concorr_me = random_field(mesh, EdgeDim, KDim)
    e_bln_c_s = random_field(mesh, CellDim, C2EDim)
    z_w_concorr_mc = zero_field(mesh, CellDim, KDim)

    ref = mo_velocity_advection_stencil_09_numpy(
        mesh.c2e,
        np.asarray(z_w_concorr_me),
        np.asarray(e_bln_c_s),
    )
    mo_velocity_advection_stencil_09(
        z_w_concorr_me,
        e_bln_c_s,
        z_w_concorr_mc,
        offset_provider={"C2E": mesh.get_c2e_offset_provider()},
    )
    assert np.allclose(z_w_concorr_mc, ref)
