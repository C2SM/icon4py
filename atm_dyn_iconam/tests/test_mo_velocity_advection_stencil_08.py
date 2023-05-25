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

from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_08 import (
    mo_velocity_advection_stencil_08,
)
from icon4py.common.dimension import C2EDim, CellDim, EdgeDim, KDim

from .test_utils.helpers import random_field, zero_field
from .test_utils.simple_mesh import SimpleMesh


def mo_velocity_advection_stencil_08_numpy(
    c2e: np.array, z_kin_hor_e: np.array, e_bln_c_s: np.array
) -> np.array:
    e_bln_c_s = np.expand_dims(e_bln_c_s, axis=-1)
    z_ekinh = np.sum(z_kin_hor_e[c2e] * e_bln_c_s, axis=1)
    return z_ekinh


def test_mo_velocity_advection_stencil_08():
    mesh = SimpleMesh()

    z_kin_hor_e = random_field(mesh, EdgeDim, KDim)
    e_bln_c_s = random_field(mesh, CellDim, C2EDim)
    z_ekinh = zero_field(mesh, CellDim, KDim)

    ref = mo_velocity_advection_stencil_08_numpy(
        mesh.c2e,
        np.asarray(z_kin_hor_e),
        np.asarray(e_bln_c_s),
    )
    mo_velocity_advection_stencil_08(
        z_kin_hor_e,
        e_bln_c_s,
        z_ekinh,
        offset_provider={"C2E": mesh.get_c2e_offset_provider()},
    )
    assert np.allclose(z_ekinh, ref)
