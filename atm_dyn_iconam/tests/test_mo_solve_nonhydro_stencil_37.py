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

from typing import Tuple

import numpy as np

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_37 import (
    mo_solve_nonhydro_stencil_37,
)
from icon4py.common.dimension import EdgeDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


def mo_solve_nonhydro_stencil_37_vn_ie_numpy(vn: np.array) -> np.array:
    vn_ie = vn
    return vn_ie


def mo_solve_nonhydro_stencil_37_z_vt_ie_numpy(vt: np.array) -> np.array:
    z_vt_ie = vt
    return z_vt_ie


def mo_solve_nonhydro_stencil_37_z_kin_hor_e_numpy(
    vn: np.array, vt: np.array
) -> np.array:
    z_kin_hor_e = 0.5 * (pow(vn, 2) + pow(vt, 2))
    return z_kin_hor_e


def mo_solve_nonhydro_stencil_37_numpy(vn: np.array, vt: np.array) -> Tuple[np.array]:
    vn_ie = mo_solve_nonhydro_stencil_37_vn_ie_numpy(vn)
    z_vt_ie = mo_solve_nonhydro_stencil_37_z_vt_ie_numpy(vt)
    z_kin_hor_e = mo_solve_nonhydro_stencil_37_z_kin_hor_e_numpy(vn, vt)
    return vn_ie, z_vt_ie, z_kin_hor_e


def test_mo_solve_nonhydro_stencil_37_z_kin_hor_e():
    mesh = SimpleMesh()

    vt = random_field(mesh, EdgeDim, KDim)
    vn = random_field(mesh, EdgeDim, KDim)
    vn_ie = zero_field(mesh, EdgeDim, KDim)
    z_kin_hor_e = zero_field(mesh, EdgeDim, KDim)
    z_vt_ie = zero_field(mesh, EdgeDim, KDim)

    vn_ie_ref, z_vt_ie_ref, z_kin_hor_e_ref = mo_solve_nonhydro_stencil_37_numpy(
        np.asarray(vn), np.asarray(vt)
    )
    mo_solve_nonhydro_stencil_37(
        vn,
        vt,
        vn_ie,
        z_vt_ie,
        z_kin_hor_e,
        offset_provider={},
    )
    assert np.allclose(z_kin_hor_e_ref, z_kin_hor_e)
    assert np.allclose(vn_ie_ref, vn_ie)
    assert np.allclose(z_vt_ie_ref, z_vt_ie)
