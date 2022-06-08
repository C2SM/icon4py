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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_24 import (
    mo_solve_nonhydro_stencil_24,
)
from icon4py.common.dimension import EdgeDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


def mo_solve_nonhydro_stencil_24_numpy(
    vn_nnow: np.array,
    ddt_vn_adv_ntl1: np.array,
    ddt_vn_phy: np.array,
    z_theta_v_e: np.array,
    z_gradh_exner: np.array,
    dtime: float,
    cpd: float,
) -> np.array:
    vn_nnew = vn_nnow + dtime * (
        ddt_vn_adv_ntl1 + ddt_vn_phy - cpd * z_theta_v_e * z_gradh_exner
    )
    return vn_nnew


def test_mo_solve_nonhydro_stencil_24():
    mesh = SimpleMesh()

    dtime, cpd = np.float64(10.0), np.float64(10.0)
    vn_nnow = random_field(mesh, EdgeDim, KDim)
    ddt_vn_adv_ntl1 = random_field(mesh, EdgeDim, KDim)
    ddt_vn_phy = random_field(mesh, EdgeDim, KDim)
    z_theta_v_e = random_field(mesh, EdgeDim, KDim)
    z_gradh_exner = random_field(mesh, EdgeDim, KDim)
    vn_nnew = zero_field(mesh, EdgeDim, KDim)

    ref = mo_solve_nonhydro_stencil_24_numpy(
        np.asarray(vn_nnow),
        np.asarray(ddt_vn_adv_ntl1),
        np.asarray(ddt_vn_phy),
        np.asarray(z_theta_v_e),
        np.asarray(z_gradh_exner),
        dtime,
        cpd,
    )
    mo_solve_nonhydro_stencil_24(
        vn_nnow,
        ddt_vn_adv_ntl1,
        ddt_vn_phy,
        z_theta_v_e,
        z_gradh_exner,
        vn_nnew,
        dtime,
        cpd,
        offset_provider={},
    )
    assert np.allclose(vn_nnew, ref)
