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

from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_02 import (
    mo_velocity_advection_stencil_02,
)
from icon4py.common.dimension import EdgeDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


def mo_velocity_advection_stencil_02_vn_ie_numpy(
    wgtfac_e: np.array, vn: np.array
) -> np.array:
    vn_ie_k_minus_1 = np.roll(vn, shift=1, axis=1)
    vn_ie = wgtfac_e * vn + (1.0 - wgtfac_e) * vn_ie_k_minus_1
    return vn_ie


def mo_velocity_advection_stencil_02_z_kin_hor_e_numpy(
    vn: np.array, vt: np.array
) -> np.array:
    z_kin_hor_e = 0.5 * (vn * vn + vt * vt)
    return z_kin_hor_e


def mo_velocity_advection_stencil_02_numpy(
    wgtfac_e: np.array, vn: np.array, vt: np.array
) -> tuple[np.array]:
    vn_ie = mo_velocity_advection_stencil_02_vn_ie_numpy(wgtfac_e, vn)
    z_kin_hor_e = mo_velocity_advection_stencil_02_z_kin_hor_e_numpy(vn, vt)

    return vn_ie, z_kin_hor_e


def test_mo_velocity_advection_stencil_02():
    mesh = SimpleMesh()

    wgtfac_e = random_field(mesh, EdgeDim, KDim)
    vn = random_field(mesh, EdgeDim, KDim)
    vt = random_field(mesh, EdgeDim, KDim)

    vn_ie = zero_field(mesh, EdgeDim, KDim)
    z_kin_hor_e = zero_field(mesh, EdgeDim, KDim)

    vn_ie_ref, z_kin_hor_e_ref = mo_velocity_advection_stencil_02_numpy(
        np.asarray(wgtfac_e),
        np.asarray(vn),
        np.asarray(vt),
    )
    mo_velocity_advection_stencil_02(
        wgtfac_e,
        vn,
        vt,
        vn_ie,
        z_kin_hor_e,
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(vn_ie, vn_ie_ref)
    assert np.allclose(z_kin_hor_e, z_kin_hor_e_ref)
