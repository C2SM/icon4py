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

from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_06 import (
    mo_velocity_advection_stencil_06,
)
from icon4py.common.dimension import EdgeDim, KDim


def mo_velocity_advection_stencil_06_numpy(
    wgtfacq_e: np.array, vn: np.array
) -> np.array:
    vn_k_minus_1 = np.roll(vn, shift=1, axis=1)
    vn_k_minus_2 = np.roll(vn, shift=2, axis=1)
    vn_k_minus_3 = np.roll(vn, shift=3, axis=1)
    wgtfacq_e_k_minus_1 = np.roll(wgtfacq_e, shift=1, axis=1)
    wgtfacq_e_k_minus_2 = np.roll(wgtfacq_e, shift=2, axis=1)
    wgtfacq_e_k_minus_3 = np.roll(wgtfacq_e, shift=3, axis=1)
    vn_ie = (
        wgtfacq_e_k_minus_1 * vn_k_minus_1
        + wgtfacq_e_k_minus_2 * vn_k_minus_2
        + wgtfacq_e_k_minus_3 * vn_k_minus_3
    )

    return vn_ie


def test_mo_velocity_advection_stencil_06():
    mesh = SimpleMesh()

    wgtfacq_e = random_field(mesh, EdgeDim, KDim)
    vn = random_field(mesh, EdgeDim, KDim)

    vn_ie = zero_field(mesh, EdgeDim, KDim)

    vn_ie_ref = mo_velocity_advection_stencil_06_numpy(
        np.asarray(wgtfacq_e), np.asarray(vn)
    )
    mo_velocity_advection_stencil_06(
        wgtfacq_e,
        vn,
        vn_ie,
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(vn_ie, vn_ie_ref)
