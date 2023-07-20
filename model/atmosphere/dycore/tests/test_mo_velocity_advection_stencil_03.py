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

from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_03 import (
    mo_velocity_advection_stencil_03,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import random_field, zero_field
from icon4py.model.common.test_utils.simple_mesh import SimpleMesh


def mo_velocity_advection_stencil_03_numpy(wgtfac_e: np.array, vt: np.array) -> np.array:
    vt_k_minus_1 = np.roll(vt, shift=1, axis=1)
    z_vt_ie = wgtfac_e * vt + (1.0 - wgtfac_e) * vt_k_minus_1

    return z_vt_ie


def test_mo_velocity_advection_stencil_03():
    mesh = SimpleMesh()

    wgtfac_e = random_field(mesh, EdgeDim, KDim)
    vt = random_field(mesh, EdgeDim, KDim)

    z_vt_ie = zero_field(mesh, EdgeDim, KDim)

    z_vt_ie_ref = mo_velocity_advection_stencil_03_numpy(np.asarray(wgtfac_e), np.asarray(vt))
    mo_velocity_advection_stencil_03(
        wgtfac_e,
        vt,
        z_vt_ie,
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(z_vt_ie, z_vt_ie_ref)
