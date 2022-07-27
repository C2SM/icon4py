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

from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_10 import (
    mo_velocity_advection_stencil_10,
)
from icon4py.common.dimension import E2C2EDim, EdgeDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field




def mo_velocity_advection_stencil_10_numpy(
    wgtfac_c: np.array, z_w_concorr_mc: np.array
) -> np.array:
    z_w_concorr_mc_k_minus_1 = np.roll(z_w_concorr_mc, shift=1, axis=1)
    w_concorr_c = wgtfac_c * z_w_concorr_mc + (1. - wgtfac_c) * z_w_concorr_mc_k_minus_1

    return w_concorr_c


def test_mo_velocity_advection_stencil_10():
    mesh = SimpleMesh()

    wgtfac_c = random_field(mesh, EdgeDim, KDim)
    z_w_concorr_mc = random_field(mesh, EdgeDim, KDim)

    w_concorr_c = zero_field(mesh, EdgeDim, KDim)

    w_concorr_c_ref = mo_velocity_advection_stencil_10_numpy(
        np.asarray(wgtfac_c),
        np.asarray(z_w_concorr_mc)
    )
    mo_velocity_advection_stencil_10(
        wgtfac_c,
        z_w_concorr_mc,
        w_concorr_c,
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(w_concorr_c, w_concorr_c_ref)

