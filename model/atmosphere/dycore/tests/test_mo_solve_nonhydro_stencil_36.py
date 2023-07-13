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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_36 import (
    mo_solve_nonhydro_stencil_36,
)
from icon4py.model.common.dimension import EdgeDim, KDim

from icon4py.model.common.test_utils.helpers import random_field, zero_field
from icon4py.model.common.test_utils.simple_mesh import SimpleMesh


def mo_solve_nonhydro_stencil_36_numpy(
    wgtfac_e: np.array,
    vn: np.array,
    vt: np.array,
) -> tuple[np.array, np.array]:
    vn_offset_1 = np.roll(vn, shift=1, axis=1)
    vt_offset_1 = np.roll(vt, shift=1, axis=1)

    vn_ie = wgtfac_e * vn + (1.0 - wgtfac_e) * vn_offset_1
    z_vt_ie = wgtfac_e * vt + (1.0 - wgtfac_e) * vt_offset_1
    z_kin_hor_e = 0.5 * (vn**2 + vt**2)
    return vn_ie, z_vt_ie, z_kin_hor_e


def test_mo_solve_nonhydro_stencil_36():
    mesh = SimpleMesh()

    wgtfac_e = zero_field(mesh, EdgeDim, KDim)
    vn = random_field(mesh, EdgeDim, KDim)
    vt = random_field(mesh, EdgeDim, KDim)

    vn_ie = zero_field(mesh, EdgeDim, KDim)
    z_vt_ie = zero_field(mesh, EdgeDim, KDim)
    z_kin_hor_e = zero_field(mesh, EdgeDim, KDim)

    vn_ie_ref, z_vt_ie_ref, z_kin_hor_e_ref = mo_solve_nonhydro_stencil_36_numpy(
        np.asarray(wgtfac_e),
        np.asarray(vn),
        np.asarray(vt),
    )

    mo_solve_nonhydro_stencil_36(
        wgtfac_e,
        vn,
        vt,
        vn_ie,
        z_vt_ie,
        z_kin_hor_e,
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(vn_ie[:, 1:], vn_ie_ref[:, 1:])
    assert np.allclose(z_vt_ie[:, 1:], z_vt_ie_ref[:, 1:])
    assert np.allclose(z_kin_hor_e[:, 1:], z_kin_hor_e_ref[:, 1:])
