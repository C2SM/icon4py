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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_38 import (
    mo_solve_nonhydro_stencil_38,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import random_field, zero_field
from icon4py.model.common.test_utils.simple_mesh import SimpleMesh


def mo_solve_nonhydro_stencil_38_numpy(
    vn: np.array,
    wgtfacq_e: np.array,
) -> np.array:
    vn_ie = (
        np.roll(wgtfacq_e, shift=1, axis=1) * np.roll(vn, shift=1, axis=1)
        + np.roll(wgtfacq_e, shift=2, axis=1) * np.roll(vn, shift=2, axis=1)
        + np.roll(wgtfacq_e, shift=3, axis=1) * np.roll(vn, shift=3, axis=1)
    )
    return vn_ie


def test_mo_solve_nonhydro_stencil_38():
    mesh = SimpleMesh()

    wgtfacq_e = random_field(mesh, EdgeDim, KDim)
    vn = random_field(mesh, EdgeDim, KDim)

    vn_ie = zero_field(mesh, EdgeDim, KDim)

    vn_ie_ref = mo_solve_nonhydro_stencil_38_numpy(
        np.asarray(vn),
        np.asarray(wgtfacq_e),
    )

    mo_solve_nonhydro_stencil_38(
        vn,
        wgtfacq_e,
        vn_ie,
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(vn_ie[:, 3:], vn_ie_ref[:, 3:])
