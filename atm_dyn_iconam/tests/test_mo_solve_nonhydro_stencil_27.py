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
from simple_mesh import SimpleMesh
from utils import random_field

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_27 import (
    mo_solve_nonhydro_stencil_27,
)
from icon4py.common.dimension import EdgeDim, KDim


def mo_solve_nonhydro_stencil_27_numpy(
    scal_divdamp: np.array,
    bdy_divdamp: np.array,
    nudgecoeff_e: np.array,
    z_graddiv2_vn: np.array,
    vn: np.array,
) -> np.array:
    nudgecoeff_e = np.expand_dims(nudgecoeff_e, axis=-1)
    vn = vn + (scal_divdamp + bdy_divdamp * nudgecoeff_e) * z_graddiv2_vn
    return vn


def test_mo_solve_nonhydro_stencil_27():
    mesh = SimpleMesh()

    scal_divdamp = random_field(mesh, KDim)
    bdy_divdamp = random_field(mesh, KDim)
    nudgecoeff_e = random_field(mesh, EdgeDim)
    z_graddiv2_vn = random_field(mesh, EdgeDim, KDim)
    vn = random_field(mesh, EdgeDim, KDim)

    ref = mo_solve_nonhydro_stencil_27_numpy(
        np.asarray(scal_divdamp),
        np.asarray(bdy_divdamp),
        np.asarray(nudgecoeff_e),
        np.asarray(z_graddiv2_vn),
        np.asarray(vn),
    )
    mo_solve_nonhydro_stencil_27(
        scal_divdamp,
        bdy_divdamp,
        nudgecoeff_e,
        z_graddiv2_vn,
        vn,
        offset_provider={},
    )
    assert np.allclose(vn, ref)
