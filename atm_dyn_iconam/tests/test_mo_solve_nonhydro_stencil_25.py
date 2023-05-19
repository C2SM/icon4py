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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_25 import (
    mo_solve_nonhydro_stencil_25,
)
from icon4py.common.dimension import E2C2EODim, EdgeDim, KDim


def mo_solve_nonhydro_stencil_25_numpy(
    e2c2eO: np.array, geofac_grdiv: np.array, z_graddiv_vn: np.array
) -> np.array:
    geofac_grdiv = np.expand_dims(geofac_grdiv, axis=-1)
    z_temp = np.sum(z_graddiv_vn[e2c2eO] * geofac_grdiv, axis=1)
    return z_temp


def test_mo_solve_nonhydro_stencil_25():
    mesh = SimpleMesh()

    z_graddiv_vn = random_field(mesh, EdgeDim, KDim)
    geofac_grdiv = random_field(mesh, EdgeDim, E2C2EODim)
    z_graddiv2_vn = zero_field(mesh, EdgeDim, KDim)

    ref = mo_solve_nonhydro_stencil_25_numpy(
        mesh.e2c2eO, np.asarray(geofac_grdiv), np.asarray(z_graddiv_vn)
    )
    mo_solve_nonhydro_stencil_25(
        geofac_grdiv,
        z_graddiv_vn,
        z_graddiv2_vn,
        offset_provider={"E2C2EO": mesh.get_e2c2eO_offset_provider()},
    )
    assert np.allclose(z_graddiv2_vn, ref)
