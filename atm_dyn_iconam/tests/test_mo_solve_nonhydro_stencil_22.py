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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_22 import (
    mo_solve_nonhydro_stencil_22,
)
from icon4py.common.dimension import EdgeDim, KDim

from .test_utils.helpers import random_field, random_mask
from .test_utils.simple_mesh import SimpleMesh


def mo_solve_nonhydro_stencil_22_numpy(
    ipeidx_dsl: np.array,
    pg_exdist: np.array,
    z_hydro_corr: np.array,
    z_gradh_exner: np.array,
) -> np.array:
    z_hydro_corr = np.expand_dims(z_hydro_corr, axis=-1)
    z_gradh_exner = np.where(
        ipeidx_dsl, z_gradh_exner + z_hydro_corr * pg_exdist, z_gradh_exner
    )
    return z_gradh_exner


def test_mo_solve_nonhydro_stencil_22():
    mesh = SimpleMesh()

    ipeidx_dsl = random_mask(mesh, EdgeDim, KDim)
    pg_exdist = random_field(mesh, EdgeDim, KDim)
    z_hydro_corr = random_field(mesh, EdgeDim)
    z_gradh_exner = random_field(mesh, EdgeDim, KDim)

    ref = mo_solve_nonhydro_stencil_22_numpy(
        np.asarray(ipeidx_dsl),
        np.asarray(pg_exdist),
        np.asarray(z_hydro_corr),
        np.asarray(z_gradh_exner),
    )
    mo_solve_nonhydro_stencil_22(
        ipeidx_dsl,
        pg_exdist,
        z_hydro_corr,
        z_gradh_exner,
        offset_provider={},
    )
    assert np.allclose(z_gradh_exner, ref)
