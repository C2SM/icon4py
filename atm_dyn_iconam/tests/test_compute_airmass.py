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

from icon4py.atm_dyn_iconam.compute_airmass import compute_airmass
from icon4py.common.dimension import CellDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field


def compute_airmass_numpy(
    rho_in: np.array, ddqz_z_full_in: np.array, deepatmo_t1mc_in: np.array
) -> np.array:
    return rho_in * ddqz_z_full_in * deepatmo_t1mc_in


def test_compute_airmass():
    mesh = SimpleMesh()

    rho_in = random_field(mesh, CellDim, KDim)
    ddqz_z_full_in = random_field(mesh, CellDim, KDim)
    deepatmo_t1mc_in = random_field(mesh, KDim)
    airmass_out = random_field(mesh, CellDim, KDim)

    ref = compute_airmass_numpy(
        np.asarray(rho_in), np.asarray(ddqz_z_full_in), np.asarray(deepatmo_t1mc_in)
    )
    compute_airmass(
        rho_in, ddqz_z_full_in, deepatmo_t1mc_in, airmass_out, offset_provider={}
    )
    assert np.allclose(airmass_out, ref)
