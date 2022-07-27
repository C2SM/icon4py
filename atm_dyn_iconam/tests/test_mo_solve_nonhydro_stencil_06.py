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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_06 import (
    mo_solve_nonhydro_stencil_06,
)
from icon4py.common.dimension import CellDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


def mo_solve_nonhydro_stencil_06_numpy(
    z_exner_ic: np.array,
    inv_ddqz_z_full: np.array,
) -> np.array:
    z_dexner_dz_c_1 = (
        z_exner_ic - np.roll(z_exner_ic, shift=-1, axis=1)
    ) * inv_ddqz_z_full
    return z_dexner_dz_c_1


def test_mo_solve_nonhydro_stencil_06():
    mesh = SimpleMesh()
    z_exner_ic = random_field(mesh, CellDim, KDim)
    inv_ddqz_z_full = random_field(mesh, CellDim, KDim)
    z_dexner_dz_c_1 = zero_field(mesh, CellDim, KDim)

    z_dexner_dz_c_1_ref = mo_solve_nonhydro_stencil_06_numpy(
        np.asarray(z_exner_ic),
        np.asarray(inv_ddqz_z_full),
    )

    mo_solve_nonhydro_stencil_06(
        z_exner_ic,
        inv_ddqz_z_full,
        z_dexner_dz_c_1,
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(
        # work around problem with field[:,:-1], i.e. without the `np.asarray`
        np.asarray(z_dexner_dz_c_1)[:, :-1],
        np.asarray(z_dexner_dz_c_1_ref)[:, :-1],
    )
