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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_63 import (
    mo_solve_nonhydro_stencil_63,
)
from icon4py.common.dimension import CellDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field


def mo_solve_nonhydro_stencil_63_numpy(
    inv_ddqz_z_full: np.array,
    w: np.array,
    w_concorr_c: np.array,
) -> np.array:
    w_offset_1 = np.roll(w, shift=-1, axis=1)
    w_concorr_c_offset_1 = np.roll(w_concorr_c, shift=-1, axis=1)
    z_dwdz_dd = inv_ddqz_z_full * (
        (w - w_offset_1) - (w_concorr_c - w_concorr_c_offset_1)
    )
    return z_dwdz_dd


def test_mo_solve_nonhydro_stencil_63():
    mesh = SimpleMesh()
    inv_ddqz_z_full = random_field(mesh, CellDim, KDim)
    w = random_field(mesh, CellDim, KDim)
    w_concorr_c = random_field(mesh, CellDim, KDim)
    z_dwdz_dd = random_field(mesh, CellDim, KDim)

    z_dwdz_dd_ref = mo_solve_nonhydro_stencil_63_numpy(
        np.asarray(inv_ddqz_z_full),
        np.asarray(w),
        np.asarray(w_concorr_c),
    )

    mo_solve_nonhydro_stencil_63(
        inv_ddqz_z_full, w, w_concorr_c, z_dwdz_dd, offset_provider={"Koff": KDim}
    )

    assert np.allclose(z_dwdz_dd_ref[:, :-1], z_dwdz_dd[:, :-1])
