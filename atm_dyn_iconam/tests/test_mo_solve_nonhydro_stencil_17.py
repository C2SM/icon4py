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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_17 import (
    mo_solve_nonhydro_stencil_17,
)
from icon4py.common.dimension import CellDim, EdgeDim, KDim

from .simple_mesh import SimpleMesh
from .utils import random_field


def mo_solve_nonhydro_stencil_17_numpy(
    e2c: np.array,
    hmask_dd3d: np.array,
    scalfac_dd3d: np.array,
    inv_dual_edge_length: np.array,
    z_dwdz_dd: np.array,
    z_graddiv_vn: np.array,
) -> np.array:
    scalfac_dd3d = np.expand_dims(scalfac_dd3d, axis=0)
    hmask_dd3d = np.expand_dims(hmask_dd3d, axis=-1)
    inv_dual_edge_length = np.expand_dims(inv_dual_edge_length, axis=-1)

    z_dwdz_dd_e2c = z_dwdz_dd[e2c]
    z_dwdz_dd_weighted = z_dwdz_dd_e2c[:, 1] - z_dwdz_dd_e2c[:, 0]

    z_graddiv_vn = z_graddiv_vn + (
        hmask_dd3d * scalfac_dd3d * inv_dual_edge_length * z_dwdz_dd_weighted
    )
    return z_graddiv_vn


def test_mo_solve_nonhydro_stencil_17():
    mesh = SimpleMesh()

    hmask_dd3d = random_field(mesh, EdgeDim)
    scalfac_dd3d = random_field(mesh, KDim)
    inv_dual_edge_length = random_field(mesh, EdgeDim)
    z_dwdz_dd = random_field(mesh, CellDim, KDim)
    z_graddiv_vn = random_field(mesh, EdgeDim, KDim)

    ref = mo_solve_nonhydro_stencil_17_numpy(
        mesh.e2c,
        np.asarray(hmask_dd3d),
        np.asarray(scalfac_dd3d),
        np.asarray(inv_dual_edge_length),
        np.asarray(z_dwdz_dd),
        np.asarray(z_graddiv_vn),
    )

    mo_solve_nonhydro_stencil_17(
        hmask_dd3d,
        scalfac_dd3d,
        inv_dual_edge_length,
        z_dwdz_dd,
        z_graddiv_vn,
        offset_provider={
            "E2C": mesh.get_e2c_offset_provider(),
        },
    )
    assert np.allclose(z_graddiv_vn, ref)
