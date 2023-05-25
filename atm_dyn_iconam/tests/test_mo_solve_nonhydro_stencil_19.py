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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_19 import (
    mo_solve_nonhydro_stencil_19,
)
from icon4py.common.dimension import CellDim, E2CDim, EdgeDim, KDim

from .test_utils.helpers import random_field
from .test_utils.simple_mesh import SimpleMesh


def mo_solve_nonhydro_stencil_19_numpy(
    e2c: np.array,
    inv_dual_edge_length: np.array,
    z_exner_ex_pr: np.array,
    ddxn_z_full: np.array,
    c_lin_e: np.array,
    z_dexner_dz_c_1: np.array,
) -> np.array:
    inv_dual_edge_length = np.expand_dims(inv_dual_edge_length, axis=-1)
    c_lin_e = np.expand_dims(c_lin_e, axis=-1)

    z_exner_ex_pr_e2c = z_exner_ex_pr[e2c]
    z_exner_ex_weighted = z_exner_ex_pr_e2c[:, 1] - z_exner_ex_pr_e2c[:, 0]

    z_gradh_exner = inv_dual_edge_length * z_exner_ex_weighted - ddxn_z_full * np.sum(
        c_lin_e * z_dexner_dz_c_1[e2c], axis=1
    )
    return z_gradh_exner


def test_mo_solve_nonhydro_stencil_19():
    mesh = SimpleMesh()

    inv_dual_edge_length = random_field(mesh, EdgeDim)
    z_exner_ex_pr = random_field(mesh, CellDim, KDim)
    ddxn_z_full = random_field(mesh, EdgeDim, KDim)
    c_lin_e = random_field(mesh, EdgeDim, E2CDim)
    z_dexner_dz_c_1 = random_field(mesh, CellDim, KDim)
    z_gradh_exner = random_field(mesh, EdgeDim, KDim)

    ref = mo_solve_nonhydro_stencil_19_numpy(
        mesh.e2c,
        np.asarray(inv_dual_edge_length),
        np.asarray(z_exner_ex_pr),
        np.asarray(ddxn_z_full),
        np.asarray(c_lin_e),
        np.asarray(z_dexner_dz_c_1),
    )

    mo_solve_nonhydro_stencil_19(
        inv_dual_edge_length,
        z_exner_ex_pr,
        ddxn_z_full,
        c_lin_e,
        z_dexner_dz_c_1,
        z_gradh_exner,
        offset_provider={
            "E2C": mesh.get_e2c_offset_provider(),
        },
    )
    assert np.allclose(z_gradh_exner, ref)
