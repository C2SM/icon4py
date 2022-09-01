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

from icon4py.common.dimension import CellDim, E2CDim, EdgeDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


def mo_solve_nonhydro_stencil_20_numpy(
    e2c: np.array,
    inv_dual_edge_length: np.array,
    z_exner_ex_pr: np.array,
    zdiff_gradp: np.array,
    ikidx: np.array,
    z_dexner_dz_c_1: np.array,
    z_dexner_dz_c_2: np.array,
) -> np.array:
    def _apply_index_field(indexed, shape, to_index, neighbor_table, index_field):
        for ic in range(shape[0]):
            for isparse in range(shape[1]):
                for ik in range(shape[2]):
                    indexed[ic, isparse, ik] = to_index[
                        neighbor_table[ic, isparse], index_field[ic, isparse, ik]
                    ]
        return indexed

    full_shape = zdiff_gradp.shape
    indexed = np.zeros_like(zdiff_gradp)
    inv_dual_edge_length = np.expand_dims(inv_dual_edge_length, -1)

    z_exner_ex_pr_at_kidx = _apply_index_field(
        indexed, full_shape, z_exner_ex_pr, e2c, ikidx
    )
    z_dexner_dz_c_1_at_kidx = _apply_index_field(
        indexed, full_shape, z_dexner_dz_c_1, e2c, ikidx
    )
    z_dexner_dz_c_2_at_kidx = _apply_index_field(
        indexed, full_shape, z_dexner_dz_c_2, e2c, ikidx
    )

    sum_expr = -(
        z_exner_ex_pr_at_kidx[:, 0, :]
        + zdiff_gradp[:, 0, :]
        * (
            z_dexner_dz_c_1_at_kidx[:, 0, :]
            + zdiff_gradp[:, 0, :] * z_dexner_dz_c_2_at_kidx[:, 0, :]
        )
        + z_exner_ex_pr_at_kidx[:, 1, :]
        + zdiff_gradp[:, 1, :]
        * (
            z_dexner_dz_c_1_at_kidx[:, 1, :]
            + zdiff_gradp[:, 1, :] * z_dexner_dz_c_2_at_kidx[:, 1, :]
        )
    )

    z_gradh_exner = inv_dual_edge_length * sum_expr
    return z_gradh_exner


def test_mo_solve_nonhydro_stencil_20():
    mesh = SimpleMesh()

    inv_dual_edge_length = random_field(mesh, EdgeDim)
    z_exner_ex_pr = random_field(mesh, CellDim, KDim)
    zdiff_gradp = random_field(mesh, EdgeDim, E2CDim, KDim)
    ikidx = zero_field(mesh, EdgeDim, E2CDim, KDim, dtype=int)
    z_dexner_dz_c_1 = random_field(mesh, CellDim, KDim)
    z_dexner_dz_c_2 = random_field(mesh, CellDim, KDim)

    z_gradh_exner_ref = mo_solve_nonhydro_stencil_20_numpy(
        mesh.e2c,
        np.asarray(inv_dual_edge_length),
        np.asarray(z_exner_ex_pr),
        np.asarray(zdiff_gradp),
        np.asarray(ikidx),
        np.asarray(z_dexner_dz_c_1),
        np.asarray(z_dexner_dz_c_2),
    )

    # todo: call gt4py stencil
    # todo: assert equality between numpy and gt4py
    assert z_gradh_exner_ref == z_gradh_exner_ref
