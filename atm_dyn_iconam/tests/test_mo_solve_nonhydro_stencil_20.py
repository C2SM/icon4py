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
from gt4py.next.ffront.fbuiltins import int32
from gt4py.next.iterator.embedded import StridedNeighborOffsetProvider

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_20 import (
    mo_solve_nonhydro_stencil_20,
)
from icon4py.common.dimension import CellDim, E2CDim, ECDim, EdgeDim, KDim

from .test_utils.helpers import flatten_first_two_dims, random_field, zero_field
from .test_utils.simple_mesh import SimpleMesh


def mo_solve_nonhydro_stencil_20_numpy(
    e2c: np.array,
    inv_dual_edge_length: np.array,
    z_exner_ex_pr: np.array,
    zdiff_gradp: np.array,
    ikoffset: np.array,
    z_dexner_dz_c_1: np.array,
    z_dexner_dz_c_2: np.array,
) -> np.array:
    def _apply_index_field(shape, to_index, neighbor_table, offset_field):
        indexed = np.zeros(shape)
        for iprimary in range(shape[0]):
            for isparse in range(shape[1]):
                for ik in range(shape[2]):
                    indexed[iprimary, isparse, ik] = to_index[
                        neighbor_table[iprimary, isparse],
                        ik + offset_field[iprimary, isparse, ik],
                    ]
        return indexed

    full_shape = zdiff_gradp.shape
    inv_dual_edge_length = np.expand_dims(inv_dual_edge_length, -1)

    z_exner_ex_pr_at_kidx = _apply_index_field(full_shape, z_exner_ex_pr, e2c, ikoffset)
    z_dexner_dz_c_1_at_kidx = _apply_index_field(
        full_shape, z_dexner_dz_c_1, e2c, ikoffset
    )
    z_dexner_dz_c_2_at_kidx = _apply_index_field(
        full_shape, z_dexner_dz_c_2, e2c, ikoffset
    )

    def at_neighbor(i):
        return z_exner_ex_pr_at_kidx[:, i, :] + zdiff_gradp[:, i, :] * (
            z_dexner_dz_c_1_at_kidx[:, i, :]
            + zdiff_gradp[:, i, :] * z_dexner_dz_c_2_at_kidx[:, i, :]
        )

    sum_expr = at_neighbor(1) - at_neighbor(0)

    z_gradh_exner = inv_dual_edge_length * sum_expr
    return z_gradh_exner


def test_mo_solve_nonhydro_stencil_20():
    mesh = SimpleMesh()

    inv_dual_edge_length = random_field(mesh, EdgeDim)
    z_exner_ex_pr = random_field(mesh, CellDim, KDim)
    zdiff_gradp = random_field(mesh, EdgeDim, E2CDim, KDim)
    ikoffset = zero_field(mesh, EdgeDim, E2CDim, KDim, dtype=int32)

    rng = np.random.default_rng()
    for k in range(mesh.k_level):
        # construct offsets that reach all k-levels except the last (because we are using the entries of this field with `+1`)
        ikoffset[:, :, k] = rng.integers(
            low=0 - k,
            high=mesh.k_level - k - 1,
            size=(ikoffset.shape[0], ikoffset.shape[1]),
        )

    zdiff_gradp_new = flatten_first_two_dims(ECDim, KDim, field=zdiff_gradp)
    ikoffset_new = flatten_first_two_dims(ECDim, KDim, field=ikoffset)

    z_dexner_dz_c_1 = random_field(mesh, CellDim, KDim)
    z_dexner_dz_c_2 = random_field(mesh, CellDim, KDim)
    z_gradh_exner = zero_field(mesh, EdgeDim, KDim)

    z_gradh_exner_ref = mo_solve_nonhydro_stencil_20_numpy(
        mesh.e2c,
        np.asarray(inv_dual_edge_length),
        np.asarray(z_exner_ex_pr),
        np.asarray(zdiff_gradp),
        np.asarray(ikoffset),
        np.asarray(z_dexner_dz_c_1),
        np.asarray(z_dexner_dz_c_2),
    )

    mo_solve_nonhydro_stencil_20(
        inv_dual_edge_length,
        z_exner_ex_pr,
        zdiff_gradp_new,
        ikoffset_new,
        z_dexner_dz_c_1,
        z_dexner_dz_c_2,
        z_gradh_exner,
        horizontal_start=int32(0),
        horizontal_end=int32(mesh.n_edges),
        vertical_start=int32(0),
        vertical_end=int32(mesh.k_level),
        offset_provider={
            "E2C": mesh.get_e2c_offset_provider(),
            "E2EC": StridedNeighborOffsetProvider(EdgeDim, ECDim, mesh.n_e2c),
            "Koff": KDim,
        },
    )

    assert np.allclose(z_gradh_exner, z_gradh_exner_ref)
