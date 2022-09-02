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


def mo_solve_nonhydro_stencil_21_numpy(
    e2c: np.array,
    theta_v: np.array,
    ikidx: np.array,
    zdiff_gradp: np.array,
    theta_v_ic: np.array,
    inv_ddqz_z_full: np.array,
    inv_dual_edge_length: np.array,
    grav_o_cpd: float,
) -> tuple[np.array]:
    def _apply_index_field(shape, to_index, neighbor_table, offset_field):
        indexed, indexed_p1 = np.zeros(shape), np.zeros(shape)
        for iprimary in range(shape[0]):
            for isparse in range(shape[1]):
                for ik in range(shape[2]):
                    indexed[iprimary, isparse, ik] = to_index[
                        neighbor_table[iprimary, isparse],
                        ik + offset_field[iprimary, isparse, ik],
                    ]
                    indexed_p1[iprimary, isparse, ik] = to_index[
                        neighbor_table[iprimary, isparse],
                        ik + offset_field[iprimary, isparse, ik],
                    ]
        return indexed, indexed_p1

    full_shape = zdiff_gradp.shape
    inv_dual_edge_length = np.expand_dims(inv_dual_edge_length, -1)

    theta_v_at_kidx, _ = _apply_index_field(full_shape, theta_v, e2c, ikidx)

    theta_v_ic_at_kidx, theta_v_ic_at_kidx_p1 = _apply_index_field(
        full_shape, theta_v_ic, e2c, ikidx
    )

    inv_ddqz_z_full_at_kidx, _ = _apply_index_field(
        full_shape, inv_ddqz_z_full, e2c, ikidx
    )

    z_theta1 = (
        theta_v_at_kidx[:, 0, :]
        + zdiff_gradp[:, 0, :]
        * (theta_v_ic_at_kidx[:, 0, :] - theta_v_ic_at_kidx_p1[:, 0, :])
        * inv_ddqz_z_full_at_kidx[:, 0, :]
    )

    z_theta2 = (
        theta_v_at_kidx[:, 1, :]
        + zdiff_gradp[:, 1, :]
        * (theta_v_ic_at_kidx[:, 1, :] - theta_v_ic_at_kidx_p1[:, 1, :])
        * inv_ddqz_z_full_at_kidx[:, 1, :]
    )

    z_hydro_corr = (
        grav_o_cpd
        * inv_dual_edge_length
        * (z_theta2 - z_theta1)
        * 4.0
        / ((z_theta1 + z_theta2) ** 2)
    )

    return z_theta1, z_theta2, z_hydro_corr


def test_mo_solve_nonhydro_stencil_21():
    mesh = SimpleMesh()

    ikidx = zero_field(mesh, EdgeDim, E2CDim, KDim, dtype=int)
    rng = np.random.default_rng()
    for k in range(mesh.k_level):
        # construct offsets that reach all k-levels except the last (because we are using the entries of this field with `+1`)
        ikidx[:, :, k] = rng.integers(
            low=0 - k,
            high=mesh.k_level - k - 1,
            size=(ikidx.shape[0], ikidx.shape[1]),
        )

    theta_v = random_field(mesh, CellDim, KDim)
    zdiff_grap = random_field(mesh, EdgeDim, E2CDim, KDim)
    theta_v_ic = random_field(mesh, CellDim, KDim)
    inv_ddqz_z_full = random_field(mesh, CellDim, KDim)
    inv_dual_edge_length = random_field(mesh, EdgeDim)
    grav_o_cpd = 10.0

    z_theta1, z_theta2, z_hydro_corr = mo_solve_nonhydro_stencil_21_numpy(
        mesh.e2c,
        np.asarray(theta_v),
        np.asarray(ikidx),
        np.asarray(zdiff_grap),
        np.asarray(theta_v_ic),
        np.asarray(inv_ddqz_z_full),
        np.asarray(inv_dual_edge_length),
        grav_o_cpd,
    )

    # todo: call gt4py stencil
    # todo: assert equality between numpy and gt4py
    assert np.allclose(z_theta1, z_theta1)
    assert np.allclose(z_theta2, z_theta2)
    assert np.allclose(z_hydro_corr, z_hydro_corr)
