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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_21 import (
    mo_solve_nonhydro_stencil_21,
)
from icon4py.model.common.dimension import CellDim, E2CDim, ECDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import (
    flatten_first_two_dims,
    random_field,
    zero_field,
)
from icon4py.model.common.test_utils.simple_mesh import SimpleMesh


def mo_solve_nonhydro_stencil_21_numpy(
    e2c: np.array,
    theta_v: np.array,
    ikoffset: np.array,
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
                        ik + offset_field[iprimary, isparse, ik] + 1,
                    ]
        return indexed, indexed_p1

    full_shape = zdiff_gradp.shape
    inv_dual_edge_length = np.expand_dims(inv_dual_edge_length, -1)

    theta_v_at_kidx, _ = _apply_index_field(full_shape, theta_v, e2c, ikoffset)

    theta_v_ic_at_kidx, theta_v_ic_at_kidx_p1 = _apply_index_field(
        full_shape, theta_v_ic, e2c, ikoffset
    )

    inv_ddqz_z_full_at_kidx, _ = _apply_index_field(full_shape, inv_ddqz_z_full, e2c, ikoffset)

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

    return z_hydro_corr


def test_mo_solve_nonhydro_stencil_21():
    mesh = SimpleMesh()

    ikoffset = zero_field(mesh, EdgeDim, E2CDim, KDim, dtype=int32)
    rng = np.random.default_rng()
    for k in range(mesh.k_level):
        # construct offsets that reach all k-levels except the last (because we are using the entries of this field with `+1`)
        ikoffset[:, :, k] = rng.integers(
            low=0 - k,
            high=mesh.k_level - k - 1,
            size=(ikoffset.shape[0], ikoffset.shape[1]),
        )

    theta_v = random_field(mesh, CellDim, KDim)
    zdiff_gradp = random_field(mesh, EdgeDim, E2CDim, KDim)
    theta_v_ic = random_field(mesh, CellDim, KDim)
    inv_ddqz_z_full = random_field(mesh, CellDim, KDim)
    inv_dual_edge_length = random_field(mesh, EdgeDim)
    grav_o_cpd = 10.0

    zdiff_gradp_new = flatten_first_two_dims(ECDim, KDim, field=zdiff_gradp)
    ikoffset_new = flatten_first_two_dims(ECDim, KDim, field=ikoffset)

    z_hydro_corr = zero_field(mesh, EdgeDim, KDim)

    z_hydro_corr_ref = mo_solve_nonhydro_stencil_21_numpy(
        mesh.e2c,
        np.asarray(theta_v),
        np.asarray(ikoffset),
        np.asarray(zdiff_gradp),
        np.asarray(theta_v_ic),
        np.asarray(inv_ddqz_z_full),
        np.asarray(inv_dual_edge_length),
        grav_o_cpd,
    )

    mo_solve_nonhydro_stencil_21(
        theta_v,
        ikoffset_new,
        zdiff_gradp_new,
        theta_v_ic,
        inv_ddqz_z_full,
        inv_dual_edge_length,
        grav_o_cpd,
        z_hydro_corr,
        offset_provider={
            "E2C": mesh.get_e2c_offset_provider(),
            "E2EC": StridedNeighborOffsetProvider(EdgeDim, ECDim, mesh.n_e2c),
            "Koff": KDim,
        },
    )

    assert np.allclose(z_hydro_corr_ref, z_hydro_corr)
