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
import pytest
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_21 import (
    mo_solve_nonhydro_stencil_21,
)
from icon4py.model.common.dimension import CellDim, E2CDim, ECDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    flatten_first_two_dims,
    random_field,
    zero_field,
)


def mo_solve_nonhydro_stencil_21_numpy(
    grid,
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

    e2c = grid.connectivities[E2CDim]
    full_shape = e2c.shape + zdiff_gradp.shape[1:]
    zdiff_gradp = zdiff_gradp.reshape(full_shape)
    ikoffset = ikoffset.reshape(full_shape)

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


class TestMoSolveNonHydroStencil21(StencilTest):
    OUTPUTS = ("z_hydro_corr",)
    PROGRAM = mo_solve_nonhydro_stencil_21

    @staticmethod
    def reference(
        grid,
        theta_v: np.array,
        ikoffset: np.array,
        zdiff_gradp: np.array,
        theta_v_ic: np.array,
        inv_ddqz_z_full: np.array,
        inv_dual_edge_length: np.array,
        grav_o_cpd: float,
        **kwargs,
    ) -> dict:

        z_hydro_corr = mo_solve_nonhydro_stencil_21_numpy(
            grid,
            theta_v,
            ikoffset,
            zdiff_gradp,
            theta_v_ic,
            inv_ddqz_z_full,
            inv_dual_edge_length,
            grav_o_cpd,
        )

        return dict(z_hydro_corr=z_hydro_corr)

    @pytest.fixture
    def input_data(self, grid):
        if np.any(grid.connectivities[E2CDim] == -1):
            pytest.xfail("Stencil does not support missing neighbors.")

        ikoffset = zero_field(grid, EdgeDim, E2CDim, KDim, dtype=int32)
        rng = np.random.default_rng()
        for k in range(grid.num_levels):
            # construct offsets that reach all k-levels except the last (because we are using the entries of this field with `+1`)
            ikoffset[:, :, k] = rng.integers(
                low=0 - k,
                high=grid.num_levels - k - 1,
                size=(ikoffset.shape[0], ikoffset.shape[1]),
            )

        theta_v = random_field(grid, CellDim, KDim)
        zdiff_gradp = random_field(grid, EdgeDim, E2CDim, KDim)
        theta_v_ic = random_field(grid, CellDim, KDim)
        inv_ddqz_z_full = random_field(grid, CellDim, KDim)
        inv_dual_edge_length = random_field(grid, EdgeDim)
        grav_o_cpd = 10.0

        zdiff_gradp_new = flatten_first_two_dims(ECDim, KDim, field=zdiff_gradp)
        ikoffset_new = flatten_first_two_dims(ECDim, KDim, field=ikoffset)

        z_hydro_corr = zero_field(grid, EdgeDim, KDim)

        return dict(
            theta_v=theta_v,
            ikoffset=ikoffset_new,
            z_hydro_corr=z_hydro_corr,
            zdiff_gradp=zdiff_gradp_new,
            theta_v_ic=theta_v_ic,
            inv_ddqz_z_full=inv_ddqz_z_full,
            inv_dual_edge_length=inv_dual_edge_length,
            grav_o_cpd=grav_o_cpd,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_edges),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
