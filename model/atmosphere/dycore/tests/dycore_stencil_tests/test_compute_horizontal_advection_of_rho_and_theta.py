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

from icon4py.model.atmosphere.dycore.compute_horizontal_advection_of_rho_and_theta import (
    compute_horizontal_advection_of_rho_and_theta,
)
from icon4py.model.common.dimension import CellDim, E2CDim, ECDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, as_1D_sparse_field, random_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestComputeBtraj(StencilTest):
    PROGRAM = compute_horizontal_advection_of_rho_and_theta
    OUTPUTS = ("z_rho_e", "z_theta_v_e")

    @staticmethod
    def compute_btraj_numpy(
        grid,
        p_vn: np.array,
        p_vt: np.array,
        pos_on_tplane_e_1: np.array,
        pos_on_tplane_e_2: np.array,
        primal_normal_cell_1: np.array,
        dual_normal_cell_1: np.array,
        primal_normal_cell_2: np.array,
        dual_normal_cell_2: np.array,
        p_dthalf: float,
        **kwargs,
    ) -> tuple[np.array, ...]:
        lvn_pos = np.where(p_vn > wpfloat("0.0"), True, False)
        pos_on_tplane_e_1 = np.expand_dims(pos_on_tplane_e_1, axis=-1)
        pos_on_tplane_e_2 = np.expand_dims(pos_on_tplane_e_2, axis=-1)
        primal_normal_cell_1 = np.expand_dims(primal_normal_cell_1, axis=-1)
        dual_normal_cell_1 = np.expand_dims(dual_normal_cell_1, axis=-1)
        primal_normal_cell_2 = np.expand_dims(primal_normal_cell_2, axis=-1)
        dual_normal_cell_2 = np.expand_dims(dual_normal_cell_2, axis=-1)

        z_ntdistv_bary_1 = -(
            p_vn * p_dthalf + np.where(lvn_pos, pos_on_tplane_e_1[:, 0], pos_on_tplane_e_1[:, 1])
        )
        z_ntdistv_bary_2 = -(
            p_vt * p_dthalf + np.where(lvn_pos, pos_on_tplane_e_2[:, 0], pos_on_tplane_e_2[:, 1])
        )

        p_distv_bary_1 = np.where(
            lvn_pos,
            z_ntdistv_bary_1 * primal_normal_cell_1[:, 0]
            + z_ntdistv_bary_2 * dual_normal_cell_1[:, 0],
            z_ntdistv_bary_1 * primal_normal_cell_1[:, 1]
            + z_ntdistv_bary_2 * dual_normal_cell_1[:, 1],
        )

        p_distv_bary_2 = np.where(
            lvn_pos,
            z_ntdistv_bary_1 * primal_normal_cell_2[:, 0]
            + z_ntdistv_bary_2 * dual_normal_cell_2[:, 0],
            z_ntdistv_bary_1 * primal_normal_cell_2[:, 1]
            + z_ntdistv_bary_2 * dual_normal_cell_2[:, 1],
        )

        return p_distv_bary_1, p_distv_bary_2

    @staticmethod
    def sten_16_numpy(
        grid,
        p_vn: np.array,
        rho_ref_me: np.array,
        theta_ref_me: np.array,
        p_distv_bary_1: np.array,
        p_distv_bary_2: np.array,
        z_grad_rth_1: np.array,
        z_grad_rth_2: np.array,
        z_grad_rth_3: np.array,
        z_grad_rth_4: np.array,
        z_rth_pr_1: np.array,
        z_rth_pr_2: np.array,
        **kwargs,
    ) -> np.array:
        e2c = grid.connectivities[E2CDim]
        z_rth_pr_1_e2c = z_rth_pr_1[e2c]
        z_rth_pr_2_e2c = z_rth_pr_2[e2c]
        z_grad_rth_1_e2c = z_grad_rth_1[e2c]
        z_grad_rth_2_e2c = z_grad_rth_2[e2c]
        z_grad_rth_3_e2c = z_grad_rth_3[e2c]
        z_grad_rth_4_e2c = z_grad_rth_4[e2c]

        z_rho_e = np.where(
            p_vn > 0,
            rho_ref_me
            + z_rth_pr_1_e2c[:, 0]
            + p_distv_bary_1 * z_grad_rth_1_e2c[:, 0]
            + p_distv_bary_2 * z_grad_rth_2_e2c[:, 0],
            rho_ref_me
            + z_rth_pr_1_e2c[:, 1]
            + p_distv_bary_1 * z_grad_rth_1_e2c[:, 1]
            + p_distv_bary_2 * z_grad_rth_2_e2c[:, 1],
        )

        z_theta_v_e = np.where(
            p_vn > 0,
            theta_ref_me
            + z_rth_pr_2_e2c[:, 0]
            + p_distv_bary_1 * z_grad_rth_3_e2c[:, 0]
            + p_distv_bary_2 * z_grad_rth_4_e2c[:, 0],
            theta_ref_me
            + z_rth_pr_2_e2c[:, 1]
            + p_distv_bary_1 * z_grad_rth_3_e2c[:, 1]
            + p_distv_bary_2 * z_grad_rth_4_e2c[:, 1],
        )

        return z_rho_e, z_theta_v_e

    @classmethod
    def reference(
        cls,
        grid,
        p_vn: np.array,
        p_vt: np.array,
        pos_on_tplane_e_1: np.array,
        pos_on_tplane_e_2: np.array,
        primal_normal_cell_1: np.array,
        dual_normal_cell_1: np.array,
        primal_normal_cell_2: np.array,
        dual_normal_cell_2: np.array,
        p_dthalf: float,
        rho_ref_me: np.array,
        theta_ref_me: np.array,
        z_grad_rth_1: np.array,
        z_grad_rth_2: np.array,
        z_grad_rth_3: np.array,
        z_grad_rth_4: np.array,
        z_rth_pr_1: np.array,
        z_rth_pr_2: np.array,
        **kwargs,
    ):
        e2c = grid.connectivities[E2CDim]
        pos_on_tplane_e_1 = pos_on_tplane_e_1.reshape(e2c.shape)
        pos_on_tplane_e_2 = pos_on_tplane_e_2.reshape(e2c.shape)
        primal_normal_cell_1 = primal_normal_cell_1.reshape(e2c.shape)
        dual_normal_cell_1 = dual_normal_cell_1.reshape(e2c.shape)
        primal_normal_cell_2 = primal_normal_cell_2.reshape(e2c.shape)
        dual_normal_cell_2 = dual_normal_cell_2.reshape(e2c.shape)

        p_distv_bary_1, p_distv_bary_2 = cls.compute_btraj_numpy(
            grid,
            p_vn,
            p_vt,
            pos_on_tplane_e_1,
            pos_on_tplane_e_2,
            primal_normal_cell_1,
            dual_normal_cell_1,
            primal_normal_cell_2,
            dual_normal_cell_2,
            p_dthalf,
        )

        z_rho_e, z_theta_v_e = cls.sten_16_numpy(
            grid,
            p_vn,
            rho_ref_me,
            theta_ref_me,
            p_distv_bary_1,
            p_distv_bary_2,
            z_grad_rth_1,
            z_grad_rth_2,
            z_grad_rth_3,
            z_grad_rth_4,
            z_rth_pr_1,
            z_rth_pr_2,
        )

        return dict(z_rho_e=z_rho_e, z_theta_v_e=z_theta_v_e)

    @pytest.fixture
    def input_data(self, grid):
        if np.any(grid.connectivities[E2CDim] == -1):
            pytest.xfail("Stencil does not support missing neighbors.")

        p_vn = random_field(grid, EdgeDim, KDim, dtype=wpfloat)
        p_vt = random_field(grid, EdgeDim, KDim, dtype=vpfloat)
        pos_on_tplane_e_1 = random_field(grid, EdgeDim, E2CDim, dtype=wpfloat)
        pos_on_tplane_e_1_new = as_1D_sparse_field(pos_on_tplane_e_1, ECDim)
        pos_on_tplane_e_2 = random_field(grid, EdgeDim, E2CDim, dtype=wpfloat)
        pos_on_tplane_e_2_new = as_1D_sparse_field(pos_on_tplane_e_2, ECDim)
        primal_normal_cell_1 = random_field(grid, EdgeDim, E2CDim, dtype=wpfloat)
        primal_normal_cell_1_new = as_1D_sparse_field(primal_normal_cell_1, ECDim)
        dual_normal_cell_1 = random_field(grid, EdgeDim, E2CDim, dtype=wpfloat)
        dual_normal_cell_1_new = as_1D_sparse_field(dual_normal_cell_1, ECDim)
        primal_normal_cell_2 = random_field(grid, EdgeDim, E2CDim, dtype=wpfloat)
        primal_normal_cell_2_new = as_1D_sparse_field(primal_normal_cell_2, ECDim)
        dual_normal_cell_2 = random_field(grid, EdgeDim, E2CDim, dtype=wpfloat)
        dual_normal_cell_2_new = as_1D_sparse_field(dual_normal_cell_2, ECDim)
        p_dthalf = wpfloat("2.0")

        rho_ref_me = random_field(grid, EdgeDim, KDim, dtype=vpfloat)
        theta_ref_me = random_field(grid, EdgeDim, KDim, dtype=vpfloat)
        z_grad_rth_1 = random_field(grid, CellDim, KDim, dtype=vpfloat)
        z_grad_rth_2 = random_field(grid, CellDim, KDim, dtype=vpfloat)
        z_grad_rth_3 = random_field(grid, CellDim, KDim, dtype=vpfloat)
        z_grad_rth_4 = random_field(grid, CellDim, KDim, dtype=vpfloat)
        z_rth_pr_1 = random_field(grid, CellDim, KDim, dtype=vpfloat)
        z_rth_pr_2 = random_field(grid, CellDim, KDim, dtype=vpfloat)
        z_rho_e = random_field(grid, EdgeDim, KDim, dtype=wpfloat)
        z_theta_v_e = random_field(grid, EdgeDim, KDim, dtype=wpfloat)

        return dict(
            p_vn=p_vn,
            p_vt=p_vt,
            pos_on_tplane_e_1=pos_on_tplane_e_1_new,
            pos_on_tplane_e_2=pos_on_tplane_e_2_new,
            primal_normal_cell_1=primal_normal_cell_1_new,
            dual_normal_cell_1=dual_normal_cell_1_new,
            primal_normal_cell_2=primal_normal_cell_2_new,
            dual_normal_cell_2=dual_normal_cell_2_new,
            p_dthalf=p_dthalf,
            rho_ref_me=rho_ref_me,
            theta_ref_me=theta_ref_me,
            z_grad_rth_1=z_grad_rth_1,
            z_grad_rth_2=z_grad_rth_2,
            z_grad_rth_3=z_grad_rth_3,
            z_grad_rth_4=z_grad_rth_4,
            z_rth_pr_1=z_rth_pr_1,
            z_rth_pr_2=z_rth_pr_2,
            z_rho_e=z_rho_e,
            z_theta_v_e=z_theta_v_e,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_edges),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
