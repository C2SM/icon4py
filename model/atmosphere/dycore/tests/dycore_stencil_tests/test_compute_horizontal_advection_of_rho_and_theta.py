# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import numpy as np
import pytest

import icon4py.model.common.utils.data_allocation as data_alloc
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_advection_of_rho_and_theta import (
    compute_horizontal_advection_of_rho_and_theta,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.testing.helpers import StencilTest


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
        lvn_pos = np.where(p_vn > ta.wpfloat("0.0"), True, False)
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
        e2c = grid.connectivities[dims.E2CDim]
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
        e2c = grid.connectivities[dims.E2CDim]
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
        if np.any(grid.connectivities[dims.E2CDim] == -1):
            pytest.xfail("Stencil does not support missing neighbors.")

        p_vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        p_vt = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat)
        pos_on_tplane_e_1 = data_alloc.random_field(grid, dims.ECDim, dtype=ta.wpfloat)
        pos_on_tplane_e_2 = data_alloc.random_field(grid, dims.ECDim, dtype=ta.wpfloat)
        primal_normal_cell_1 = data_alloc.random_field(grid, dims.ECDim, dtype=ta.wpfloat)
        dual_normal_cell_1 = data_alloc.random_field(grid, dims.ECDim, dtype=ta.wpfloat)
        primal_normal_cell_2 = data_alloc.random_field(grid, dims.ECDim, dtype=ta.wpfloat)
        dual_normal_cell_2 = data_alloc.random_field(grid, dims.ECDim, dtype=ta.wpfloat)
        p_dthalf = ta.wpfloat("2.0")

        rho_ref_me = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat)
        theta_ref_me = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat)
        z_grad_rth_1 = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        z_grad_rth_2 = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        z_grad_rth_3 = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        z_grad_rth_4 = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        z_rth_pr_1 = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        z_rth_pr_2 = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        z_rho_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        z_theta_v_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)

        return dict(
            p_vn=p_vn,
            p_vt=p_vt,
            pos_on_tplane_e_1=pos_on_tplane_e_1,
            pos_on_tplane_e_2=pos_on_tplane_e_2,
            primal_normal_cell_1=primal_normal_cell_1,
            dual_normal_cell_1=dual_normal_cell_1,
            primal_normal_cell_2=primal_normal_cell_2,
            dual_normal_cell_2=dual_normal_cell_2,
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
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
