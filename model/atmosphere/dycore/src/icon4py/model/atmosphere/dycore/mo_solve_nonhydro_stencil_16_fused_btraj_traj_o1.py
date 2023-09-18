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

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, broadcast, where

from icon4py.model.common.dimension import E2C, E2EC, CellDim, ECDim, EdgeDim, KDim


@field_operator
def _compute_btraj(
    p_vn: Field[[EdgeDim, KDim], float],
    p_vt: Field[[EdgeDim, KDim], float],
    pos_on_tplane_e_1: Field[[ECDim], float],
    pos_on_tplane_e_2: Field[[ECDim], float],
    primal_normal_cell_1: Field[[ECDim], float],
    dual_normal_cell_1: Field[[ECDim], float],
    primal_normal_cell_2: Field[[ECDim], float],
    dual_normal_cell_2: Field[[ECDim], float],
    p_dthalf: float,
) -> tuple[Field[[EdgeDim, KDim], float], Field[[EdgeDim, KDim], float]]:
    lvn_pos = where(p_vn >= 0.0, True, False)

    z_ntdistv_bary_1 = -(
        p_vn * p_dthalf
        + where(lvn_pos, pos_on_tplane_e_1(E2EC[0]), pos_on_tplane_e_1(E2EC[1]))
    )

    z_ntdistv_bary_2 = -(
        p_vt * p_dthalf
        + where(lvn_pos, pos_on_tplane_e_2(E2EC[0]), pos_on_tplane_e_2(E2EC[1]))
    )

    p_distv_bary_1 = where(
        lvn_pos,
        z_ntdistv_bary_1 * primal_normal_cell_1(E2EC[0])
        + z_ntdistv_bary_2 * dual_normal_cell_1(E2EC[0]),
        z_ntdistv_bary_1 * primal_normal_cell_1(E2EC[1])
        + z_ntdistv_bary_2 * dual_normal_cell_1(E2EC[1]),
    )

    p_distv_bary_2 = where(
        lvn_pos,
        z_ntdistv_bary_1 * primal_normal_cell_2(E2EC[0])
        + z_ntdistv_bary_2 * dual_normal_cell_2(E2EC[0]),
        z_ntdistv_bary_1 * primal_normal_cell_2(E2EC[1])
        + z_ntdistv_bary_2 * dual_normal_cell_2(E2EC[1]),
    )

    return p_distv_bary_1, p_distv_bary_2


@field_operator
def _sten_16(
    p_vn: Field[[EdgeDim, KDim], float],
    rho_ref_me: Field[[EdgeDim, KDim], float],
    theta_ref_me: Field[[EdgeDim, KDim], float],
    p_distv_bary_1: Field[[EdgeDim, KDim], float],
    p_distv_bary_2: Field[[EdgeDim, KDim], float],
    z_grad_rth_1: Field[[CellDim, KDim], float],
    z_grad_rth_2: Field[[CellDim, KDim], float],
    z_grad_rth_3: Field[[CellDim, KDim], float],
    z_grad_rth_4: Field[[CellDim, KDim], float],
    z_rth_pr_1: Field[[CellDim, KDim], float],
    z_rth_pr_2: Field[[CellDim, KDim], float],
) -> tuple[Field[[EdgeDim, KDim], float], Field[[EdgeDim, KDim], float]]:
    z_rho_e = broadcast(0, (EdgeDim, KDim))
    z_theta_v_e = broadcast(0, (EdgeDim, KDim))

    z_rho_e = where(
        p_vn >= 0.0,
        rho_ref_me
        + z_rth_pr_1(E2C[0])
        + p_distv_bary_1 * z_grad_rth_1(E2C[0])
        + p_distv_bary_2 * z_grad_rth_2(E2C[0]),
        rho_ref_me
        + z_rth_pr_1(E2C[1])
        + p_distv_bary_1 * z_grad_rth_1(E2C[1])
        + p_distv_bary_2 * z_grad_rth_2(E2C[1]),
    )

    z_theta_v_e = where(
        p_vn >= 0.0,
        theta_ref_me
        + z_rth_pr_2(E2C[0])
        + p_distv_bary_1 * z_grad_rth_3(E2C[0])
        + p_distv_bary_2 * z_grad_rth_4(E2C[0]),
        theta_ref_me
        + z_rth_pr_2(E2C[1])
        + p_distv_bary_1 * z_grad_rth_3(E2C[1])
        + p_distv_bary_2 * z_grad_rth_4(E2C[1]),
    )

    return z_rho_e, z_theta_v_e


@field_operator
def _mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1(
    p_vn: Field[[EdgeDim, KDim], float],
    p_vt: Field[[EdgeDim, KDim], float],
    pos_on_tplane_e_1: Field[[ECDim], float],
    pos_on_tplane_e_2: Field[[ECDim], float],
    primal_normal_cell_1: Field[[ECDim], float],
    dual_normal_cell_1: Field[[ECDim], float],
    primal_normal_cell_2: Field[[ECDim], float],
    dual_normal_cell_2: Field[[ECDim], float],
    p_dthalf: float,
    rho_ref_me: Field[[EdgeDim, KDim], float],
    theta_ref_me: Field[[EdgeDim, KDim], float],
    z_grad_rth_1: Field[[CellDim, KDim], float],
    z_grad_rth_2: Field[[CellDim, KDim], float],
    z_grad_rth_3: Field[[CellDim, KDim], float],
    z_grad_rth_4: Field[[CellDim, KDim], float],
    z_rth_pr_1: Field[[CellDim, KDim], float],
    z_rth_pr_2: Field[[CellDim, KDim], float],
) -> tuple[Field[[EdgeDim, KDim], float], Field[[EdgeDim, KDim], float]]:
    (p_distv_bary_1, p_distv_bary_2) = _compute_btraj(
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

    z_rho_e, z_theta_v_e = _sten_16(
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

    return z_rho_e, z_theta_v_e


@program(grid_type=GridType.UNSTRUCTURED)
def mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1(
    p_vn: Field[[EdgeDim, KDim], float],
    p_vt: Field[[EdgeDim, KDim], float],
    pos_on_tplane_e_1: Field[[ECDim], float],
    pos_on_tplane_e_2: Field[[ECDim], float],
    primal_normal_cell_1: Field[[ECDim], float],
    dual_normal_cell_1: Field[[ECDim], float],
    primal_normal_cell_2: Field[[ECDim], float],
    dual_normal_cell_2: Field[[ECDim], float],
    p_dthalf: float,
    rho_ref_me: Field[[EdgeDim, KDim], float],
    theta_ref_me: Field[[EdgeDim, KDim], float],
    z_grad_rth_1: Field[[CellDim, KDim], float],
    z_grad_rth_2: Field[[CellDim, KDim], float],
    z_grad_rth_3: Field[[CellDim, KDim], float],
    z_grad_rth_4: Field[[CellDim, KDim], float],
    z_rth_pr_1: Field[[CellDim, KDim], float],
    z_rth_pr_2: Field[[CellDim, KDim], float],
    z_rho_e: Field[[EdgeDim, KDim], float],
    z_theta_v_e: Field[[EdgeDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1(
        p_vn,
        p_vt,
        pos_on_tplane_e_1,
        pos_on_tplane_e_2,
        primal_normal_cell_1,
        dual_normal_cell_1,
        primal_normal_cell_2,
        dual_normal_cell_2,
        p_dthalf,
        rho_ref_me,
        theta_ref_me,
        z_grad_rth_1,
        z_grad_rth_2,
        z_grad_rth_3,
        z_grad_rth_4,
        z_rth_pr_1,
        z_rth_pr_2,
        out=(z_rho_e, z_theta_v_e),
    )
