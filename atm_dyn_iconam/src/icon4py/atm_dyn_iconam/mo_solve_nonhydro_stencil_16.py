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

from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Field, broadcast, where

from icon4py.common.dimension import E2C, CellDim, EdgeDim, KDim


@field_operator
def _mo_solve_nonhydro_stencil_16(
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
        p_vn > 0.0,
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
        p_vn > 0.0,
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


@program
def mo_solve_nonhydro_stencil_16(
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
    z_rho_e: Field[[EdgeDim, KDim], float],
    z_theta_v_e: Field[[EdgeDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_16(
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
        out=(z_rho_e, z_theta_v_e),
    )
