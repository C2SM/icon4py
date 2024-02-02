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
from gt4py.next.ffront.fbuiltins import Field, astype, int32

from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_solver_coefficients_matrix(
    exner_nnow: Field[[CellDim, KDim], wpfloat],
    rho_nnow: Field[[CellDim, KDim], wpfloat],
    theta_v_nnow: Field[[CellDim, KDim], wpfloat],
    inv_ddqz_z_full: Field[[CellDim, KDim], vpfloat],
    vwind_impl_wgt: Field[[CellDim], wpfloat],
    theta_v_ic: Field[[CellDim, KDim], wpfloat],
    rho_ic: Field[[CellDim, KDim], wpfloat],
    dtime: wpfloat,
    rd: wpfloat,
    cvd: wpfloat,
) -> tuple[Field[[CellDim, KDim], vpfloat], Field[[CellDim, KDim], vpfloat]]:
    """Formerly known as _mo_solve_nonhydro_stencil_44."""
    inv_ddqz_z_full_wp = astype(inv_ddqz_z_full, wpfloat)

    z_beta_wp = dtime * rd * exner_nnow / (cvd * rho_nnow * theta_v_nnow) * inv_ddqz_z_full_wp
    z_alpha_wp = vwind_impl_wgt * theta_v_ic * rho_ic
    return astype((z_beta_wp, z_alpha_wp), vpfloat)


@program(grid_type=GridType.UNSTRUCTURED)
def compute_solver_coefficients_matrix(
    z_beta: Field[[CellDim, KDim], vpfloat],
    exner_nnow: Field[[CellDim, KDim], wpfloat],
    rho_nnow: Field[[CellDim, KDim], wpfloat],
    theta_v_nnow: Field[[CellDim, KDim], wpfloat],
    inv_ddqz_z_full: Field[[CellDim, KDim], vpfloat],
    z_alpha: Field[[CellDim, KDim], vpfloat],
    vwind_impl_wgt: Field[[CellDim], wpfloat],
    theta_v_ic: Field[[CellDim, KDim], wpfloat],
    rho_ic: Field[[CellDim, KDim], wpfloat],
    dtime: wpfloat,
    rd: wpfloat,
    cvd: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_solver_coefficients_matrix(
        exner_nnow,
        rho_nnow,
        theta_v_nnow,
        inv_ddqz_z_full,
        vwind_impl_wgt,
        theta_v_ic,
        rho_ic,
        dtime,
        rd,
        cvd,
        out=(z_beta, z_alpha),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
