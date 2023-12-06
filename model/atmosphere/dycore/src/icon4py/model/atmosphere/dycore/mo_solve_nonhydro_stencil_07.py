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
def _mo_solve_nonhydro_stencil_07(
    rho: Field[[CellDim, KDim], wpfloat],
    rho_ref_mc: Field[[CellDim, KDim], vpfloat],
    theta_v: Field[[CellDim, KDim], wpfloat],
    theta_ref_mc: Field[[CellDim, KDim], vpfloat],
) -> tuple[Field[[CellDim, KDim], vpfloat], Field[[CellDim, KDim], vpfloat]]:
    rho_ref_mc_wp, theta_ref_mc_wp = astype((rho_ref_mc, theta_ref_mc), wpfloat)

    z_rth_pr_1_wp = rho - rho_ref_mc_wp
    z_rth_pr_2_wp = theta_v - theta_ref_mc_wp
    return astype((z_rth_pr_1_wp, z_rth_pr_2_wp), vpfloat)


@program(grid_type=GridType.UNSTRUCTURED)
def mo_solve_nonhydro_stencil_07(
    rho: Field[[CellDim, KDim], wpfloat],
    rho_ref_mc: Field[[CellDim, KDim], vpfloat],
    theta_v: Field[[CellDim, KDim], wpfloat],
    theta_ref_mc: Field[[CellDim, KDim], vpfloat],
    z_rth_pr_1: Field[[CellDim, KDim], vpfloat],
    z_rth_pr_2: Field[[CellDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _mo_solve_nonhydro_stencil_07(
        rho,
        rho_ref_mc,
        theta_v,
        theta_ref_mc,
        out=(z_rth_pr_1, z_rth_pr_2),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
