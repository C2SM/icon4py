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
from functional.ffront.fbuiltins import Field

from icon4py.common.dimension import CellDim, KDim


@field_operator
def _mo_solve_nonhydro_stencil_13_z_rth_pr_1(
    rho: Field[[CellDim, KDim], float], rho_ref_mc: Field[[CellDim, KDim], float]
) -> Field[[CellDim, KDim], float]:
    z_rth_pr_1 = rho - rho_ref_mc
    return z_rth_pr_1


@field_operator
def _mo_solve_nonhydro_stencil_13_z_rth_pr_2(
    theta_v: Field[[CellDim, KDim], float],
    theta_ref_mc: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], float]:
    z_rth_pr_2 = theta_v - theta_ref_mc
    return z_rth_pr_2


@program
def mo_solve_nonhydro_stencil_13(
    rho: Field[[CellDim, KDim], float],
    rho_ref_mc: Field[[CellDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
    theta_ref_mc: Field[[CellDim, KDim], float],
    z_rth_pr_1: Field[[CellDim, KDim], float],
    z_rth_pr_2: Field[[CellDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_13_z_rth_pr_1(rho, rho_ref_mc, out=z_rth_pr_1)
    _mo_solve_nonhydro_stencil_13_z_rth_pr_2(theta_v, theta_ref_mc, out=z_rth_pr_2)
