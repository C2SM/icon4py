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

from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field
from gt4py.next.program_processors.runners import gtfn_cpu

from icon4py.common.dimension import CellDim, KDim


@field_operator
def _mo_solve_nonhydro_stencil_13(
    rho: Field[[CellDim, KDim], float],
    rho_ref_mc: Field[[CellDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
    theta_ref_mc: Field[[CellDim, KDim], float],
) -> tuple[Field[[CellDim, KDim], float], Field[[CellDim, KDim], float]]:
    z_rth_pr_1 = rho - rho_ref_mc
    z_rth_pr_2 = theta_v - theta_ref_mc
    return z_rth_pr_1, z_rth_pr_2


@program(backend=gtfn_cpu.run_gtfn)
def mo_solve_nonhydro_stencil_13(
    rho: Field[[CellDim, KDim], float],
    rho_ref_mc: Field[[CellDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
    theta_ref_mc: Field[[CellDim, KDim], float],
    z_rth_pr_1: Field[[CellDim, KDim], float],
    z_rth_pr_2: Field[[CellDim, KDim], float],
    horizontal_start: int,
    horizontal_end: int,
    vertical_start: int,
    vertical_end: int,
):
    _mo_solve_nonhydro_stencil_13(
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
