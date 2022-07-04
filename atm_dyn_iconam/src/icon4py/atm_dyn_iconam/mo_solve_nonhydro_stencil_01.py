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
from functional.ffront.fbuiltins import Field, float32

from icon4py.common.dimension import CellDim, KDim


@field_operator
def _mo_solve_nonhydro_stencil_01_z_rth_pr_1() -> Field[[CellDim, KDim], float32]:
    z_rth_pr_1 = float32(0.0)
    return z_rth_pr_1


@program
def mo_solve_nonhydro_stencil_01_z_rth_pr_1(
    z_rth_pr_1: Field[[CellDim, KDim], float32]
):
    _mo_solve_nonhydro_stencil_01_z_rth_pr_1(out=z_rth_pr_1)


@field_operator
def _mo_solve_nonhydro_stencil_01_z_rth_pr_2() -> Field[[CellDim, KDim], float32]:
    z_rth_pr_2 = float32(0.0)
    return z_rth_pr_2


@program
def mo_solve_nonhydro_stencil_01_z_rth_pr_2(
    z_rth_pr_2: Field[[CellDim, KDim], float32]
):
    _mo_solve_nonhydro_stencil_01_z_rth_pr_2(out=z_rth_pr_2)


@program
def mo_solve_nonhydro_stencil_01(
    z_rth_pr_1: Field[[CellDim, KDim], float32],
    z_rth_pr_2: Field[[CellDim, KDim], float32],
):
    _mo_solve_nonhydro_stencil_01_z_rth_pr_1(out=z_rth_pr_1)
    _mo_solve_nonhydro_stencil_01_z_rth_pr_2(out=z_rth_pr_2)
