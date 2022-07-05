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
from functional.ffront.fbuiltins import Field, float

from icon4py.common.dimension import CellDim, KDim


@field_operator
def _mo_solve_nonhydro_stencil_46_w_nnew() -> Field[[CellDim, KDim], float]:
    w_nnew = 0.0
    return w_nnew


@program
def mo_solve_nonhydro_stencil_46_w_nnew(w_nnew: Field[[CellDim, KDim], float]):
    _mo_solve_nonhydro_stencil_46_w_nnew(out=w_nnew)


@field_operator
def _mo_solve_nonhydro_stencil_46_z_contr_w_fl_l() -> Field[[CellDim, KDim], float]:
    z_contr_w_fl_l = 0.0
    return z_contr_w_fl_l


@program
def mo_solve_nonhydro_stencil_46_z_contr_w_fl_l(
    z_contr_w_fl_l: Field[[CellDim, KDim], float]
):
    _mo_solve_nonhydro_stencil_46_z_contr_w_fl_l(out=z_contr_w_fl_l)


@program
def mo_solve_nonhydro_stencil_46(
    w_nnew: Field[[CellDim, KDim], float],
    z_contr_w_fl_l: Field[[CellDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_46_w_nnew(out=w_nnew)
    _mo_solve_nonhydro_stencil_46_z_contr_w_fl_l(out=z_contr_w_fl_l)
