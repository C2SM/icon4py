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
from gt4py.next.ffront.fbuiltins import Field, neighbor_sum

from icon4py.model.common.dimension import E2C, CellDim, E2CDim, EdgeDim, KDim


@field_operator
def _mo_solve_nonhydro_stencil_19(
    inv_dual_edge_length: Field[[EdgeDim], float],
    z_exner_ex_pr: Field[[CellDim, KDim], float],
    ddxn_z_full: Field[[EdgeDim, KDim], float],
    c_lin_e: Field[[EdgeDim, E2CDim], float],
    z_dexner_dz_c_1: Field[[CellDim, KDim], float],
) -> Field[[EdgeDim, KDim], float]:
    z_gradh_exner = inv_dual_edge_length * (
        z_exner_ex_pr(E2C[1]) - z_exner_ex_pr(E2C[0])
    ) - ddxn_z_full * neighbor_sum(z_dexner_dz_c_1(E2C) * c_lin_e, axis=E2CDim)
    return z_gradh_exner


@program
def mo_solve_nonhydro_stencil_19(
    inv_dual_edge_length: Field[[EdgeDim], float],
    z_exner_ex_pr: Field[[CellDim, KDim], float],
    ddxn_z_full: Field[[EdgeDim, KDim], float],
    c_lin_e: Field[[EdgeDim, E2CDim], float],
    z_dexner_dz_c_1: Field[[CellDim, KDim], float],
    z_gradh_exner: Field[[EdgeDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_19(
        inv_dual_edge_length,
        z_exner_ex_pr,
        ddxn_z_full,
        c_lin_e,
        z_dexner_dz_c_1,
        out=z_gradh_exner,
    )
