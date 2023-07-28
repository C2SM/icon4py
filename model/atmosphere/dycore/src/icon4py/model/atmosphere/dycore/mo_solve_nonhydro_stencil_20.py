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
from gt4py.next.ffront.experimental import as_offset
from gt4py.next.ffront.fbuiltins import Field, int32

from icon4py.model.common.dimension import (
    E2C,
    E2EC,
    CellDim,
    ECDim,
    EdgeDim,
    KDim,
    Koff,
)


@field_operator
def _mo_solve_nonhydro_stencil_20(
    inv_dual_edge_length: Field[[EdgeDim], float],
    z_exner_ex_pr: Field[[CellDim, KDim], float],
    zdiff_gradp: Field[[ECDim, KDim], float],
    ikoffset: Field[[ECDim, KDim], int32],
    z_dexner_dz_c_1: Field[[CellDim, KDim], float],
    z_dexner_dz_c_2: Field[[CellDim, KDim], float],
) -> Field[[EdgeDim, KDim], float]:
    z_exner_ex_pr_0 = z_exner_ex_pr(as_offset(Koff, ikoffset(E2EC[0])))
    z_exner_ex_pr_1 = z_exner_ex_pr(as_offset(Koff, ikoffset(E2EC[1])))

    z_dexner_dz_c1_0 = z_dexner_dz_c_1(as_offset(Koff, ikoffset(E2EC[0])))
    z_dexner_dz_c1_1 = z_dexner_dz_c_1(as_offset(Koff, ikoffset(E2EC[1])))

    z_dexner_dz_c2_0 = z_dexner_dz_c_2(as_offset(Koff, ikoffset(E2EC[0])))
    z_dexner_dz_c2_1 = z_dexner_dz_c_2(as_offset(Koff, ikoffset(E2EC[1])))

    z_gradh_exner = inv_dual_edge_length * (
        (
            z_exner_ex_pr_1(E2C[1])
            + zdiff_gradp(E2EC[1])
            * (z_dexner_dz_c1_1(E2C[1]) + zdiff_gradp(E2EC[1]) * z_dexner_dz_c2_1(E2C[1]))
        )
        - (
            z_exner_ex_pr_0(E2C[0])
            + zdiff_gradp(E2EC[0])
            * (z_dexner_dz_c1_0(E2C[0]) + zdiff_gradp(E2EC[0]) * z_dexner_dz_c2_0(E2C[0]))
        )
    )

    return z_gradh_exner


@program(grid_type=GridType.UNSTRUCTURED)
def mo_solve_nonhydro_stencil_20(
    inv_dual_edge_length: Field[[EdgeDim], float],
    z_exner_ex_pr: Field[[CellDim, KDim], float],
    zdiff_gradp: Field[[ECDim, KDim], float],
    ikoffset: Field[[ECDim, KDim], int32],
    z_dexner_dz_c_1: Field[[CellDim, KDim], float],
    z_dexner_dz_c_2: Field[[CellDim, KDim], float],
    z_gradh_exner: Field[[EdgeDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_20(
        inv_dual_edge_length,
        z_exner_ex_pr,
        zdiff_gradp,
        ikoffset,
        z_dexner_dz_c_1,
        z_dexner_dz_c_2,
        out=z_gradh_exner,
    )
