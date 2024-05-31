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
from gt4py.next.ffront.fbuiltins import Field, int32, neighbor_sum
from model.common.tests import field_type_aliases as fa

from icon4py.model.common.dimension import C2E2C2E, C2E2C2EDim, CellDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _edge_2_cell_vector_rbf_interpolation(
    p_e_in: fa.EKvpField,
    ptr_coeff_1: Field[[CellDim, C2E2C2EDim], wpfloat],
    ptr_coeff_2: Field[[CellDim, C2E2C2EDim], wpfloat],
) -> tuple[fa.CKvpField, fa.CKvpField]:
    p_u_out = neighbor_sum(ptr_coeff_1 * p_e_in(C2E2C2E), axis=C2E2C2EDim)
    p_v_out = neighbor_sum(ptr_coeff_2 * p_e_in(C2E2C2E), axis=C2E2C2EDim)
    return p_u_out, p_v_out


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def edge_2_cell_vector_rbf_interpolation(
    p_e_in: fa.EKvpField,
    ptr_coeff_1: Field[[CellDim, C2E2C2EDim], wpfloat],
    ptr_coeff_2: Field[[CellDim, C2E2C2EDim], wpfloat],
    p_u_out: fa.CKvpField,
    p_v_out: fa.CKvpField,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _edge_2_cell_vector_rbf_interpolation(
        p_e_in,
        ptr_coeff_1,
        ptr_coeff_2,
        out=(p_u_out, p_v_out),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
