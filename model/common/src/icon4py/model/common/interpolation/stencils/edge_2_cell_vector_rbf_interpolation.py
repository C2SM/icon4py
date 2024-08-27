# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, int32, neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2E2C2E, C2E2C2EDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _edge_2_cell_vector_rbf_interpolation(
    p_e_in: fa.EdgeKField[vpfloat],
    ptr_coeff_1: Field[[dims.CellDim, C2E2C2EDim], wpfloat],
    ptr_coeff_2: Field[[dims.CellDim, C2E2C2EDim], wpfloat],
) -> tuple[fa.CellKField[vpfloat], fa.CellKField[vpfloat]]:
    p_u_out = neighbor_sum(ptr_coeff_1 * p_e_in(C2E2C2E), axis=C2E2C2EDim)
    p_v_out = neighbor_sum(ptr_coeff_2 * p_e_in(C2E2C2E), axis=C2E2C2EDim)
    return p_u_out, p_v_out


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def edge_2_cell_vector_rbf_interpolation(
    p_e_in: fa.EdgeKField[vpfloat],
    ptr_coeff_1: Field[[dims.CellDim, C2E2C2EDim], wpfloat],
    ptr_coeff_2: Field[[dims.CellDim, C2E2C2EDim], wpfloat],
    p_u_out: fa.CellKField[vpfloat],
    p_v_out: fa.CellKField[vpfloat],
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
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
