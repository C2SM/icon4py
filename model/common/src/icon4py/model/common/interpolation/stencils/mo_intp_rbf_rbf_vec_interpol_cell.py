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
from icon4py.model.common.dimension import C2E, C2EDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import wpfloat


# TODO: this will have to be removed once domain allows for imports
CellDim = dims.CellDim
KDim = dims.KDim


@field_operator
def _mo_intp_rbf_rbf_vec_interpol_cell(
    p_vn_in: fa.EdgeKField[wpfloat],
    ptr_coeff_1: Field[[dims.CellDim, C2EDim], wpfloat],
    ptr_coeff_2: Field[[dims.CellDim, C2EDim], wpfloat],
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    p_u_out = neighbor_sum(ptr_coeff_1 * p_vn_in(C2E), axis=C2EDim)
    p_v_out = neighbor_sum(ptr_coeff_2 * p_vn_in(C2E), axis=C2EDim)
    return p_u_out, p_v_out


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def mo_intp_rbf_rbf_vec_interpol_cell(
    p_vn_in: fa.EdgeKField[wpfloat],
    ptr_coeff_1: Field[[dims.CellDim, C2EDim], wpfloat],
    ptr_coeff_2: Field[[dims.CellDim, C2EDim], wpfloat],
    p_u_out: fa.CellKField[wpfloat],
    p_v_out: fa.CellKField[wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _mo_intp_rbf_rbf_vec_interpol_cell(
        p_vn_in,
        ptr_coeff_1,
        ptr_coeff_2,
        out=(p_u_out, p_v_out),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
