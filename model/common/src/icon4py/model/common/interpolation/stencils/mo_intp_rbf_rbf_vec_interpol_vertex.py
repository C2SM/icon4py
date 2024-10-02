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
from icon4py.model.common.dimension import V2E, V2EDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _mo_intp_rbf_rbf_vec_interpol_vertex(
    p_e_in: fa.EdgeKField[wpfloat],
    ptr_coeff_1: Field[[dims.VertexDim, V2EDim], wpfloat],
    ptr_coeff_2: Field[[dims.VertexDim, V2EDim], wpfloat],
) -> tuple[
    Field[[dims.VertexDim, dims.KDim], wpfloat], Field[[dims.VertexDim, dims.KDim], wpfloat]
]:
    p_u_out = neighbor_sum(ptr_coeff_1 * p_e_in(V2E), axis=V2EDim)
    p_v_out = neighbor_sum(ptr_coeff_2 * p_e_in(V2E), axis=V2EDim)
    return p_u_out, p_v_out


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def mo_intp_rbf_rbf_vec_interpol_vertex(
    p_e_in: fa.EdgeKField[wpfloat],
    ptr_coeff_1: Field[[dims.VertexDim, V2EDim], wpfloat],
    ptr_coeff_2: Field[[dims.VertexDim, V2EDim], wpfloat],
    p_u_out: Field[[dims.VertexDim, dims.KDim], wpfloat],
    p_v_out: Field[[dims.VertexDim, dims.KDim], wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _mo_intp_rbf_rbf_vec_interpol_vertex(
        p_e_in,
        ptr_coeff_1,
        ptr_coeff_2,
        out=(p_u_out, p_v_out),
        domain={
            dims.VertexDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
