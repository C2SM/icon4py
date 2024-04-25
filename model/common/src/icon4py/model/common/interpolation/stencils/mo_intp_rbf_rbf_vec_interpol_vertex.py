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

from icon4py.model.common.dimension import V2E, EdgeDim, KDim, V2EDim, VertexDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _mo_intp_rbf_rbf_vec_interpol_vertex(
    p_e_in: Field[[EdgeDim, KDim], wpfloat],
    ptr_coeff_1: Field[[VertexDim, V2EDim], wpfloat],
    ptr_coeff_2: Field[[VertexDim, V2EDim], wpfloat],
) -> tuple[Field[[VertexDim, KDim], wpfloat], Field[[VertexDim, KDim], wpfloat]]:
    p_u_out = neighbor_sum(ptr_coeff_1 * p_e_in(V2E), axis=V2EDim)
    p_v_out = neighbor_sum(ptr_coeff_2 * p_e_in(V2E), axis=V2EDim)
    return p_u_out, p_v_out


# We temporarility define two here for caching reasons, until we can stop passing sizes to the toolchain
# as this stencil is used more than once in diffusion.
@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def mo_intp_rbf_rbf_vec_interpol_vertex(
    p_e_in: Field[[EdgeDim, KDim], wpfloat],
    ptr_coeff_1: Field[[VertexDim, V2EDim], wpfloat],
    ptr_coeff_2: Field[[VertexDim, V2EDim], wpfloat],
    p_u_out: Field[[VertexDim, KDim], wpfloat],
    p_v_out: Field[[VertexDim, KDim], wpfloat],
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
            VertexDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )

