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
from icon4py.model.common.dimension import V2E, EdgeDim, KDim, V2EDim, VertexDim


@field_operator
def _mo_intp_rbf_rbf_vec_interpol_vertex(
    p_e_in: Field[[EdgeDim, KDim], float],
    ptr_coeff_1: Field[[VertexDim, V2EDim], float],
    ptr_coeff_2: Field[[VertexDim, V2EDim], float],
) -> tuple[Field[[VertexDim, KDim], float], Field[[VertexDim, KDim], float]]:
    p_u_out = neighbor_sum(p_e_in(V2E) * ptr_coeff_1, axis=V2EDim)
    p_v_out = neighbor_sum(p_e_in(V2E) * ptr_coeff_2, axis=V2EDim)
    return p_u_out, p_v_out


@program
def mo_intp_rbf_rbf_vec_interpol_vertex(
    p_e_in: Field[[EdgeDim, KDim], float],
    ptr_coeff_1: Field[[VertexDim, V2EDim], float],
    ptr_coeff_2: Field[[VertexDim, V2EDim], float],
    p_u_out: Field[[VertexDim, KDim], float],
    p_v_out: Field[[VertexDim, KDim], float],
):
    _mo_intp_rbf_rbf_vec_interpol_vertex(
        p_e_in, ptr_coeff_1, ptr_coeff_2, out=(p_u_out, p_v_out)
    )
