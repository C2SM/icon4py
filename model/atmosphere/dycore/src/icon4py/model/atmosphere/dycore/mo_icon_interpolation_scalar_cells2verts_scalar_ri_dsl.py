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
from gt4py.next.ffront.fbuiltins import Field, neighbor_sum

from icon4py.model.common.dimension import V2C, CellDim, KDim, V2CDim, VertexDim


@field_operator
def _mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
    p_cell_in: Field[[CellDim, KDim], float],
    c_intp: Field[[VertexDim, V2CDim], float],
) -> Field[[VertexDim, KDim], float]:
    p_vert_out = neighbor_sum(p_cell_in(V2C) * c_intp, axis=V2CDim)
    return p_vert_out


@program(grid_type=GridType.UNSTRUCTURED)
def mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
    p_cell_in: Field[[CellDim, KDim], float],
    c_intp: Field[[VertexDim, V2CDim], float],
    p_vert_out: Field[[VertexDim, KDim], float],
):
    _mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(p_cell_in, c_intp, out=p_vert_out)
