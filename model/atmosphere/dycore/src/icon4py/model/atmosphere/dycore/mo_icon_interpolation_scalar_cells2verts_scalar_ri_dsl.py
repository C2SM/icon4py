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
from gt4py.next.ffront.fbuiltins import Field, astype, int32, neighbor_sum

from icon4py.model.common.dimension import V2C, CellDim, KDim, V2CDim, VertexDim
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.model_backend import backend


@field_operator
def _mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
    p_cell_in: Field[[CellDim, KDim], wpfloat],
    c_intp: Field[[VertexDim, V2CDim], wpfloat],
) -> Field[[VertexDim, KDim], vpfloat]:
    p_vert_out_wp = neighbor_sum(p_cell_in(V2C) * c_intp, axis=V2CDim)
    return astype(p_vert_out_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
    p_cell_in: Field[[CellDim, KDim], wpfloat],
    c_intp: Field[[VertexDim, V2CDim], wpfloat],
    p_vert_out: Field[[VertexDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
        p_cell_in,
        c_intp,
        out=p_vert_out,
        domain={
            VertexDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
