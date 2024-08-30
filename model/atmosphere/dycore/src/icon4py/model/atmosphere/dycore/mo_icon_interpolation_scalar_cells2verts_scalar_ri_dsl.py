# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, astype, int32, neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import V2C, V2CDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


# TODO: this will have to be removed once domain allows for imports
VertexDim = dims.VertexDim
KDim = dims.KDim


@field_operator
def _mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
    p_cell_in: fa.CellKField[wpfloat],
    c_intp: Field[[dims.VertexDim, V2CDim], wpfloat],
) -> fa.VertexKField[vpfloat]:
    p_vert_out_wp = neighbor_sum(p_cell_in(V2C) * c_intp, axis=V2CDim)
    return astype(p_vert_out_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
    p_cell_in: fa.CellKField[wpfloat],
    c_intp: Field[[dims.VertexDim, V2CDim], wpfloat],
    p_vert_out: fa.VertexKField[vpfloat],
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
