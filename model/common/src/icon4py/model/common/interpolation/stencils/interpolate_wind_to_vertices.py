# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.interpolation.stencils.compute_cell_2_vertex_interpolation import (
    _compute_cell_2_vertex_interpolation,
)
from icon4py.model.common.interpolation.stencils.mo_intp_rbf_rbf_vec_interpol_vertex import (
    _mo_intp_rbf_rbf_vec_interpol_vertex,
)
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _interpolate_wind_to_vertices(
    w: fa.CellKField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
    cells_aw_verts: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], wpfloat],
    rbf_coeff_v1: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], wpfloat],
    rbf_coeff_v2: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], wpfloat],
) -> tuple[fa.VertexKField[wpfloat], fa.VertexKField[wpfloat], fa.VertexKField[wpfloat]]:
    """
    Interpolate the wind components to the vertices.

    The vertical component is averaged from the V2C neighbor cells with the
    weights ``cells_aw_verts``; the horizontal components are reconstructed
    from the edge-normal component with the RBF coefficients ``rbf_coeff_v1``
    and ``rbf_coeff_v2``.

    Note that ``w`` lives on half levels and ``vn`` on full levels, so the
    three outputs do not share a vertical domain.
    """
    w_vert = _compute_cell_2_vertex_interpolation(w, cells_aw_verts)
    u_vert, v_vert = _mo_intp_rbf_rbf_vec_interpol_vertex(
        p_e_in=vn, ptr_coeff_1=rbf_coeff_v1, ptr_coeff_2=rbf_coeff_v2
    )
    return w_vert, u_vert, v_vert


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def interpolate_wind_to_vertices(
    w: fa.CellKField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
    cells_aw_verts: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], wpfloat],
    rbf_coeff_v1: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], wpfloat],
    rbf_coeff_v2: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], wpfloat],
    w_vert: fa.VertexKField[wpfloat],
    u_vert: fa.VertexKField[wpfloat],
    v_vert: fa.VertexKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
    vertical_end_half: gtx.int32,
) -> None:
    _interpolate_wind_to_vertices(
        w=w,
        vn=vn,
        cells_aw_verts=cells_aw_verts,
        rbf_coeff_v1=rbf_coeff_v1,
        rbf_coeff_v2=rbf_coeff_v2,
        out=(w_vert, u_vert, v_vert),
        domain=(
            # w_vert: half levels
            {
                dims.VertexDim: (horizontal_start, horizontal_end),
                dims.KDim: (vertical_start, vertical_end_half),
            },
            # u_vert / v_vert: full levels
            {
                dims.VertexDim: (horizontal_start, horizontal_end),
                dims.KDim: (vertical_start, vertical_end),
            },
            {
                dims.VertexDim: (horizontal_start, horizontal_end),
                dims.KDim: (vertical_start, vertical_end),
            },
        ),
    )
