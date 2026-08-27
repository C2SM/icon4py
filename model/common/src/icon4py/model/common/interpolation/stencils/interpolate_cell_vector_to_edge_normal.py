# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C, E2CDim
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _interpolate_cell_vector_to_edge_normal(
    vector_x: fa.CellKField[wpfloat],
    vector_y: fa.CellKField[wpfloat],
    primal_normal_cell_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    primal_normal_cell_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
) -> fa.EdgeKField[wpfloat]:
    """
    Interpolate a cell-centered vector field to the edges and take its normal component.

    Inverse of the RBF reconstruction in
    :mod:`edge_2_cell_vector_rbf_interpolation`::

        normal_component(e, k) = sum over the two E2C neighbor cells c of
            c_lin_e(e, c) * (vector_x(c, k) * primal_normal_cell_x(e, c)
                             + vector_y(c, k) * primal_normal_cell_y(e, c))
    """
    return neighbor_sum(
        c_lin_e * (vector_x(E2C) * primal_normal_cell_x + vector_y(E2C) * primal_normal_cell_y),
        axis=E2CDim,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def interpolate_cell_vector_to_edge_normal(
    vector_x: fa.CellKField[wpfloat],
    vector_y: fa.CellKField[wpfloat],
    primal_normal_cell_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    primal_normal_cell_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    normal_component: fa.EdgeKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _interpolate_cell_vector_to_edge_normal(
        vector_x=vector_x,
        vector_y=vector_y,
        primal_normal_cell_x=primal_normal_cell_x,
        primal_normal_cell_y=primal_normal_cell_y,
        c_lin_e=c_lin_e,
        out=normal_component,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
