from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import (
    Field,
    float32,
    neighbor_sum,
)

from icon4py.common.dimension import KDim, VertexDim, CellDim, V2CDim, V2C


@field_operator
def _mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
    p_cell_in: Field[[CellDim, KDim], float32],
    c_intp: Field[[VertexDim, V2CDim], float32],
) -> Field[[VertexDim, KDim], float32]:
    p_vert_out = neighbor_sum(c_intp * p_cell_in(V2C), axis=V2CDim)
    return p_vert_out


@program
def mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
    p_cell_in: Field[[CellDim, KDim], float32],
    c_intp: Field[[VertexDim, V2CDim], float32],
    p_vert_out: Field[[VertexDim, KDim], float32],
):
    _mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
        p_cell_in, c_intp, out=p_vert_out
    )
