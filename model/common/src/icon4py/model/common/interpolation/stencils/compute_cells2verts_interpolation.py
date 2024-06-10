import gt4py.next as gt
from gt4py.next import neighbor_sum

from icon4py.model.common.dimension import V2C, CellDim, KDim, V2CDim, VertexDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import wpfloat


@gt.field_operator
def _compute_cells2verts(
    cell_in: gt.Field[[CellDim, KDim], wpfloat],
    c_int: gt.Field[[VertexDim, V2CDim], wpfloat],
) -> gt.Field[[VertexDim, KDim], wpfloat]:
    vert_out = neighbor_sum(c_int * cell_in(V2C), axis=V2CDim)
    return vert_out


@gt.program(grid_type=gt.GridType.UNSTRUCTURED, backend=backend)
def compute_cells2verts_interpolation(
    cell_in: gt.Field[[CellDim, KDim], wpfloat],
    c_int: gt.Field[[VertexDim, V2CDim], wpfloat],
    vert_out: gt.Field[[VertexDim, KDim], wpfloat],
    horizontal_start: gt.int32,
    horizontal_end: gt.int32,
    vertical_start: gt.int32,
    vertical_end: gt.int32,
):
    _compute_cells2verts(
        cell_in,
        c_int,
        out=vert_out,
        domain={
            VertexDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )