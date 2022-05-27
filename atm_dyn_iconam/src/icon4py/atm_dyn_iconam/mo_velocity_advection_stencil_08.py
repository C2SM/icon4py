from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Field, float32, neighbor_sum

from icon4py.common.dimension import KDim, CellDim, C2EDim, EdgeDim, C2E


@field_operator
def _mo_velocity_advection_stencil_08(
    z_kin_hor_e: Field[[EdgeDim, KDim], float32],
    e_bln_c_s: Field[[CellDim, C2EDim], float32],
) -> Field[[CellDim, KDim], float32]:
    z_ekinh = neighbor_sum(e_bln_c_s * z_kin_hor_e(C2E), axis=C2EDim)
    return z_ekinh


@program
def mo_velocity_advection_stencil_08(
    z_kin_hor_e: Field[[EdgeDim, KDim], float32],
    e_bln_c_s: Field[[CellDim, C2EDim], float32],
    z_ekinh: Field[[CellDim, KDim], float32],
):
    _mo_velocity_advection_stencil_08(z_kin_hor_e, e_bln_c_s, out=z_ekinh)
