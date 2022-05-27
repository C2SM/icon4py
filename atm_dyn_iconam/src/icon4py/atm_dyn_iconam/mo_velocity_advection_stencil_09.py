from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Field, float32, neighbor_sum

from icon4py.common.dimension import KDim, CellDim, C2EDim, EdgeDim, C2E


@field_operator
def _mo_velocity_advection_stencil_09(
    z_w_concorr_me: Field[[EdgeDim, KDim], float32],
    e_bln_c_s: Field[[CellDim, C2EDim], float32],
) -> Field[[CellDim, KDim], float32]:
    z_w_concorr_mc = neighbor_sum(e_bln_c_s * z_w_concorr_me(C2E), axis=C2EDim)
    return z_w_concorr_mc


@program
def mo_velocity_advection_stencil_09(
    z_w_concorr_me: Field[[EdgeDim, KDim], float32],
    e_bln_c_s: Field[[CellDim, C2EDim], float32],
    z_w_concorr_mc: Field[[CellDim, KDim], float32],
):
    _mo_velocity_advection_stencil_09(z_w_concorr_me, e_bln_c_s, out=z_w_concorr_mc)
