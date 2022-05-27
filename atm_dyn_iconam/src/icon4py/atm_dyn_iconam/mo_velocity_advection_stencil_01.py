from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Field, float32, neighbor_sum

from icon4py.common.dimension import KDim, EdgeDim, E2C2EDim, E2C2E


@field_operator
def _mo_velocity_advection_stencil_01(
    vn: Field[[EdgeDim, KDim], float32],
    rbf_vec_coeff_e: Field[[EdgeDim, E2C2EDim], float32],
) -> Field[[EdgeDim, KDim], float32]:
    vt = neighbor_sum(vn(E2C2E) * rbf_vec_coeff_e, axis=E2C2EDim)
    return vt


@program
def mo_velocity_advection_stencil_01(
    vn: Field[[EdgeDim, KDim], float32],
    rbf_vec_coeff_e: Field[[EdgeDim, E2C2EDim], float32],
    vt: Field[[EdgeDim, KDim], float32],
):
    _mo_velocity_advection_stencil_01(vn, rbf_vec_coeff_e, out=vt)
