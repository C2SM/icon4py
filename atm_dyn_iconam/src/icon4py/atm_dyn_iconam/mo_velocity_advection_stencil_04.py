from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Field, float32

from icon4py.common.dimension import KDim, EdgeDim


@field_operator
def _mo_velocity_advection_stencil_04(
    vn: Field[[EdgeDim, KDim], float32],
    ddxn_z_full: Field[[EdgeDim, KDim], float32],
    ddxt_z_full: Field[[EdgeDim, KDim], float32],
    vt: Field[[EdgeDim, KDim], float32],
) -> Field[[EdgeDim, KDim], float32]:
    z_w_concorr_me = vn * ddxn_z_full + vt * ddxt_z_full
    return z_w_concorr_me


@program
def mo_velocity_advection_stencil_04(
    vn: Field[[EdgeDim, KDim], float32],
    ddxn_z_full: Field[[EdgeDim, KDim], float32],
    ddxt_z_full: Field[[EdgeDim, KDim], float32],
    vt: Field[[EdgeDim, KDim], float32],
    z_w_concorr_me: Field[[EdgeDim, KDim], float32],
):
    _mo_velocity_advection_stencil_04(
        vn, ddxn_z_full, ddxt_z_full, vt, out=z_w_concorr_me
    )
