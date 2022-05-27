from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Field, float32

from icon4py.common.dimension import KDim, EdgeDim


@field_operator
def _mo_velocity_advection_stencil_05_vn_ie(
    vn: Field[[EdgeDim, KDim], float32]
) -> Field[[EdgeDim, KDim], float32]:
    vn_ie = vn
    return vn_ie


@program
def mo_velocity_advection_stencil_05_vn_ie(
    vn: Field[[EdgeDim, KDim], float32], vn_ie: Field[[EdgeDim, KDim], float32]
):
    _mo_velocity_advection_stencil_05_vn_ie(vn, out=vn_ie)


@field_operator
def _mo_velocity_advection_stencil_05_z_vt_ie(
    vt: Field[[EdgeDim, KDim], float32]
) -> Field[[EdgeDim, KDim], float32]:
    z_vt_ie = vt
    return z_vt_ie


@program
def mo_velocity_advection_stencil_05_z_vt_ie(
    vt: Field[[EdgeDim, KDim], float32], z_vt_ie: Field[[EdgeDim, KDim], float32]
):
    _mo_velocity_advection_stencil_05_z_vt_ie(vt, out=z_vt_ie)


@field_operator
def _mo_velocity_advection_stencil_05_z_kin_hor_e(
    vn: Field[[EdgeDim, KDim], float32], vt: Field[[EdgeDim, KDim], float32]
) -> Field[[EdgeDim, KDim], float32]:
    z_kin_hor_e = float32(0.5) * ((vn * vn) + (vt * vt))
    return z_kin_hor_e


@program
def mo_velocity_advection_stencil_05_z_kin_hor_e(
    vn: Field[[EdgeDim, KDim], float32],
    vt: Field[[EdgeDim, KDim], float32],
    z_kin_hor_e: Field[[EdgeDim, KDim], float32],
):
    _mo_velocity_advection_stencil_05_z_kin_hor_e(vn, vt, out=z_kin_hor_e)


@program
def mo_velocity_advection_stencil_05(
    vn: Field[[EdgeDim, KDim], float32],
    vt: Field[[EdgeDim, KDim], float32],
    vn_ie: Field[[EdgeDim, KDim], float32],
    z_vt_ie: Field[[EdgeDim, KDim], float32],
    z_kin_hor_e: Field[[EdgeDim, KDim], float32],
):
    _mo_velocity_advection_stencil_05_vn_ie(vn, out=vn_ie)
    _mo_velocity_advection_stencil_05_z_vt_ie(vt, out=z_vt_ie)
    _mo_velocity_advection_stencil_05_z_kin_hor_e(vn, vt, out=z_kin_hor_e)
