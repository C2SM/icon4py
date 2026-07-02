# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C, E2C2V, E2C2VDim, KDim
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_shear_and_div_of_stress(
    u_vert: fa.VertexKField[wpfloat],
    v_vert: fa.VertexKField[wpfloat],
    w_vert: fa.VertexKField[wpfloat],
    w: fa.CellKField[wpfloat],
    vn_ie: fa.EdgeKField[wpfloat],
    vt_ie: fa.EdgeKField[wpfloat],
    w_ie: fa.EdgeKField[wpfloat],
    primal_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    inv_ddqz_z_full_e: fa.EdgeKField[wpfloat],
) -> tuple[fa.EdgeKField[wpfloat], fa.EdgeKField[wpfloat]]:
    """
    Compute shear and divergence of stress at edges of full levels.

    Fuses the ICON TMX subroutines 'compute_velocity_gradient_tensor' and
    'compute_shear' (mo_vdf_atmo.f90). The 3x3 velocity gradient tensor
    (first index: velocity component, second index: derivative direction;
    1: normal, 2: tangential, 3: vertical) is kept in local temporaries and
    contracted into

        shear      = 2 * |S|^2 = 4 * (T_11^2 + T_22^2 + T_33^2)
                     + 2 * (D_12^2 + D_13^2 + D_23^2),  D_ij = T_ij + T_ji
        div_stress = trace(S_ij) = T_11 + T_22 + T_33

    Half-level (interface) input fields (w_vert, w, vn_ie, vt_ie, w_ie) must
    provide num_levels + 1 vertical levels; outputs live on full levels.

    Domains (from the Fortran caller in mo_vdf_atmo.f90): all full levels
    (jk = 1..nlev, uniformly -- the vertical differences read the interface
    fields at jk and jk+1, so no special top/bottom rows exist in this
    stencil; the surface/top extrapolation lives in the stencils producing
    vn_ie/vt_ie); edges from rl_start = 4 (LATERAL_BOUNDARY_LEVEL_4) to
    rl_end = min_rledge_int - 2 (HALO_LEVEL_2).
    """
    # Normal/tangential velocity components at the four E2C2V vertices
    # (0, 1: edge endpoints; 2, 3: far vertices of the adjacent cells).
    vn_vert = u_vert(E2C2V) * primal_normal_vert_x + v_vert(E2C2V) * primal_normal_vert_y
    vt_vert = u_vert(E2C2V) * dual_normal_vert_x + v_vert(E2C2V) * dual_normal_vert_y

    # Vertical velocity at full levels: cell centers (E2C) and edge endpoints (E2C2V 0, 1).
    w_full_c1 = wpfloat("0.5") * (w(E2C[0]) + w(E2C[0])(KDim + 1))
    w_full_c2 = wpfloat("0.5") * (w(E2C[1]) + w(E2C[1])(KDim + 1))
    w_full_v1 = wpfloat("0.5") * (w_vert(E2C2V[0]) + w_vert(E2C2V[0])(KDim + 1))
    w_full_v2 = wpfloat("0.5") * (w_vert(E2C2V[1]) + w_vert(E2C2V[1])(KDim + 1))

    # Velocity gradient tensor at edge of full levels, e.g. T_12 = du_1/dx_2.
    vgrad_11 = (vn_vert[E2C2VDim(3)] - vn_vert[E2C2VDim(2)]) * inv_vert_vert_length
    vgrad_12 = (
        (vn_vert[E2C2VDim(1)] - vn_vert[E2C2VDim(0)]) * tangent_orientation * inv_primal_edge_length
    )
    vgrad_13 = (vn_ie - vn_ie(KDim + 1)) * inv_ddqz_z_full_e

    vgrad_21 = (vt_vert[E2C2VDim(3)] - vt_vert[E2C2VDim(2)]) * inv_vert_vert_length
    vgrad_22 = (
        (vt_vert[E2C2VDim(1)] - vt_vert[E2C2VDim(0)]) * tangent_orientation * inv_primal_edge_length
    )
    vgrad_23 = (vt_ie - vt_ie(KDim + 1)) * inv_ddqz_z_full_e

    vgrad_31 = (w_full_c2 - w_full_c1) * inv_dual_edge_length
    vgrad_32 = (w_full_v2 - w_full_v1) * tangent_orientation * inv_primal_edge_length
    vgrad_33 = (w_ie - w_ie(KDim + 1)) * inv_ddqz_z_full_e

    # Strain rates at edge center, D_ij = 2 * S_ij = du_i/dx_j + du_j/dx_i.
    d_12 = vgrad_12 + vgrad_21
    d_13 = vgrad_13 + vgrad_31
    d_23 = vgrad_23 + vgrad_32

    # shear = 2 * |S|^2 with |S| = sqrt(2 * S_ij * S_ij);
    # mechanical production is half of this value multiplied by km.
    shear = wpfloat("4.0") * (
        vgrad_11 * vgrad_11 + vgrad_22 * vgrad_22 + vgrad_33 * vgrad_33
    ) + wpfloat("2.0") * (d_12 * d_12 + d_13 * d_13 + d_23 * d_23)

    # Trace of the strain-rate tensor S_ij: trace(S_ij) = S_jj = 0.5 * D_jj = du_j/dx_j.
    div_stress = vgrad_11 + vgrad_22 + vgrad_33

    return shear, div_stress


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_shear_and_div_of_stress(
    u_vert: fa.VertexKField[wpfloat],
    v_vert: fa.VertexKField[wpfloat],
    w_vert: fa.VertexKField[wpfloat],
    w: fa.CellKField[wpfloat],
    vn_ie: fa.EdgeKField[wpfloat],
    vt_ie: fa.EdgeKField[wpfloat],
    w_ie: fa.EdgeKField[wpfloat],
    primal_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    inv_ddqz_z_full_e: fa.EdgeKField[wpfloat],
    shear: fa.EdgeKField[wpfloat],
    div_stress: fa.EdgeKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_shear_and_div_of_stress(
        u_vert=u_vert,
        v_vert=v_vert,
        w_vert=w_vert,
        w=w,
        vn_ie=vn_ie,
        vt_ie=vt_ie,
        w_ie=w_ie,
        primal_normal_vert_x=primal_normal_vert_x,
        primal_normal_vert_y=primal_normal_vert_y,
        dual_normal_vert_x=dual_normal_vert_x,
        dual_normal_vert_y=dual_normal_vert_y,
        tangent_orientation=tangent_orientation,
        inv_primal_edge_length=inv_primal_edge_length,
        inv_vert_vert_length=inv_vert_vert_length,
        inv_dual_edge_length=inv_dual_edge_length,
        inv_ddqz_z_full_e=inv_ddqz_z_full_e,
        out=(shear, div_stress),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
