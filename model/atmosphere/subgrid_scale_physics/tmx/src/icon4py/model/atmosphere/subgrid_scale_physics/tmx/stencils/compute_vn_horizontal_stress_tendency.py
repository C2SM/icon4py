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
def _compute_vn_horizontal_stress_tendency(
    u_vert: fa.VertexKField[wpfloat],
    v_vert: fa.VertexKField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
    km_c: fa.CellKField[wpfloat],
    div_c: fa.CellKField[wpfloat],
    km_iv: fa.VertexKField[wpfloat],
    inv_rhoe: fa.EdgeKField[wpfloat],
    primal_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
) -> fa.EdgeKField[wpfloat]:
    """
    Compute the horizontal divergence of the 3D stress tensor acting on vn.

    Port of the '1) First get the horizontal tendencies' loop of
    'Compute_diffusion_hor_wind' (mo_vdf.f90):

        flux_up_c = km_c(E2C[1]) * (4 * (vn_vert4 - vn) * inv_vert_vert_length
                                    - 2/3 * div_c(E2C[1]))
        flux_dn_c = km_c(E2C[0]) * (4 * (vn - vn_vert3) * inv_vert_vert_length
                                    - 2/3 * div_c(E2C[0]))
        flux_up_v = (km_iv(v2, k) + km_iv(v2, k+1))
                    * (tangent_orientation * (vn_vert2 - vn) * inv_primal_edge_length
                       + 0.5 * dvt * inv_vert_vert_length)
        flux_dn_v = (km_iv(v1, k) + km_iv(v1, k+1))
                    * (tangent_orientation * (vn - vn_vert1) * inv_primal_edge_length
                       + 0.5 * dvt * inv_vert_vert_length)
        tot_tend  = ((flux_up_c - flux_dn_c) * inv_dual_edge_length
                     + 2 * tangent_orientation * (flux_up_v - flux_dn_v)
                       * inv_primal_edge_length) * inv_rhoe

    with vn_vert1..4 the normal projections of (u_vert, v_vert) at the four
    E2C2V vertices (0, 1: edge endpoints v1/v2; 2, 3: far vertices of the
    adjacent cells) and dvt the tangential velocity difference between the two
    far vertices. km_iv is a half-level (nlev + 1) vertex field; all other
    K fields live on full levels.

    The vertical part of the vn diffusion is added to this tendency later by
    the edge-based tridiagonal solve ('solve_vertical_diffusion_edges').

    Domains (Fortran caller): jk = 1..nlev; edges from
    rl_start = grf_bdywidth_e + 1 -> 'h_grid.Zone.NUDGING_LEVEL_2' to
    rl_end = min_rledge_int -> 'h_grid.Zone.LOCAL'.
    """
    z_2by3 = wpfloat("2.0") / wpfloat("3.0")

    # Normal/tangential velocity components at the four E2C2V vertices.
    vn_vert = u_vert(E2C2V) * primal_normal_vert_x + v_vert(E2C2V) * primal_normal_vert_y
    vt_vert = u_vert(E2C2V) * dual_normal_vert_x + v_vert(E2C2V) * dual_normal_vert_y

    # Tangential velocity difference between the two far vertices.
    dvt = vt_vert[E2C2VDim(3)] - vt_vert[E2C2VDim(2)]

    # Tendency in normal direction: flux = visc * (D_11 - 2/3 DIV)
    #   = visc * (2 * delta_v / (vert_vert_len/2) - 2/3 * div_of_stress).
    flux_up_c = km_c(E2C[1]) * (
        wpfloat("4.0") * (vn_vert[E2C2VDim(3)] - vn) * inv_vert_vert_length - z_2by3 * div_c(E2C[1])
    )
    flux_dn_c = km_c(E2C[0]) * (
        wpfloat("4.0") * (vn - vn_vert[E2C2VDim(2)]) * inv_vert_vert_length - z_2by3 * div_c(E2C[0])
    )

    # Tendency in tangential direction: flux = D_12 * visc, D_12 between edge
    # center and vertex; km_iv(k) + km_iv(k+1) is (twice) the full-level
    # viscosity at the vertex.
    flux_up_v = (km_iv(E2C2V[1]) + km_iv(E2C2V[1])(KDim + 1)) * (
        tangent_orientation * (vn_vert[E2C2VDim(1)] - vn) * inv_primal_edge_length
        + wpfloat("0.5") * dvt * inv_vert_vert_length
    )
    flux_dn_v = (km_iv(E2C2V[0]) + km_iv(E2C2V[0])(KDim + 1)) * (
        tangent_orientation * (vn - vn_vert[E2C2VDim(0)]) * inv_primal_edge_length
        + wpfloat("0.5") * dvt * inv_vert_vert_length
    )

    return (
        (flux_up_c - flux_dn_c) * inv_dual_edge_length
        + wpfloat("2.0") * tangent_orientation * (flux_up_v - flux_dn_v) * inv_primal_edge_length
    ) * inv_rhoe


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_vn_horizontal_stress_tendency(
    u_vert: fa.VertexKField[wpfloat],
    v_vert: fa.VertexKField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
    km_c: fa.CellKField[wpfloat],
    div_c: fa.CellKField[wpfloat],
    km_iv: fa.VertexKField[wpfloat],
    inv_rhoe: fa.EdgeKField[wpfloat],
    primal_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    tot_tend: fa.EdgeKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_vn_horizontal_stress_tendency(
        u_vert=u_vert,
        v_vert=v_vert,
        vn=vn,
        km_c=km_c,
        div_c=div_c,
        km_iv=km_iv,
        inv_rhoe=inv_rhoe,
        primal_normal_vert_x=primal_normal_vert_x,
        primal_normal_vert_y=primal_normal_vert_y,
        dual_normal_vert_x=dual_normal_vert_x,
        dual_normal_vert_y=dual_normal_vert_y,
        tangent_orientation=tangent_orientation,
        inv_primal_edge_length=inv_primal_edge_length,
        inv_vert_vert_length=inv_vert_vert_length,
        inv_dual_edge_length=inv_dual_edge_length,
        out=tot_tend,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
