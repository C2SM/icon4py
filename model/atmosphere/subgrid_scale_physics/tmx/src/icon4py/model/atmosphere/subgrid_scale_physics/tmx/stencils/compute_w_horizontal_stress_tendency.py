# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C, E2C2V, E2C2VDim, E2CDim, KDim
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_w_horizontal_stress_tendency(
    u: fa.CellKField[wpfloat],
    v: fa.CellKField[wpfloat],
    km_ic: fa.CellKField[wpfloat],
    inv_ddqz_z_half: fa.CellKField[wpfloat],
    u_vert: fa.VertexKField[wpfloat],
    v_vert: fa.VertexKField[wpfloat],
    w_vert: fa.VertexKField[wpfloat],
    km_iv: fa.VertexKField[wpfloat],
    inv_ddqz_z_half_v: fa.VertexKField[wpfloat],
    w_ie: fa.EdgeKField[wpfloat],
    vt_e: fa.EdgeKField[wpfloat],
    primal_normal_cell_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    primal_normal_cell_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    dual_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    edge_cell_length: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
) -> fa.EdgeKField[wpfloat]:
    """
    Compute the horizontal D31/D32 stress tendency of w at half-level edges.

    Port of the '1) Get horizontal tendencies at half level edges' loop of
    'Compute_diffusion_vert_wind' (mo_vdf.f90). Normal direction (flux =
    visc_c * D_31 at the half-level cell centers, E2C neighbors 0/1 with the
    far E2C2V vertices 2/3):

        dvn_i = (u(c_i, k-1) - u(c_i, k)) * primal_normal_cell_x(i)
                + (v(c_i, k-1) - v(c_i, k)) * primal_normal_cell_y(i)
        flux_up_c = km_ic(c2, k) * (dvn2 * inv_dzh(c2, k)
                    + (w_vert(v4, k) - w_ie(k)) * 2 * inv_vert_vert_length)
        flux_dn_c = km_ic(c1, k) * (dvn1 * inv_dzh(c1, k)
                    + (w_ie(k) - w_vert(v3, k)) * 2 * inv_vert_vert_length)

    Tangential direction (flux = visc_v * D_32 between the edge-endpoint
    vertices v1/v2 = E2C2V neighbors 0/1 and the edge center):

        dvt_i = 0.5 * (u_vert(v_i, k-1) * dual_normal_vert_x(i)
                       + v_vert(v_i, k-1) * dual_normal_vert_y(i) + vt_e(k-1))
                - 0.5 * (u_vert(v_i, k) * dual_normal_vert_x(i)
                         + v_vert(v_i, k) * dual_normal_vert_y(i) + vt_e(k))
        flux_up_v = km_iv(v2, k) * (dvt2 * inv_ddqz_z_half_v(v2, k)
                    + tangent_orientation * (w_vert(v2, k) - w_ie(k))
                      / edge_cell_length(1))
        flux_dn_v = km_iv(v1, k) * (dvt1 * inv_ddqz_z_half_v(v1, k)
                    + tangent_orientation * (w_ie(k) - w_vert(v1, k))
                      / edge_cell_length(0))

        hori_tend_e = (flux_up_c - flux_dn_c) * inv_dual_edge_length
                      + (flux_up_v - flux_dn_v) * tangent_orientation
                        * 2 * inv_primal_edge_length

    km_ic, inv_ddqz_z_half ('inv_dzh'), w_vert, km_iv, inv_ddqz_z_half_v and
    w_ie are half-level fields (nlev + 1 rows); u, v, u_vert, v_vert and vt_e
    are full-level fields read at the full levels above (k-1) and at (k) the
    half level. The output lives on half levels; rows jk = 2..nlev (1-based),
    i.e. the program must be called with vertical bounds (1, nlev).

    Domains (Fortran caller): edges from rl_start = grf_bdywidth_e ->
    'h_grid.Zone.NUDGING' to rl_end = min_rledge_int - 1 -> 'h_grid.Zone.HALO'.
    """
    # Normal direction: D_31 at the half-level centers of the two E2C cells.
    dvn2 = (u(E2C[1])(KDim - 1) - u(E2C[1])) * primal_normal_cell_x[E2CDim(1)] + (
        v(E2C[1])(KDim - 1) - v(E2C[1])
    ) * primal_normal_cell_y[E2CDim(1)]
    flux_up_c = km_ic(E2C[1]) * (
        dvn2 * inv_ddqz_z_half(E2C[1])
        + (w_vert(E2C2V[3]) - w_ie) * wpfloat("2.0") * inv_vert_vert_length
    )

    dvn1 = (u(E2C[0])(KDim - 1) - u(E2C[0])) * primal_normal_cell_x[E2CDim(0)] + (
        v(E2C[0])(KDim - 1) - v(E2C[0])
    ) * primal_normal_cell_y[E2CDim(0)]
    flux_dn_c = km_ic(E2C[0]) * (
        dvn1 * inv_ddqz_z_half(E2C[0])
        + (w_ie - w_vert(E2C2V[2])) * wpfloat("2.0") * inv_vert_vert_length
    )

    # Tangential direction: D_32 between the edge-endpoint vertices and the
    # edge center. The tangential velocity at the half level is the mean of
    # the vertex projection and vt_e over the two adjacent full levels.
    dvt2 = wpfloat("0.5") * (
        u_vert(E2C2V[1])(KDim - 1) * dual_normal_vert_x[E2C2VDim(1)]
        + v_vert(E2C2V[1])(KDim - 1) * dual_normal_vert_y[E2C2VDim(1)]
        + vt_e(KDim - 1)
    ) - wpfloat("0.5") * (
        u_vert(E2C2V[1]) * dual_normal_vert_x[E2C2VDim(1)]
        + v_vert(E2C2V[1]) * dual_normal_vert_y[E2C2VDim(1)]
        + vt_e
    )
    flux_up_v = km_iv(E2C2V[1]) * (
        dvt2 * inv_ddqz_z_half_v(E2C2V[1])
        + tangent_orientation * (w_vert(E2C2V[1]) - w_ie) / edge_cell_length[E2CDim(1)]
    )

    dvt1 = wpfloat("0.5") * (
        u_vert(E2C2V[0])(KDim - 1) * dual_normal_vert_x[E2C2VDim(0)]
        + v_vert(E2C2V[0])(KDim - 1) * dual_normal_vert_y[E2C2VDim(0)]
        + vt_e(KDim - 1)
    ) - wpfloat("0.5") * (
        u_vert(E2C2V[0]) * dual_normal_vert_x[E2C2VDim(0)]
        + v_vert(E2C2V[0]) * dual_normal_vert_y[E2C2VDim(0)]
        + vt_e
    )
    flux_dn_v = km_iv(E2C2V[0]) * (
        dvt1 * inv_ddqz_z_half_v(E2C2V[0])
        + tangent_orientation * (w_ie - w_vert(E2C2V[0])) / edge_cell_length[E2CDim(0)]
    )

    return (flux_up_c - flux_dn_c) * inv_dual_edge_length + (
        flux_up_v - flux_dn_v
    ) * tangent_orientation * wpfloat("2.0") * inv_primal_edge_length


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_w_horizontal_stress_tendency(
    u: fa.CellKField[wpfloat],
    v: fa.CellKField[wpfloat],
    km_ic: fa.CellKField[wpfloat],
    inv_ddqz_z_half: fa.CellKField[wpfloat],
    u_vert: fa.VertexKField[wpfloat],
    v_vert: fa.VertexKField[wpfloat],
    w_vert: fa.VertexKField[wpfloat],
    km_iv: fa.VertexKField[wpfloat],
    inv_ddqz_z_half_v: fa.VertexKField[wpfloat],
    w_ie: fa.EdgeKField[wpfloat],
    vt_e: fa.EdgeKField[wpfloat],
    primal_normal_cell_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    primal_normal_cell_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    dual_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    edge_cell_length: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    hori_tend_e: fa.EdgeKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_w_horizontal_stress_tendency(
        u=u,
        v=v,
        km_ic=km_ic,
        inv_ddqz_z_half=inv_ddqz_z_half,
        u_vert=u_vert,
        v_vert=v_vert,
        w_vert=w_vert,
        km_iv=km_iv,
        inv_ddqz_z_half_v=inv_ddqz_z_half_v,
        w_ie=w_ie,
        vt_e=vt_e,
        primal_normal_cell_x=primal_normal_cell_x,
        primal_normal_cell_y=primal_normal_cell_y,
        dual_normal_vert_x=dual_normal_vert_x,
        dual_normal_vert_y=dual_normal_vert_y,
        edge_cell_length=edge_cell_length,
        tangent_orientation=tangent_orientation,
        inv_primal_edge_length=inv_primal_edge_length,
        inv_vert_vert_length=inv_vert_vert_length,
        inv_dual_edge_length=inv_dual_edge_length,
        out=hori_tend_e,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
