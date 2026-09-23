# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Stencils of the tmx wind diffusion (horizontal wind vn and vertical wind w)."""

import gt4py.next as gtx
from gt4py.next import broadcast, neighbor_sum
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.vertical_diffusion import (
    _assemble_vertical_diffusion_matrix_on_cell_half_levels,
    _assemble_vertical_diffusion_matrix_on_edges,
    _solve_implicit_vertical_diffusion_on_cell_half_levels,
    _solve_implicit_vertical_diffusion_on_edges,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2E, E2C, E2C2V, C2EDim, E2C2VDim, E2CDim
from icon4py.model.common.interpolation.stencils.compute_tangential_wind import (
    _compute_tangential_wind,
)
from icon4py.model.common.interpolation.stencils.edge_2_cell_vector_rbf_interpolation import (
    _edge_2_cell_vector_rbf_interpolation,
)
from icon4py.model.common.interpolation.stencils.interpolate_cell_field_to_edge import (
    _interpolate_cell_field_to_edge,
)
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_vn_horizontal_stress_tendency(
    u_vert: fa.VertexKField[wpfloat],
    v_vert: fa.VertexKField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
    km_c: fa.CellKField[wpfloat],
    div_c: fa.CellKField[wpfloat],
    km_iv: fa.VertexKHalfField[wpfloat],
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
    vn tendency of the horizontal divergence of the turbulent stress.

    The stress is evaluated at the faces of the edge's control volume: the two adjacent
    cell centers (E2C) and the two edge endpoints (E2C2V 0 and 1).
    """
    z_2by3 = wpfloat("2.0") / wpfloat("3.0")

    vn_vert = u_vert(E2C2V) * primal_normal_vert_x + v_vert(E2C2V) * primal_normal_vert_y
    vt_vert = u_vert(E2C2V) * dual_normal_vert_x + v_vert(E2C2V) * dual_normal_vert_y
    dvt = vt_vert[E2C2VDim(3)] - vt_vert[E2C2VDim(2)]

    flux_up_c = km_c(E2C[1]) * (
        wpfloat("4.0") * (vn_vert[E2C2VDim(3)] - vn) * inv_vert_vert_length - z_2by3 * div_c(E2C[1])
    )
    flux_dn_c = km_c(E2C[0]) * (
        wpfloat("4.0") * (vn - vn_vert[E2C2VDim(2)]) * inv_vert_vert_length - z_2by3 * div_c(E2C[0])
    )

    # the sum of the two half levels is twice the full-level viscosity at the vertex
    flux_up_v = (km_iv(E2C2V[1])(dims.KDim - 0.5) + km_iv(E2C2V[1])(dims.KDim + 0.5)) * (
        tangent_orientation * (vn_vert[E2C2VDim(1)] - vn) * inv_primal_edge_length
        + wpfloat("0.5") * dvt * inv_vert_vert_length
    )
    flux_dn_v = (km_iv(E2C2V[0])(dims.KDim - 0.5) + km_iv(E2C2V[0])(dims.KDim + 0.5)) * (
        tangent_orientation * (vn - vn_vert[E2C2VDim(0)]) * inv_primal_edge_length
        + wpfloat("0.5") * dvt * inv_vert_vert_length
    )

    return (
        (flux_up_c - flux_dn_c) * inv_dual_edge_length
        + wpfloat("2.0") * tangent_orientation * (flux_up_v - flux_dn_v) * inv_primal_edge_length
    ) * inv_rhoe


@gtx.field_operator
def _solve_vn_vertical_diffusion(
    w: fa.CellKHalfField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
    km_ie: fa.EdgeKHalfField[wpfloat],
    inv_rhoe: fa.EdgeKField[wpfloat],
    inv_ddqz_z_full_e: fa.EdgeKField[wpfloat],
    inv_ddqz_z_half_e: fa.EdgeKHalfField[wpfloat],
    u_stress: fa.CellField[wpfloat],
    v_stress: fa.CellField[wpfloat],
    primal_normal_cell_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    primal_normal_cell_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    tend: fa.EdgeKField[wpfloat],
    dtime: wpfloat,
    minlvl: gtx.int32,
    maxlvl: gtx.int32,
) -> fa.EdgeKField[wpfloat]:
    """
    tend plus the vn tendency of the implicit vertical diffusion over full levels minlvl..maxlvl.

    The right-hand side holds the vertical flux of the dw/dn stress, with no flux through the
    top and the surface momentum stress (u_stress, v_stress) through the bottom.
    """
    inv_air_mass = inv_ddqz_z_full_e * inv_rhoe

    dwdn_flux_above = (
        km_ie(dims.KDim - 0.5)
        * inv_dual_edge_length
        * (w(E2C[1])(dims.KDim - 0.5) - w(E2C[0])(dims.KDim - 0.5))
    )
    dwdn_flux_below = (
        km_ie(dims.KDim + 0.5)
        * inv_dual_edge_length
        * (w(E2C[1])(dims.KDim + 0.5) - w(E2C[0])(dims.KDim + 0.5))
    )
    surface_stress = neighbor_sum(
        (u_stress(E2C) * primal_normal_cell_x + v_stress(E2C) * primal_normal_cell_y) * c_lin_e,
        axis=E2CDim,
    )
    rhs = concat_where(
        dims.KDim > minlvl,
        (dwdn_flux_above - dwdn_flux_below) * inv_air_mass,
        (wpfloat("0.0") - dwdn_flux_below) * inv_air_mass,
    )
    rhs = concat_where(
        dims.KDim < maxlvl,
        rhs,
        dwdn_flux_above * inv_air_mass - surface_stress * inv_air_mass,
    )

    a, b, c = _assemble_vertical_diffusion_matrix_on_edges(
        km_ie, inv_ddqz_z_half_e, inv_air_mass, wpfloat("1.0"), minlvl, maxlvl
    )
    return _solve_implicit_vertical_diffusion_on_edges(vn, a, b, c, rhs, tend, dtime)


@gtx.field_operator
def _compute_vn_diffusion_tendency(
    rho: fa.CellKField[wpfloat],
    w: fa.CellKHalfField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
    u_vert: fa.VertexKField[wpfloat],
    v_vert: fa.VertexKField[wpfloat],
    km_c: fa.CellKField[wpfloat],
    div_c: fa.CellKField[wpfloat],
    km_iv: fa.VertexKHalfField[wpfloat],
    km_ie: fa.EdgeKHalfField[wpfloat],
    u_stress: fa.CellField[wpfloat],
    v_stress: fa.CellField[wpfloat],
    inv_ddqz_z_full_e: fa.EdgeKField[wpfloat],
    inv_ddqz_z_half_e: fa.EdgeKHalfField[wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    primal_normal_cell_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    primal_normal_cell_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    primal_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    dtime: wpfloat,
    minlvl: gtx.int32,
    maxlvl: gtx.int32,
) -> fa.EdgeKField[wpfloat]:
    inv_rhoe = wpfloat("1.0") / _interpolate_cell_field_to_edge(rho, c_lin_e)
    horizontal_tendency = _compute_vn_horizontal_stress_tendency(
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
    )
    return _solve_vn_vertical_diffusion(
        w=w,
        vn=vn,
        km_ie=km_ie,
        inv_rhoe=inv_rhoe,
        inv_ddqz_z_full_e=inv_ddqz_z_full_e,
        inv_ddqz_z_half_e=inv_ddqz_z_half_e,
        u_stress=u_stress,
        v_stress=v_stress,
        primal_normal_cell_x=primal_normal_cell_x,
        primal_normal_cell_y=primal_normal_cell_y,
        c_lin_e=c_lin_e,
        inv_dual_edge_length=inv_dual_edge_length,
        tend=horizontal_tendency,
        dtime=dtime,
        minlvl=minlvl,
        maxlvl=maxlvl,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_vn_diffusion_tendency(
    rho: fa.CellKField[wpfloat],
    w: fa.CellKHalfField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
    u_vert: fa.VertexKField[wpfloat],
    v_vert: fa.VertexKField[wpfloat],
    km_c: fa.CellKField[wpfloat],
    div_c: fa.CellKField[wpfloat],
    km_iv: fa.VertexKHalfField[wpfloat],
    km_ie: fa.EdgeKHalfField[wpfloat],
    u_stress: fa.CellField[wpfloat],
    v_stress: fa.CellField[wpfloat],
    inv_ddqz_z_full_e: fa.EdgeKField[wpfloat],
    inv_ddqz_z_half_e: fa.EdgeKHalfField[wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    primal_normal_cell_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    primal_normal_cell_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    primal_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    vn_tendency: fa.EdgeKField[wpfloat],
    dtime: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_vn_diffusion_tendency(
        rho=rho,
        w=w,
        vn=vn,
        u_vert=u_vert,
        v_vert=v_vert,
        km_c=km_c,
        div_c=div_c,
        km_iv=km_iv,
        km_ie=km_ie,
        u_stress=u_stress,
        v_stress=v_stress,
        inv_ddqz_z_full_e=inv_ddqz_z_full_e,
        inv_ddqz_z_half_e=inv_ddqz_z_half_e,
        c_lin_e=c_lin_e,
        primal_normal_cell_x=primal_normal_cell_x,
        primal_normal_cell_y=primal_normal_cell_y,
        primal_normal_vert_x=primal_normal_vert_x,
        primal_normal_vert_y=primal_normal_vert_y,
        dual_normal_vert_x=dual_normal_vert_x,
        dual_normal_vert_y=dual_normal_vert_y,
        tangent_orientation=tangent_orientation,
        inv_primal_edge_length=inv_primal_edge_length,
        inv_vert_vert_length=inv_vert_vert_length,
        inv_dual_edge_length=inv_dual_edge_length,
        dtime=dtime,
        minlvl=vertical_start,
        maxlvl=vertical_end - 1,
        out=vn_tendency,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _interpolate_and_update_horizontal_wind(
    vn_tendency: fa.EdgeKField[wpfloat],
    u: fa.CellKField[wpfloat],
    v: fa.CellKField[wpfloat],
    rbf_coeff_c1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2EDim], wpfloat],
    rbf_coeff_c2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2EDim], wpfloat],
    dtime: wpfloat,
) -> tuple[
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
]:
    """The (u, v) tendencies RBF-reconstructed from vn_tendency, and the updated u, v."""
    tend_u, tend_v = _edge_2_cell_vector_rbf_interpolation(vn_tendency, rbf_coeff_c1, rbf_coeff_c2)
    return tend_u, tend_v, u + tend_u * dtime, v + tend_v * dtime


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def interpolate_and_update_horizontal_wind(
    vn_tendency: fa.EdgeKField[wpfloat],
    u: fa.CellKField[wpfloat],
    v: fa.CellKField[wpfloat],
    rbf_coeff_c1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2EDim], wpfloat],
    rbf_coeff_c2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2EDim], wpfloat],
    tend_u: fa.CellKField[wpfloat],
    tend_v: fa.CellKField[wpfloat],
    new_u: fa.CellKField[wpfloat],
    new_v: fa.CellKField[wpfloat],
    dtime: wpfloat,
    tendency_horizontal_start: gtx.int32,
    update_horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _interpolate_and_update_horizontal_wind(
        vn_tendency=vn_tendency,
        u=u,
        v=v,
        rbf_coeff_c1=rbf_coeff_c1,
        rbf_coeff_c2=rbf_coeff_c2,
        dtime=dtime,
        out=(tend_u, tend_v, new_u, new_v),
        domain=(
            {
                dims.CellDim: (tendency_horizontal_start, horizontal_end),
                dims.KDim: (vertical_start, vertical_end),
            },
            {
                dims.CellDim: (tendency_horizontal_start, horizontal_end),
                dims.KDim: (vertical_start, vertical_end),
            },
            {
                dims.CellDim: (update_horizontal_start, horizontal_end),
                dims.KDim: (vertical_start, vertical_end),
            },
            {
                dims.CellDim: (update_horizontal_start, horizontal_end),
                dims.KDim: (vertical_start, vertical_end),
            },
        ),
    )


@gtx.field_operator
def _compute_w_horizontal_stress_tendency(
    u: fa.CellKField[wpfloat],
    v: fa.CellKField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
    u_vert: fa.VertexKField[wpfloat],
    v_vert: fa.VertexKField[wpfloat],
    w_vert: fa.VertexKHalfField[wpfloat],
    w_ie: fa.EdgeKHalfField[wpfloat],
    km_ic: fa.CellKHalfField[wpfloat],
    km_iv: fa.VertexKHalfField[wpfloat],
    inv_ddqz_z_half: fa.CellKHalfField[wpfloat],
    inv_ddqz_z_half_v: fa.VertexKHalfField[wpfloat],
    rbf_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], wpfloat],
    primal_normal_cell_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    primal_normal_cell_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    dual_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    edge_cell_length: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
) -> fa.EdgeKHalfField[wpfloat]:
    """
    w tendency (times density) of the horizontal divergence of the turbulent stress, at half-level
    edges.

    The stress is evaluated at the faces of the edge's control volume: the two adjacent cell
    centers (E2C) and the two edge endpoints (E2C2V 0 and 1). Valid on interior half levels only.
    """
    vt_e = _compute_tangential_wind(vn, rbf_coeff_e)

    dvn_up = (
        u(E2C[1])(dims.KHalfDim - 0.5) - u(E2C[1])(dims.KHalfDim + 0.5)
    ) * primal_normal_cell_x[E2CDim(1)] + (
        v(E2C[1])(dims.KHalfDim - 0.5) - v(E2C[1])(dims.KHalfDim + 0.5)
    ) * primal_normal_cell_y[E2CDim(1)]
    flux_up_c = km_ic(E2C[1]) * (
        dvn_up * inv_ddqz_z_half(E2C[1])
        + (w_vert(E2C2V[3]) - w_ie) * wpfloat("2.0") * inv_vert_vert_length
    )
    dvn_dn = (
        u(E2C[0])(dims.KHalfDim - 0.5) - u(E2C[0])(dims.KHalfDim + 0.5)
    ) * primal_normal_cell_x[E2CDim(0)] + (
        v(E2C[0])(dims.KHalfDim - 0.5) - v(E2C[0])(dims.KHalfDim + 0.5)
    ) * primal_normal_cell_y[E2CDim(0)]
    flux_dn_c = km_ic(E2C[0]) * (
        dvn_dn * inv_ddqz_z_half(E2C[0])
        + (w_ie - w_vert(E2C2V[2])) * wpfloat("2.0") * inv_vert_vert_length
    )

    # the tangential wind between vertex and edge center is the mean of the two
    vt_up_above = (
        u_vert(E2C2V[1])(dims.KHalfDim - 0.5) * dual_normal_vert_x[E2C2VDim(1)]
        + v_vert(E2C2V[1])(dims.KHalfDim - 0.5) * dual_normal_vert_y[E2C2VDim(1)]
        + vt_e(dims.KHalfDim - 0.5)
    )
    vt_up_below = (
        u_vert(E2C2V[1])(dims.KHalfDim + 0.5) * dual_normal_vert_x[E2C2VDim(1)]
        + v_vert(E2C2V[1])(dims.KHalfDim + 0.5) * dual_normal_vert_y[E2C2VDim(1)]
        + vt_e(dims.KHalfDim + 0.5)
    )
    dvt_up = wpfloat("0.5") * vt_up_above - wpfloat("0.5") * vt_up_below
    flux_up_v = km_iv(E2C2V[1]) * (
        dvt_up * inv_ddqz_z_half_v(E2C2V[1])
        + tangent_orientation * (w_vert(E2C2V[1]) - w_ie) / edge_cell_length[E2CDim(1)]
    )
    vt_dn_above = (
        u_vert(E2C2V[0])(dims.KHalfDim - 0.5) * dual_normal_vert_x[E2C2VDim(0)]
        + v_vert(E2C2V[0])(dims.KHalfDim - 0.5) * dual_normal_vert_y[E2C2VDim(0)]
        + vt_e(dims.KHalfDim - 0.5)
    )
    vt_dn_below = (
        u_vert(E2C2V[0])(dims.KHalfDim + 0.5) * dual_normal_vert_x[E2C2VDim(0)]
        + v_vert(E2C2V[0])(dims.KHalfDim + 0.5) * dual_normal_vert_y[E2C2VDim(0)]
        + vt_e(dims.KHalfDim + 0.5)
    )
    dvt_dn = wpfloat("0.5") * vt_dn_above - wpfloat("0.5") * vt_dn_below
    flux_dn_v = km_iv(E2C2V[0]) * (
        dvt_dn * inv_ddqz_z_half_v(E2C2V[0])
        + tangent_orientation * (w_ie - w_vert(E2C2V[0])) / edge_cell_length[E2CDim(0)]
    )

    return (flux_up_c - flux_dn_c) * inv_dual_edge_length + (
        flux_up_v - flux_dn_v
    ) * tangent_orientation * wpfloat("2.0") * inv_primal_edge_length


@gtx.field_operator
def _solve_w_vertical_diffusion(
    w: fa.CellKHalfField[wpfloat],
    inv_rho_ic: fa.CellKHalfField[wpfloat],
    inv_ddqz_z_half: fa.CellKHalfField[wpfloat],
    inv_ddqz_z_full: fa.CellKField[wpfloat],
    km_c: fa.CellKField[wpfloat],
    div_c: fa.CellKField[wpfloat],
    tend: fa.CellKHalfField[wpfloat],
    dtime: wpfloat,
    minlvl: gtx.int32,
    maxlvl: gtx.int32,
) -> fa.CellKHalfField[wpfloat]:
    """
    tend plus the w tendency of the implicit vertical diffusion over half levels minlvl..maxlvl.

    w = 0 is imposed on the half levels minlvl - 1 and maxlvl + 1 bounding the system.
    """
    z_1by3 = wpfloat("1.0") / wpfloat("3.0")
    inv_air_mass = inv_rho_ic * inv_ddqz_z_half
    rhs = (
        wpfloat("2.0")
        * inv_air_mass
        * (
            km_c(dims.KHalfDim + 0.5) * z_1by3 * div_c(dims.KHalfDim + 0.5)
            - km_c(dims.KHalfDim - 0.5) * z_1by3 * div_c(dims.KHalfDim - 0.5)
        )
    )
    a, b, c = _assemble_vertical_diffusion_matrix_on_cell_half_levels(
        km_c, inv_ddqz_z_full, inv_air_mass, wpfloat("2.0"), minlvl, maxlvl
    )
    b = concat_where(
        dims.KHalfDim > minlvl,
        b,
        b
        + wpfloat("2.0")
        * km_c(dims.KHalfDim - 0.5)
        * inv_ddqz_z_full(dims.KHalfDim - 0.5)
        * inv_air_mass,
    )
    b = concat_where(
        dims.KHalfDim < maxlvl,
        b,
        b
        + wpfloat("2.0")
        * km_c(dims.KHalfDim + 0.5)
        * inv_ddqz_z_full(dims.KHalfDim + 0.5)
        * inv_air_mass,
    )
    return _solve_implicit_vertical_diffusion_on_cell_half_levels(w, a, b, c, rhs, tend, dtime)


@gtx.field_operator
def _compute_w_diffusion_tendency_and_update(
    w: fa.CellKHalfField[wpfloat],
    rho_ic: fa.CellKHalfField[wpfloat],
    km_c: fa.CellKField[wpfloat],
    div_c: fa.CellKField[wpfloat],
    horizontal_stress_tendency: fa.EdgeKHalfField[wpfloat],
    inv_ddqz_z_half: fa.CellKHalfField[wpfloat],
    inv_ddqz_z_full: fa.CellKField[wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    dtime: wpfloat,
    minlvl: gtx.int32,
    maxlvl: gtx.int32,
) -> tuple[fa.CellKHalfField[wpfloat], fa.CellKHalfField[wpfloat]]:
    """The w tendency of the vertical and horizontal diffusion, and the updated w."""
    inv_rho_ic = wpfloat("1.0") / rho_ic
    horizontal_tendency = inv_rho_ic * neighbor_sum(
        horizontal_stress_tendency(C2E) * e_bln_c_s, axis=C2EDim
    )
    tend_w = _solve_w_vertical_diffusion(
        w=w,
        inv_rho_ic=inv_rho_ic,
        inv_ddqz_z_half=inv_ddqz_z_half,
        inv_ddqz_z_full=inv_ddqz_z_full,
        km_c=km_c,
        div_c=div_c,
        tend=horizontal_tendency,
        dtime=dtime,
        minlvl=minlvl,
        maxlvl=maxlvl,
    )
    return tend_w, w + tend_w * dtime


@gtx.field_operator
def _zero_on_cell_half_levels() -> fa.CellKHalfField[wpfloat]:
    return broadcast(wpfloat("0.0"), (dims.CellDim, dims.KHalfDim))


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_w_diffusion_tendency_and_update(
    w: fa.CellKHalfField[wpfloat],
    u: fa.CellKField[wpfloat],
    v: fa.CellKField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
    rho_ic: fa.CellKHalfField[wpfloat],
    u_vert: fa.VertexKField[wpfloat],
    v_vert: fa.VertexKField[wpfloat],
    w_vert: fa.VertexKHalfField[wpfloat],
    w_ie: fa.EdgeKHalfField[wpfloat],
    km_c: fa.CellKField[wpfloat],
    km_ic: fa.CellKHalfField[wpfloat],
    km_iv: fa.VertexKHalfField[wpfloat],
    div_c: fa.CellKField[wpfloat],
    inv_ddqz_z_full: fa.CellKField[wpfloat],
    inv_ddqz_z_half: fa.CellKHalfField[wpfloat],
    inv_ddqz_z_half_v: fa.VertexKHalfField[wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    rbf_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], wpfloat],
    primal_normal_cell_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    primal_normal_cell_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    dual_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    edge_cell_length: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    horizontal_stress_tendency: fa.EdgeKHalfField[wpfloat],
    tend_w: fa.CellKHalfField[wpfloat],
    new_w: fa.CellKHalfField[wpfloat],
    dtime: wpfloat,
    edge_start: gtx.int32,
    edge_end: gtx.int32,
    cell_start: gtx.int32,
    cell_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """
    Diffuse w on half levels vertical_start..vertical_end - 1 of the cells; new_w is zero on the
    half levels vertical_start - 1 and vertical_end bounding them.
    """
    _compute_w_horizontal_stress_tendency(
        u=u,
        v=v,
        vn=vn,
        u_vert=u_vert,
        v_vert=v_vert,
        w_vert=w_vert,
        w_ie=w_ie,
        km_ic=km_ic,
        km_iv=km_iv,
        inv_ddqz_z_half=inv_ddqz_z_half,
        inv_ddqz_z_half_v=inv_ddqz_z_half_v,
        rbf_coeff_e=rbf_coeff_e,
        primal_normal_cell_x=primal_normal_cell_x,
        primal_normal_cell_y=primal_normal_cell_y,
        dual_normal_vert_x=dual_normal_vert_x,
        dual_normal_vert_y=dual_normal_vert_y,
        edge_cell_length=edge_cell_length,
        tangent_orientation=tangent_orientation,
        inv_primal_edge_length=inv_primal_edge_length,
        inv_vert_vert_length=inv_vert_vert_length,
        inv_dual_edge_length=inv_dual_edge_length,
        out=horizontal_stress_tendency,
        domain={
            dims.EdgeDim: (edge_start, edge_end),
            dims.KHalfDim: (vertical_start, vertical_end),
        },
    )
    _compute_w_diffusion_tendency_and_update(
        w=w,
        rho_ic=rho_ic,
        km_c=km_c,
        div_c=div_c,
        horizontal_stress_tendency=horizontal_stress_tendency,
        inv_ddqz_z_half=inv_ddqz_z_half,
        inv_ddqz_z_full=inv_ddqz_z_full,
        e_bln_c_s=e_bln_c_s,
        dtime=dtime,
        minlvl=vertical_start,
        maxlvl=vertical_end - 1,
        out=(tend_w, new_w),
        domain={
            dims.CellDim: (cell_start, cell_end),
            dims.KHalfDim: (vertical_start, vertical_end),
        },
    )
    _zero_on_cell_half_levels(
        out=new_w,
        domain={
            dims.CellDim: (cell_start, cell_end),
            dims.KHalfDim: (vertical_start - 1, vertical_start),
        },
    )
    _zero_on_cell_half_levels(
        out=new_w,
        domain={
            dims.CellDim: (cell_start, cell_end),
            dims.KHalfDim: (vertical_end, vertical_end + 1),
        },
    )
