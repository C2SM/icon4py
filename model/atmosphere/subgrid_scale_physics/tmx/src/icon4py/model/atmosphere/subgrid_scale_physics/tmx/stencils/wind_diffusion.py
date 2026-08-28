# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Stencils of the tmx momentum diffusion.

Ports ``Compute_diffusion_hor_wind`` (mo_vdf.f90 l. 1207, run by
:meth:`Tmx.run_horizontal_wind_diffusion`) and ``Compute_diffusion_vert_wind``
(mo_vdf.f90 l. 1601, run by :meth:`Tmx.run_vertical_wind_diffusion`).

Both solves fuse their right-hand side, their tridiagonal matrix rows and the
solve itself into one program, so the Fortran scratch arrays (``rhs``,
``inv_maire``, ``a``, ``b``, ``c``) never leave the kernel.

Bottom rows are selected with ``concat_where(dims.KDim < nlev - 1, ...)`` /
``concat_where(dims.KDim > minlvl, ...)`` rather than an equality test because
``concat_where(dims.KDim == nlev - 1, ...)`` is broken in GT4Py
(GridTools/gt4py#2205); see :mod:`vertical_diffusion` for the bounded-zero
branch idiom the boundary rows rely on.
"""

import gt4py.next as gtx
from gt4py.next import neighbor_sum
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.vertical_diffusion import (
    _prepare_tridiagonal_matrix_cells_half,
    _prepare_tridiagonal_matrix_edges,
    _solve_vertical_diffusion_cells,
    _solve_vertical_diffusion_edges,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2E, E2C, E2C2V, C2EDim, E2C2VDim, E2CDim, KDim
from icon4py.model.common.interpolation.stencils.cell_2_edge_interpolation import (
    _cell_2_edge_interpolation,
)
from icon4py.model.common.interpolation.stencils.compute_tangential_wind import (
    _compute_tangential_wind_wp,
)
from icon4py.model.common.interpolation.stencils.edge_2_cell_vector_rbf_interpolation import (
    _edge_2_cell_vector_rbf_interpolation,
)
from icon4py.model.common.math.operators import (
    _broadcast_value_on_cell_k,
    _compute_reciprocal_on_edge_k,
)
from icon4py.model.common.math.stencils.update_two_cell_kdim_fields_with_tendency import (
    _update_two_cell_kdim_fields_with_tendency,
)
from icon4py.model.common.type_alias import wpfloat


# ---------------------------------------------------------------------------
# Compute_diffusion_hor_wind (vn diffusion)
# ---------------------------------------------------------------------------
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
    the edge-based tridiagonal solve.

    Domains (Fortran caller): jk = 1..nlev; edges from
    ``rl_start = grf_bdywidth_e + 1`` -> ``h_grid.Zone.NUDGING_LEVEL_2`` to
    ``rl_end = min_rledge_int`` -> ``h_grid.Zone.LOCAL``.
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


@gtx.field_operator
def _compute_inverse_density_and_vn_stress_tendency(
    rho: fa.CellKField[wpfloat],
    u_vert: fa.VertexKField[wpfloat],
    v_vert: fa.VertexKField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
    km_c: fa.CellKField[wpfloat],
    div_c: fa.CellKField[wpfloat],
    km_iv: fa.VertexKField[wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    primal_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
) -> tuple[fa.EdgeKField[wpfloat], fa.EdgeKField[wpfloat]]:
    """
    Inverse edge density and horizontal vn stress tendency.

    Fuses the 'density at edge' block of 'Compute_diffusion_hor_wind'
    (``cells2edges_scalar`` with the linear E2C weights ``c_lin_e`` followed by
    the in-place reciprocal loop) with the horizontal stress tendency that
    scales with it. ``inv_rhoe`` is also needed by the vertical solve and is
    therefore returned.
    """
    inv_rhoe = _compute_reciprocal_on_edge_k(input_field=_cell_2_edge_interpolation(rho, c_lin_e))
    tot_tend = _compute_vn_horizontal_stress_tendency(
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
    return inv_rhoe, tot_tend


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_inverse_density_and_vn_stress_tendency(
    rho: fa.CellKField[wpfloat],
    u_vert: fa.VertexKField[wpfloat],
    v_vert: fa.VertexKField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
    km_c: fa.CellKField[wpfloat],
    div_c: fa.CellKField[wpfloat],
    km_iv: fa.VertexKField[wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    primal_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    inv_rhoe: fa.EdgeKField[wpfloat],
    tot_tend: fa.EdgeKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_inverse_density_and_vn_stress_tendency(
        rho=rho,
        u_vert=u_vert,
        v_vert=v_vert,
        vn=vn,
        km_c=km_c,
        div_c=div_c,
        km_iv=km_iv,
        c_lin_e=c_lin_e,
        primal_normal_vert_x=primal_normal_vert_x,
        primal_normal_vert_y=primal_normal_vert_y,
        dual_normal_vert_x=dual_normal_vert_x,
        dual_normal_vert_y=dual_normal_vert_y,
        tangent_orientation=tangent_orientation,
        inv_primal_edge_length=inv_primal_edge_length,
        inv_vert_vert_length=inv_vert_vert_length,
        inv_dual_edge_length=inv_dual_edge_length,
        out=(inv_rhoe, tot_tend),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _compute_vn_vertical_diffusion_rhs(
    w: fa.CellKField[wpfloat],
    km_ie: fa.EdgeKField[wpfloat],
    inv_rhoe: fa.EdgeKField[wpfloat],
    inv_ddqz_z_full_e: fa.EdgeKField[wpfloat],
    u_stress: fa.CellField[wpfloat],
    v_stress: fa.CellField[wpfloat],
    primal_normal_cell_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    primal_normal_cell_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    nlev: gtx.int32,
) -> tuple[fa.EdgeKField[wpfloat], fa.EdgeKField[wpfloat]]:
    """
    Compute the right-hand side and inverse layer air mass of the vn tridiagonal solve.

    Port of the '2) Vertical tendency' loops of 'Compute_diffusion_hor_wind'
    (mo_vdf.f90). With grad(k) = km_ie(k) * inv_dual_edge_length
    * (w(E2C[1], k) - w(E2C[0], k)) (all half-level rows):

        inv_maire(k) = inv_ddqz_z_full_e(k) * inv_rhoe(k)      for all k
        rhs(k) = (grad(k) - grad(k+1)) * inv_maire(k)          interior rows
        rhs(0) = -grad(1) * inv_maire(0)                       top row (jk = 1)
        rhs(nlev-1) = grad(nlev-1) * inv_maire(nlev-1)
                      - flux_dn_e * inv_maire(nlev-1)          bottom row (jk = nlev)

    where the bottom row replaces the surface flux grad(nlev) by the net surface
    shear stress projected on the edge normal:

        flux_dn_e = sum_{c in E2C} c_lin_e
                    * (u_stress(c) * primal_normal_cell_x
                       + v_stress(c) * primal_normal_cell_y)

    w (the ICON 'w_wind_ic' input, 'pwp1') and km_ie are half-level fields
    (nlev + 1 rows); rhs and inv_maire live on full levels. Note that the
    Fortran top row is the interior formula with the (zero-flux) term
    grad(1) = grad at the model top omitted, and the bottom row is the interior
    formula with grad(nlev+1) replaced by the surface stress.

    Domains (Fortran caller): jk = 1..nlev; edges from
    ``rl_start = grf_bdywidth_e + 1`` -> ``h_grid.Zone.NUDGING_LEVEL_2`` to
    ``rl_end = min_rledge_int`` -> ``h_grid.Zone.LOCAL``.

    Returns:
        rhs: right-hand side of the vn diffusion solve (edges, full levels)
        inv_maire: inverse air mass per unit area of the edge layer,
            1 / (rho_e * dz_e) (edges, full levels)
    """
    inv_maire = inv_ddqz_z_full_e * inv_rhoe

    # Vertical flux of the horizontal (dw/dn) stress at half levels k and k+1.
    grad_k = km_ie * inv_dual_edge_length * (w(E2C[1]) - w(E2C[0]))
    grad_kp1 = km_ie(KDim + 1) * inv_dual_edge_length * (w(E2C[1])(KDim + 1) - w(E2C[0])(KDim + 1))

    # Net surface shear stress in the direction of vn at the edge.
    flux_dn_e = neighbor_sum(
        (u_stress(E2C) * primal_normal_cell_x + v_stress(E2C) * primal_normal_cell_y) * c_lin_e,
        axis=E2CDim,
    )

    rhs_interior = (grad_k - grad_kp1) * inv_maire
    rhs_top = (wpfloat("0.0") - grad_kp1) * inv_maire
    rhs_bottom = grad_k * inv_maire - flux_dn_e * inv_maire

    rhs = concat_where(dims.KDim == 0, rhs_top, rhs_interior)
    rhs = concat_where(dims.KDim < nlev - 1, rhs, rhs_bottom)
    return rhs, inv_maire


@gtx.field_operator
def _solve_vn_vertical_diffusion(
    w: fa.CellKField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
    km_ie: fa.EdgeKField[wpfloat],
    inv_rhoe: fa.EdgeKField[wpfloat],
    inv_ddqz_z_full_e: fa.EdgeKField[wpfloat],
    inv_ddqz_z_half_e: fa.EdgeKField[wpfloat],
    u_stress: fa.CellField[wpfloat],
    v_stress: fa.CellField[wpfloat],
    primal_normal_cell_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    primal_normal_cell_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    tot_tend: fa.EdgeKField[wpfloat],
    dtime: wpfloat,
    minlvl: gtx.int32,
    maxlvl: gtx.int32,
    nlev: gtx.int32,
) -> fa.EdgeKField[wpfloat]:
    """
    Right-hand side, tridiagonal matrix and implicit solve of the vn diffusion.

    Accumulates onto the total edge tendency of ``Compute_diffusion_hor_wind``,
    which already holds the horizontal stress tendency.
    """
    rhs, inv_maire = _compute_vn_vertical_diffusion_rhs(
        w=w,
        km_ie=km_ie,
        inv_rhoe=inv_rhoe,
        inv_ddqz_z_full_e=inv_ddqz_z_full_e,
        u_stress=u_stress,
        v_stress=v_stress,
        primal_normal_cell_x=primal_normal_cell_x,
        primal_normal_cell_y=primal_normal_cell_y,
        c_lin_e=c_lin_e,
        inv_dual_edge_length=inv_dual_edge_length,
        nlev=nlev,
    )
    a, b, c = _prepare_tridiagonal_matrix_edges(
        inv_mair=inv_maire,
        inv_dz=inv_ddqz_z_half_e,
        zk=km_ie,
        zprefac=wpfloat("1.0"),
        minlvl=minlvl,
        maxlvl=maxlvl,
    )
    return _solve_vertical_diffusion_edges(
        a=a, b=b, c=c, rhs=rhs, var=vn, tend=tot_tend, dtime=dtime
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def solve_vn_vertical_diffusion(
    w: fa.CellKField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
    km_ie: fa.EdgeKField[wpfloat],
    inv_rhoe: fa.EdgeKField[wpfloat],
    inv_ddqz_z_full_e: fa.EdgeKField[wpfloat],
    inv_ddqz_z_half_e: fa.EdgeKField[wpfloat],
    u_stress: fa.CellField[wpfloat],
    v_stress: fa.CellField[wpfloat],
    primal_normal_cell_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    primal_normal_cell_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    tot_tend: fa.EdgeKField[wpfloat],
    dtime: wpfloat,
    nlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _solve_vn_vertical_diffusion(
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
        tot_tend=tot_tend,
        dtime=dtime,
        minlvl=vertical_start,
        maxlvl=vertical_end - 1,
        nlev=nlev,
        out=tot_tend,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _interpolate_and_update_horizontal_wind(
    tot_tend: fa.EdgeKField[wpfloat],
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
    """
    Cell wind tendencies and updated winds of ``Compute_diffusion_hor_wind``.

    Fuses ``rbf_vec_interpol_cell`` (tot_tend -> tend_u, tend_v) with the final
    update loop ``new_u/v = u/v + tend_u/v * dtime``.
    """
    tend_u, tend_v = _edge_2_cell_vector_rbf_interpolation(
        p_e_in=tot_tend, ptr_coeff_1=rbf_coeff_c1, ptr_coeff_2=rbf_coeff_c2
    )
    new_u, new_v = _update_two_cell_kdim_fields_with_tendency(
        field_1=u, field_2=v, tendency_1=tend_u, tendency_2=tend_v, dtime=dtime
    )
    return tend_u, tend_v, new_u, new_v


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def interpolate_and_update_horizontal_wind(
    tot_tend: fa.EdgeKField[wpfloat],
    u: fa.CellKField[wpfloat],
    v: fa.CellKField[wpfloat],
    rbf_coeff_c1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2EDim], wpfloat],
    rbf_coeff_c2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2EDim], wpfloat],
    tend_u: fa.CellKField[wpfloat],
    tend_v: fa.CellKField[wpfloat],
    new_u: fa.CellKField[wpfloat],
    new_v: fa.CellKField[wpfloat],
    dtime: wpfloat,
    cell_start_nudging: gtx.int32,
    cell_start_lateral_boundary_level_2: gtx.int32,
    cell_end_local: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _interpolate_and_update_horizontal_wind(
        tot_tend=tot_tend,
        u=u,
        v=v,
        rbf_coeff_c1=rbf_coeff_c1,
        rbf_coeff_c2=rbf_coeff_c2,
        dtime=dtime,
        out=(tend_u, tend_v, new_u, new_v),
        domain=(
            # tend_u / tend_v: cells rl 2..min_rlcell_int (the Fortran default
            # opt_rlstart = 2 of rbf_vec_interpol_cell), all full levels
            {
                dims.CellDim: (cell_start_lateral_boundary_level_2, cell_end_local),
                dims.KDim: (vertical_start, vertical_end),
            },
            {
                dims.CellDim: (cell_start_lateral_boundary_level_2, cell_end_local),
                dims.KDim: (vertical_start, vertical_end),
            },
            # new_u / new_v: the tmx t_domain cells, all full levels
            {
                dims.CellDim: (cell_start_nudging, cell_end_local),
                dims.KDim: (vertical_start, vertical_end),
            },
            {
                dims.CellDim: (cell_start_nudging, cell_end_local),
                dims.KDim: (vertical_start, vertical_end),
            },
        ),
    )


# ---------------------------------------------------------------------------
# Compute_diffusion_vert_wind (w diffusion)
# ---------------------------------------------------------------------------
@gtx.field_operator
def _compute_w_vertical_diffusion_rhs(
    rho_ic: fa.CellKField[wpfloat],
    inv_ddqz_z_half: fa.CellKField[wpfloat],
    km_c: fa.CellKField[wpfloat],
    div_c: fa.CellKField[wpfloat],
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """
    Compute the right-hand side of the w tridiagonal solve and the inverse
    half-level density and air mass.

    Port of the first loop of 'Compute_diffusion_vert_wind' (mo_vdf.f90):

        inv_rho_ic(k)  = 1 / rho_ic(k)
        inv_mair_ic(k) = inv_rho_ic(k) * inv_dzh(k)
        rhs(k) = 2 * inv_mair_ic(k) * (km_c(k) * 1/3 * div_c(k)
                                       - km_c(k-1) * 1/3 * div_c(k-1))

    All outputs live on half levels (rows jk = 2..nlev in the 1-based Fortran,
    i.e. rows 1..nlev-1 with 0-based indexing). rho_ic and inv_ddqz_z_half
    ('inv_dz_ic') are half-level fields, km_c and div_c are full-level fields
    read at the full levels directly above (k-1) and below (k) the half level k.
    """
    z_1by3 = wpfloat("1.0") / wpfloat("3.0")
    inv_rho_ic = wpfloat("1.0") / rho_ic
    inv_mair_ic = inv_rho_ic * inv_ddqz_z_half
    rhs = (
        wpfloat("2.0")
        * inv_mair_ic
        * (km_c * z_1by3 * div_c - km_c(KDim - 1) * z_1by3 * div_c(KDim - 1))
    )
    return rhs, inv_rho_ic, inv_mair_ic


@gtx.field_operator
def _modify_w_diffusion_matrix_boundary(
    b: fa.CellKField[wpfloat],
    km_c: fa.CellKField[wpfloat],
    inv_dz: fa.CellKField[wpfloat],
    inv_mair_ic: fa.CellKField[wpfloat],
    minlvl: gtx.int32,
    maxlvl: gtx.int32,
) -> fa.CellKField[wpfloat]:
    """
    Add the w = 0 top/bottom boundary-condition terms to the main diagonal of
    the w-diffusion tridiagonal matrix.

    Port of the boundary-row loop of 'Compute_diffusion_vert_wind'
    (mo_vdf.f90), applied to the 'b' produced by
    ``_prepare_tridiagonal_matrix_cells_half`` (zprefac = 2):

        b(2)    += 2 * km_c(1)    * inv_dzf(1)    * inv_mair_ic(2)
        b(nlev) += 2 * km_c(nlev) * inv_dzf(nlev) * inv_mair_ic(nlev)

    (1-based rows; the terms result from the condition w = 0 at the top and
    bottom boundaries.) km_c and inv_dz ('inv_dzf' = inv_dz_c) are full-level
    fields; inv_mair_ic lives on half levels. The two rows are selected with
    ``concat_where`` instead of two single-row program domains; the interior
    rows add exactly zero.
    """
    zero = _broadcast_value_on_cell_k(wpfloat("0.0"), inv_mair_ic)
    top_term = wpfloat("2.0") * km_c(KDim - 1) * inv_dz(KDim - 1) * inv_mair_ic
    bottom_term = wpfloat("2.0") * km_c * inv_dz * inv_mair_ic
    return (
        b
        + concat_where(dims.KDim > minlvl, zero, top_term)
        + concat_where(dims.KDim < maxlvl, zero, bottom_term)
    )


@gtx.field_operator
def _solve_w_vertical_diffusion(
    w: fa.CellKField[wpfloat],
    rho_ic: fa.CellKField[wpfloat],
    inv_ddqz_z_half: fa.CellKField[wpfloat],
    inv_ddqz_z_full: fa.CellKField[wpfloat],
    km_c: fa.CellKField[wpfloat],
    div_c: fa.CellKField[wpfloat],
    tend: fa.CellKField[wpfloat],
    dtime: wpfloat,
    minlvl: gtx.int32,
    maxlvl: gtx.int32,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """
    Right-hand side, tridiagonal matrix and implicit solve of the w diffusion.

    The Fortran w solve is implicit regardless of the configured solver type.
    ``inv_rho_ic`` is also needed by the horizontal w diffusion and is
    therefore returned.
    """
    rhs, inv_rho_ic, inv_mair_ic = _compute_w_vertical_diffusion_rhs(
        rho_ic=rho_ic, inv_ddqz_z_half=inv_ddqz_z_half, km_c=km_c, div_c=div_c
    )
    a, b, c = _prepare_tridiagonal_matrix_cells_half(
        inv_mair=inv_mair_ic,
        inv_dz=inv_ddqz_z_full,
        zk=km_c,
        zprefac=wpfloat("2.0"),
        minlvl=minlvl,
        maxlvl=maxlvl,
    )
    b = _modify_w_diffusion_matrix_boundary(
        b=b,
        km_c=km_c,
        inv_dz=inv_ddqz_z_full,
        inv_mair_ic=inv_mair_ic,
        minlvl=minlvl,
        maxlvl=maxlvl,
    )
    new_tend = _solve_vertical_diffusion_cells(
        a=a, b=b, c=c, rhs=rhs, var=w, tend=tend, dtime=dtime
    )
    return inv_rho_ic, new_tend


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def solve_w_vertical_diffusion(
    w: fa.CellKField[wpfloat],
    rho_ic: fa.CellKField[wpfloat],
    inv_ddqz_z_half: fa.CellKField[wpfloat],
    inv_ddqz_z_full: fa.CellKField[wpfloat],
    km_c: fa.CellKField[wpfloat],
    div_c: fa.CellKField[wpfloat],
    inv_rho_ic: fa.CellKField[wpfloat],
    tend: fa.CellKField[wpfloat],
    dtime: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _solve_w_vertical_diffusion(
        w=w,
        rho_ic=rho_ic,
        inv_ddqz_z_half=inv_ddqz_z_half,
        inv_ddqz_z_full=inv_ddqz_z_full,
        km_c=km_c,
        div_c=div_c,
        tend=tend,
        dtime=dtime,
        minlvl=vertical_start,
        maxlvl=vertical_end - 1,
        out=(inv_rho_ic, tend),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


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
    half level. The output lives on half levels, rows jk = 2..nlev (1-based).

    Domains (Fortran caller): edges from ``rl_start = grf_bdywidth_e`` ->
    ``h_grid.Zone.NUDGING`` to ``rl_end = min_rledge_int - 1`` ->
    ``h_grid.Zone.HALO``.
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


@gtx.field_operator
def _compute_w_horizontal_stress_tendency_from_vn(
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
    vn: fa.EdgeKField[wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], wpfloat],
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
    Horizontal w stress tendency, with the full-level tangential wind
    (``rbf_vec_interpol_edge`` of vn) it needs computed in place.
    """
    vt_e = _compute_tangential_wind_wp(vn=vn, rbf_vec_coeff_e=rbf_vec_coeff_e)
    return _compute_w_horizontal_stress_tendency(
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
    )


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
    vn: fa.EdgeKField[wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], wpfloat],
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
    _compute_w_horizontal_stress_tendency_from_vn(
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
        vn=vn,
        rbf_vec_coeff_e=rbf_vec_coeff_e,
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


@gtx.field_operator
def _apply_w_horizontal_diffusion_and_update(
    hori_tend_e: fa.EdgeKField[wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    inv_rho_ic: fa.CellKField[wpfloat],
    w: fa.CellKField[wpfloat],
    tend: fa.CellKField[wpfloat],
    dtime: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """
    Add the horizontal w-diffusion tendency (interpolated to cell centers) to
    the w tendency and update w.

    Port of the last two loops of 'Compute_diffusion_vert_wind' (mo_vdf.f90):

        tend(k) += inv_rho_ic(k) * sum_{e in C2E} e_bln_c_s * hori_tend_e(e, k)
        new_w(k) = w(k) + tend(k) * dtime

    All K fields live on half levels (nlev + 1 rows). The top and bottom half
    levels are excluded (w = 0 boundary condition): rows jk = 2..nlev
    (1-based), i.e. the program must be called with vertical bounds (1, nlev).

    Domains (Fortran caller, the tmx ``domain`` cell loop bounds): cells from
    ``rl_start = grf_bdywidth_c + 1`` -> ``h_grid.Zone.NUDGING`` to
    ``rl_end = min_rlcell_int`` -> ``h_grid.Zone.LOCAL``.
    """
    new_tend = tend + inv_rho_ic * neighbor_sum(e_bln_c_s * hori_tend_e(C2E), axis=C2EDim)
    new_w = w + new_tend * dtime
    return new_w, new_tend


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def apply_w_horizontal_diffusion_and_update(
    hori_tend_e: fa.EdgeKField[wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    inv_rho_ic: fa.CellKField[wpfloat],
    w: fa.CellKField[wpfloat],
    new_w: fa.CellKField[wpfloat],
    tend: fa.CellKField[wpfloat],
    dtime: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _apply_w_horizontal_diffusion_and_update(
        hori_tend_e=hori_tend_e,
        e_bln_c_s=e_bln_c_s,
        inv_rho_ic=inv_rho_ic,
        w=w,
        tend=tend,
        dtime=dtime,
        out=(new_w, tend),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
