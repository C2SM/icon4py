# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import neighbor_sum
from gt4py.next.experimental import concat_where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C, E2CDim, KDim
from icon4py.model.common.type_alias import wpfloat


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

    The bottom boundary row is selected with 'dims.KDim < nlev - 1' because
    'concat_where(dims.KDim == nlev - 1, ...)' is currently broken in GT4Py
    (GridTools/gt4py#2205).

    Domains (Fortran caller): jk = 1..nlev; edges from
    rl_start = grf_bdywidth_e + 1 -> 'h_grid.Zone.NUDGING_LEVEL_2' to
    rl_end = min_rledge_int -> 'h_grid.Zone.LOCAL'.

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


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_vn_vertical_diffusion_rhs(
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
    rhs: fa.EdgeKField[wpfloat],
    inv_maire: fa.EdgeKField[wpfloat],
    nlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_vn_vertical_diffusion_rhs(
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
        out=(rhs, inv_maire),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
