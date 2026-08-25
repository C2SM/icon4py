# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import KDim
from icon4py.model.common.type_alias import wpfloat


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
    i.e. rows 1..nlev-1 with 0-based indexing; the program must be called with
    vertical bounds (1, nlev), rows 0 and nlev stay untouched). rho_ic and
    inv_ddqz_z_half ('inv_dz_ic') are half-level fields, km_c and div_c are
    full-level fields read at the full levels directly above (k-1) and below
    (k) the half level k.

    The three outputs are fused because the Fortran computes them in a single
    loop and both inv_rho_ic (horizontal w diffusion) and inv_mair_ic
    (tridiagonal matrix setup and its boundary-row modification) are reused
    downstream.

    Domains (Fortran caller, the tmx 'domain' cell loop bounds): cells from
    rl_start = grf_bdywidth_c + 1 -> 'h_grid.Zone.NUDGING' to
    rl_end = min_rlcell_int -> 'h_grid.Zone.LOCAL'.
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


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_w_vertical_diffusion_rhs(
    rho_ic: fa.CellKField[wpfloat],
    inv_ddqz_z_half: fa.CellKField[wpfloat],
    km_c: fa.CellKField[wpfloat],
    div_c: fa.CellKField[wpfloat],
    rhs: fa.CellKField[wpfloat],
    inv_rho_ic: fa.CellKField[wpfloat],
    inv_mair_ic: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_w_vertical_diffusion_rhs(
        rho_ic=rho_ic,
        inv_ddqz_z_half=inv_ddqz_z_half,
        km_c=km_c,
        div_c=div_c,
        out=(rhs, inv_rho_ic, inv_mair_ic),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
