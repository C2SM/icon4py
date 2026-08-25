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
def _modify_w_diffusion_matrix_boundary_top(
    b: fa.CellKField[wpfloat],
    km_c: fa.CellKField[wpfloat],
    inv_dz: fa.CellKField[wpfloat],
    inv_mair_ic: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """Top boundary row (jk = 2, i.e. row minlvl): b += 2 * km_c(k-1) * inv_dz(k-1) * inv_mair_ic(k)."""
    return b + wpfloat("2.0") * km_c(KDim - 1) * inv_dz(KDim - 1) * inv_mair_ic


@gtx.field_operator
def _modify_w_diffusion_matrix_boundary_bottom(
    b: fa.CellKField[wpfloat],
    km_c: fa.CellKField[wpfloat],
    inv_dz: fa.CellKField[wpfloat],
    inv_mair_ic: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """Bottom boundary row (jk = nlev, i.e. row maxlvl): b += 2 * km_c(k) * inv_dz(k) * inv_mair_ic(k)."""
    return b + wpfloat("2.0") * km_c * inv_dz * inv_mair_ic


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def modify_w_diffusion_matrix_boundary(
    b: fa.CellKField[wpfloat],
    km_c: fa.CellKField[wpfloat],
    inv_dz: fa.CellKField[wpfloat],
    inv_mair_ic: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """
    Add the w = 0 top/bottom boundary-condition terms to the main diagonal of
    the w-diffusion tridiagonal matrix.

    Port of the boundary-row loop of 'Compute_diffusion_vert_wind'
    (mo_vdf.f90), applied to the 'b' produced by
    'prepare_tridiagonal_matrix_cells_half' (zprefac = 2):

        b(2)    += 2 * km_c(1)    * inv_dzf(1)    * inv_mair_ic(2)
        b(nlev) += 2 * km_c(nlev) * inv_dzf(nlev) * inv_mair_ic(nlev)

    (1-based rows; the terms result from the condition w = 0 at the top and
    bottom boundaries.) 'b' is modified in place on the two rows only; the two
    rows are addressed with single-row program domains (mirroring
    'prepare_tridiagonal_matrix_cells_half') instead of 'concat_where' because
    'concat_where(dims.KDim == nlev - 1, ...)' is currently broken in GT4Py
    (GridTools/gt4py#2205).

    'vertical_start'/'vertical_end' are the vertical bounds of the w solve
    (minlvl = 2, maxlvl = nlev in the Fortran -> (1, nlev) 0-based): the top
    term is added at row 'vertical_start', the bottom term at row
    'vertical_end - 1'. km_c and inv_dz ('inv_dzf' = inv_dz_c) are full-level
    fields; inv_mair_ic lives on half levels.

    Domains (Fortran caller, the tmx 'domain' cell loop bounds): cells from
    rl_start = grf_bdywidth_c + 1 -> 'h_grid.Zone.NUDGING' to
    rl_end = min_rlcell_int -> 'h_grid.Zone.LOCAL'.
    """
    _modify_w_diffusion_matrix_boundary_top(
        b=b,
        km_c=km_c,
        inv_dz=inv_dz,
        inv_mair_ic=inv_mair_ic,
        out=b,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_start + 1),
        },
    )
    _modify_w_diffusion_matrix_boundary_bottom(
        b=b,
        km_c=km_c,
        inv_dz=inv_dz,
        inv_mair_ic=inv_mair_ic,
        out=b,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_end - 1, vertical_end),
        },
    )
