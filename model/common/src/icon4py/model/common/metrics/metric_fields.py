# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from gt4py.next import Field, GridType, abs, field_operator, int32, maximum, minimum, program, where

from icon4py.model.common.dimension import C2E, CellDim, EdgeDim, KDim, Koff
from icon4py.model.common.math.helpers import (
    average_k_level_up,
    difference_k_level_down,
    difference_k_level_up,
)
from icon4py.model.common.type_alias import wpfloat


"""
Contains metric fields calculations for the vertical grid, ported from mo_vertical_grid.f90.
"""


@program(grid_type=GridType.UNSTRUCTURED)
def compute_z_mc(
    z_ifc: Field[[CellDim, KDim], wpfloat],
    z_mc: Field[[CellDim, KDim], wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    """
    Compute the geometric height of full levels from the geometric height of half levels (z_ifc).

    This assumes that the input field z_ifc is defined on half levels (KHalfDim) and the
    returned fields is defined on full levels (KDim)

    Args:
        z_ifc: Field[[CellDim, KDim], wpfloat] geometric height on half levels
        z_mc: Field[[CellDim, KDim], wpfloat] output, geometric height defined on full levels
        horizontal_start:int32 start index of horizontal domain
        horizontal_end:int32 end index of horizontal domain
        vertical_start:int32 start index of vertical domain
        vertical_end:int32 end index of vertical domain

    """
    average_k_level_up(
        z_ifc,
        out=z_mc,
        domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )


@field_operator
def _compute_ddqz_z_half(
    z_ifc: Field[[CellDim, KDim], wpfloat],
    z_mc: Field[[CellDim, KDim], wpfloat],
    k: Field[[KDim], int32],
    nlev: int32,
) -> Field[[CellDim, KDim], wpfloat]:
    ddqz_z_half = where(
        (k > int32(0)) & (k < nlev),
        difference_k_level_down(z_mc),
        where(k == 0, 2.0 * (z_ifc - z_mc), 2.0 * (z_mc(Koff[-1]) - z_ifc)),
    )
    return ddqz_z_half


@program(grid_type=GridType.UNSTRUCTURED, backend=None)
def compute_ddqz_z_half(
    z_ifc: Field[[CellDim, KDim], wpfloat],
    z_mc: Field[[CellDim, KDim], wpfloat],
    k: Field[[KDim], int32],
    ddqz_z_half: Field[[CellDim, KDim], wpfloat],
    nlev: int32,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    """
    Compute functional determinant of the metrics (is positive) on half levels.

    See mo_vertical_grid.f90

    Args:
        z_ifc: geometric height on half levels
        z_mc: geometric height on full levels
        k: vertical dimension index
        nlev: total number of levels
        ddqz_z_half: (output)
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
        vertical_start: vertical start index
        vertical_end: vertical end index
    """
    _compute_ddqz_z_half(
        z_ifc,
        z_mc,
        k,
        nlev,
        out=ddqz_z_half,
        domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )


@field_operator
def _compute_ddqz_z_full(
    z_ifc: Field[[CellDim, KDim], wpfloat],
) -> tuple[Field[[CellDim, KDim], wpfloat], Field[[CellDim, KDim], wpfloat]]:
    ddqz_z_full = difference_k_level_up(z_ifc)
    inverse_ddqz_z_full = 1.0 / ddqz_z_full
    return ddqz_z_full, inverse_ddqz_z_full


@program(grid_type=GridType.UNSTRUCTURED)
def compute_ddqz_z_full(
    z_ifc: Field[[CellDim, KDim], wpfloat],
    ddqz_z_full: Field[[CellDim, KDim], wpfloat],
    inv_ddqz_z_full: Field[[CellDim, KDim], wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    """
    Compute ddqz_z_full and its inverse inv_ddqz_z_full.

    Functional determinant of the metrics (is positive) on full levels and inverse inverse layer thickness(for runtime optimization).
    See mo_vertical_grid.f90

    Args:
        z_ifc: geometric height on half levels
        ddqz_z_full: (output)
        inv_ddqz_z_full: (output)
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
        vertical_start: vertical start index
        vertical_end: vertical end index

    """
    _compute_ddqz_z_full(
        z_ifc,
        out=(ddqz_z_full, inv_ddqz_z_full),
        domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )


@field_operator
def _compute_vwind_impl_wgt(
    z_ddxn_z_half_e: Field[[EdgeDim], wpfloat],
    z_ddxt_z_half_e: Field[[EdgeDim], wpfloat],
    dual_edge_length: Field[[EdgeDim], wpfloat],
    vwind_offctr: wpfloat,
) -> Field[[CellDim], wpfloat]:
    z_ddx_1 = maximum(abs(z_ddxn_z_half_e(C2E[1])), abs(z_ddxt_z_half_e(C2E[1])))
    z_ddx_2 = maximum(abs(z_ddxn_z_half_e(C2E[2])), abs(z_ddxt_z_half_e(C2E[2])))
    z_ddx_3 = maximum(abs(z_ddxn_z_half_e(C2E[3])), abs(z_ddxt_z_half_e(C2E[3])))
    z_ddx_1_2 = maximum(z_ddx_1, z_ddx_2)
    z_maxslope = maximum(z_ddx_1_2, z_ddx_3)

    z_diff_1_2 = maximum(
        abs(z_ddxn_z_half_e(C2E[1]) * dual_edge_length(C2E[1])),
        abs(z_ddxn_z_half_e(C2E[2]) * dual_edge_length(C2E[2])),
    )
    z_diff = maximum(z_diff_1_2, abs(z_ddxn_z_half_e(C2E[3]) * dual_edge_length(C2E[3])))
    z_offctr_1 = maximum(vwind_offctr, 0.425 * z_maxslope**0.75)
    z_offctr = maximum(z_offctr_1, minimum(0.25, 2.5e-4 * (z_diff - 250.0)))
    z_offctr = minimum(maximum(vwind_offctr, 0.75), z_offctr)
    vwind_impl_wgt = 0.5 + z_offctr
    return vwind_impl_wgt


@program(grid_type=GridType.UNSTRUCTURED)
def compute_vwind_impl_wgt(
    z_ddxn_z_half_e: Field[[EdgeDim], wpfloat],
    z_ddxt_z_half_e: Field[[EdgeDim], wpfloat],
    dual_edge_length: Field[[EdgeDim], wpfloat],
    vwind_impl_wgt: Field[[CellDim], wpfloat],
    vwind_offctr: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
):
    """
    Compute vwind_impl_wgt.

    See mo_vertical_grid.f90

    Args:
        z_ddxn_z_half_e: intermediate storage for field
        z_ddxt_z_half_e: intermediate storage for field
        dual_edge_length: dual_edge_length
        vwind_impl_wgt: (output) offcentering in vertical mass flux
        vwind_offctr: off-centering in vertical wind solver
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index

    """
    _compute_vwind_impl_wgt(
        z_ddxn_z_half_e=z_ddxn_z_half_e,
        z_ddxt_z_half_e=z_ddxt_z_half_e,
        dual_edge_length=dual_edge_length,
        vwind_offctr=vwind_offctr,
        out=vwind_impl_wgt,
        domain={CellDim: (horizontal_start, horizontal_end)},
    )


@field_operator
def _compute_vwind_expl_wgt(vwind_impl_wgt: Field[[CellDim], wpfloat]) -> Field[[CellDim], wpfloat]:
    return 1.0 - vwind_impl_wgt


@program(grid_type=GridType.UNSTRUCTURED)
def compute_vwind_expl_wgt(
    vwind_impl_wgt: Field[[CellDim], wpfloat],
    vwind_expl_wgt: Field[[CellDim], wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
):
    """
    Compute vwind_expl_wgt.

    See mo_vertical_grid.f90

    Args:
        vwind_impl_wgt: offcentering in vertical mass flux
        vwind_expl_wgt: (output) 1 - of vwind_impl_wgt
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index

    """

    _compute_vwind_expl_wgt(
        vwind_impl_wgt=vwind_impl_wgt,
        out=vwind_expl_wgt,
        domain={CellDim: (horizontal_start, horizontal_end)},
    )
