# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next.experimental import concat_where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.math.tridiagonal import (
    _solve_tridiagonal_matrix_back_substitution,
    _solve_tridiagonal_matrix_back_substitution_on_half_levels_wp,
    _solve_tridiagonal_matrix_forward_sweep,
    _solve_tridiagonal_matrix_forward_sweep_on_half_levels_wp,
)
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def _assemble_vertical_diffusion_matrix_on_cells(
    diffusivity: fa.CellKHalfField[wpfloat],
    inv_dz: fa.CellKHalfField[wpfloat],
    inv_air_mass: fa.CellKField[wpfloat],
    prefactor: wpfloat,
    minlvl: gtx.int32,
    maxlvl: gtx.int32,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """
    Sub-, main and super-diagonal of the vertical diffusion matrix for a full-level field.

    The column spans full levels minlvl..maxlvl, with no flux through its top and bottom.
    """
    # embedded rejects a scalar branch on an unbounded region, so the zeros are a field
    zero = wpfloat("0.0") * inv_air_mass
    a = concat_where(
        dims.KDim > minlvl,
        wpfloat("0.0")
        - prefactor * diffusivity(dims.KDim - 0.5) * inv_dz(dims.KDim - 0.5) * inv_air_mass,
        zero,
    )
    c = concat_where(
        dims.KDim < maxlvl,
        wpfloat("0.0")
        - prefactor * diffusivity(dims.KDim + 0.5) * inv_dz(dims.KDim + 0.5) * inv_air_mass,
        zero,
    )
    return a, wpfloat("0.0") - a - c, c


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def _assemble_vertical_diffusion_matrix_on_cell_half_levels(
    diffusivity: fa.CellKField[wpfloat],
    inv_dz: fa.CellKField[wpfloat],
    inv_air_mass: fa.CellKHalfField[wpfloat],
    prefactor: wpfloat,
    minlvl: gtx.int32,
    maxlvl: gtx.int32,
) -> tuple[fa.CellKHalfField[wpfloat], fa.CellKHalfField[wpfloat], fa.CellKHalfField[wpfloat]]:
    """
    Sub-, main and super-diagonal of the vertical diffusion matrix for a half-level field.

    The column spans half levels minlvl..maxlvl, with no flux through its top and bottom.
    """
    # embedded rejects a scalar branch on an unbounded region, so the zeros are a field
    zero = wpfloat("0.0") * inv_air_mass
    a = concat_where(
        dims.KHalfDim > minlvl,
        wpfloat("0.0")
        - prefactor * diffusivity(dims.KHalfDim - 0.5) * inv_dz(dims.KHalfDim - 0.5) * inv_air_mass,
        zero,
    )
    c = concat_where(
        dims.KHalfDim < maxlvl,
        wpfloat("0.0")
        - prefactor * diffusivity(dims.KHalfDim + 0.5) * inv_dz(dims.KHalfDim + 0.5) * inv_air_mass,
        zero,
    )
    return a, wpfloat("0.0") - a - c, c


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def _assemble_vertical_diffusion_matrix_on_edges(
    diffusivity: fa.EdgeKHalfField[wpfloat],
    inv_dz: fa.EdgeKHalfField[wpfloat],
    inv_air_mass: fa.EdgeKField[wpfloat],
    prefactor: wpfloat,
    minlvl: gtx.int32,
    maxlvl: gtx.int32,
) -> tuple[fa.EdgeKField[wpfloat], fa.EdgeKField[wpfloat], fa.EdgeKField[wpfloat]]:
    """
    Sub-, main and super-diagonal of the vertical diffusion matrix for a full-level field.

    The column spans full levels minlvl..maxlvl, with no flux through its top and bottom.
    """
    # embedded rejects a scalar branch on an unbounded region, so the zeros are a field
    zero = wpfloat("0.0") * inv_air_mass
    a = concat_where(
        dims.KDim > minlvl,
        wpfloat("0.0")
        - prefactor * diffusivity(dims.KDim - 0.5) * inv_dz(dims.KDim - 0.5) * inv_air_mass,
        zero,
    )
    c = concat_where(
        dims.KDim < maxlvl,
        wpfloat("0.0")
        - prefactor * diffusivity(dims.KDim + 0.5) * inv_dz(dims.KDim + 0.5) * inv_air_mass,
        zero,
    )
    return a, wpfloat("0.0") - a - c, c


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def _solve_implicit_vertical_diffusion_on_cells(
    var: fa.CellKField[wpfloat],
    a: fa.CellKField[wpfloat],
    b: fa.CellKField[wpfloat],
    c: fa.CellKField[wpfloat],
    rhs: fa.CellKField[wpfloat],
    tend: fa.CellKField[wpfloat],
    dtime: wpfloat,
) -> fa.CellKField[wpfloat]:
    """
    tend plus the tendency of one implicit step of the vertical diffusion with matrix (a, b, c).

    The system spans the call's vertical domain; a on its first row and c on its last row
    have no effect.
    """
    inv_dtime = wpfloat("1.0") / dtime
    q, d_prime = _solve_tridiagonal_matrix_forward_sweep(a, inv_dtime + b, c, var * inv_dtime + rhs)
    return tend + (_solve_tridiagonal_matrix_back_substitution(q, d_prime) - var) * inv_dtime


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def _solve_implicit_vertical_diffusion_on_cell_half_levels(
    var: fa.CellKHalfField[wpfloat],
    a: fa.CellKHalfField[wpfloat],
    b: fa.CellKHalfField[wpfloat],
    c: fa.CellKHalfField[wpfloat],
    rhs: fa.CellKHalfField[wpfloat],
    tend: fa.CellKHalfField[wpfloat],
    dtime: wpfloat,
) -> fa.CellKHalfField[wpfloat]:
    """
    tend plus the tendency of one implicit step of the vertical diffusion with matrix (a, b, c).

    The system spans the call's vertical domain; a on its first row and c on its last row
    have no effect.
    """
    inv_dtime = wpfloat("1.0") / dtime
    q, d_prime = _solve_tridiagonal_matrix_forward_sweep_on_half_levels_wp(
        a, inv_dtime + b, c, var * inv_dtime + rhs
    )
    return (
        tend
        + (_solve_tridiagonal_matrix_back_substitution_on_half_levels_wp(q, d_prime) - var)
        * inv_dtime
    )


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def _solve_implicit_vertical_diffusion_on_edges(
    var: fa.EdgeKField[wpfloat],
    a: fa.EdgeKField[wpfloat],
    b: fa.EdgeKField[wpfloat],
    c: fa.EdgeKField[wpfloat],
    rhs: fa.EdgeKField[wpfloat],
    tend: fa.EdgeKField[wpfloat],
    dtime: wpfloat,
) -> fa.EdgeKField[wpfloat]:
    """
    tend plus the tendency of one implicit step of the vertical diffusion with matrix (a, b, c).

    The system spans the call's vertical domain; a on its first row and c on its last row
    have no effect.
    """
    inv_dtime = wpfloat("1.0") / dtime
    q, d_prime = _solve_tridiagonal_matrix_forward_sweep(a, inv_dtime + b, c, var * inv_dtime + rhs)
    return tend + (_solve_tridiagonal_matrix_back_substitution(q, d_prime) - var) * inv_dtime
