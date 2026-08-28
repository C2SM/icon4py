# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Tridiagonal machinery of the tmx vertical diffusion.

Field operators shared by the scalar diffusion (``Compute_diffusion_hydrometeors``
and ``Compute_diffusion_temperature``, cells on full levels), the horizontal
wind diffusion (``Compute_diffusion_hor_wind``, edges on full levels) and the
vertical wind diffusion (``Compute_diffusion_vert_wind``, cells on half levels).
The callers assemble them into one program per halo-exchange interval, so the
matrix rows and the right-hand side never leave the fused kernel.

The boundary rows are selected with ``concat_where(dims.KDim > minlvl, ...)`` /
``concat_where(dims.KDim < maxlvl, ...)`` rather than an equality test because
``concat_where(dims.KDim == maxlvl, ...)`` is broken in GT4Py
(GridTools/gt4py#2205). Both branches must be fields with a bounded K range;
a bare scalar or ``broadcast`` leaves the range open, which raises "Cannot
compute length of open 'UnitRange'" on the embedded backend and silently
computes the wrong values with gtfn. The constant branches are therefore built
with ``_broadcast_value_on_cell_k`` / ``_broadcast_value_on_edge_k``, which
anchor the value to an input field's K range; the anchor is always an inverse
mass or density, which is strictly positive and finite on the tmx domains.
"""

from typing import NamedTuple

import gt4py.next as gtx
from gt4py.next.experimental import concat_where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import KDim
from icon4py.model.common.math.operators import (
    _broadcast_value_on_cell_k,
    _broadcast_value_on_edge_k,
)
from icon4py.model.common.math.tridiagonal import (
    _tridiagonal_back_substitution,
    _tridiagonal_forward_sweep,
)
from icon4py.model.common.type_alias import wpfloat


class TridiagonalMatrixCells(NamedTuple):
    """Sub-, main and super-diagonal of a tridiagonal matrix on cells."""

    a: fa.CellKField[wpfloat]
    b: fa.CellKField[wpfloat]
    c: fa.CellKField[wpfloat]


class TridiagonalMatrixEdges(NamedTuple):
    """Sub-, main and super-diagonal of a tridiagonal matrix on edges."""

    a: fa.EdgeKField[wpfloat]
    b: fa.EdgeKField[wpfloat]
    c: fa.EdgeKField[wpfloat]


@gtx.field_operator
def _prepare_tridiagonal_matrix_cells(
    inv_mair: fa.CellKField[wpfloat],
    inv_dz: fa.CellKField[wpfloat],
    zk: fa.CellKField[wpfloat],
    zprefac: wpfloat,
    minlvl: gtx.int32,
    maxlvl: gtx.int32,
) -> TridiagonalMatrixCells:
    """
    Rows of the full-level tridiagonal diffusion matrix on cells.

    Port of 'prepare_diffusion_matrix_wp' (mo_vdf_atmo.f90) with
    lhalflvl=.FALSE. (lvlcorr_a=0, lvlcorr_c=1):
        a(jk) = - zprefac * zk(jk)   * inv_dz(jk)   * inv_mair(jk)
        c(jk) = - zprefac * zk(jk+1) * inv_dz(jk+1) * inv_mair(jk)
        b(jk) = - a(jk) - c(jk)
    with a = 0 on the upper boundary row ``minlvl`` and c = 0 on the lower
    boundary row ``maxlvl``.
    """
    zero = _broadcast_value_on_cell_k(wpfloat("0.0"), inv_mair)
    a_interior = wpfloat("0.0") - zprefac * zk * inv_dz * inv_mair
    c_interior = wpfloat("0.0") - zprefac * zk(KDim + 1) * inv_dz(KDim + 1) * inv_mair
    a = concat_where(dims.KDim > minlvl, a_interior, zero)
    c = concat_where(dims.KDim < maxlvl, c_interior, zero)
    b = wpfloat("0.0") - a - c
    return TridiagonalMatrixCells(a=a, b=b, c=c)


@gtx.field_operator
def _prepare_tridiagonal_matrix_cells_half(
    inv_mair: fa.CellKField[wpfloat],
    inv_dz: fa.CellKField[wpfloat],
    zk: fa.CellKField[wpfloat],
    zprefac: wpfloat,
    minlvl: gtx.int32,
    maxlvl: gtx.int32,
) -> TridiagonalMatrixCells:
    """
    Rows of the half-level tridiagonal diffusion matrix on cells (w solve).

    Port of 'prepare_diffusion_matrix_wp' (mo_vdf_atmo.f90) with
    lhalflvl=.TRUE. (lvlcorr_a=-1, lvlcorr_c=0):
        a(jk) = - zprefac * zk(jk-1) * inv_dz(jk-1) * inv_mair(jk)
        c(jk) = - zprefac * zk(jk)   * inv_dz(jk)   * inv_mair(jk)
        b(jk) = - a(jk) - c(jk)
    with a = 0 on the upper boundary row ``minlvl`` and c = 0 on the lower
    boundary row ``maxlvl``. The unknowns live on half levels (rows
    minlvl..maxlvl, i.e. 2..nlev in the Fortran w solve), zk and inv_dz are
    full-level fields. The extra w = 0 boundary terms of the w solve are added
    by ``_modify_w_diffusion_matrix_boundary``.
    """
    zero = _broadcast_value_on_cell_k(wpfloat("0.0"), inv_mair)
    a_interior = wpfloat("0.0") - zprefac * zk(KDim - 1) * inv_dz(KDim - 1) * inv_mair
    c_interior = wpfloat("0.0") - zprefac * zk * inv_dz * inv_mair
    a = concat_where(dims.KDim > minlvl, a_interior, zero)
    c = concat_where(dims.KDim < maxlvl, c_interior, zero)
    b = wpfloat("0.0") - a - c
    return TridiagonalMatrixCells(a=a, b=b, c=c)


@gtx.field_operator
def _prepare_tridiagonal_matrix_edges(
    inv_mair: fa.EdgeKField[wpfloat],
    inv_dz: fa.EdgeKField[wpfloat],
    zk: fa.EdgeKField[wpfloat],
    zprefac: wpfloat,
    minlvl: gtx.int32,
    maxlvl: gtx.int32,
) -> TridiagonalMatrixEdges:
    """
    Rows of the full-level tridiagonal diffusion matrix on edges (vn solve).

    Same as ``_prepare_tridiagonal_matrix_cells`` on the edge grid; used by the
    vn diffusion of 'Compute_diffusion_hor_wind' (mo_vdf.f90).
    """
    zero = _broadcast_value_on_edge_k(wpfloat("0.0"), inv_mair)
    a_interior = wpfloat("0.0") - zprefac * zk * inv_dz * inv_mair
    c_interior = wpfloat("0.0") - zprefac * zk(KDim + 1) * inv_dz(KDim + 1) * inv_mair
    a = concat_where(dims.KDim > minlvl, a_interior, zero)
    c = concat_where(dims.KDim < maxlvl, c_interior, zero)
    b = wpfloat("0.0") - a - c
    return TridiagonalMatrixEdges(a=a, b=b, c=c)


@gtx.field_operator
def _solve_vertical_diffusion_cells(
    a: fa.CellKField[wpfloat],
    b: fa.CellKField[wpfloat],
    c: fa.CellKField[wpfloat],
    rhs: fa.CellKField[wpfloat],
    var: fa.CellKField[wpfloat],
    tend: fa.CellKField[wpfloat],
    dtime: wpfloat,
) -> fa.CellKField[wpfloat]:
    """
    Implicit vertical diffusion solve on cells.

    Port of 'diffuse_vertical_implicit' (mo_tmx_numerics.f90):
        b_tot   = 1/dtime + b               (b is 'bb' in the Fortran)
        d       = var/dtime + rhs
        new_var = tridiagonal_solve(a, b_tot, c, d)
        tend    = tend + (new_var - var)/dtime

    As in the Fortran, the tridiagonal solution only enters through the
    tendency, so only the accumulated tendency is returned.
    """
    rdtime = wpfloat("1.0") / dtime
    c_prime, d_prime = _tridiagonal_forward_sweep(a, rdtime + b, c, var * rdtime + rhs)
    new_var = _tridiagonal_back_substitution(c_prime, d_prime)
    return tend + (new_var - var) * rdtime


@gtx.field_operator
def _solve_vertical_diffusion_edges(
    a: fa.EdgeKField[wpfloat],
    b: fa.EdgeKField[wpfloat],
    c: fa.EdgeKField[wpfloat],
    rhs: fa.EdgeKField[wpfloat],
    var: fa.EdgeKField[wpfloat],
    tend: fa.EdgeKField[wpfloat],
    dtime: wpfloat,
) -> fa.EdgeKField[wpfloat]:
    """
    Implicit vertical diffusion solve on edges (vn diffusion).

    Same math as ``_solve_vertical_diffusion_cells`` with EdgeDim as horizontal
    dimension.
    """
    rdtime = wpfloat("1.0") / dtime
    c_prime, d_prime = _tridiagonal_forward_sweep(a, rdtime + b, c, var * rdtime + rhs)
    new_var = _tridiagonal_back_substitution(c_prime, d_prime)
    return tend + (new_var - var) * rdtime


@gtx.field_operator
def _apply_explicit_vertical_diffusion_cells(
    a: fa.CellKField[wpfloat],
    b: fa.CellKField[wpfloat],
    c: fa.CellKField[wpfloat],
    rhs: fa.CellKField[wpfloat],
    var: fa.CellKField[wpfloat],
    tend: fa.CellKField[wpfloat],
    minlvl: gtx.int32,
    maxlvl: gtx.int32,
) -> fa.CellKField[wpfloat]:
    """
    Explicit vertical diffusion tendency on cells.

    Port of 'diffuse_vertical_explicit' (mo_tmx_numerics.f90). The matrix
    coefficients a, b, c were built for the implicit scheme, hence the signs:
        tend(k) = tend(k) - a(k)*var(k-1) - b(k)*var(k) - c(k)*var(k+1) + rhs(k)
    The boundary rows drop the term that would read outside the column
    (a(minlvl) and c(maxlvl) are zero anyway, but ``var`` must not be shifted
    past the column bounds).
    """
    interior = tend - a * var(KDim - 1) - b * var - c * var(KDim + 1) + rhs
    top = tend - b * var - c * var(KDim + 1) + rhs
    bottom = tend - a * var(KDim - 1) - b * var + rhs
    new_tend = concat_where(dims.KDim > minlvl, interior, top)
    return concat_where(dims.KDim < maxlvl, new_tend, bottom)
