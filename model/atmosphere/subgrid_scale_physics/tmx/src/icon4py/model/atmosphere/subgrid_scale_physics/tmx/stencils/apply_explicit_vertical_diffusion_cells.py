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
def _apply_explicit_vertical_diffusion_cells_interior(
    a: fa.CellKField[wpfloat],
    b: fa.CellKField[wpfloat],
    c: fa.CellKField[wpfloat],
    rhs: fa.CellKField[wpfloat],
    var: fa.CellKField[wpfloat],
    tend: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """
    Explicit vertical diffusion tendency, interior rows.

    Port of 'diffuse_vertical_explicit' (mo_tmx_numerics.f90). The matrix
    coefficients a, b, c were built for the implicit scheme, hence the signs:
        tend(k) = tend(k) - a(k)*var(k-1) - b(k)*var(k) - c(k)*var(k+1) + rhs(k)
    """
    return tend - a * var(KDim - 1) - b * var - c * var(KDim + 1) + rhs


@gtx.field_operator
def _apply_explicit_vertical_diffusion_cells_top(
    b: fa.CellKField[wpfloat],
    c: fa.CellKField[wpfloat],
    rhs: fa.CellKField[wpfloat],
    var: fa.CellKField[wpfloat],
    tend: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """Upper boundary row (minlvl): no a-term."""
    return tend - b * var - c * var(KDim + 1) + rhs


@gtx.field_operator
def _apply_explicit_vertical_diffusion_cells_bottom(
    a: fa.CellKField[wpfloat],
    b: fa.CellKField[wpfloat],
    rhs: fa.CellKField[wpfloat],
    var: fa.CellKField[wpfloat],
    tend: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """Lower boundary row (maxlvl): no c-term."""
    return tend - a * var(KDim - 1) - b * var + rhs


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def apply_explicit_vertical_diffusion_cells(
    a: fa.CellKField[wpfloat],
    b: fa.CellKField[wpfloat],
    c: fa.CellKField[wpfloat],
    rhs: fa.CellKField[wpfloat],
    var: fa.CellKField[wpfloat],
    tend: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _apply_explicit_vertical_diffusion_cells_interior(
        a=a,
        b=b,
        c=c,
        rhs=rhs,
        var=var,
        tend=tend,
        out=tend,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start + 1, vertical_end - 1),
        },
    )
    _apply_explicit_vertical_diffusion_cells_top(
        b=b,
        c=c,
        rhs=rhs,
        var=var,
        tend=tend,
        out=tend,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_start + 1),
        },
    )
    _apply_explicit_vertical_diffusion_cells_bottom(
        a=a,
        b=b,
        rhs=rhs,
        var=var,
        tend=tend,
        out=tend,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_end - 1, vertical_end),
        },
    )
