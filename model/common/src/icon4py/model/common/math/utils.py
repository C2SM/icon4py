# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
General-purpose mathematical utility functions.

Contains typed wrappers around standard math operations for use in factories
and validation, and general-purpose GT4Py field operators.
"""

from gt4py import next as gtx
from gt4py.next import where

from icon4py.model.common import dimension as dims, field_type_aliases as fa


@gtx.field_operator
def invert_edge_field(f: fa.EdgeField[gtx.float64]) -> fa.EdgeField[gtx.float64]:
    """
    Invert values.
    Args:
        f: values

    Returns:
        1/f where f is not zero.
    """
    return where(f != 0.0, 1.0 / f, f)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_inverse_on_edges(
    f: fa.EdgeField[gtx.float64],
    f_inverse: fa.EdgeField[gtx.float64],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    invert_edge_field(f, out=f_inverse, domain={dims.EdgeDim: (horizontal_start, horizontal_end)})


@gtx.field_operator
def _compute_inverse_on_cell_khalf(
    f: fa.CellKHalfField[gtx.float64],
) -> fa.CellKHalfField[gtx.float64]:
    return where(f != 0.0, 1.0 / f, f)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_inverse_on_cell_khalf(  # noqa: PLR0917 [too-many-positional-arguments]
    f: fa.CellKHalfField[gtx.float64],
    f_inverse: fa.CellKHalfField[gtx.float64],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_inverse_on_cell_khalf(
        f,
        out=f_inverse,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KHalfDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _compute_inverse_on_edge_k(f: fa.EdgeKField[gtx.float64]) -> fa.EdgeKField[gtx.float64]:
    return where(f != 0.0, 1.0 / f, f)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_inverse_on_edge_k(  # noqa: PLR0917 [too-many-positional-arguments]
    f: fa.EdgeKField[gtx.float64],
    f_inverse: fa.EdgeKField[gtx.float64],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_inverse_on_edge_k(
        f,
        out=f_inverse,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
