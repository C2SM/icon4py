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
