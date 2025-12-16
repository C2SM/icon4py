# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# mypy: ignore-errors
import gt4py.next as gtx

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid.simple import simple_grid
from icon4py.model.common.type_alias import wpfloat
from icon4py.tools.py2fgen.wrappers.icon4py_export import export


grid = simple_grid()


@gtx.field_operator
def _square(
    inp: gtx.Field[[dims.CellDim, dims.KDim], wpfloat],
) -> gtx.Field[[dims.CellDim, dims.KDim], wpfloat]:
    return inp**2


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def square(
    inp: gtx.Field[[dims.CellDim, dims.KDim], wpfloat],
    result: gtx.Field[[dims.CellDim, dims.KDim], wpfloat],
):
    _square(inp, out=result)


@export
def square_from_function(
    inp: gtx.Field[[dims.CellDim, dims.KDim], wpfloat],
    result: gtx.Field[[dims.CellDim, dims.KDim], wpfloat],
):
    square(inp, result, offset_provider={})


@export
def square_error(
    inp: gtx.Field[[dims.CellDim, dims.KDim], wpfloat],
    result: gtx.Field[[dims.CellDim, dims.KDim], wpfloat],
):
    raise Exception("Exception foo occurred")
