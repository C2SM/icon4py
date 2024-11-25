# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import neighbor_sum

import icon4py.model.common.test_utils.helpers as test_utils
from icon4py.model.common import dimension as dims
from icon4py.model.common.dimension import C2E, C2EDim
from icon4py.model.common.grid import simple


@gtx.field_operator
def field_op(
    in_field: gtx.Field[gtx.Dims[dims.EdgeDim], float],
    coeff: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], float],
) -> gtx.Field[gtx.Dims[dims.CellDim], float]:
    return neighbor_sum(in_field(C2E) * coeff, axis=C2EDim)


def test_call_field_operator(backend):
    grid = simple.SimpleGrid()
    hstart = 0
    hend = grid.num_cells
    coefficient = test_utils.constant_field(grid, 0.8, dims.CellDim, dims.C2EDim, dtype=float)
    in_field = test_utils.constant_field(grid, 1.0, dims.EdgeDim, dtype=float)
    out_field = test_utils.zero_field(grid, dims.CellDim, dtype=float)
    expected = test_utils.constant_field(grid, 2.4, dims.CellDim, dtype=float)
    field_op.with_backend(backend)(
        in_field=in_field,
        coeff=coefficient,
        out=out_field,
        offset_provider={"C2E": grid.get_offset_provider("C2E")},
        domain={dims.CellDim: (hstart, hend)},
    )
    test_utils.dallclose(out_field.asnumpy(), expected.asnumpy())
