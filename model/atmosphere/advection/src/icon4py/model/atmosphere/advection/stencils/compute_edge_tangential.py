# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, int32, neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C2E
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import wpfloat


# TODO (dastrm): this stencil is a duplicate of compute_tangential_wind
# TODO (dastrm): this stencil has no test


@field_operator
def _compute_edge_tangential(
    p_vn_in: fa.EdgeKField[wpfloat],
    ptr_coeff: Field[[dims.EdgeDim, dims.E2C2EDim], wpfloat],
) -> fa.EdgeKField[wpfloat]:
    p_vt_out = neighbor_sum(p_vn_in(E2C2E) * ptr_coeff, axis=dims.E2C2EDim)
    return p_vt_out


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_edge_tangential(
    p_vn_in: fa.EdgeKField[wpfloat],
    ptr_coeff: Field[[dims.EdgeDim, dims.E2C2EDim], wpfloat],
    p_vt_out: fa.EdgeKField[wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_edge_tangential(
        p_vn_in,
        ptr_coeff,
        out=p_vt_out,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
