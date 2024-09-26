# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from gt4py.next import gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _compute_vn_on_lateral_boundary(
    grf_tend_vn: fa.EdgeKField[wpfloat],
    vn_now: fa.EdgeKField[wpfloat],
    dtime: wpfloat,
) -> fa.EdgeKField[wpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_29."""
    vn_new_wp = vn_now + dtime * grf_tend_vn
    return vn_new_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_vn_on_lateral_boundary(
    grf_tend_vn: fa.EdgeKField[wpfloat],
    vn_now: fa.EdgeKField[wpfloat],
    vn_new: fa.EdgeKField[wpfloat],
    dtime: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_vn_on_lateral_boundary(
        grf_tend_vn,
        vn_now,
        dtime,
        out=vn_new,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
