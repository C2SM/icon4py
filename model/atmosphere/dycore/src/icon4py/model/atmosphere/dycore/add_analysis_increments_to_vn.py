# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import astype

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _add_analysis_increments_to_vn(
    vn_incr: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    iau_wgt_dyn: wpfloat,
) -> fa.EdgeKField[wpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_28."""
    vn_incr_wp = astype(vn_incr, wpfloat)

    vn_wp = vn + (iau_wgt_dyn * vn_incr_wp)
    return vn_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def add_analysis_increments_to_vn(
    vn_incr: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    iau_wgt_dyn: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _add_analysis_increments_to_vn(
        vn_incr,
        vn,
        iau_wgt_dyn,
        out=vn,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
