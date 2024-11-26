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
from icon4py.model.common.dimension import Koff
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _extrapolate_at_top(
    wgtfacq_e: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
) -> fa.EdgeKField[vpfloat]:
    """Formerly known as mo_velocity_advection_stencil_06 or mo_solve_nonhydro_stencil_38."""
    wgtfacq_e_wp = astype(wgtfacq_e, wpfloat)

    vn_ie_wp = (
        wgtfacq_e_wp(Koff[-1]) * vn(Koff[-1])
        + wgtfacq_e_wp(Koff[-2]) * vn(Koff[-2])
        + wgtfacq_e_wp(Koff[-3]) * vn(Koff[-3])
    )

    return astype(vn_ie_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def extrapolate_at_top(
    wgtfacq_e: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    vn_ie: fa.EdgeKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _extrapolate_at_top(
        wgtfacq_e,
        vn,
        out=vn_ie,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )