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
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _interpolate_vt_to_interface_edges(
    wgtfac_e: fa.EdgeKField[vpfloat],
    vt: fa.EdgeKField[vpfloat],
) -> fa.EdgeKField[vpfloat]:
    """Formerly known as _mo_velocity_advection_stencil_03."""
    wgtfac_e_wp, vt_wp = astype((wgtfac_e, vt), wpfloat)

    z_vt_ie_wp = astype(wgtfac_e * vt, wpfloat) + (wpfloat("1.0") - wgtfac_e_wp) * vt_wp(Koff[-1])

    return astype(z_vt_ie_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED)
def interpolate_vt_to_interface_edges(
    wgtfac_e: fa.EdgeKField[vpfloat],
    vt: fa.EdgeKField[vpfloat],
    z_vt_ie: fa.EdgeKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _interpolate_vt_to_interface_edges(
        wgtfac_e,
        vt,
        out=z_vt_ie,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
