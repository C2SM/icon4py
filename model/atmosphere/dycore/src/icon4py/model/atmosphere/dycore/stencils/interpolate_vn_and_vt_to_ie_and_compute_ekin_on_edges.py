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

from icon4py.model.atmosphere.dycore.stencils.interpolate_vn_to_ie_and_compute_ekin_on_edges import (
    _interpolate_vn_to_ie_and_compute_ekin_on_edges,
)
from icon4py.model.atmosphere.dycore.stencils.interpolate_vt_to_interface_edges import (
    _interpolate_vt_to_interface_edges,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _interpolate_vn_and_vt_to_ie_and_compute_ekin_on_edges(
    wgtfac_e: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    vt: fa.EdgeKField[vpfloat],
) -> tuple[
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
]:
    """Formerly known as _mo_solve_nonhydro_stencil_36."""
    z_vt_ie = _interpolate_vt_to_interface_edges(wgtfac_e=wgtfac_e, vt=vt)
    vn_ie, z_kin_hor_e = _interpolate_vn_to_ie_and_compute_ekin_on_edges(
        wgtfac_e=wgtfac_e, vn=vn, vt=vt
    )
    return vn_ie, z_vt_ie, z_kin_hor_e


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def interpolate_vn_and_vt_to_ie_and_compute_ekin_on_edges(
    wgtfac_e: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    vt: fa.EdgeKField[vpfloat],
    vn_ie: fa.EdgeKField[vpfloat],
    z_vt_ie: fa.EdgeKField[vpfloat],
    z_kin_hor_e: fa.EdgeKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _interpolate_vn_and_vt_to_ie_and_compute_ekin_on_edges(
        wgtfac_e,
        vn,
        vt,
        out=(vn_ie, z_vt_ie, z_kin_hor_e),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
