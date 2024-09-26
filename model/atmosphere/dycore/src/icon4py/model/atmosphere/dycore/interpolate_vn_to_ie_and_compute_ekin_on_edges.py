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

from icon4py.model.atmosphere.dycore.compute_horizontal_kinetic_energy import (
    _compute_horizontal_kinetic_energy,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _interpolate_vn_to_ie_and_compute_ekin_on_edges(
    wgtfac_e: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    vt: fa.EdgeKField[vpfloat],
) -> tuple[fa.EdgeKField[vpfloat], fa.EdgeKField[vpfloat]]:
    """Formerly known as _mo_velocity_advection_stencil_02."""
    wgtfac_e_wp = astype(wgtfac_e, wpfloat)

    vn_ie_wp = wgtfac_e_wp * vn + (wpfloat("1.0") - wgtfac_e_wp) * vn(Koff[-1])
    _, _, z_kin_hor_e = _compute_horizontal_kinetic_energy(vn=vn, vt=vt)

    return astype(vn_ie_wp, vpfloat), z_kin_hor_e


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def interpolate_vn_to_ie_and_compute_ekin_on_edges(
    wgtfac_e: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    vt: fa.EdgeKField[vpfloat],
    vn_ie: fa.EdgeKField[vpfloat],
    z_kin_hor_e: fa.EdgeKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _interpolate_vn_to_ie_and_compute_ekin_on_edges(
        wgtfac_e,
        vn,
        vt,
        out=(vn_ie, z_kin_hor_e),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
