# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype
from gt4py.next.experimental import concat_where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _interpolate_to_half_levels(
    wgtfac_e: fa.EdgeKHalfField[ta.vpfloat],
    x: fa.EdgeKField[ta.wpfloat],
) -> fa.EdgeKHalfField[ta.vpfloat]:
    wgtfac_e_wp = astype(wgtfac_e, wpfloat)
    x_ie_wp = wgtfac_e_wp * x(dims.KHalfDim + 0.5) + (wpfloat("1.0") - wgtfac_e_wp) * x(
        dims.KHalfDim - 0.5
    )
    return concat_where(
        dims.KHalfDim > 0, astype(x_ie_wp, vpfloat), astype(x(dims.KHalfDim + 0.5), vpfloat)
    )


@gtx.field_operator
def _compute_horizontal_kinetic_energy(
    vn: fa.EdgeKField[ta.wpfloat],
    vt: fa.EdgeKField[ta.vpfloat],
) -> fa.EdgeKField[ta.vpfloat]:
    z_kin_hor_e_wp = wpfloat("0.5") * (vn * vn + astype(vt * vt, wpfloat))
    return astype(z_kin_hor_e_wp, vpfloat)
