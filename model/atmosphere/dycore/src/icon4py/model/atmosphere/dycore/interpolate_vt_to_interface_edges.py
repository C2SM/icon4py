# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, astype, int32

from icon4py.model.common.dimension import EdgeDim, KDim, Koff
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _interpolate_vt_to_interface_edges(
    wgtfac_e: Field[[EdgeDim, KDim], vpfloat],
    vt: Field[[EdgeDim, KDim], vpfloat],
) -> Field[[EdgeDim, KDim], vpfloat]:
    """Formerly known as _mo_velocity_advection_stencil_03."""
    wgtfac_e_wp, vt_wp = astype((wgtfac_e, vt), wpfloat)

    z_vt_ie_wp = astype(wgtfac_e * vt, wpfloat) + (wpfloat("1.0") - wgtfac_e_wp) * vt_wp(Koff[-1])

    return astype(z_vt_ie_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def interpolate_vt_to_interface_edges(
    wgtfac_e: Field[[EdgeDim, KDim], vpfloat],
    vt: Field[[EdgeDim, KDim], vpfloat],
    z_vt_ie: Field[[EdgeDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _interpolate_vt_to_interface_edges(
        wgtfac_e,
        vt,
        out=z_vt_ie,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
