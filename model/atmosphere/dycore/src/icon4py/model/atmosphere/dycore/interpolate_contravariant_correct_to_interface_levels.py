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
from gt4py.next.ffront.fbuiltins import Field, int32

from icon4py.model.common.dimension import CellDim, KDim, Koff
from icon4py.model.common.type_alias import vpfloat


@field_operator
def _interpolate_contravariant_correct_to_interface_levels(
    z_w_concorr_mc: Field[[CellDim, KDim], vpfloat],
    wgtfac_c: Field[[CellDim, KDim], vpfloat],
) -> Field[[CellDim, KDim], vpfloat]:
    """Formerly know as _mo_velocity_advection_stencil_10."""
    w_concorr_c_vp = wgtfac_c * z_w_concorr_mc + (vpfloat("1.0") - wgtfac_c) * z_w_concorr_mc(
        Koff[-1]
    )

    return w_concorr_c_vp


@program(grid_type=GridType.UNSTRUCTURED)
def interpolate_contravariant_correct_to_interface_levels(
    z_w_concorr_mc: Field[[CellDim, KDim], vpfloat],
    wgtfac_c: Field[[CellDim, KDim], vpfloat],
    w_concorr_c: Field[[CellDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _interpolate_contravariant_correct_to_interface_levels(
        z_w_concorr_mc,
        wgtfac_c,
        out=w_concorr_c,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
