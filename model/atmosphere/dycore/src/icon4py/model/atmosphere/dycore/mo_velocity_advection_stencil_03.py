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

from icon4py.model.common.dimension import EdgeDim, KDim, Koff


@field_operator
def _mo_velocity_advection_stencil_03(
    wgtfac_e: Field[[EdgeDim, KDim], float],
    vt: Field[[EdgeDim, KDim], float],
) -> Field[[EdgeDim, KDim], float]:
    z_vt_ie = wgtfac_e * vt + (1.0 - wgtfac_e) * vt(Koff[-1])

    return z_vt_ie


@program(grid_type=GridType.UNSTRUCTURED)
def mo_velocity_advection_stencil_03(
    wgtfac_e: Field[[EdgeDim, KDim], float],
    vt: Field[[EdgeDim, KDim], float],
    z_vt_ie: Field[[EdgeDim, KDim], float],
):
    _mo_velocity_advection_stencil_03(
        wgtfac_e,
        vt,
        out=z_vt_ie[:, 1:],
    )
