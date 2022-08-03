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

from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Field, FieldOffset

from icon4py.common.dimension import EdgeDim, KDim


Koff = FieldOffset("Koff", source=KDim, target=(KDim,))


@field_operator
def _mo_velocity_advection_stencil_02(
    wgtfac_e: Field[[EdgeDim, KDim], float],
    vn: Field[[EdgeDim, KDim], float],
    vt: Field[[EdgeDim, KDim], float],
) -> tuple[Field[[EdgeDim, KDim], float], Field[[EdgeDim, KDim], float]]:
    vn_ie = wgtfac_e * vn + (1.0 - wgtfac_e) * vn(Koff[-1])
    z_kin_hor_e = 0.5 * (vn * vn + vt * vt)

    return vn_ie, z_kin_hor_e


@program
def mo_velocity_advection_stencil_02(
    wgtfac_e: Field[[EdgeDim, KDim], float],
    vn: Field[[EdgeDim, KDim], float],
    vt: Field[[EdgeDim, KDim], float],
    vn_ie: Field[[EdgeDim, KDim], float],
    z_kin_hor_e: Field[[EdgeDim, KDim], float],
):
    _mo_velocity_advection_stencil_02(wgtfac_e, vn, vt, out=(vn_ie, z_kin_hor_e))
