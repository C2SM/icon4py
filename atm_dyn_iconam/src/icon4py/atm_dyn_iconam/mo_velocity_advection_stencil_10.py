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
def _mo_velocity_advection_stencil_10(
    wgtfac_c: Field[[EdgeDim, KDim], float],
    z_w_concorr_mc: Field[[EdgeDim, KDim], float],
    ) -> Field[[EdgeDim, KDim], float]:
    w_concorr_c = wgtfac_c * z_w_concorr_mc + (1. - wgtfac_c) * z_w_concorr_mc(Koff[-1])

    return w_concorr_c

@program
def mo_velocity_advection_stencil_10(
    wgtfac_c: Field[[EdgeDim, KDim], float],
    z_w_concorr_mc: Field[[EdgeDim, KDim], float],
    w_concorr_c: Field[[EdgeDim, KDim], float],
):
    _mo_velocity_advection_stencil_10(
        wgtfac_c, z_w_concorr_mc, out=w_concorr_c
    )
