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
def _mo_solve_nonhydro_stencil_38(
    wgtfacq_e: Field[[EdgeDim, KDim], float],
    vn: Field[[EdgeDim, KDim], float],
) -> Field[[EdgeDim, KDim], float]:
    vn_ie = (
        wgtfacq_e(Koff[-1]) * vn(Koff[-1])
        + wgtfacq_e(Koff[-2]) * vn(Koff[-2])
        + wgtfacq_e(Koff[-3]) * vn(Koff[-3])
    )
    return vn_ie


@program
def mo_solve_nonhydro_stencil_38(
    wgtfac_e: Field[[EdgeDim, KDim], float],
    vn: Field[[EdgeDim, KDim], float],
    vn_ie: Field[[EdgeDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_38(wgtfac_e, vn, out=vn_ie)
