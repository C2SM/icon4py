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
from gt4py.next.ffront.fbuiltins import Field

from icon4py.model.common.dimension import CellDim, KDim, Koff


@field_operator
def _calculate_diagnostics_for_turbulence(
    div: Field[[CellDim, KDim], float],
    kh_c: Field[[CellDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
) -> tuple[Field[[CellDim, KDim], float], Field[[CellDim, KDim], float]]:
    div_ic = wgtfac_c * div + (1.0 - wgtfac_c) * div(Koff[-1])
    # TODO(magdalena): change exponent back to int (workaround for gt4py)
    hdef_ic = (wgtfac_c * kh_c + (1.0 - wgtfac_c) * kh_c(Koff[-1])) ** 2
    return div_ic, hdef_ic


@program(grid_type=GridType.UNSTRUCTURED)
def calculate_diagnostics_for_turbulence(
    div: Field[[CellDim, KDim], float],
    kh_c: Field[[CellDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    div_ic: Field[[CellDim, KDim], float],
    hdef_ic: Field[[CellDim, KDim], float],
):
    _calculate_diagnostics_for_turbulence(div, kh_c, wgtfac_c, out=(div_ic[:, 1:], hdef_ic[:, 1:]))
