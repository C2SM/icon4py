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

from functional.common import Field
from functional.ffront.decorator import field_operator, program

from icon4py.common.dimension import C2CEC, C2E2C, CECDim, CellDim, KDim


@field_operator
def _upwind_hflux_miura_stencil_02(
    p_cc: Field[[CellDim, KDim], float],
    lsq_pseudoinv_1: Field[[CECDim], float],
    lsq_pseudoinv_2: Field[[CECDim], float],
) -> tuple[Field[[CellDim, KDim], float], Field[[CellDim, KDim], float], Field[[CellDim, KDim], float]]:
    p_coeff_1 = p_cc
    p_coeff_2 = (
        lsq_pseudoinv_1(C2CEC[0]) * (p_cc(C2E2C[0]) - p_cc)
        + lsq_pseudoinv_1(C2CEC[1]) * (p_cc(C2E2C[1]) - p_cc)
        + lsq_pseudoinv_1(C2CEC[2]) * (p_cc(C2E2C[2]) - p_cc)
    )
    p_coeff_3 = (
        lsq_pseudoinv_2(C2CEC[0]) * (p_cc(C2E2C[0]) - p_cc)
        + lsq_pseudoinv_2(C2CEC[1]) * (p_cc(C2E2C[1]) - p_cc)
        + lsq_pseudoinv_2(C2CEC[2]) * (p_cc(C2E2C[2]) - p_cc)
    )
    return p_coeff_1, p_coeff_2, p_coeff_3


@program
def upwind_hflux_miura_stencil_02(
    p_cc: Field[[CellDim, KDim], float],
    lsq_pseudoinv_1: Field[[CECDim], float],
    lsq_pseudoinv_2: Field[[CECDim], float],
    p_coeff_1: Field[[CellDim, KDim], float],
    p_coeff_2: Field[[CellDim, KDim], float],
    p_coeff_3: Field[[CellDim, KDim], float],
):
    _upwind_hflux_miura_stencil_02(
        p_cc, lsq_pseudoinv_1, lsq_pseudoinv_2, out=(p_coeff_1, p_coeff_2, p_coeff_3)
    )
