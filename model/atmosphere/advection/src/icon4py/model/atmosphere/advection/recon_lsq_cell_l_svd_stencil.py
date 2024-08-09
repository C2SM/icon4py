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

from gt4py.next.common import Field, GridType
from gt4py.next.ffront.decorator import field_operator, program

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import C2CEC, C2E2C, CECDim


@field_operator
def _recon_lsq_cell_l_svd_stencil(
    p_cc: fa.CellKField[float],
    lsq_pseudoinv_1: Field[[CECDim], float],
    lsq_pseudoinv_2: Field[[CECDim], float],
) -> tuple[
    fa.CellKField[float],
    fa.CellKField[float],
    fa.CellKField[float],
]:
    p_coeff_1_dsl = p_cc
    p_coeff_2_dsl = (
        lsq_pseudoinv_1(C2CEC[0]) * (p_cc(C2E2C[0]) - p_cc)
        + lsq_pseudoinv_1(C2CEC[1]) * (p_cc(C2E2C[1]) - p_cc)
        + lsq_pseudoinv_1(C2CEC[2]) * (p_cc(C2E2C[2]) - p_cc)
    )
    p_coeff_3_dsl = (
        lsq_pseudoinv_2(C2CEC[0]) * (p_cc(C2E2C[0]) - p_cc)
        + lsq_pseudoinv_2(C2CEC[1]) * (p_cc(C2E2C[1]) - p_cc)
        + lsq_pseudoinv_2(C2CEC[2]) * (p_cc(C2E2C[2]) - p_cc)
    )
    return p_coeff_1_dsl, p_coeff_2_dsl, p_coeff_3_dsl


@program(grid_type=GridType.UNSTRUCTURED)
def recon_lsq_cell_l_svd_stencil(
    p_cc: fa.CellKField[float],
    lsq_pseudoinv_1: Field[[CECDim], float],
    lsq_pseudoinv_2: Field[[CECDim], float],
    p_coeff_1_dsl: fa.CellKField[float],
    p_coeff_2_dsl: fa.CellKField[float],
    p_coeff_3_dsl: fa.CellKField[float],
):
    _recon_lsq_cell_l_svd_stencil(
        p_cc,
        lsq_pseudoinv_1,
        lsq_pseudoinv_2,
        out=(p_coeff_1_dsl, p_coeff_2_dsl, p_coeff_3_dsl),
    )
