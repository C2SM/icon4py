# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import Field, GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2CEC, C2E2C
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _reconstruct_linear_coefficients_svd(
    p_cc: fa.CellKField[wpfloat],
    lsq_pseudoinv_1: Field[[dims.CECDim], wpfloat],
    lsq_pseudoinv_2: Field[[dims.CECDim], wpfloat],
) -> tuple[
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
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


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def reconstruct_linear_coefficients_svd(
    p_cc: fa.CellKField[wpfloat],
    lsq_pseudoinv_1: Field[[dims.CECDim], wpfloat],
    lsq_pseudoinv_2: Field[[dims.CECDim], wpfloat],
    p_coeff_1_dsl: fa.CellKField[wpfloat],
    p_coeff_2_dsl: fa.CellKField[wpfloat],
    p_coeff_3_dsl: fa.CellKField[wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _reconstruct_linear_coefficients_svd(
        p_cc,
        lsq_pseudoinv_1,
        lsq_pseudoinv_2,
        out=(p_coeff_1_dsl, p_coeff_2_dsl, p_coeff_3_dsl),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
