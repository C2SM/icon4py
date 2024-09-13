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
from icon4py.model.common.dimension import C2CECEC, C2E2C2E2C
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _reconstruct_cubic_coefficients_svd(
    p_cc: fa.CellKField[wpfloat],
    lsq_pseudoinv_1: Field[[dims.CECECDim], wpfloat],
    lsq_pseudoinv_2: Field[[dims.CECECDim], wpfloat],
    lsq_pseudoinv_3: Field[[dims.CECECDim], wpfloat],
    lsq_pseudoinv_4: Field[[dims.CECECDim], wpfloat],
    lsq_pseudoinv_5: Field[[dims.CECECDim], wpfloat],
    lsq_pseudoinv_6: Field[[dims.CECECDim], wpfloat],
    lsq_pseudoinv_7: Field[[dims.CECECDim], wpfloat],
    lsq_pseudoinv_8: Field[[dims.CECECDim], wpfloat],
    lsq_pseudoinv_9: Field[[dims.CECECDim], wpfloat],
    lsq_moments_1: fa.CellField[wpfloat],
    lsq_moments_2: fa.CellField[wpfloat],
    lsq_moments_3: fa.CellField[wpfloat],
    lsq_moments_4: fa.CellField[wpfloat],
    lsq_moments_5: fa.CellField[wpfloat],
    lsq_moments_6: fa.CellField[wpfloat],
    lsq_moments_7: fa.CellField[wpfloat],
    lsq_moments_8: fa.CellField[wpfloat],
    lsq_moments_9: fa.CellField[wpfloat],
) -> tuple[
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
]:
    p_coeff_10_dsl = (
        lsq_pseudoinv_9(C2CECEC[0]) * (p_cc(C2E2C2E2C[0]) - p_cc)
        + lsq_pseudoinv_9(C2CECEC[1]) * (p_cc(C2E2C2E2C[1]) - p_cc)
        + lsq_pseudoinv_9(C2CECEC[2]) * (p_cc(C2E2C2E2C[2]) - p_cc)
        + lsq_pseudoinv_9(C2CECEC[3]) * (p_cc(C2E2C2E2C[3]) - p_cc)
        + lsq_pseudoinv_9(C2CECEC[4]) * (p_cc(C2E2C2E2C[4]) - p_cc)
        + lsq_pseudoinv_9(C2CECEC[5]) * (p_cc(C2E2C2E2C[5]) - p_cc)
        + lsq_pseudoinv_9(C2CECEC[6]) * (p_cc(C2E2C2E2C[6]) - p_cc)
        + lsq_pseudoinv_9(C2CECEC[7]) * (p_cc(C2E2C2E2C[7]) - p_cc)
        + lsq_pseudoinv_9(C2CECEC[8]) * (p_cc(C2E2C2E2C[8]) - p_cc)
    )
    p_coeff_9_dsl = (
        lsq_pseudoinv_8(C2CECEC[0]) * (p_cc(C2E2C2E2C[0]) - p_cc)
        + lsq_pseudoinv_8(C2CECEC[1]) * (p_cc(C2E2C2E2C[1]) - p_cc)
        + lsq_pseudoinv_8(C2CECEC[2]) * (p_cc(C2E2C2E2C[2]) - p_cc)
        + lsq_pseudoinv_8(C2CECEC[3]) * (p_cc(C2E2C2E2C[3]) - p_cc)
        + lsq_pseudoinv_8(C2CECEC[4]) * (p_cc(C2E2C2E2C[4]) - p_cc)
        + lsq_pseudoinv_8(C2CECEC[5]) * (p_cc(C2E2C2E2C[5]) - p_cc)
        + lsq_pseudoinv_8(C2CECEC[6]) * (p_cc(C2E2C2E2C[6]) - p_cc)
        + lsq_pseudoinv_8(C2CECEC[7]) * (p_cc(C2E2C2E2C[7]) - p_cc)
        + lsq_pseudoinv_8(C2CECEC[8]) * (p_cc(C2E2C2E2C[8]) - p_cc)
    )
    p_coeff_8_dsl = (
        lsq_pseudoinv_7(C2CECEC[0]) * (p_cc(C2E2C2E2C[0]) - p_cc)
        + lsq_pseudoinv_7(C2CECEC[1]) * (p_cc(C2E2C2E2C[1]) - p_cc)
        + lsq_pseudoinv_7(C2CECEC[2]) * (p_cc(C2E2C2E2C[2]) - p_cc)
        + lsq_pseudoinv_7(C2CECEC[3]) * (p_cc(C2E2C2E2C[3]) - p_cc)
        + lsq_pseudoinv_7(C2CECEC[4]) * (p_cc(C2E2C2E2C[4]) - p_cc)
        + lsq_pseudoinv_7(C2CECEC[5]) * (p_cc(C2E2C2E2C[5]) - p_cc)
        + lsq_pseudoinv_7(C2CECEC[6]) * (p_cc(C2E2C2E2C[6]) - p_cc)
        + lsq_pseudoinv_7(C2CECEC[7]) * (p_cc(C2E2C2E2C[7]) - p_cc)
        + lsq_pseudoinv_7(C2CECEC[8]) * (p_cc(C2E2C2E2C[8]) - p_cc)
    )
    p_coeff_7_dsl = (
        lsq_pseudoinv_6(C2CECEC[0]) * (p_cc(C2E2C2E2C[0]) - p_cc)
        + lsq_pseudoinv_6(C2CECEC[1]) * (p_cc(C2E2C2E2C[1]) - p_cc)
        + lsq_pseudoinv_6(C2CECEC[2]) * (p_cc(C2E2C2E2C[2]) - p_cc)
        + lsq_pseudoinv_6(C2CECEC[3]) * (p_cc(C2E2C2E2C[3]) - p_cc)
        + lsq_pseudoinv_6(C2CECEC[4]) * (p_cc(C2E2C2E2C[4]) - p_cc)
        + lsq_pseudoinv_6(C2CECEC[5]) * (p_cc(C2E2C2E2C[5]) - p_cc)
        + lsq_pseudoinv_6(C2CECEC[6]) * (p_cc(C2E2C2E2C[6]) - p_cc)
        + lsq_pseudoinv_6(C2CECEC[7]) * (p_cc(C2E2C2E2C[7]) - p_cc)
        + lsq_pseudoinv_6(C2CECEC[8]) * (p_cc(C2E2C2E2C[8]) - p_cc)
    )
    p_coeff_6_dsl = (
        lsq_pseudoinv_5(C2CECEC[0]) * (p_cc(C2E2C2E2C[0]) - p_cc)
        + lsq_pseudoinv_5(C2CECEC[1]) * (p_cc(C2E2C2E2C[1]) - p_cc)
        + lsq_pseudoinv_5(C2CECEC[2]) * (p_cc(C2E2C2E2C[2]) - p_cc)
        + lsq_pseudoinv_5(C2CECEC[3]) * (p_cc(C2E2C2E2C[3]) - p_cc)
        + lsq_pseudoinv_5(C2CECEC[4]) * (p_cc(C2E2C2E2C[4]) - p_cc)
        + lsq_pseudoinv_5(C2CECEC[5]) * (p_cc(C2E2C2E2C[5]) - p_cc)
        + lsq_pseudoinv_5(C2CECEC[6]) * (p_cc(C2E2C2E2C[6]) - p_cc)
        + lsq_pseudoinv_5(C2CECEC[7]) * (p_cc(C2E2C2E2C[7]) - p_cc)
        + lsq_pseudoinv_5(C2CECEC[8]) * (p_cc(C2E2C2E2C[8]) - p_cc)
    )
    p_coeff_5_dsl = (
        lsq_pseudoinv_4(C2CECEC[0]) * (p_cc(C2E2C2E2C[0]) - p_cc)
        + lsq_pseudoinv_4(C2CECEC[1]) * (p_cc(C2E2C2E2C[1]) - p_cc)
        + lsq_pseudoinv_4(C2CECEC[2]) * (p_cc(C2E2C2E2C[2]) - p_cc)
        + lsq_pseudoinv_4(C2CECEC[3]) * (p_cc(C2E2C2E2C[3]) - p_cc)
        + lsq_pseudoinv_4(C2CECEC[4]) * (p_cc(C2E2C2E2C[4]) - p_cc)
        + lsq_pseudoinv_4(C2CECEC[5]) * (p_cc(C2E2C2E2C[5]) - p_cc)
        + lsq_pseudoinv_4(C2CECEC[6]) * (p_cc(C2E2C2E2C[6]) - p_cc)
        + lsq_pseudoinv_4(C2CECEC[7]) * (p_cc(C2E2C2E2C[7]) - p_cc)
        + lsq_pseudoinv_4(C2CECEC[8]) * (p_cc(C2E2C2E2C[8]) - p_cc)
    )
    p_coeff_4_dsl = (
        lsq_pseudoinv_3(C2CECEC[0]) * (p_cc(C2E2C2E2C[0]) - p_cc)
        + lsq_pseudoinv_3(C2CECEC[1]) * (p_cc(C2E2C2E2C[1]) - p_cc)
        + lsq_pseudoinv_3(C2CECEC[2]) * (p_cc(C2E2C2E2C[2]) - p_cc)
        + lsq_pseudoinv_3(C2CECEC[3]) * (p_cc(C2E2C2E2C[3]) - p_cc)
        + lsq_pseudoinv_3(C2CECEC[4]) * (p_cc(C2E2C2E2C[4]) - p_cc)
        + lsq_pseudoinv_3(C2CECEC[5]) * (p_cc(C2E2C2E2C[5]) - p_cc)
        + lsq_pseudoinv_3(C2CECEC[6]) * (p_cc(C2E2C2E2C[6]) - p_cc)
        + lsq_pseudoinv_3(C2CECEC[7]) * (p_cc(C2E2C2E2C[7]) - p_cc)
        + lsq_pseudoinv_3(C2CECEC[8]) * (p_cc(C2E2C2E2C[8]) - p_cc)
    )
    p_coeff_3_dsl = (
        lsq_pseudoinv_2(C2CECEC[0]) * (p_cc(C2E2C2E2C[0]) - p_cc)
        + lsq_pseudoinv_2(C2CECEC[1]) * (p_cc(C2E2C2E2C[1]) - p_cc)
        + lsq_pseudoinv_2(C2CECEC[2]) * (p_cc(C2E2C2E2C[2]) - p_cc)
        + lsq_pseudoinv_2(C2CECEC[3]) * (p_cc(C2E2C2E2C[3]) - p_cc)
        + lsq_pseudoinv_2(C2CECEC[4]) * (p_cc(C2E2C2E2C[4]) - p_cc)
        + lsq_pseudoinv_2(C2CECEC[5]) * (p_cc(C2E2C2E2C[5]) - p_cc)
        + lsq_pseudoinv_2(C2CECEC[6]) * (p_cc(C2E2C2E2C[6]) - p_cc)
        + lsq_pseudoinv_2(C2CECEC[7]) * (p_cc(C2E2C2E2C[7]) - p_cc)
        + lsq_pseudoinv_2(C2CECEC[8]) * (p_cc(C2E2C2E2C[8]) - p_cc)
    )
    p_coeff_2_dsl = (
        lsq_pseudoinv_1(C2CECEC[0]) * (p_cc(C2E2C2E2C[0]) - p_cc)
        + lsq_pseudoinv_1(C2CECEC[1]) * (p_cc(C2E2C2E2C[1]) - p_cc)
        + lsq_pseudoinv_1(C2CECEC[2]) * (p_cc(C2E2C2E2C[2]) - p_cc)
        + lsq_pseudoinv_1(C2CECEC[3]) * (p_cc(C2E2C2E2C[3]) - p_cc)
        + lsq_pseudoinv_1(C2CECEC[4]) * (p_cc(C2E2C2E2C[4]) - p_cc)
        + lsq_pseudoinv_1(C2CECEC[5]) * (p_cc(C2E2C2E2C[5]) - p_cc)
        + lsq_pseudoinv_1(C2CECEC[6]) * (p_cc(C2E2C2E2C[6]) - p_cc)
        + lsq_pseudoinv_1(C2CECEC[7]) * (p_cc(C2E2C2E2C[7]) - p_cc)
        + lsq_pseudoinv_1(C2CECEC[8]) * (p_cc(C2E2C2E2C[8]) - p_cc)
    )

    p_coeff_1_dsl = p_cc - (
        p_coeff_2_dsl * lsq_moments_1
        + p_coeff_3_dsl * lsq_moments_2
        + p_coeff_4_dsl * lsq_moments_3
        + p_coeff_5_dsl * lsq_moments_4
        + p_coeff_6_dsl * lsq_moments_5
        + p_coeff_7_dsl * lsq_moments_6
        + p_coeff_8_dsl * lsq_moments_7
        + p_coeff_9_dsl * lsq_moments_8
        + p_coeff_10_dsl * lsq_moments_9
    )

    return (
        p_coeff_1_dsl,
        p_coeff_2_dsl,
        p_coeff_3_dsl,
        p_coeff_4_dsl,
        p_coeff_5_dsl,
        p_coeff_6_dsl,
        p_coeff_7_dsl,
        p_coeff_8_dsl,
        p_coeff_9_dsl,
        p_coeff_10_dsl,
    )


@program(grid_type=GridType.UNSTRUCTURED)
def reconstruct_cubic_coefficients_svd(
    p_cc: fa.CellKField[wpfloat],
    lsq_pseudoinv_1: Field[[dims.CECECDim], wpfloat],
    lsq_pseudoinv_2: Field[[dims.CECECDim], wpfloat],
    lsq_pseudoinv_3: Field[[dims.CECECDim], wpfloat],
    lsq_pseudoinv_4: Field[[dims.CECECDim], wpfloat],
    lsq_pseudoinv_5: Field[[dims.CECECDim], wpfloat],
    lsq_pseudoinv_6: Field[[dims.CECECDim], wpfloat],
    lsq_pseudoinv_7: Field[[dims.CECECDim], wpfloat],
    lsq_pseudoinv_8: Field[[dims.CECECDim], wpfloat],
    lsq_pseudoinv_9: Field[[dims.CECECDim], wpfloat],
    lsq_moments_1: fa.CellField[wpfloat],
    lsq_moments_2: fa.CellField[wpfloat],
    lsq_moments_3: fa.CellField[wpfloat],
    lsq_moments_4: fa.CellField[wpfloat],
    lsq_moments_5: fa.CellField[wpfloat],
    lsq_moments_6: fa.CellField[wpfloat],
    lsq_moments_7: fa.CellField[wpfloat],
    lsq_moments_8: fa.CellField[wpfloat],
    lsq_moments_9: fa.CellField[wpfloat],
    p_coeff_1_dsl: fa.CellKField[wpfloat],
    p_coeff_2_dsl: fa.CellKField[wpfloat],
    p_coeff_3_dsl: fa.CellKField[wpfloat],
    p_coeff_4_dsl: fa.CellKField[wpfloat],
    p_coeff_5_dsl: fa.CellKField[wpfloat],
    p_coeff_6_dsl: fa.CellKField[wpfloat],
    p_coeff_7_dsl: fa.CellKField[wpfloat],
    p_coeff_8_dsl: fa.CellKField[wpfloat],
    p_coeff_9_dsl: fa.CellKField[wpfloat],
    p_coeff_10_dsl: fa.CellKField[wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _reconstruct_cubic_coefficients_svd(
        p_cc,
        lsq_pseudoinv_1,
        lsq_pseudoinv_2,
        lsq_pseudoinv_3,
        lsq_pseudoinv_4,
        lsq_pseudoinv_5,
        lsq_pseudoinv_6,
        lsq_pseudoinv_7,
        lsq_pseudoinv_8,
        lsq_pseudoinv_9,
        lsq_moments_1,
        lsq_moments_2,
        lsq_moments_3,
        lsq_moments_4,
        lsq_moments_5,
        lsq_moments_6,
        lsq_moments_7,
        lsq_moments_8,
        lsq_moments_9,
        out=(
            p_coeff_1_dsl,
            p_coeff_2_dsl,
            p_coeff_3_dsl,
            p_coeff_4_dsl,
            p_coeff_5_dsl,
            p_coeff_6_dsl,
            p_coeff_7_dsl,
            p_coeff_8_dsl,
            p_coeff_9_dsl,
            p_coeff_10_dsl,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
