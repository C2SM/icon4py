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
from icon4py.model.common.dimension import C2CECEC, C2E2C2E2C, CECECDim


@field_operator
def _recon_lsq_cell_c_svd_stencil(
    p_cc: fa.CKfloatField,
    lsq_pseudoinv_1: Field[[CECECDim], float],
    lsq_pseudoinv_2: Field[[CECECDim], float],
    lsq_pseudoinv_3: Field[[CECECDim], float],
    lsq_pseudoinv_4: Field[[CECECDim], float],
    lsq_pseudoinv_5: Field[[CECECDim], float],
    lsq_pseudoinv_6: Field[[CECECDim], float],
    lsq_pseudoinv_7: Field[[CECECDim], float],
    lsq_pseudoinv_8: Field[[CECECDim], float],
    lsq_pseudoinv_9: Field[[CECECDim], float],
    lsq_moments_1: fa.CfloatField,
    lsq_moments_2: fa.CfloatField,
    lsq_moments_3: fa.CfloatField,
    lsq_moments_4: fa.CfloatField,
    lsq_moments_5: fa.CfloatField,
    lsq_moments_6: fa.CfloatField,
    lsq_moments_7: fa.CfloatField,
    lsq_moments_8: fa.CfloatField,
    lsq_moments_9: fa.CfloatField,
) -> tuple[
    fa.CKfloatField,
    fa.CKfloatField,
    fa.CKfloatField,
    fa.CKfloatField,
    fa.CKfloatField,
    fa.CKfloatField,
    fa.CKfloatField,
    fa.CKfloatField,
    fa.CKfloatField,
    fa.CKfloatField,
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
def recon_lsq_cell_c_svd_stencil(
    p_cc: fa.CKfloatField,
    lsq_pseudoinv_1: Field[[CECECDim], float],
    lsq_pseudoinv_2: Field[[CECECDim], float],
    lsq_pseudoinv_3: Field[[CECECDim], float],
    lsq_pseudoinv_4: Field[[CECECDim], float],
    lsq_pseudoinv_5: Field[[CECECDim], float],
    lsq_pseudoinv_6: Field[[CECECDim], float],
    lsq_pseudoinv_7: Field[[CECECDim], float],
    lsq_pseudoinv_8: Field[[CECECDim], float],
    lsq_pseudoinv_9: Field[[CECECDim], float],
    lsq_moments_1: fa.CfloatField,
    lsq_moments_2: fa.CfloatField,
    lsq_moments_3: fa.CfloatField,
    lsq_moments_4: fa.CfloatField,
    lsq_moments_5: fa.CfloatField,
    lsq_moments_6: fa.CfloatField,
    lsq_moments_7: fa.CfloatField,
    lsq_moments_8: fa.CfloatField,
    lsq_moments_9: fa.CfloatField,
    p_coeff_1_dsl: fa.CKfloatField,
    p_coeff_2_dsl: fa.CKfloatField,
    p_coeff_3_dsl: fa.CKfloatField,
    p_coeff_4_dsl: fa.CKfloatField,
    p_coeff_5_dsl: fa.CKfloatField,
    p_coeff_6_dsl: fa.CKfloatField,
    p_coeff_7_dsl: fa.CKfloatField,
    p_coeff_8_dsl: fa.CKfloatField,
    p_coeff_9_dsl: fa.CKfloatField,
    p_coeff_10_dsl: fa.CKfloatField,
):
    _recon_lsq_cell_c_svd_stencil(
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
    )
