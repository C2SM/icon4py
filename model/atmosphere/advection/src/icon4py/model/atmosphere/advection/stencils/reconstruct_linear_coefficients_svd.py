# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import C2CEC, C2E2C
from icon4py.model.common.settings import backend


# TODO (dastrm): this stencil has no test
# TODO (dastrm): move to common


@gtx.field_operator
def _reconstruct_linear_coefficients_svd(
    p_cc: fa.CellKField[ta.wpfloat],
    lsq_pseudoinv_1: gtx.Field[gtx.Dims[dims.CECDim], ta.wpfloat],
    lsq_pseudoinv_2: gtx.Field[gtx.Dims[dims.CECDim], ta.wpfloat],
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
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


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def reconstruct_linear_coefficients_svd(
    p_cc: fa.CellKField[ta.wpfloat],
    lsq_pseudoinv_1: gtx.Field[gtx.Dims[dims.CECDim], ta.wpfloat],
    lsq_pseudoinv_2: gtx.Field[gtx.Dims[dims.CECDim], ta.wpfloat],
    p_coeff_1_dsl: fa.CellKField[ta.wpfloat],
    p_coeff_2_dsl: fa.CellKField[ta.wpfloat],
    p_coeff_3_dsl: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
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
