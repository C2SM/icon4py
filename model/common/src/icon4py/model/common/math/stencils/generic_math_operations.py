# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.math.operators import (
    _compute_difference_on_cell_k,
    _compute_field_a_plus_coeff_times_field_b_on_cell_k,
    _copy_field_on_cell_k,
)


@gtx.program
def compute_difference_on_cell_k(
    field_a: fa.CellKField[ta.wpfloat],
    field_b: fa.CellKField[ta.vpfloat],
    output_field: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_difference_on_cell_k(
        field_a,
        field_b,
        out=output_field,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.program
def compute_field_a_plus_coeff_times_field_b_on_cell_k(
    field_a: fa.CellKField[ta.wpfloat],
    coeff: ta.wpfloat,
    field_b: fa.CellKField[ta.wpfloat],
    output_field: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_field_a_plus_coeff_times_field_b_on_cell_k(
        field_a,
        coeff,
        field_b,
        out=output_field,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.program
def copy_field_on_cell_k(
    field: fa.CellKField[ta.wpfloat],
    output_field: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _copy_field_on_cell_k(
        field,
        out=output_field,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
