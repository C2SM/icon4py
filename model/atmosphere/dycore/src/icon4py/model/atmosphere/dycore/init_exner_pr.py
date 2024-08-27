# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat


@field_operator
def _init_exner_pr(
    exner: fa.CellKField[vpfloat],
    exner_ref: fa.CellKField[vpfloat],
) -> fa.CellKField[vpfloat]:
    exner_pr = exner - exner_ref
    return exner_pr


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def init_exner_pr(
    exner: fa.CellKField[vpfloat],
    exner_ref: fa.CellKField[vpfloat],
    exner_pr: fa.CellKField[vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _init_exner_pr(
        exner,
        exner_ref,
        out=exner_pr,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
