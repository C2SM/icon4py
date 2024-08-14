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

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat


@field_operator
def _correct_contravariant_vertical_velocity(
    z_w_con_c: fa.CellKField[vpfloat],
    w_concorr_c: fa.CellKField[vpfloat],
) -> fa.CellKField[vpfloat]:
    """Formerly known as _mo_velocity_advection_stencil_13."""
    z_w_con_c_vp = z_w_con_c - w_concorr_c
    return z_w_con_c_vp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def correct_contravariant_vertical_velocity(
    w_concorr_c: fa.CellKField[vpfloat],
    z_w_con_c: fa.CellKField[vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _correct_contravariant_vertical_velocity(
        z_w_con_c,
        w_concorr_c,
        out=z_w_con_c,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
