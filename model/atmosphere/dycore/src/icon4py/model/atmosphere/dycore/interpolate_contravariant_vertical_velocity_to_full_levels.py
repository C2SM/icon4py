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
from icon4py.model.common.dimension import CellDim, KDim, Koff
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat


@field_operator
def _interpolate_contravariant_vertical_velocity_to_full_levels(
    z_w_con_c: fa.CellKField[vpfloat],
) -> fa.CellKField[vpfloat]:
    """Formerly know as _mo_velocity_advection_stencil_15."""
    z_w_con_c_full_vp = vpfloat("0.5") * (z_w_con_c + z_w_con_c(Koff[1]))
    return z_w_con_c_full_vp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def interpolate_contravariant_vertical_velocity_to_full_levels(
    z_w_con_c: fa.CellKField[vpfloat],
    z_w_con_c_full: fa.CellKField[vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _interpolate_contravariant_vertical_velocity_to_full_levels(
        z_w_con_c,
        out=z_w_con_c_full,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
