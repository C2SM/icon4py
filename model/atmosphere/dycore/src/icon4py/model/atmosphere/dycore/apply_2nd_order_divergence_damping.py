# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import astype, int32

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _apply_2nd_order_divergence_damping(
    z_graddiv_vn: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    scal_divdamp_o2: wpfloat,
) -> fa.EdgeKField[wpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_26."""
    z_graddiv_vn_wp = astype(z_graddiv_vn, wpfloat)

    vn_wp = vn + (scal_divdamp_o2 * z_graddiv_vn_wp)
    return vn_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def apply_2nd_order_divergence_damping(
    z_graddiv_vn: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    scal_divdamp_o2: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _apply_2nd_order_divergence_damping(
        z_graddiv_vn,
        vn,
        scal_divdamp_o2,
        out=vn,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
