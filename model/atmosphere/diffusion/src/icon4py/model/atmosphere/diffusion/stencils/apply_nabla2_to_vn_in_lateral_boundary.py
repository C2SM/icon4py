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
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _apply_nabla2_to_vn_in_lateral_boundary(
    z_nabla2_e: fa.EdgeKField[wpfloat],
    area_edge: fa.EdgeField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
    fac_bdydiff_v: wpfloat,
) -> fa.EdgeKField[wpfloat]:
    vn_wp = vn + (area_edge * fac_bdydiff_v * z_nabla2_e)
    return vn_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def apply_nabla2_to_vn_in_lateral_boundary(
    z_nabla2_e: fa.EdgeKField[wpfloat],
    area_edge: fa.EdgeField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
    fac_bdydiff_v: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _apply_nabla2_to_vn_in_lateral_boundary(
        z_nabla2_e,
        area_edge,
        vn,
        fac_bdydiff_v,
        out=vn,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
