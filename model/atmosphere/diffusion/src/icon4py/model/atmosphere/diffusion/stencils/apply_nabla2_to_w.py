# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, astype, int32, neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2E2CO, C2E2CODim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


# TODO: this will have to be removed once domain allows for imports
CellDim = dims.CellDim
KDim = dims.KDim


@field_operator
def _apply_nabla2_to_w(
    area: fa.CellField[wpfloat],
    z_nabla2_c: fa.CellKField[vpfloat],
    geofac_n2s: Field[[dims.CellDim, C2E2CODim], wpfloat],
    w: fa.CellKField[wpfloat],
    diff_multfac_w: wpfloat,
) -> fa.CellKField[wpfloat]:
    z_nabla2_c_wp = astype(z_nabla2_c, wpfloat)

    w_wp = w - diff_multfac_w * (area * area) * neighbor_sum(
        z_nabla2_c_wp(C2E2CO) * geofac_n2s, axis=C2E2CODim
    )
    return w_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def apply_nabla2_to_w(
    area: fa.CellField[wpfloat],
    z_nabla2_c: fa.CellKField[vpfloat],
    geofac_n2s: Field[[dims.CellDim, C2E2CODim], wpfloat],
    w: fa.CellKField[wpfloat],
    diff_multfac_w: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _apply_nabla2_to_w(
        area,
        z_nabla2_c,
        geofac_n2s,
        w,
        diff_multfac_w,
        out=w,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
