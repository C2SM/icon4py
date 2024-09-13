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
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _apply_vertical_density_increment(
    rhodz_ast: fa.CellKField[wpfloat],
    p_mflx_contra_v: fa.CellKField[wpfloat],
    deepatmo_divzl: fa.KField[wpfloat],
    deepatmo_divzu: fa.KField[wpfloat],
    p_dtime: wpfloat,
) -> fa.CellKField[wpfloat]:
    k_offset_up_low = p_dtime * (
        p_mflx_contra_v(Koff[1]) * deepatmo_divzl - p_mflx_contra_v * deepatmo_divzu
    )
    return rhodz_ast + k_offset_up_low


@program(grid_type=GridType.UNSTRUCTURED)
def apply_vertical_density_increment(
    rhodz_ast: fa.CellKField[wpfloat],
    p_mflx_contra_v: fa.CellKField[wpfloat],
    deepatmo_divzl: fa.KField[wpfloat],
    deepatmo_divzu: fa.KField[wpfloat],
    p_dtime: wpfloat,
    rhodz_ast2: fa.CellKField[wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _apply_vertical_density_increment(
        rhodz_ast,
        p_mflx_contra_v,
        deepatmo_divzl,
        deepatmo_divzu,
        p_dtime,
        out=rhodz_ast2,
        domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )
