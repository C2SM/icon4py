# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import maximum

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import Koff


@field_operator
def _step_advection_stencil_02(
    p_rhodz_new: fa.CellKField[float],
    p_mflx_contra_v: fa.CellKField[float],
    deepatmo_divzl: fa.KField[float],
    deepatmo_divzu: fa.KField[float],
    p_dtime: float,
) -> fa.CellKField[float]:
    return maximum(0.1 * p_rhodz_new, p_rhodz_new) - p_dtime * (
        p_mflx_contra_v(Koff[1]) * deepatmo_divzl - p_mflx_contra_v * deepatmo_divzu
    )


@program(grid_type=GridType.UNSTRUCTURED)
def step_advection_stencil_02(
    p_rhodz_new: fa.CellKField[float],
    p_mflx_contra_v: fa.CellKField[float],
    deepatmo_divzl: fa.KField[float],
    deepatmo_divzu: fa.KField[float],
    p_dtime: float,
    rhodz_ast2: fa.CellKField[float],
):
    _step_advection_stencil_02(
        p_rhodz_new,
        p_mflx_contra_v,
        deepatmo_divzl,
        deepatmo_divzu,
        p_dtime,
        out=rhodz_ast2,
    )
