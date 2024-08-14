# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import minimum, where

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import E2C


@field_operator
def _hflx_limiter_mo_stencil_04(
    z_anti: fa.EdgeKField[float],
    r_m: fa.CellKField[float],
    r_p: fa.CellKField[float],
    z_mflx_low: fa.EdgeKField[float],
) -> fa.EdgeKField[float]:
    r_frac = where(
        z_anti >= 0.0,
        minimum(r_m(E2C[0]), r_p(E2C[1])),
        minimum(r_m(E2C[1]), r_p(E2C[0])),
    )
    return z_mflx_low + minimum(1.0, r_frac) * z_anti


@program
def hflx_limiter_mo_stencil_04(
    z_anti: fa.EdgeKField[float],
    r_m: fa.CellKField[float],
    r_p: fa.CellKField[float],
    z_mflx_low: fa.EdgeKField[float],
    p_mflx_tracer_h: fa.EdgeKField[float],
):
    _hflx_limiter_mo_stencil_04(z_anti, r_m, r_p, z_mflx_low, out=p_mflx_tracer_h)
