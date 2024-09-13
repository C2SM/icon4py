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
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _hflx_limiter_mo_stencil_05(
    z_anti: fa.EdgeKField[wpfloat],
    z_mflx_low: fa.EdgeKField[wpfloat],
    r_m: fa.CellKField[wpfloat],
    r_p: fa.CellKField[wpfloat],
) -> fa.EdgeKField[wpfloat]:
    z_signum = where((z_anti > wpfloat(0.0)), wpfloat(1.0), -wpfloat(1.0))

    r_frac = wpfloat(0.5) * (
        (wpfloat(1.0) + z_signum) * minimum(r_m(E2C[0]), r_p(E2C[1]))
        + (wpfloat(1.0) - z_signum) * minimum(r_m(E2C[1]), r_p(E2C[0]))
    )

    p_mflx_tracer_h = z_mflx_low + minimum(wpfloat(1.0), r_frac) * z_anti

    return p_mflx_tracer_h


@program
def hflx_limiter_mo_stencil_05(
    z_anti: fa.EdgeKField[wpfloat],
    z_mflx_low: fa.EdgeKField[wpfloat],
    r_m: fa.CellKField[wpfloat],
    r_p: fa.CellKField[wpfloat],
    p_mflx_tracer_h: fa.EdgeKField[wpfloat],
):
    _hflx_limiter_mo_stencil_05(z_anti, z_mflx_low, r_m, r_p, out=p_mflx_tracer_h)
