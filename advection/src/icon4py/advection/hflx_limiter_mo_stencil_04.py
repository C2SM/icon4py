# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from functional.common import Field
from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import minimum, where

from icon4py.common.dimension import E2C, CellDim, EdgeDim, KDim


@field_operator
def _hflx_limiter_mo_stencil_04(
    z_anti: Field[[EdgeDim, KDim], float],
    r_m: Field[[CellDim, KDim], float],
    r_p: Field[[CellDim, KDim], float],
    z_mflx_low: Field[[EdgeDim, KDim], float],
) -> Field[[EdgeDim, KDim], float]:
    r_frac = where(
        z_anti >= 0.0,
        minimum(r_m(E2C[0]), r_p(E2C[1])),
        minimum(r_m(E2C[1]), r_p(E2C[0])),
    )
    return z_mflx_low + minimum(1.0, r_frac) * z_anti


@program
def hflx_limiter_mo_stencil_04(
    z_anti: Field[[EdgeDim, KDim], float],
    r_m: Field[[CellDim, KDim], float],
    r_p: Field[[CellDim, KDim], float],
    z_mflx_low: Field[[EdgeDim, KDim], float],
    p_mflx_tracer_h: Field[[EdgeDim, KDim], float],
):
    _hflx_limiter_mo_stencil_04(z_anti, r_m, r_p, z_mflx_low, out=p_mflx_tracer_h)
