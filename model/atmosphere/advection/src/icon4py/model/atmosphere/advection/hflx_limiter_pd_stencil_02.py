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

from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, where

from icon4py.model.common.dimension import E2C, CellDim, EdgeDim, KDim


@field_operator
def _hflx_limiter_pd_stencil_02(
    r_m: Field[[CellDim, KDim], float],
    p_mflx_tracer_h: Field[[EdgeDim, KDim], float],
) -> Field[[EdgeDim, KDim], float]:
    p_mflx_tracer_h_out = where(
        p_mflx_tracer_h >= 0.0,
        p_mflx_tracer_h * r_m(E2C[0]),
        p_mflx_tracer_h * r_m(E2C[1]),
    )
    return p_mflx_tracer_h_out


@program
def hflx_limiter_pd_stencil_02(
    r_m: Field[[CellDim, KDim], float],
    p_mflx_tracer_h: Field[[EdgeDim, KDim], float],
):
    _hflx_limiter_pd_stencil_02(
        r_m,
        p_mflx_tracer_h,
        out=p_mflx_tracer_h,
    )
