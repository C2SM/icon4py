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

from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Field, where

from icon4py.common.dimension import CellDim, EdgeDim, KDim, E2C

@field_operator
def _hflx_limiter_pd_stencil_02(
    refin_ctrl: Field[[EdgeDim], float], 
    r_m: Field[[CellDim, KDim], float],
    p_mflx_tracer_h_in: Field[[EdgeDim, KDim], float],
    bound: float,
) -> Field[[EdgeDim, KDim], float]:
    p_mflx_tracer_h_out = where (
        refin_ctrl == bound,
        p_mflx_tracer_h_in,
        where (
            (p_mflx_tracer_h_in > 0.0) | (p_mflx_tracer_h_in==0.0),
            p_mflx_tracer_h_in * r_m(E2C[0]),
            p_mflx_tracer_h_in * r_m(E2C[1]),
        ),
    )
    return p_mflx_tracer_h_out

@program
def hflx_limiter_pd_stencil_02(
    refin_ctrl: Field[[EdgeDim], float],
    r_m: Field[[CellDim, KDim], float],
    p_mflx_tracer_h_in: Field[[EdgeDim, KDim], float],
    p_mflx_tracer_h_out: Field[[EdgeDim, KDim], float],
    bound: float,
):
    _hflx_limiter_pd_stencil_02(
    refin_ctrl,
    r_m,
    p_mflx_tracer_h_in,
    bound,
    out=p_mflx_tracer_h_out,
)
