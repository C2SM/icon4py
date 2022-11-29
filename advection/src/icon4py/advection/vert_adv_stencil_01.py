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
from functional.ffront.fbuiltins import Field

from icon4py.common.dimension import CellDim, KDim, Koff


@field_operator
def _vert_adv_stencil_01(
    tracer_now: Field[[CellDim, KDim], float],
    rhodz_now: Field[[CellDim, KDim], float],
    p_mflx_tracer_v: Field[[CellDim, KDim], float],
    deepatmo_divzl: Field[[KDim], float],
    deepatmo_divzu: Field[[KDim], float],
    rhodz_new: Field[[CellDim, KDim], float],
    p_dtime: float,
) -> Field[[CellDim, KDim], float]:
    tracer_new = (
        tracer_now * rhodz_now
        + p_dtime
        * (p_mflx_tracer_v(Koff[1]) * deepatmo_divzl - p_mflx_tracer_v * deepatmo_divzu)
    ) / rhodz_new

    return tracer_new


@program
def vert_adv_stencil_01(
    tracer_now: Field[[CellDim, KDim], float],
    rhodz_now: Field[[CellDim, KDim], float],
    p_mflx_tracer_v: Field[[CellDim, KDim], float],
    deepatmo_divzl: Field[[KDim], float],
    deepatmo_divzu: Field[[KDim], float],
    rhodz_new: Field[[CellDim, KDim], float],
    tracer_new: Field[[CellDim, KDim], float],
    p_dtime: float,
):
    _vert_adv_stencil_01(
        tracer_now,
        rhodz_now,
        p_mflx_tracer_v,
        deepatmo_divzl,
        deepatmo_divzu,
        rhodz_new,
        p_dtime,
        out=tracer_new,
    )
