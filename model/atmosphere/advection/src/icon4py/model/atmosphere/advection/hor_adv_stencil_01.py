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
from gt4py.next.ffront.fbuiltins import Field, neighbor_sum

from icon4py.model.common.dimension import C2CE, C2E, C2EDim, CEDim, CellDim, EdgeDim, KDim


@field_operator
def _hor_adv_stencil_01(
    p_mflx_tracer_h: Field[[EdgeDim, KDim], float],
    deepatmo_divh: Field[[KDim], float],
    tracer_now: Field[[CellDim, KDim], float],
    rhodz_now: Field[[CellDim, KDim], float],
    rhodz_new: Field[[CellDim, KDim], float],
    geofac_div: Field[[CEDim], float],
    p_dtime: float,
) -> Field[[CellDim, KDim], float]:
    tracer_new_hor = (
        tracer_now * rhodz_now
        - p_dtime
        * deepatmo_divh
        * neighbor_sum(p_mflx_tracer_h(C2E) * geofac_div(C2CE), axis=C2EDim)
    ) / rhodz_new

    return tracer_new_hor


@program
def hor_adv_stencil_01(
    p_mflx_tracer_h: Field[[EdgeDim, KDim], float],
    deepatmo_divh: Field[[KDim], float],
    tracer_now: Field[[CellDim, KDim], float],
    rhodz_now: Field[[CellDim, KDim], float],
    rhodz_new: Field[[CellDim, KDim], float],
    geofac_div: Field[[CEDim], float],
    tracer_new_hor: Field[[CellDim, KDim], float],
    p_dtime: float,
):
    _hor_adv_stencil_01(
        p_mflx_tracer_h,
        deepatmo_divh,
        tracer_now,
        rhodz_now,
        rhodz_new,
        geofac_div,
        p_dtime,
        out=tracer_new_hor,
    )
