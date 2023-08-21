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

from gt4py.next.common import Field
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.common.dimension import EdgeDim, KDim


@field_operator
def _upwind_hflux_miura_cycl_stencil_03b(
    z_tracer_mflx_1_dsl: Field[[EdgeDim, KDim], float],
    z_tracer_mflx_2_dsl: Field[[EdgeDim, KDim], float],
    z_tracer_mflx_3_dsl: Field[[EdgeDim, KDim], float],
) -> Field[[EdgeDim, KDim], float]:
    p_out_e = (z_tracer_mflx_1_dsl + z_tracer_mflx_2_dsl + z_tracer_mflx_3_dsl) / float(
        3
    )
    return p_out_e


@program
def upwind_hflux_miura_cycl_stencil_03b(
    z_tracer_mflx_1_dsl: Field[[EdgeDim, KDim], float],
    z_tracer_mflx_2_dsl: Field[[EdgeDim, KDim], float],
    z_tracer_mflx_3_dsl: Field[[EdgeDim, KDim], float],
    p_out_e: Field[[EdgeDim, KDim], float],
):
    _upwind_hflux_miura_cycl_stencil_03b(
        z_tracer_mflx_1_dsl, z_tracer_mflx_2_dsl, z_tracer_mflx_3_dsl, out=(p_out_e)
    )
