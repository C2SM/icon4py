# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import float64

from icon4py.model.common import field_type_aliases as fa


@field_operator
def _upwind_hflux_miura_cycl_stencil_03a(
    z_tracer_mflx_1_dsl: fa.EdgeKField[float],
    z_tracer_mflx_2_dsl: fa.EdgeKField[float],
) -> fa.EdgeKField[float]:
    p_out_e = (z_tracer_mflx_1_dsl + z_tracer_mflx_2_dsl) / float64(2)
    return p_out_e


@program(grid_type=GridType.UNSTRUCTURED)
def upwind_hflux_miura_cycl_stencil_03a(
    z_tracer_mflx_1_dsl: fa.EdgeKField[float],
    z_tracer_mflx_2_dsl: fa.EdgeKField[float],
    p_out_e: fa.EdgeKField[float],
):
    _upwind_hflux_miura_cycl_stencil_03a(z_tracer_mflx_1_dsl, z_tracer_mflx_2_dsl, out=(p_out_e))
