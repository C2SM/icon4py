# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _average_horizontal_flux_subcycling_2(
    z_tracer_mflx_1_dsl: fa.EdgeKField[wpfloat],
    z_tracer_mflx_2_dsl: fa.EdgeKField[wpfloat],
) -> fa.EdgeKField[wpfloat]:
    p_out_e = (z_tracer_mflx_1_dsl + z_tracer_mflx_2_dsl) / wpfloat(2.0)
    return p_out_e


@program(grid_type=GridType.UNSTRUCTURED)
def average_horizontal_flux_subcycling_2(
    z_tracer_mflx_1_dsl: fa.EdgeKField[wpfloat],
    z_tracer_mflx_2_dsl: fa.EdgeKField[wpfloat],
    p_out_e: fa.EdgeKField[wpfloat],
):
    _average_horizontal_flux_subcycling_2(z_tracer_mflx_1_dsl, z_tracer_mflx_2_dsl, out=(p_out_e))
