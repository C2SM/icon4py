# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Final

import gt4py.next as gtx
from gt4py.next import astype

from icon4py.model.common import constants, dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


dycore_consts: Final = constants.PhysicsConstants()


@gtx.field_operator
def _add_temporal_tendencies_to_vn(
    current_vn: fa.EdgeKField[wpfloat],
    predictor_normal_wind_advective_tendency: fa.EdgeKField[vpfloat],
    normal_wind_tendency_due_to_slow_physics_process: fa.EdgeKField[vpfloat],
    theta_v_at_edges_on_model_levels: fa.EdgeKField[wpfloat],
    horizontal_pressure_gradient: fa.EdgeKField[vpfloat],
    dtime: wpfloat,
) -> fa.EdgeKField[wpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_24."""
    z_gradh_exner_wp = astype(horizontal_pressure_gradient, wpfloat)

    vn_nnew_wp = current_vn + dtime * (
        astype(predictor_normal_wind_advective_tendency, wpfloat)
        - dycore_consts.cpd * theta_v_at_edges_on_model_levels * z_gradh_exner_wp
        + astype(normal_wind_tendency_due_to_slow_physics_process, wpfloat)
    )
    return vn_nnew_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def add_temporal_tendencies_to_vn(
    current_vn: fa.EdgeKField[wpfloat],
    predictor_normal_wind_advective_tendency: fa.EdgeKField[vpfloat],
    normal_wind_tendency_due_to_slow_physics_process: fa.EdgeKField[vpfloat],
    theta_v_at_edges_on_model_levels: fa.EdgeKField[wpfloat],
    horizontal_pressure_gradient: fa.EdgeKField[vpfloat],
    vn_nnew: fa.EdgeKField[wpfloat],
    dtime: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _add_temporal_tendencies_to_vn(
        current_vn,
        predictor_normal_wind_advective_tendency,
        normal_wind_tendency_due_to_slow_physics_process,
        theta_v_at_edges_on_model_levels,
        horizontal_pressure_gradient,
        dtime,
        out=vn_nnew,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
