# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any, Final

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.dycore.stencils.add_temporal_tendencies_to_vn import (
    add_temporal_tendencies_to_vn,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.stencil_tests import StencilTest


dycore_consts: Final = constants.PhysicsConstants()


def add_temporal_tendencies_to_vn_numpy(
    current_vn: np.ndarray,
    predictor_normal_wind_advective_tendency: np.ndarray,
    normal_wind_tendency_due_to_slow_physics_process: np.ndarray,
    theta_v_at_edges_on_model_levels: np.ndarray,
    horizontal_pressure_gradient: np.ndarray,
    dtime: float,
) -> np.ndarray:
    vn_nnew = current_vn + dtime * (
        predictor_normal_wind_advective_tendency + normal_wind_tendency_due_to_slow_physics_process - dycore_consts.cpd * theta_v_at_edges_on_model_levels * horizontal_pressure_gradient
    )
    return vn_nnew


class TestAddTemporalTendenciesToVn(StencilTest):
    PROGRAM = add_temporal_tendencies_to_vn
    OUTPUTS = ("vn_nnew",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        current_vn: np.ndarray,
        predictor_normal_wind_advective_tendency: np.ndarray,
        normal_wind_tendency_due_to_slow_physics_process: np.ndarray,
        theta_v_at_edges_on_model_levels: np.ndarray,
        horizontal_pressure_gradient: np.ndarray,
        dtime: float,
        **kwargs: Any,
    ) -> dict:
        vn_nnew = add_temporal_tendencies_to_vn_numpy(
            current_vn, predictor_normal_wind_advective_tendency, normal_wind_tendency_due_to_slow_physics_process, theta_v_at_edges_on_model_levels, horizontal_pressure_gradient, dtime
        )
        return dict(vn_nnew=vn_nnew)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        dtime = wpfloat("10.0")
        current_vn = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        predictor_normal_wind_advective_tendency = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        normal_wind_tendency_due_to_slow_physics_process = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        theta_v_at_edges_on_model_levels = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        horizontal_pressure_gradient = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        vn_nnew = zero_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)

        return dict(
            current_vn=current_vn,
            predictor_normal_wind_advective_tendency=predictor_normal_wind_advective_tendency,
            normal_wind_tendency_due_to_slow_physics_process=normal_wind_tendency_due_to_slow_physics_process,
            theta_v_at_edges_on_model_levels=theta_v_at_edges_on_model_levels,
            horizontal_pressure_gradient=horizontal_pressure_gradient,
            vn_nnew=vn_nnew,
            dtime=dtime,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
