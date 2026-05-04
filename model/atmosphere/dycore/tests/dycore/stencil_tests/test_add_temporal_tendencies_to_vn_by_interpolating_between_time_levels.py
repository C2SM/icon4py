# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

import icon4py.model.common.type_alias as ta
from icon4py.model.atmosphere.dycore.stencils.add_temporal_tendencies_to_vn_by_interpolating_between_time_levels import (
    add_temporal_tendencies_to_vn_by_interpolating_between_time_levels,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.stencil_tests import StencilTest


def add_temporal_tendencies_to_vn_by_interpolating_between_time_levels_numpy(
    current_vn: np.ndarray,
    predictor_normal_wind_advective_tendency: np.ndarray,
    corrector_normal_wind_advective_tendency: np.ndarray,
    normal_wind_tendency_due_to_slow_physics_process: np.ndarray,
    theta_v_at_edges_on_model_levels: np.ndarray,
    horizontal_pressure_gradient: np.ndarray,
    dtime: ta.wpfloat,
    advection_explicit_weight_parameter: ta.wpfloat,
    advection_implicit_weight_parameter: ta.wpfloat,
    cpd: ta.wpfloat,
) -> np.ndarray:
    vn_nnew = current_vn + dtime * (
        advection_explicit_weight_parameter * predictor_normal_wind_advective_tendency
        + advection_implicit_weight_parameter * corrector_normal_wind_advective_tendency
        + normal_wind_tendency_due_to_slow_physics_process
        - cpd * theta_v_at_edges_on_model_levels * horizontal_pressure_gradient
    )
    return vn_nnew


class TestAddTemporalTendenciesToVnByInterpolatingBetweenTimeLevels(StencilTest):
    PROGRAM = add_temporal_tendencies_to_vn_by_interpolating_between_time_levels
    OUTPUTS = ("vn_nnew",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        current_vn: np.ndarray,
        predictor_normal_wind_advective_tendency: np.ndarray,
        corrector_normal_wind_advective_tendency: np.ndarray,
        normal_wind_tendency_due_to_slow_physics_process: np.ndarray,
        theta_v_at_edges_on_model_levels: np.ndarray,
        horizontal_pressure_gradient: np.ndarray,
        dtime: ta.wpfloat,
        advection_explicit_weight_parameter: ta.wpfloat,
        advection_implicit_weight_parameter: ta.wpfloat,
        cpd: ta.wpfloat,
        **kwargs: Any,
    ) -> dict:
        vn_nnew = add_temporal_tendencies_to_vn_by_interpolating_between_time_levels_numpy(
            current_vn,
            predictor_normal_wind_advective_tendency,
            corrector_normal_wind_advective_tendency,
            normal_wind_tendency_due_to_slow_physics_process,
            theta_v_at_edges_on_model_levels,
            horizontal_pressure_gradient,
            dtime,
            advection_explicit_weight_parameter,
            advection_implicit_weight_parameter,
            cpd,
        )
        return dict(vn_nnew=vn_nnew)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        current_vn = random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        predictor_normal_wind_advective_tendency = random_field(
            grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat
        )
        corrector_normal_wind_advective_tendency = random_field(
            grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat
        )
        normal_wind_tendency_due_to_slow_physics_process = random_field(
            grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat
        )
        theta_v_at_edges_on_model_levels = random_field(
            grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat
        )
        horizontal_pressure_gradient = random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat)
        vn_nnew = zero_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        dtime = ta.wpfloat("5.0")
        advection_explicit_weight_parameter = ta.wpfloat("8.0")
        advection_implicit_weight_parameter = ta.wpfloat("7.0")
        cpd = ta.wpfloat("2.0")

        return dict(
            current_vn=current_vn,
            predictor_normal_wind_advective_tendency=predictor_normal_wind_advective_tendency,
            corrector_normal_wind_advective_tendency=corrector_normal_wind_advective_tendency,
            normal_wind_tendency_due_to_slow_physics_process=normal_wind_tendency_due_to_slow_physics_process,
            theta_v_at_edges_on_model_levels=theta_v_at_edges_on_model_levels,
            horizontal_pressure_gradient=horizontal_pressure_gradient,
            vn_nnew=vn_nnew,
            dtime=dtime,
            advection_explicit_weight_parameter=advection_explicit_weight_parameter,
            advection_implicit_weight_parameter=advection_implicit_weight_parameter,
            cpd=cpd,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
