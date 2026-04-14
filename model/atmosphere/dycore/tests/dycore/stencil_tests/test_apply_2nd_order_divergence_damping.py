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
from icon4py.model.atmosphere.dycore.stencils.apply_2nd_order_divergence_damping import (
    apply_2nd_order_divergence_damping,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field
from icon4py.model.testing.stencil_tests import StencilTest


def apply_2nd_order_divergence_damping_numpy(
    horizontal_gradient_of_normal_wind_divergence: np.ndarray, vn: np.ndarray, second_order_divdamp_scaling_coeff: wpfloat
) -> np.ndarray:
    vn = vn + (second_order_divdamp_scaling_coeff * horizontal_gradient_of_normal_wind_divergence)
    return vn


class TestApply2ndOrderDivergenceDamping(StencilTest):
    PROGRAM = apply_2nd_order_divergence_damping
    OUTPUTS = ("vn",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        horizontal_gradient_of_normal_wind_divergence: np.ndarray,
        vn: np.ndarray,
        second_order_divdamp_scaling_coeff: ta.wpfloat,
        **kwargs: Any,
    ) -> dict:
        vn = apply_2nd_order_divergence_damping_numpy(horizontal_gradient_of_normal_wind_divergence, vn, second_order_divdamp_scaling_coeff)
        return dict(vn=vn)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        horizontal_gradient_of_normal_wind_divergence = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        vn = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        second_order_divdamp_scaling_coeff = wpfloat("5.0")

        return dict(
            horizontal_gradient_of_normal_wind_divergence=horizontal_gradient_of_normal_wind_divergence,
            vn=vn,
            second_order_divdamp_scaling_coeff=second_order_divdamp_scaling_coeff,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
