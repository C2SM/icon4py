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

import icon4py.model.testing.stencil_tests as test_helpers
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base as base_grid
from icon4py.model.common.interpolation.stencils.interpolate_edge_field_to_half_levels_wp import (
    interpolate_edge_field_to_half_levels_wp,
)
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils.data_allocation import random_field, zero_field


def interpolate_edge_field_to_half_levels_wp_numpy(
    wgtfac_e: np.ndarray, interpolant: np.ndarray
) -> np.ndarray:
    interpolant_offset_1 = np.roll(interpolant, shift=1, axis=1)
    interpolation_to_half_levels_wp = (
        wgtfac_e * interpolant + (1.0 - wgtfac_e) * interpolant_offset_1
    )
    interpolation_to_half_levels_wp[:, 0] = 0

    return interpolation_to_half_levels_wp


class TestInterpolateToHalfLevelsWp(test_helpers.StencilTest):
    PROGRAM = interpolate_edge_field_to_half_levels_wp
    OUTPUTS = ("interpolation_to_half_levels_wp",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        wgtfac_e: np.ndarray,
        interpolant: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        interpolation_to_half_levels_wp = interpolate_edge_field_to_half_levels_wp_numpy(
            wgtfac_e=wgtfac_e, interpolant=interpolant
        )
        return dict(interpolation_to_half_levels_wp=interpolation_to_half_levels_wp)

    @pytest.fixture
    def input_data(self, grid: base_grid.Grid) -> dict:
        interpolant = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        wgtfac_e = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        interpolation_to_half_levels_wp = zero_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)

        return dict(
            wgtfac_e=wgtfac_e,
            interpolant=interpolant,
            interpolation_to_half_levels_wp=interpolation_to_half_levels_wp,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=1,
            vertical_end=gtx.int32(grid.num_levels),
        )
