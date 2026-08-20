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
from icon4py.model.common.interpolation.stencils.interpolate_edge_field_to_half_levels_vp import (
    interpolate_edge_field_to_half_levels_vp,
)
from icon4py.model.common.type_alias import vpfloat


def interpolate_edge_field_to_half_levels_vp_numpy(
    wgtfac_e: np.ndarray, interpolant: np.ndarray
) -> np.ndarray:
    nlev = interpolant.shape[1]
    interpolation_to_half_levels_vp = np.zeros((interpolant.shape[0], nlev + 1))
    w = wgtfac_e[:, 1:nlev]
    interpolation_to_half_levels_vp[:, 1:nlev] = (
        w * interpolant[:, 1:nlev] + (1.0 - w) * interpolant[:, 0 : nlev - 1]
    )
    return interpolation_to_half_levels_vp


class TestInterpolateToHalfLevelsVp(test_helpers.StencilTest):
    PROGRAM = interpolate_edge_field_to_half_levels_vp
    OUTPUTS = ("interpolation_to_half_levels_vp",)

    @test_helpers.static_reference
    def reference(
        grid: base_grid.Grid,
        *,
        wgtfac_e: np.ndarray,
        interpolant: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        interpolation_to_half_levels_vp = interpolate_edge_field_to_half_levels_vp_numpy(
            wgtfac_e=wgtfac_e, interpolant=interpolant
        )
        return dict(interpolation_to_half_levels_vp=interpolation_to_half_levels_vp)

    @test_helpers.input_data_fixture
    def input_data(data_alloc: test_helpers.DataAllocationWrapper, grid: base_grid.Grid) -> dict:
        interpolant = data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=vpfloat)
        wgtfac_e = data_alloc.random_field(dims.EdgeDim, dims.KHalfDim, dtype=vpfloat)
        interpolation_to_half_levels_vp = data_alloc.zero_field(
            dims.EdgeDim, dims.KHalfDim, dtype=vpfloat
        )

        return dict(
            wgtfac_e=wgtfac_e,
            interpolant=interpolant,
            interpolation_to_half_levels_vp=interpolation_to_half_levels_vp,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=1,
            vertical_end=gtx.int32(grid.num_levels),
        )
