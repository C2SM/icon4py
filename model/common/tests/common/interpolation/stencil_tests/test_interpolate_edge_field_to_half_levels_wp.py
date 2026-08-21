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

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.interpolation.stencils.interpolate_edge_field_to_half_levels_wp import (
    interpolate_edge_field_to_half_levels_wp,
)
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


def interpolate_edge_field_to_half_levels_wp_numpy(
    wgtfac_e: np.ndarray, interpolant: np.ndarray
) -> np.ndarray:
    interpolant_offset_1 = np.roll(interpolant, shift=1, axis=1)
    interpolation_to_half_levels_wp = (
        wgtfac_e * interpolant + (1.0 - wgtfac_e) * interpolant_offset_1
    )
    interpolation_to_half_levels_wp[:, 0] = 0

    return interpolation_to_half_levels_wp


class TestInterpolateToHalfLevelsWp(stencil_tests.StencilTest):
    PROGRAM = interpolate_edge_field_to_half_levels_wp
    OUTPUTS = ("interpolation_to_half_levels_wp",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        wgtfac_e: np.ndarray,
        interpolant: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        interpolation_to_half_levels_wp = interpolate_edge_field_to_half_levels_wp_numpy(
            wgtfac_e=wgtfac_e, interpolant=interpolant
        )
        return dict(interpolation_to_half_levels_wp=interpolation_to_half_levels_wp)

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        interpolant = data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=wpfloat)
        wgtfac_e = data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=wpfloat)
        interpolation_to_half_levels_wp = data_alloc.zero_field(
            dims.EdgeDim, dims.KDim, dtype=wpfloat
        )

        return dict(
            wgtfac_e=wgtfac_e,
            interpolant=interpolant,
            interpolation_to_half_levels_wp=interpolation_to_half_levels_wp,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=1,
            vertical_end=gtx.int32(grid.num_levels),
        )
