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
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.interpolation.stencils.interpolate_cell_field_to_half_levels import (
    _interpolate_cell_field_to_half_levels_vp,
    _interpolate_cell_field_to_half_levels_wp,
)
from icon4py.model.testing import stencil_tests


def interpolate_cell_field_to_half_levels_vp_numpy(
    wgtfac_c: np.ndarray, interpolant: np.ndarray
) -> np.ndarray:
    interpolant_offset_1 = np.roll(interpolant, shift=1, axis=1)
    interpolation_to_half_levels_vp = (
        wgtfac_c * interpolant + (1.0 - wgtfac_c) * interpolant_offset_1
    )
    interpolation_to_half_levels_vp[:, 0] = 0

    return interpolation_to_half_levels_vp


def interpolate_cell_field_to_half_levels_wp_numpy(
    wgtfac_c: np.ndarray, interpolant: np.ndarray
) -> np.ndarray:
    interpolant_offset_1 = np.roll(interpolant, shift=1, axis=1)
    interpolation_to_half_levels_wp = (
        wgtfac_c * interpolant + (1.0 - wgtfac_c) * interpolant_offset_1
    )
    interpolation_to_half_levels_wp[:, 0] = 0

    return interpolation_to_half_levels_wp


class TestInterpolateToHalfLevelsVp(stencil_tests.StencilTest):
    PROGRAM = _interpolate_cell_field_to_half_levels_vp
    OUTPUTS = ("out",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        wgtfac_c: np.ndarray,
        interpolant: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        return dict(
            out=interpolate_cell_field_to_half_levels_vp_numpy(
                wgtfac_c=wgtfac_c, interpolant=interpolant
            )
        )

    @stencil_tests.input_data_fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        interpolant = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        wgtfac_c = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        out = self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=ta.vpfloat)

        return dict(
            wgtfac_c=wgtfac_c,
            interpolant=interpolant,
            out=out,
            domain={
                dims.CellDim: (0, gtx.int32(grid.num_cells)),
                dims.KDim: (1, gtx.int32(grid.num_levels)),
            },
        )


class TestInterpolateToHalfLevelsWp(stencil_tests.StencilTest):
    PROGRAM = _interpolate_cell_field_to_half_levels_wp
    OUTPUTS = ("out",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        wgtfac_c: np.ndarray,
        interpolant: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        return dict(
            out=interpolate_cell_field_to_half_levels_wp_numpy(
                wgtfac_c=wgtfac_c, interpolant=interpolant
            )
        )

    @stencil_tests.input_data_fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        interpolant = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        wgtfac_c = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        out = self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=ta.wpfloat)

        return dict(
            wgtfac_c=wgtfac_c,
            interpolant=interpolant,
            out=out,
            domain={
                dims.CellDim: (0, gtx.int32(grid.num_cells)),
                dims.KDim: (1, gtx.int32(grid.num_levels)),
            },
        )
