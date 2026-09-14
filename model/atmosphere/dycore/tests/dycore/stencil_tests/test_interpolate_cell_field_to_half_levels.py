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
    nlev = interpolant.shape[1]
    interpolation_to_half_levels_vp = np.zeros((interpolant.shape[0], nlev + 1))
    w = wgtfac_c[:, 1:nlev]
    interpolation_to_half_levels_vp[:, 1:nlev] = (
        w * interpolant[:, 1:nlev] + (1.0 - w) * interpolant[:, 0 : nlev - 1]
    )

    return interpolation_to_half_levels_vp


def interpolate_cell_field_to_half_levels_wp_numpy(
    wgtfac_c: np.ndarray, interpolant: np.ndarray
) -> np.ndarray:
    nlev = interpolant.shape[1]
    interpolation_to_half_levels_wp = np.zeros((interpolant.shape[0], nlev + 1))
    w = wgtfac_c[:, 1:nlev]
    interpolation_to_half_levels_wp[:, 1:nlev] = (
        w * interpolant[:, 1:nlev] + (1.0 - w) * interpolant[:, 0 : nlev - 1]
    )

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
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, Any]:
        interpolant = data_alloc.random_field(dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        wgtfac_c = data_alloc.random_field(dims.CellDim, dims.KHalfDim, dtype=ta.vpfloat)
        out = data_alloc.zero_field(dims.CellDim, dims.KHalfDim, dtype=ta.vpfloat)

        return dict(
            wgtfac_c=wgtfac_c,
            interpolant=interpolant,
            out=out,
            domain={
                dims.CellDim: (0, gtx.int32(grid.num_cells)),
                dims.KHalfDim: (1, gtx.int32(grid.num_levels)),
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
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, Any]:
        interpolant = data_alloc.random_field(dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        wgtfac_c = data_alloc.random_field(dims.CellDim, dims.KHalfDim, dtype=ta.wpfloat)
        out = data_alloc.zero_field(dims.CellDim, dims.KHalfDim, dtype=ta.wpfloat)

        return dict(
            wgtfac_c=wgtfac_c,
            interpolant=interpolant,
            out=out,
            domain={
                dims.CellDim: (0, gtx.int32(grid.num_cells)),
                dims.KHalfDim: (1, gtx.int32(grid.num_levels)),
            },
        )
