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
from icon4py.model.common.interpolation.stencils.interpolate_edge_field_to_half_levels_with_boundaries_wp import (
    interpolate_edge_field_to_half_levels_with_boundaries_wp,
)
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


def interpolate_edge_field_to_half_levels_with_boundaries_numpy(
    *,
    interpolant: np.ndarray,
    wgtfac_e: np.ndarray,
    wgtfacq1_e_1: np.ndarray,
    wgtfacq1_e_2: np.ndarray,
    wgtfacq1_e_3: np.ndarray,
    wgtfacq_e_1: np.ndarray,
    wgtfacq_e_2: np.ndarray,
    wgtfacq_e_3: np.ndarray,
) -> np.ndarray:
    nlev = interpolant.shape[1]
    interpolation = np.zeros((interpolant.shape[0], nlev + 1), dtype=interpolant.dtype)
    interpolation[:, 0] = (
        wgtfacq1_e_1 * interpolant[:, 0]
        + wgtfacq1_e_2 * interpolant[:, 1]
        + wgtfacq1_e_3 * interpolant[:, 2]
    )
    interpolation[:, 1:nlev] = (
        wgtfac_e[:, 1:nlev] * interpolant[:, 1:nlev]
        + (1.0 - wgtfac_e[:, 1:nlev]) * interpolant[:, 0 : nlev - 1]
    )
    interpolation[:, nlev] = (
        wgtfacq_e_1 * interpolant[:, nlev - 1]
        + wgtfacq_e_2 * interpolant[:, nlev - 2]
        + wgtfacq_e_3 * interpolant[:, nlev - 3]
    )
    return interpolation


class TestInterpolateEdgeFieldToHalfLevelsWithBoundariesWp(stencil_tests.StencilTest):
    PROGRAM = interpolate_edge_field_to_half_levels_with_boundaries_wp
    OUTPUTS = ("interpolation",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        interpolant: np.ndarray,
        wgtfac_e: np.ndarray,
        wgtfacq1_e_1: np.ndarray,
        wgtfacq1_e_2: np.ndarray,
        wgtfacq1_e_3: np.ndarray,
        wgtfacq_e_1: np.ndarray,
        wgtfacq_e_2: np.ndarray,
        wgtfacq_e_3: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        interpolation = interpolate_edge_field_to_half_levels_with_boundaries_numpy(
            interpolant=interpolant,
            wgtfac_e=wgtfac_e,
            wgtfacq1_e_1=wgtfacq1_e_1,
            wgtfacq1_e_2=wgtfacq1_e_2,
            wgtfacq1_e_3=wgtfacq1_e_3,
            wgtfacq_e_1=wgtfacq_e_1,
            wgtfacq_e_2=wgtfacq_e_2,
            wgtfacq_e_3=wgtfacq_e_3,
        )
        return dict(interpolation=interpolation)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        interpolant = data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=wpfloat)
        wgtfac_e = data_alloc.random_field(
            dims.EdgeDim, dims.KDim, dtype=wpfloat, extend={dims.KDim: 1}
        )
        wgtfacq1_e_1 = data_alloc.random_field(dims.EdgeDim, dtype=wpfloat)
        wgtfacq1_e_2 = data_alloc.random_field(dims.EdgeDim, dtype=wpfloat)
        wgtfacq1_e_3 = data_alloc.random_field(dims.EdgeDim, dtype=wpfloat)
        wgtfacq_e_1 = data_alloc.random_field(dims.EdgeDim, dtype=wpfloat)
        wgtfacq_e_2 = data_alloc.random_field(dims.EdgeDim, dtype=wpfloat)
        wgtfacq_e_3 = data_alloc.random_field(dims.EdgeDim, dtype=wpfloat)
        interpolation = data_alloc.zero_field(
            dims.EdgeDim, dims.KDim, dtype=wpfloat, extend={dims.KDim: 1}
        )

        return dict(
            interpolant=interpolant,
            wgtfac_e=wgtfac_e,
            wgtfacq1_e_1=wgtfacq1_e_1,
            wgtfacq1_e_2=wgtfacq1_e_2,
            wgtfacq1_e_3=wgtfacq1_e_3,
            wgtfacq_e_1=wgtfacq_e_1,
            wgtfacq_e_2=wgtfacq_e_2,
            wgtfacq_e_3=wgtfacq_e_3,
            interpolation=interpolation,
            nlev=gtx.int32(grid.num_levels),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels + 1),
        )
