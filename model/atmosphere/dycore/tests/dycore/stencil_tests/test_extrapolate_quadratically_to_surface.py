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

from icon4py.model.atmosphere.dycore.stencils.extrapolate_quadratically_to_surface import (
    extrapolate_quadratically_to_surface,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat
from icon4py.model.testing import stencil_tests


def extrapolate_quadratically_to_surface_numpy(
    interpolant: np.ndarray, wgtfacq_c: np.ndarray, interpolation_to_surface: np.ndarray
) -> np.ndarray:
    # half level k is extrapolated from the three model levels below it
    interpolation_to_surface = np.copy(interpolation_to_surface)
    nlev = interpolant.shape[1]
    khalf = np.arange(3, nlev + 1)
    interpolation_to_surface[:, 3 : nlev + 1] = (
        wgtfacq_c[:, khalf - 1] * interpolant[:, khalf - 1]
        + wgtfacq_c[:, khalf - 2] * interpolant[:, khalf - 2]
        + wgtfacq_c[:, khalf - 3] * interpolant[:, khalf - 3]
    )
    return interpolation_to_surface


class TestInterpolateToSurface(stencil_tests.StencilTest):
    PROGRAM = extrapolate_quadratically_to_surface
    OUTPUTS = ("interpolation_to_surface",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        interpolant: np.ndarray,
        wgtfacq_c: np.ndarray,
        interpolation_to_surface: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        interpolation_to_surface = extrapolate_quadratically_to_surface_numpy(
            wgtfacq_c=wgtfacq_c,
            interpolant=interpolant,
            interpolation_to_surface=interpolation_to_surface,
        )
        return dict(interpolation_to_surface=interpolation_to_surface)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        interpolant = data_alloc.random_field(dims.CellDim, dims.KDim, dtype=vpfloat)
        wgtfacq_c = data_alloc.random_field(dims.CellDim, dims.KDim, dtype=vpfloat)
        interpolation_to_surface = data_alloc.zero_field(dims.CellDim, dims.KHalfDim, dtype=vpfloat)

        return dict(
            interpolant=interpolant,
            wgtfacq_c=wgtfacq_c,
            interpolation_to_surface=interpolation_to_surface,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=3,
            vertical_end=gtx.int32(grid.num_levels + 1),
        )
