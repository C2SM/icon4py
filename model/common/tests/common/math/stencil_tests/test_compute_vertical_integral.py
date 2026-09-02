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
from icon4py.model.common.math.vertical_operations import compute_vertical_integral
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


class TestComputeVerticalIntegral(stencil_tests.StencilTest):
    PROGRAM = compute_vertical_integral
    OUTPUTS = ("vertical_integral",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        integrand: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        return dict(vertical_integral=np.cumsum(integrand, axis=1))

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        return dict(
            integrand=data_alloc.random_field(dims.CellDim, dims.KDim, dtype=wpfloat),
            vertical_integral=data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )


class TestComputeVerticalIntegralReadAtBottomLevel(stencil_tests.StencilTest):
    """
    Verify only the last full level of the running sum against the column sum.

    This emulates the ``*_vi`` diagnostics consumers (``Update_diagnostics`` in ICON's
    ``mo_vdf_atmo.f90``) that read the running sum at the bottom level as the column
    integral.
    """

    PROGRAM = compute_vertical_integral
    OUTPUTS = (
        stencil_tests.Output(
            "vertical_integral",
            refslice=(slice(None), slice(-1, None)),
            gtslice=(slice(None), slice(-1, None)),
        ),
    )

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        integrand: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        vertical_integral = np.zeros_like(integrand)
        vertical_integral[:, -1] = np.sum(integrand, axis=1)
        return dict(vertical_integral=vertical_integral)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        return dict(
            integrand=data_alloc.random_field(dims.CellDim, dims.KDim, dtype=wpfloat),
            vertical_integral=data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
