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

from icon4py.model.atmosphere.dycore.stencils.compute_diagnostics_from_normal_wind import (
    _interpolate_to_half_levels,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.testing import stencil_tests

from .test_velocity_advection_terms import interpolate_vn_to_half_levels_numpy


class TestInterpolateToHalfLevels(stencil_tests.StencilTest):
    PROGRAM = _interpolate_to_half_levels
    # the operator reads x at K+0.5, so the surface half level is out of reach
    OUTPUTS = (
        stencil_tests.Output(
            "out", refslice=(slice(None), slice(0, -1)), gtslice=(slice(None), slice(0, -1))
        ),
    )

    @stencil_tests.static_reference
    def reference(grid: base.Grid, *, wgtfac_e: np.ndarray, x: np.ndarray, **kwargs: Any) -> dict:
        return dict(out=interpolate_vn_to_half_levels_numpy(wgtfac_e, x))

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        return dict(
            wgtfac_e=data_alloc.random_field(
                dims.EdgeDim, dims.KHalfDim, low=0.0, high=1.0, dtype=ta.vpfloat
            ),
            x=data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=ta.wpfloat),
            out=data_alloc.random_field(dims.EdgeDim, dims.KHalfDim, dtype=ta.vpfloat),
            domain={
                dims.EdgeDim: (0, gtx.int32(grid.num_edges)),
                dims.KHalfDim: (0, gtx.int32(grid.num_levels)),
            },
        )
