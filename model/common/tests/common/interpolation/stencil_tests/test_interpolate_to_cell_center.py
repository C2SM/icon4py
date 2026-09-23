# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from collections.abc import Mapping
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.interpolation.stencils.interpolate_to_cell_center import (
    _interpolate_to_cell_center,
)
from icon4py.model.common.states import utils as state_utils
from icon4py.model.testing import stencil_tests
from icon4py.model.testing.reference_funcs import interpolate_to_cell_center_numpy


class TestInterpolateToCellCenter(stencil_tests.StencilTest):
    PROGRAM = _interpolate_to_cell_center
    OUTPUTS = ("out",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        interpolant: np.ndarray,
        e_bln_c_s: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        interpolation = interpolate_to_cell_center_numpy(connectivities, interpolant, e_bln_c_s)
        return dict(out=interpolation)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        interpolant = data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        e_bln_c_s = data_alloc.random_field(dims.CellDim, dims.C2EDim, dtype=ta.wpfloat)
        return dict(
            interpolant=interpolant,
            e_bln_c_s=e_bln_c_s,
            domain={
                dims.CellDim: (0, gtx.int32(grid.num_cells)),
                dims.KDim: (0, gtx.int32(grid.num_levels)),
            },
            out=data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=ta.wpfloat),
        )
