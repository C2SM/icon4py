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

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.math.stencils.generic_math_operations import copy_field_on_edge_k
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


class TestCopyFieldOnEdgeK(stencil_tests.StencilTest):
    PROGRAM = copy_field_on_edge_k
    OUTPUTS = ("output_field",)
    STATIC_PARAMS = {
        stencil_tests.StandardStaticVariants.NONE: (),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_DOMAIN: (
            "horizontal_start",
            "horizontal_end",
            "vertical_start",
            "vertical_end",
        ),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_VERTICAL: (
            "vertical_start",
            "vertical_end",
        ),
    }

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        field: np.ndarray,
        output_field: np.ndarray,
        horizontal_start: int,
        horizontal_end: int,
        vertical_start: int,
        vertical_end: int,
        **kwargs: Any,
    ) -> dict:
        output = output_field.copy()
        output[horizontal_start:horizontal_end, vertical_start:vertical_end] = field[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ]
        return dict(output_field=output)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType | gtx.common.DomainLike]:
        field = data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=wpfloat)
        return dict(
            field=field,
            output_field=data_alloc.zero_field(dims.EdgeDim, dims.KDim, dtype=wpfloat),
            horizontal_start=gtx.int32(0),
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(grid.num_levels),
        )
