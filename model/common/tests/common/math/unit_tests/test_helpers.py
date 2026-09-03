# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import numpy as np
import pytest

import icon4py.model.testing.test_utils
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, simple
from icon4py.model.common.math import (
    vector_operations as vector_ops,
    vertical_operations as vertical_ops,
)
from icon4py.model.common.utils import data_allocation
from icon4py.model.testing import stencil_tests
from icon4py.model.testing.fixtures.datatest import backend, backend_like
from icon4py.model.testing.fixtures.stencil_tests import data_alloc, grid, grid_manager


def test_cross_product(backend: gtx_typing.Backend) -> None:
    mesh = simple.simple_grid(allocator=backend)
    x1 = data_allocation.random_field(mesh, dims.EdgeDim, allocator=backend)
    y1 = data_allocation.random_field(mesh, dims.EdgeDim, allocator=backend)
    z1 = data_allocation.random_field(mesh, dims.EdgeDim, allocator=backend)
    x2 = data_allocation.random_field(mesh, dims.EdgeDim, allocator=backend)
    y2 = data_allocation.random_field(mesh, dims.EdgeDim, allocator=backend)
    z2 = data_allocation.random_field(mesh, dims.EdgeDim, allocator=backend)
    x = data_allocation.zero_field(mesh, dims.EdgeDim, allocator=backend)
    y = data_allocation.zero_field(mesh, dims.EdgeDim, allocator=backend)
    z = data_allocation.zero_field(mesh, dims.EdgeDim, allocator=backend)

    vector_ops.cross_product_on_edges.with_backend(backend)(
        x1, x2, y1, y2, z1, z2, out=(x, y, z), offset_provider={}
    )
    a = np.column_stack((x1.asnumpy(), y1.asnumpy(), z1.asnumpy()))
    b = np.column_stack((x2.asnumpy(), y2.asnumpy(), z2.asnumpy()))
    c = np.cross(a, b)

    # The inputs are unseeded, so a component of the cross product occasionally lands near
    # zero. There the compiled backend's FMA contraction of 'a * b - c * d' shifts one
    # product by an ulp and the cancellation amplifies it beyond the default
    # 'rtol = 1e-12, atol = 0.0'. The inputs are in [-1, 1], so the deviation is bounded by
    # an ulp of 1.0 (2.2e-16 measured) regardless of how small the component gets.
    atol = 1.0e-15
    assert icon4py.model.testing.test_utils.dallclose(c[:, 0], x.asnumpy(), atol=atol)
    assert icon4py.model.testing.test_utils.dallclose(c[:, 1], y.asnumpy(), atol=atol)
    assert icon4py.model.testing.test_utils.dallclose(c[:, 2], z.asnumpy(), atol=atol)


class TestAverageTwoVerticalLevelsDownwardsOnEdges(stencil_tests.StencilTest):
    PROGRAM = vertical_ops.average_two_vertical_levels_downwards_on_edges
    OUTPUTS = ("average",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        input_field: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        shp = input_field.shape
        res = 0.5 * (input_field + np.roll(input_field, shift=-1, axis=1))[:, : shp[1] - 1]
        return dict(average=res)

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        input_field = data_alloc.zero_field(dims.EdgeDim, dims.KHalfDim)
        result = data_alloc.random_field(dims.EdgeDim, dims.KDim)
        return dict(
            input_field=input_field,
            average=result,
            horizontal_start=gtx.int32(0),
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(grid.num_levels),
        )


class TestAverageTwoVerticalLevelsDownwardsOnCells(stencil_tests.StencilTest):
    PROGRAM = vertical_ops.average_two_vertical_levels_downwards_on_cells
    OUTPUTS = ("average",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        input_field: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        shp = input_field.shape
        res = 0.5 * (input_field + np.roll(input_field, shift=-1, axis=1))[:, : shp[1] - 1]
        return dict(average=res)

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        input_field = data_alloc.random_field(dims.CellDim, dims.KHalfDim)
        result = data_alloc.zero_field(dims.CellDim, dims.KDim)
        return dict(
            input_field=input_field,
            average=result,
            horizontal_start=gtx.int32(0),
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(grid.num_levels),
        )
