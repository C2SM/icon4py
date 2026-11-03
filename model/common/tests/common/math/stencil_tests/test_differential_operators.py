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

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.math import differential_operators
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.testing import reference_funcs, stencil_tests


def cell_horizontal_gradients_by_green_gauss_method_numpy(
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
    scalar_field: np.ndarray,
    geofac_grg_x: np.ndarray,
    geofac_grg_y: np.ndarray,
) -> tuple[np.ndarray, ...]:
    c2e2cO = connectivities[dims.C2E2CO]
    geofac_grg_x = np.expand_dims(geofac_grg_x, axis=-1)
    p_grad_1_u = np.sum(
        np.where((c2e2cO != -1)[:, :, np.newaxis], geofac_grg_x * scalar_field[c2e2cO], 0), axis=1
    )
    geofac_grg_y = np.expand_dims(geofac_grg_y, axis=-1)
    p_grad_1_v = np.sum(
        np.where((c2e2cO != -1)[:, :, np.newaxis], geofac_grg_y * scalar_field[c2e2cO], 0), axis=1
    )
    return (
        p_grad_1_u,
        p_grad_1_v,
    )


@pytest.mark.embedded_remap_error
class TestMoMathGradientsGradGreenGaussCellDsl(stencil_tests.StencilTest):
    PROGRAM = differential_operators.cell_horizontal_gradients_by_green_gauss_method
    OUTPUTS = ("out",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        scalar_field: np.ndarray,
        geofac_grg_x: np.ndarray,
        geofac_grg_y: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        (
            p_grad_1_u,
            p_grad_1_v,
        ) = cell_horizontal_gradients_by_green_gauss_method_numpy(
            connectivities, scalar_field, geofac_grg_x, geofac_grg_y
        )
        return dict(
            out=(p_grad_1_u, p_grad_1_v),
        )

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType | gtx.Domain | tuple[gtx.Field, ...]]:
        scalar_field = data_alloc.random_field(dims.CellDim, dims.KDim, dtype=vpfloat)
        geofac_grg_x = data_alloc.random_field(dims.CellDim, dims.C2E2CODim, dtype=wpfloat)
        geofac_grg_y = data_alloc.random_field(dims.CellDim, dims.C2E2CODim, dtype=wpfloat)
        p_grad_1_u = data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=vpfloat)
        p_grad_1_v = data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=vpfloat)

        return dict(
            scalar_field=scalar_field,
            geofac_grg_x=geofac_grg_x,
            geofac_grg_y=geofac_grg_y,
            out=(p_grad_1_u, p_grad_1_v),
            domain=gtx.domain(
                {
                    dims.CellDim: gtx.int32(grid.num_cells),
                    dims.KDim: gtx.int32(grid.num_levels),
                }
            ),
        )


class TestComputeCurl(stencil_tests.StencilTest):
    PROGRAM = differential_operators.compute_curl
    OUTPUTS = ("out",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        vec_e: np.ndarray,
        geofac_rot: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        return dict(out=reference_funcs.compute_curl_numpy(connectivities, vec_e, geofac_rot))

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType | gtx.common.DomainLike]:
        vec_e = data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=wpfloat)
        geofac_rot = data_alloc.random_field(dims.VertexDim, dims.V2EDim, dtype=wpfloat)
        return dict(
            vec_e=vec_e,
            geofac_rot=geofac_rot,
            out=data_alloc.zero_field(dims.VertexDim, dims.KDim, dtype=wpfloat),
            domain={
                dims.VertexDim: (0, gtx.int32(grid.num_vertices)),
                dims.KDim: (0, gtx.int32(grid.num_levels)),
            },
        )


def laplacian_on_edge_numpy(
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
    *,
    vec_normal: np.ndarray,
    vec_eastward: np.ndarray,
    vec_northward: np.ndarray,
    primal_normal_vert_x: np.ndarray,
    primal_normal_vert_y: np.ndarray,
    inv_primal_edge_length: np.ndarray,
    inv_vert_vert_length: np.ndarray,
) -> np.ndarray:
    e2c2v = connectivities[dims.E2C2V]
    normal_at_vertices = (
        vec_eastward[e2c2v] * primal_normal_vert_x[:, :, np.newaxis]
        + vec_northward[e2c2v] * primal_normal_vert_y[:, :, np.newaxis]
    )
    inv_primal_edge_length = inv_primal_edge_length[:, np.newaxis]
    inv_vert_vert_length = inv_vert_vert_length[:, np.newaxis]
    return 4.0 * (
        (normal_at_vertices[:, 0] + normal_at_vertices[:, 1] - 2.0 * vec_normal)
        * inv_primal_edge_length**2
        + (normal_at_vertices[:, 2] + normal_at_vertices[:, 3] - 2.0 * vec_normal)
        * inv_vert_vert_length**2
    )


class TestLaplacianOnEdge(stencil_tests.StencilTest):
    PROGRAM = differential_operators.laplacian_on_edge
    OUTPUTS = ("out",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        vec_normal: np.ndarray,
        vec_eastward: np.ndarray,
        vec_northward: np.ndarray,
        primal_normal_vert_x: np.ndarray,
        primal_normal_vert_y: np.ndarray,
        inv_primal_edge_length: np.ndarray,
        inv_vert_vert_length: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        return dict(
            out=laplacian_on_edge_numpy(
                connectivities,
                vec_normal=vec_normal,
                vec_eastward=vec_eastward,
                vec_northward=vec_northward,
                primal_normal_vert_x=primal_normal_vert_x,
                primal_normal_vert_y=primal_normal_vert_y,
                inv_primal_edge_length=inv_primal_edge_length,
                inv_vert_vert_length=inv_vert_vert_length,
            )
        )

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType | gtx.common.DomainLike]:
        return dict(
            vec_normal=data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=wpfloat),
            vec_eastward=data_alloc.random_field(dims.VertexDim, dims.KDim, dtype=wpfloat),
            vec_northward=data_alloc.random_field(dims.VertexDim, dims.KDim, dtype=wpfloat),
            primal_normal_vert_x=data_alloc.random_field(
                dims.EdgeDim, dims.E2C2VDim, dtype=wpfloat
            ),
            primal_normal_vert_y=data_alloc.random_field(
                dims.EdgeDim, dims.E2C2VDim, dtype=wpfloat
            ),
            inv_primal_edge_length=data_alloc.random_field(dims.EdgeDim, dtype=wpfloat),
            inv_vert_vert_length=data_alloc.random_field(dims.EdgeDim, dtype=wpfloat),
            out=data_alloc.zero_field(dims.EdgeDim, dims.KDim, dtype=wpfloat),
            domain={
                dims.EdgeDim: (0, gtx.int32(grid.num_edges)),
                dims.KDim: (0, gtx.int32(grid.num_levels)),
            },
        )


class TestNormalGradient(stencil_tests.StencilTest):
    PROGRAM = differential_operators.normal_gradient
    OUTPUTS = ("out",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        scalar_field: np.ndarray,
        inv_dual_edge_length: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        e2c = stencil_tests.connectivities_asnumpy(grid)[dims.E2C]
        return dict(
            out=inv_dual_edge_length[:, np.newaxis]
            * (scalar_field[e2c[:, 1]] - scalar_field[e2c[:, 0]])
        )

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType | gtx.common.DomainLike]:
        return dict(
            scalar_field=data_alloc.random_field(dims.CellDim, dims.KDim, dtype=wpfloat),
            inv_dual_edge_length=data_alloc.random_field(dims.EdgeDim, dtype=wpfloat),
            out=data_alloc.zero_field(dims.EdgeDim, dims.KDim, dtype=wpfloat),
            domain={
                dims.EdgeDim: (0, gtx.int32(grid.num_edges)),
                dims.KDim: (0, gtx.int32(grid.num_levels)),
            },
        )


class TestLaplacianOnCellKHalf(stencil_tests.StencilTest):
    PROGRAM = differential_operators.laplacian_on_cell_khalf
    OUTPUTS = ("out",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        scalar_field: np.ndarray,
        geofac_n2s: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        c2e2cO = stencil_tests.connectivities_asnumpy(grid)[dims.C2E2CO]
        return dict(
            out=np.sum(
                np.where(
                    (c2e2cO != -1)[:, :, np.newaxis],
                    scalar_field[c2e2cO] * geofac_n2s[:, :, np.newaxis],
                    0.0,
                ),
                axis=1,
            )
        )

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType | gtx.common.DomainLike]:
        return dict(
            scalar_field=data_alloc.random_field(dims.CellDim, dims.KHalfDim, dtype=wpfloat),
            geofac_n2s=data_alloc.random_field(dims.CellDim, dims.C2E2CODim, dtype=wpfloat),
            out=data_alloc.zero_field(dims.CellDim, dims.KHalfDim, dtype=wpfloat),
            domain={
                dims.CellDim: (0, gtx.int32(grid.num_cells)),
                dims.KHalfDim: (0, gtx.int32(grid.num_levels + 1)),
            },
        )


class TestDivergence(stencil_tests.StencilTest):
    PROGRAM = differential_operators.divergence
    OUTPUTS = ("out",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        vec_e: np.ndarray,
        geofac_div: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        c2e = stencil_tests.connectivities_asnumpy(grid)[dims.C2E]
        return dict(out=np.sum(vec_e[c2e] * geofac_div[:, :, np.newaxis], axis=1))

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType | gtx.common.DomainLike]:
        return dict(
            vec_e=data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=wpfloat),
            geofac_div=data_alloc.random_field(dims.CellDim, dims.C2EDim, dtype=wpfloat),
            out=data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
            domain={
                dims.CellDim: (0, gtx.int32(grid.num_cells)),
                dims.KDim: (0, gtx.int32(grid.num_levels)),
            },
        )


class TestCellHorizontalGradientsOnHalfLevels(stencil_tests.StencilTest):
    PROGRAM = differential_operators.cell_horizontal_gradients_by_green_gauss_method_on_half_levels
    OUTPUTS = ("out",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        scalar_field: np.ndarray,
        weights_x: np.ndarray,
        weights_y: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        c2e2cO = stencil_tests.connectivities_asnumpy(grid)[dims.C2E2CO]
        valid = (c2e2cO != -1)[:, :, np.newaxis]
        gradx = np.sum(
            np.where(valid, weights_x[:, :, np.newaxis] * scalar_field[c2e2cO], 0.0), axis=1
        )
        grady = np.sum(
            np.where(valid, weights_y[:, :, np.newaxis] * scalar_field[c2e2cO], 0.0), axis=1
        )
        return dict(out=(gradx, grady))

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[
        str, gtx.Field | tuple[gtx.Field, ...] | state_utils.ScalarType | gtx.common.DomainLike
    ]:
        return dict(
            scalar_field=data_alloc.random_field(dims.CellDim, dims.KHalfDim, dtype=wpfloat),
            weights_x=data_alloc.random_field(dims.CellDim, dims.C2E2CODim, dtype=wpfloat),
            weights_y=data_alloc.random_field(dims.CellDim, dims.C2E2CODim, dtype=wpfloat),
            out=(
                data_alloc.zero_field(dims.CellDim, dims.KHalfDim, dtype=vpfloat),
                data_alloc.zero_field(dims.CellDim, dims.KHalfDim, dtype=vpfloat),
            ),
            domain={
                dims.CellDim: (0, gtx.int32(grid.num_cells)),
                dims.KHalfDim: (0, gtx.int32(grid.num_levels + 1)),
            },
        )
