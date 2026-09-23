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

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.grid import base
from icon4py.model.common.math.tridiagonal import (
    _solve_tridiagonal_matrix_back_substitution_on_half_levels_mixed_precision,
    _solve_tridiagonal_matrix_forward_sweep_on_half_levels_mixed_precision,
    _solve_tridiagonal_matrix_on_cell_half_levels,
    _solve_tridiagonal_matrix_on_cells,
    _solve_tridiagonal_matrix_on_edges,
)
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.testing import stencil_tests


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def _solve_on_half_levels_mixed_precision(
    a: fa.CellKHalfField[vpfloat],  # type: ignore[valid-type]
    b: fa.CellKHalfField[vpfloat],  # type: ignore[valid-type]
    c: fa.CellKHalfField[vpfloat],  # type: ignore[valid-type]
    d: fa.CellKHalfField[wpfloat],
) -> fa.CellKHalfField[wpfloat]:
    q, d_prime = _solve_tridiagonal_matrix_forward_sweep_on_half_levels_mixed_precision(a, b, c, d)
    return _solve_tridiagonal_matrix_back_substitution_on_half_levels_mixed_precision(q, d_prime)


def solve_tridiagonal_numpy(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> np.ndarray:
    """Solve each column's system; a on the first row and c on the last row are not part of it."""
    x = np.empty_like(d)
    for cell in range(d.shape[0]):
        matrix = np.diag(b[cell]) + np.diag(a[cell, 1:], -1) + np.diag(c[cell, :-1], 1)
        x[cell] = np.linalg.solve(matrix, d[cell])
    return x


def tridiagonal_input_data(
    data_alloc: stencil_tests.DataAllocationWrapper,
    grid: base.Grid,
    horizontal_dim: gtx.Dimension,
    vertical_dim: gtx.Dimension,
    coefficient_dtype: type = wpfloat,
) -> dict[str, Any]:
    # diagonally dominant, so that the system is well conditioned
    return dict(
        a=data_alloc.random_field(
            horizontal_dim, vertical_dim, low=-1.0, high=1.0, dtype=coefficient_dtype
        ),
        b=data_alloc.random_field(
            horizontal_dim, vertical_dim, low=3.0, high=4.0, dtype=coefficient_dtype
        ),
        c=data_alloc.random_field(
            horizontal_dim, vertical_dim, low=-1.0, high=1.0, dtype=coefficient_dtype
        ),
        d=data_alloc.random_field(horizontal_dim, vertical_dim, dtype=wpfloat),
        domain={
            horizontal_dim: (0, grid.size[horizontal_dim]),
            vertical_dim: (0, grid.size[vertical_dim]),
        },
        out=data_alloc.zero_field(horizontal_dim, vertical_dim, dtype=wpfloat),
    )


class TestSolveTridiagonalMatrixOnCells(stencil_tests.StencilTest):
    PROGRAM = _solve_tridiagonal_matrix_on_cells
    OUTPUTS = ("out",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        d: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        return dict(out=solve_tridiagonal_numpy(a, b, c, d))

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, Any]:
        return tridiagonal_input_data(data_alloc, grid, dims.CellDim, dims.KDim)


class TestSolveTridiagonalMatrixOnEdges(stencil_tests.StencilTest):
    PROGRAM = _solve_tridiagonal_matrix_on_edges
    OUTPUTS = ("out",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        d: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        return dict(out=solve_tridiagonal_numpy(a, b, c, d))

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, Any]:
        return tridiagonal_input_data(data_alloc, grid, dims.EdgeDim, dims.KDim)


class TestSolveTridiagonalMatrixOnCellHalfLevels(stencil_tests.StencilTest):
    PROGRAM = _solve_tridiagonal_matrix_on_cell_half_levels
    OUTPUTS = ("out",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        d: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        return dict(out=solve_tridiagonal_numpy(a, b, c, d))

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, Any]:
        return tridiagonal_input_data(data_alloc, grid, dims.CellDim, dims.KHalfDim)


class TestSolveTridiagonalMatrixOnHalfLevelsMixedPrecision(stencil_tests.StencilTest):
    PROGRAM = _solve_on_half_levels_mixed_precision
    OUTPUTS = ("out",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        d: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        return dict(
            out=solve_tridiagonal_numpy(a.astype(wpfloat), b.astype(wpfloat), c.astype(wpfloat), d)
        )

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, Any]:
        return tridiagonal_input_data(data_alloc, grid, dims.CellDim, dims.KHalfDim, vpfloat)
