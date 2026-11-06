# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import numpy as np

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.grid import simple as simple_grid
from icon4py.model.common.math.tensor_operations import (
    Mat3OnEdges,
    squared_norm_of_symmetric_on_edges,
    trace_on_edges,
    twice_symmetric_part_on_edges,
)
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc


NUM_LEVELS = 4
GRID = simple_grid.simple_grid(num_levels=NUM_LEVELS)


@gtx.field_operator
def _offdiagonal_of_twice_symmetric_part(
    m: Mat3OnEdges,
) -> tuple[fa.EdgeKField[wpfloat], fa.EdgeKField[wpfloat], fa.EdgeKField[wpfloat]]:
    d = twice_symmetric_part_on_edges(m)
    return d[0][1], d[0][2], d[1][2]


def _edge_k_field() -> gtx.Field:
    return data_alloc.zero_field(GRID, dims.EdgeDim, dims.KDim, dtype=wpfloat)


def _random_components() -> list[list[data_alloc.NDArray]]:
    """Nine random edge fields as numpy arrays, indexed as ``components[i][j]``."""
    return [
        [
            data_alloc.random_field(GRID, dims.EdgeDim, dims.KDim, dtype=wpfloat).asnumpy()
            for _ in range(3)
        ]
        for _ in range(3)
    ]


def _as_tensor(components: list[list[data_alloc.NDArray]]) -> Mat3OnEdges:
    def component(i: int, j: int) -> fa.EdgeKField[wpfloat]:
        return gtx.as_field(
            (dims.EdgeDim, dims.KDim),
            components[i][j],  # type: ignore [arg-type]
            dtype=wpfloat,
        )

    return (
        (component(0, 0), component(0, 1), component(0, 2)),
        (component(1, 0), component(1, 1), component(1, 2)),
        (component(2, 0), component(2, 1), component(2, 2)),
    )


def test_trace_sums_the_diagonal() -> None:
    components = _random_components()
    trace = _edge_k_field()

    trace_on_edges(_as_tensor(components), out=trace, offset_provider={})

    expected = components[0][0] + components[1][1] + components[2][2]
    np.testing.assert_allclose(trace.asnumpy(), expected, rtol=1e-15)


def test_twice_symmetric_part_adds_the_transpose() -> None:
    components = _random_components()
    out = (_edge_k_field(), _edge_k_field(), _edge_k_field())

    _offdiagonal_of_twice_symmetric_part(_as_tensor(components), out=out, offset_provider={})

    for field, (i, j) in zip(out, [(0, 1), (0, 2), (1, 2)], strict=True):
        np.testing.assert_allclose(field.asnumpy(), components[i][j] + components[j][i], rtol=1e-15)


def test_squared_norm_contracts_a_symmetric_tensor() -> None:
    components = _random_components()
    symmetric = [[0.5 * (components[i][j] + components[j][i]) for j in range(3)] for i in range(3)]
    norm = _edge_k_field()

    squared_norm_of_symmetric_on_edges(_as_tensor(symmetric), out=norm, offset_provider={})

    expected = sum(symmetric[i][j] * symmetric[i][j] for i in range(3) for j in range(3))
    np.testing.assert_allclose(norm.asnumpy(), expected, rtol=1e-14)
