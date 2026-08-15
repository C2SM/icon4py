# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""FFSL's patch1/patch2 gather: reading a cell field through the edge butterfly offset.

ICON identifies the two outer patches of an edge's departure region by ABSOLUTE cell index
(butterfly_idx, mo_model_domimp_setup.f90:380-443). GT4Py cannot index a field by a runtime
index, so the port has to carry a relative SLOT into the E2C2E2C offset instead. This pins
that the substitution is expressible and exact, independently of the rest of FFSL, which
does not exist yet.
"""

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import numpy as np
from gt4py.next import where

from icon4py.model.common import (
    dimension as dims,
    field_type_aliases as fa,
    model_backends,
    type_alias as ta,
)
from icon4py.model.common.dimension import E2C2E2C
from icon4py.model.common.grid import base
from icon4py.model.common.utils import data_allocation as data_alloc


@gtx.field_operator
def _gather_butterfly_cell(
    p_cc: fa.CellKField[ta.wpfloat], slot: fa.EdgeKField[gtx.int32]
) -> fa.EdgeKField[ta.wpfloat]:
    """The value of the butterfly cell in 'slot', gathered onto the edge.

    A chain of where() over compile-time-constant slots is how GT4Py expresses a runtime
    choice of neighbour; there are only four, so the chain is short.
    """
    return where(
        slot == 0,
        p_cc(E2C2E2C[0]),
        where(
            slot == 1,
            p_cc(E2C2E2C[1]),
            where(slot == 2, p_cc(E2C2E2C[2]), p_cc(E2C2E2C[3])),
        ),
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def gather_butterfly_cell(  # noqa: PLR0917 [too-many-positional-arguments]  # a gtx.program's arguments must stay positional
    p_cc: fa.CellKField[ta.wpfloat],
    slot: fa.EdgeKField[gtx.int32],
    out_e: fa.EdgeKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _gather_butterfly_cell(
        p_cc=p_cc,
        slot=slot,
        out=out_e,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


def test_runtime_slot_gather_through_the_butterfly_offset(
    grid: base.Grid, backend: gtx_typing.Backend
) -> None:
    allocator = model_backends.get_allocator(backend)
    p_cc = data_alloc.random_field(grid, dims.CellDim, dims.KDim, allocator=allocator)
    out_e = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, allocator=allocator)

    # every slot exercised on every edge, so a dropped branch cannot hide
    rng = np.random.default_rng(3)
    slot_values = rng.integers(0, 4, size=(grid.num_edges, grid.num_levels)).astype(np.int32)
    slot = gtx.as_field((dims.EdgeDim, dims.KDim), slot_values, allocator=allocator)

    gather_butterfly_cell.with_backend(backend)(
        p_cc=p_cc,
        slot=slot,
        out_e=out_e,
        horizontal_start=gtx.int32(0),
        horizontal_end=gtx.int32(grid.num_edges),
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(grid.num_levels),
        offset_provider=grid.connectivities,
    )

    butterfly = grid.get_connectivity(dims.E2C2E2C).asnumpy()
    edges = np.arange(grid.num_edges)[:, np.newaxis]
    levels = np.arange(grid.num_levels)[np.newaxis, :]
    expected = p_cc.asnumpy()[butterfly[edges, slot_values], levels]

    np.testing.assert_array_equal(out_e.asnumpy(), expected)
