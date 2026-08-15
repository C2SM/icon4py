# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The edge-to-butterfly-cell connectivity that FFSL's patch1/patch2 gather needs.

The tests below are written against the defining property rather than against a stored
table, because the failure mode that matters is a PERMUTED SLOT ORDER: a permutation still
gathers real cells and still produces plausible-looking fluxes, so a test that only checked
the set of four cells would pass while the scheme silently read the wrong one.
"""

import numpy as np
import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import gridfile, simple
from icon4py.model.common.grid.grid_manager import _construct_edge_butterfly_cells


INVALID = gridfile.GridFile.INVALID_INDEX


@pytest.fixture
def tables() -> dict:
    data = simple.SimpleGridData()
    return {
        "e2c": np.asarray(data.e2c_table),
        "e2v": np.asarray(data.e2v_table),
        "c2e2c": np.asarray(data.c2e2c_table),
        "c2v": np.asarray(data.c2v_table),
        "e2c2e2c": np.asarray(data.e2c2e2c_table),
    }


def test_e2c2e2c_table_matches_the_construction(tables: dict) -> None:
    """The literal in simple.py must stay equal to the algorithm that generated it.

    The hand-written c2e2c2e2c table there does NOT match grid_manager's construction, and
    that divergence has cost debugging time before; this pins the edge one so it cannot
    happen again.
    """
    constructed = _construct_edge_butterfly_cells(
        tables["e2c"], tables["e2v"], tables["c2e2c"], tables["c2v"]
    )
    np.testing.assert_array_equal(tables["e2c2e2c"], constructed)


def test_slot_order_is_side_then_shared_vertex(tables: dict) -> None:
    """slot = 2 * side + vertex, the property ICON's construction defines.

    For every edge and slot, the cell there must BOTH flank e2c[side] AND contain
    e2v[vertex]. Permuting any two slots breaks one of the two halves.
    """
    e2c, e2v, c2e2c, c2v = (tables[k] for k in ("e2c", "e2v", "c2e2c", "c2v"))
    butterfly = tables["e2c2e2c"]

    for edge in range(e2c.shape[0]):
        for side in range(2):
            for vertex in range(2):
                cell = butterfly[edge, 2 * side + vertex]
                if cell == INVALID:
                    continue
                assert cell in c2e2c[e2c[edge, side]], (
                    f"edge {edge} slot {2 * side + vertex}: cell {cell} does not neighbour "
                    f"e2c[{side}] = {e2c[edge, side]}"
                )
                assert e2v[edge, vertex] in c2v[cell], (
                    f"edge {edge} slot {2 * side + vertex}: cell {cell} does not contain "
                    f"e2v[{vertex}] = {e2v[edge, vertex]}"
                )


def test_butterfly_cells_exclude_the_edges_own_cells(tables: dict) -> None:
    """The four flanking cells are never the two cells the edge itself separates."""
    e2c, butterfly = tables["e2c"], tables["e2c2e2c"]
    for edge in range(e2c.shape[0]):
        for slot in range(4):
            cell = butterfly[edge, slot]
            if cell == INVALID:
                continue
            assert cell not in tuple(e2c[edge]), (
                f"edge {edge} slot {slot}: {cell} is one of the edge's own cells {e2c[edge]}"
            )


def test_all_four_slots_are_filled_and_distinct_on_a_periodic_grid(tables: dict) -> None:
    """A boundary-free grid has all four, and they are distinct.

    Distinctness is what makes a single gather per slot well defined; it is not guaranteed
    on a grid with boundaries, which is why this asserts it only for the periodic case.
    """
    butterfly = tables["e2c2e2c"]
    assert (butterfly != INVALID).all(), "the simple grid is periodic, so no slot may be empty"
    for edge in range(butterfly.shape[0]):
        row = butterfly[edge].tolist()
        assert len(set(row)) == 4, f"edge {edge} has repeated butterfly cells: {row}"


def test_a_permuted_slot_order_is_rejected(tables: dict) -> None:
    """The order test has teeth: swapping the two sides must make it fail."""
    e2c, e2v, c2e2c, c2v = (tables[k] for k in ("e2c", "e2v", "c2e2c", "c2v"))
    permuted = tables["e2c2e2c"][:, [2, 3, 0, 1]]

    def side_and_vertex_hold() -> bool:
        return all(
            permuted[edge, 2 * side + vertex] in c2e2c[e2c[edge, side]]
            and e2v[edge, vertex] in c2v[permuted[edge, 2 * side + vertex]]
            for edge in range(e2c.shape[0])
            for side in range(2)
            for vertex in range(2)
        )

    assert not side_and_vertex_hold()


def test_grid_exposes_the_offset(tables: dict) -> None:
    grid = simple.simple_grid()
    connectivity = grid.get_connectivity(dims.E2C2E2C).asnumpy()
    np.testing.assert_array_equal(connectivity, tables["e2c2e2c"])
