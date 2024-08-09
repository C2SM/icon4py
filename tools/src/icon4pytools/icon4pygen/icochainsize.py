# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""
We encode the grid as follows.

 \|{-1, 1}                 \|{0, 1}                  \|
 -*-------------------------*-------------------------*-
  |\     {-1, 1, 0}         |\     {0, 1, 0}          |\
  | \                       | \                       |
  |  \                      |  \                      |
  |   \       {-1, 0, 1}    |   \       {0, 0, 1}     |
  |    \                    |    \                    |
  |     \                   |     \                   |
  |      \                  |      \                  |
  |       \                 |       \                 |
  |        \                |        \                |
  |         \               |         \               |
  |          \{-1, 0, 1}    |          \{0, 0, 1}     |
  |           \             |           \             |
  |            \            |            \            |
  |             \           |             \           |
  |{-1, 0, 2}    \          |{0, 0, 2}     \          |
  |               \         |               \         |
  |                \        |                \        |
  |                 \       |                 \       |
  |                  \      |                  \      |
  |                   \     |                   \     |
  |                    \    |                    \    |
  |   {-1, 0, 0}        \   |   {0, 0, 0}         \   |
  |                      \  |                      \  |
  |                       \ |                       \ |
 \|{-1, 0}                 \|{0, 0}                  \|
 -*-------------------------*-------------------------*-
  |\     {-1, 0, 0}         |\     {0, 0, 0}          |\
  | \                       | \                       |
  |  \                      |  \                      |
  |   \       {-1, -1, 1}   |   \       {0, -1, 1}    |
  |    \                    |    \                    |
  |     \                   |     \                   |
  |      \                  |      \                  |
  |       \                 |       \                 |
  |        \                |        \                |
  |         \               |         \               |
  |          \{-1, -1, 1}   |          \{0, -1, 1}    |
  |           \             |           \             |
  |            \            |            \            |
  |             \           |             \           |
  |{-1, -1, 2}   \          |{0, -1, 2}    \          |
  |               \         |               \         |
  |                \        |                \        |
  |                 \       |                 \       |
  |                  \      |                  \      |
  |                   \     |                   \     |
  |                    \    |                    \    |
  |   {-1, -1, 0}       \   |   {0, -1, 0}        \   |
  |                      \  |                      \  |
  |                       \ |                       \ |
 \|{-1, -1}                \|{0, -1}                 \|
 -*-------------------------*-------------------------*-
  |\     {-1, -1, 0}        |\     {0, -1, 0}         |\


 Which is described by this general pattern:

  |\
  | \
  |  \
  |   \       {x, y, 1}
  |    \
  |     \
  |      \
  |       \
  |        \
  |         \
  |          \{x, y, 1}
  |           \
  |            \
  |             \
  |{x, y, 2}     \
  |               \
  |                \
  |                 \
  |                  \
  |                   \
  |                    \
  |   {x, y, 0}         \
  |                      \
  |                       \
  |{x, y}                  \
  *-------------------------
         {x, y, 0}

 Note: Each location type uses a separate _id-space_.
 {x, y, 0} can both mean an edge or cell. It's up to the user to ensure
 they know what location type is meant.
 /
"""

from dataclasses import dataclass
from typing import ClassVar, List, TypeAlias

from gt4py.next.common import Dimension
from icon4py.model.common import dimension as dims


@dataclass
class Connection:
    start: Dimension
    end: Dimension


Position: TypeAlias = tuple[int, int, int]
HexCell: TypeAlias = tuple[Position, Position, Position, Position, Position, Position]


def vertex_to_edge(vertex: Position) -> HexCell:
    (x, y, _) = vertex
    return (
        (x, y, 0),
        (x, y, 2),
        (x - 1, y, 0),
        (x - 1, y, 1),
        (x, y - 1, 1),
        (x, y - 1, 2),
    )


def vertex_to_cell(vertex: Position) -> HexCell:
    (x, y, _) = vertex
    return (
        (x, y, 0),
        (x - 1, y, 0),
        (x - 1, y, 1),
        (x, y - 1, 0),
        (x, y - 1, 1),
        (x - 1, y - 1, 1),
    )


def edge_to_vertex(edge: Position) -> tuple[Position, Position]:
    (x, y, e) = edge
    if e == 0:
        return ((x, y, 0), (x + 1, y, 0))
    elif e == 1:
        return ((x + 1, y, 0), (x, y + 1, 0))
    elif e == 2:
        return ((x, y, 0), (x, y + 1, 0))
    else:
        raise Exception("Invalid edge type")


def edge_to_cell(edge: Position) -> tuple[Position, Position]:
    (x, y, e) = edge
    if e == 0:
        return ((x, y, 0), (x, y - 1, 1))
    elif e == 1:
        return ((x, y, 0), (x, y, 1))
    elif e == 2:
        return ((x, y, 0), (x - 1, y, 1))
    else:
        raise Exception("Invalid edge type")


def cell_to_vertex(cell: Position) -> tuple[Position, Position, Position]:
    (x, y, c) = cell
    if c == 0:
        return ((x, y, 0), (x + 1, y, 0), (x, y + 1, 0))
    elif c == 1:
        return ((x + 1, y + 1, 0), (x + 1, y, 0), (x, y + 1, 0))
    else:
        raise Exception("Invalid edge type")


def cell_to_edge(cell: Position) -> tuple[Position, Position, Position]:
    (x, y, c) = cell
    if c == 0:
        return ((x, y, 0), (x, y, 1), (x, y, 2))
    elif c == 1:
        return ((x, y, 1), (x + 1, y, 2), (x, y + 1, 0))
    else:
        raise Exception("Invalid edge type")


class IcoChainSize:
    """A class to compute the number of neighbors for a given neighbor chain."""

    _CHAIN_DISPATCHER: ClassVar[dict[str, list[Dimension]]] = {
        "vertex_to_edge": [dims.VertexDim, dims.EdgeDim],
        "vertex_to_cell": [dims.VertexDim, dims.CellDim],
        "edge_to_vertex": [dims.EdgeDim, dims.VertexDim],
        "edge_to_cell": [dims.EdgeDim, dims.CellDim],
        "cell_to_vertex": [dims.CellDim, dims.VertexDim],
        "cell_to_edge": [dims.CellDim, dims.EdgeDim],
    }

    @classmethod
    def get(cls, chain: List[Dimension]) -> int:
        previous_location_type = chain[0]
        previous_locations = {(0, 0, 0)}

        for loc_type in chain[1::]:
            assert loc_type != previous_location_type
            connection = Connection(previous_location_type, loc_type)

            current_locations = set()
            for func_name, _dims in cls._CHAIN_DISPATCHER.items():
                if [connection.start, connection.end] == _dims:
                    func = globals()[func_name]
                    for previous_location in previous_locations:
                        neighbors = func(previous_location)
                        current_locations.update(neighbors)

            previous_locations = current_locations
            previous_location_type = loc_type

        if chain[0] == chain[-1]:
            return len(previous_locations) - 1
        return len(previous_locations)
