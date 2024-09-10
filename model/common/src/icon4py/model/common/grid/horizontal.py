# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
 This module handles several aspects of the horizontal grid in ICON.

 Among which most importantly:

 Horizontal domain zones
 -------------------------
 ICON provides three routines `get_indices_c`, `get_indices_e` and `get_indices_v` which return indices into Fields of the given dimension
 that mark the start and end of specific horizontal grid domains such as the lateral boundaries, nudging zones etc.

 Those routines get passed an integer value normally called `rl_start` or `rl_end`. The values ranges over a custom index range
 for each dimension, some of which are denoted by constants defined in `mo_impl_constants.f90` and `mo_impl_constants_grf.f90`.

 Internally ICON uses a double indexing scheme for those start and end indices. They are 
 stored in arrays `start_idx` and `end_idx` originally read from the grid file ICON accesses those indices by a custom index range
 denoted by the constants mentioned above. However, some entries into these arrays contain invalid Field indices and must not 
 be used ever.

 horizontal.py provides an interface to a Python port of constants wrapped in a custom `Domain` class, which takes care of the
 custom index range and makes sure that for each dimension only legal values can be passed. 

 The horizontal domain zones are denoted by a set of named enums for the different zones: 
 see Fig. 8.2 in the official [ICON tutorial](https://www.dwd.de/DE/leistungen/nwv_icon_tutorial/pdf_einzelbaende/icon_tutorial2024.html).


"""

import dataclasses
import enum
import functools
from abc import abstractmethod
from typing import ClassVar, Final, Protocol

import gt4py.next as gtx

from icon4py.model.common import dimension as dims


NUM_GHOST_ROWS: Final[int] = 2
# values from mo_impl_constants.f90
_ICON_INDEX_OFFSET_CELLS: Final[int] = 8
_GRF_BOUNDARY_WIDTH_CELL: Final[int] = 4
_MIN_RL_CELL_INT: Final[int] = -4
_MIN_RL_CELL: Final[int] = _MIN_RL_CELL_INT - 2 * NUM_GHOST_ROWS
_MAX_RL_CELL: Final[int] = 5

_ICON_INDEX_OFFSET_VERTEX: Final[int] = 7
_MIN_RL_VERTEX_INT: Final[int] = _MIN_RL_CELL_INT
_MIN_RL_VERTEX: Final[int] = _MIN_RL_VERTEX_INT - (NUM_GHOST_ROWS + 1)
_MAX_RL_VERTEX: Final[int] = _MAX_RL_CELL

_ICON_INDEX_OFFSET_EDGES: Final[int] = 13
_GRF_BOUNDARY_WIDTH_EDGES: Final[int] = 9
_GRF_NUDGEZONE_START_EDGES: Final[int] = _GRF_BOUNDARY_WIDTH_EDGES + 1
_GRF_NUDGEZONE_WIDTH: Final[int] = 8
_MIN_RL_EDGE_INT: Final[int] = 2 * _MIN_RL_CELL_INT
_MIN_RL_EDGE: Final[int] = _MIN_RL_EDGE_INT - (2 * NUM_GHOST_ROWS + 1)
_MAX_RL_EDGE: Final[int] = 2 * _MAX_RL_CELL

_LATERAL_BOUNDARY_EDGES: Final[int] = 1 + _ICON_INDEX_OFFSET_EDGES  # 14
_INTERIOR_EDGES: Final[int] = _ICON_INDEX_OFFSET_EDGES  # 13
_NUDGING_EDGES: Final[int] = _GRF_BOUNDARY_WIDTH_EDGES + _ICON_INDEX_OFFSET_EDGES  # 22
_HALO_EDGES: Final[int] = _MIN_RL_EDGE_INT - 1 + _ICON_INDEX_OFFSET_EDGES  # 4
_LOCAL_EDGES: Final[int] = _MIN_RL_EDGE_INT + _ICON_INDEX_OFFSET_EDGES  # 5
_END_EDGES: Final[int] = 0

_LATERAL_BOUNDARY_CELLS: Final[int] = 1 + _ICON_INDEX_OFFSET_CELLS  # 9
_INTERIOR_CELLS: Final[int] = _ICON_INDEX_OFFSET_CELLS  # 8
_NUDGING_CELLS: Final[int] = _GRF_BOUNDARY_WIDTH_CELL + 1 + _ICON_INDEX_OFFSET_CELLS  # 13
_HALO_CELLS: Final[int] = _MIN_RL_CELL_INT - 1 + _ICON_INDEX_OFFSET_CELLS  # 3
_LOCAL_CELLS: Final[int] = _MIN_RL_CELL_INT + _ICON_INDEX_OFFSET_CELLS  # 4
_END_CELLS: Final[int] = 0

_LATERAL_BOUNDARY_VERTICES = 1 + _ICON_INDEX_OFFSET_VERTEX  # 8
_INTERIOR_VERTICES: Final[int] = _ICON_INDEX_OFFSET_VERTEX  # 7
_NUDGING_VERTICES: Final[int] = 0
_HALO_VERTICES: Final[int] = _MIN_RL_VERTEX_INT - 1 + _ICON_INDEX_OFFSET_VERTEX  # 2
_LOCAL_VERTICES: Final[int] = _MIN_RL_VERTEX_INT + _ICON_INDEX_OFFSET_VERTEX  # 3
_END_VERTICES: Final[int] = 0


_EDGE_GRF: Final[int] = 24
_CELL_GRF: Final[int] = 14
_VERTEX_GRF: Final[int] = 13


_LATERAL_BOUNDARY = {
    dims.CellDim: _LATERAL_BOUNDARY_CELLS,
    dims.EdgeDim: _LATERAL_BOUNDARY_EDGES,
    dims.VertexDim: _LATERAL_BOUNDARY_VERTICES,
}
_LOCAL = {
    dims.CellDim: _LOCAL_CELLS,
    dims.EdgeDim: _LOCAL_EDGES,
    dims.VertexDim: _LOCAL_VERTICES,
}
_HALO = {
    dims.CellDim: _HALO_CELLS,
    dims.EdgeDim: _HALO_EDGES,
    dims.VertexDim: _HALO_VERTICES,
}
_INTERIOR = {
    dims.CellDim: _INTERIOR_CELLS,
    dims.EdgeDim: _INTERIOR_EDGES,
    dims.VertexDim: _INTERIOR_VERTICES,
}
_NUDGING = {
    dims.CellDim: _NUDGING_CELLS,
    dims.EdgeDim: _NUDGING_EDGES,
    dims.VertexDim: _NUDGING_VERTICES,
}
_END = {
    dims.CellDim: _END_CELLS,
    dims.EdgeDim: _END_EDGES,
    dims.VertexDim: _END_VERTICES,
}

_BOUNDS = {
    dims.CellDim: (0, _CELL_GRF - 1),
    dims.EdgeDim: (0, _EDGE_GRF - 1),
    dims.VertexDim: (0, _VERTEX_GRF - 1),
}


class LineNumber(enum.IntEnum):
    HALO = -1
    FIRST = 0
    SECOND = 1
    THIRD = 2
    FOURTH = 3
    FIFTH = 4
    SIXTH = 5
    SEVENTH = 6
    EIGHTH = 7


def _lateral_boundary(dim: gtx.Dimension, offset=LineNumber.FIRST) -> int:
    """Indicate lateral boundary.

    These points correspond to the sorted points in ICON, the marker can be incremented in order
    to access higher order boundary lines
    """
    return _domain_index(_LATERAL_BOUNDARY, dim, offset)


def _domain_index(value_dict, dim: gtx.Dimension, offset: LineNumber) -> int:
    index = value_dict[dim] + offset
    assert index <= _BOUNDS[dim][1], f"Index {index} out of bounds for {dim}:  {_BOUNDS[dim]}"
    assert index >= _BOUNDS[dim][0], f"Index {index} out of bounds for {dim}: {_BOUNDS[dim]}"
    return index


def _local(dim: gtx.Dimension, offset=LineNumber.FIRST) -> int:
    """
    Indicate points that are owned by the processing unit, i.e. non halo points.

    This is true to the exception that it excludes points in the halo lines. For classical ICON ordering
    this zone might include halo points that are part of lateral boundary and are ordered in the lateral boundary zone.
    It is there _not_ identical to the fully correct owner mask in the [DecompositionInfo](../../../../../decomposition/definitions.py)
    """
    return _domain_index(_LOCAL, dim, offset)


def _halo(dim: gtx.Dimension, offset=LineNumber.FIRST) -> int:
    return _domain_index(_HALO, dim, offset)


def _nudging(dim: gtx.Dimension, offset=LineNumber.FIRST) -> int:
    """Indicate the nudging zone."""
    return _domain_index(_NUDGING, dim, offset)


def _interior(dim: gtx.Dimension, offset=LineNumber.FIRST) -> int:
    """Indicate interior i.e. unordered prognostic cells in ICON."""
    return _domain_index(_INTERIOR, dim, offset)


def _end(dim: gtx.Dimension) -> int:
    return _END[dim]


class Zone(str, enum.Enum):
    """
    Enum of different zones on the horizontal ICON grid.
    The mapping to the constant used in ICON is as follows: (note that not all values exist for all dimensions


    ## CellDim
    | ICON constant or value    | ICON4py Name               |
    |:------------------------- |:-------------------------- |
    | `min_rlcell` (-8)         | `END`                      |
    | `min_rlcell_int-2`,  (-6) | `HALO_LEVEL_2`             |
    | `min_rlcell_int-1` (-5)   | `HALO`                     |
    | `min_rlcell_int`(-4)      | `LOCAL`                    |
    | `0`                       | `INTERIOR`                 |
    | `1`                       | `LATERAL_BOUNDARY`         |
    | `2`                       | `LATERAL_BOUNDARY_LEVEL_2` |
    | `3`                       | `LATERAL_BOUNDARY_LEVEL_3` |
    | `grf_bdywidth_c` (4)      | `LATERAL_BOUNDARY_LEVEL_4` |
    | `grf_bdywith_c +1` (5)    | `NUDGING`                  |
    | `grf_bdywidth_c+2` (6)    | `NUDGING_LEVEL_2`          |

    Lateral boundary and nudging are only relevant for LAM runs, halo lines only for distributed domains.
    The constants are defined in `mo_impl_constants.f90` and `mo_impl_constants_grf.f90`
    ## VertexDim


    | ICON constant or value                  | ICON4Py Name               |
    |:--------------------------------------- |:-------------------------- |
    | `min_rlvert` (-7)                       | `END`                      |
    | `min_rlvert+1`, `min_rlvert_int-2` (-6) | `HALO_LEVEL_2`             |
    | `min_rlvert_int-1` (-5)                 | `HALO`                     |
    | `min_rlvert_int` (-4)                   | `LOCAL`                    |
    | `0`                                     | `INTERIOR`                 |
    | `1`                                     | `LATERAL_BOUNDARY`         |
    | `2`                                     | `LATERAL_BOUNDARY_LEVEL_2` |
    | `3`                                     | `LATERAL_BOUNDARY_LEVEL_3` |
    | `4`                                     | `LATERAL_BOUNDARY_LEVEL_4` |
    | `max_rlvert` (5)                        |                            |

    For the meaning see above.

    ##EdgeDim


    | ICON constant or value                 | ICON4Py Name               |
    |:-------------------------------------- |:-------------------------- |
    | `min_rledge` (-13)                     | `END`                      |
    | `min_rledge_int-2`            (-10)    | `HALO_LEVEL_2`             |
    | `min_rledge_int-1` (-9)                | `HALO`                     |
    | `min_rledge_int` (-8)                  | `LOCAL`                    |
    | `0`                                    | `INTERIOR`                 |
    | `1`                                    | `LATERAL_BOUNDARY`         |
    | `2`                                    | `LATERAL_BOUNDARY_LEVEL_2` |
    | `3`                                    | `LATERAL_BOUNDARY_LEVEL_3` |
    | `4`                                    | `LATERAL_BOUNDARY_LEVEL_4` |
    | `5`                                    | `LATERAL_BOUNDARY_LEVEL_5` |
    | `6`                                    | `LATERAL_BOUNDARY_LEVEL_6` |
    | `7`                                    | `LATERAL_BOUNDARY_LEVEL_7` |
    | `8`                                    | `LATERAL_BOUNDARY_LEVEL_8` |
    | `grf_bdywidth_e`   (9)                 | `NUDGING`                  |
    | `grf_bdywidth_e+1` `max_rledge`   (10) | `NUDGING_LEVEL_2`          |


    """

    #: points the the number of entries in a local grid
    END = "end"

    #: interior unordered prognostic entries
    INTERIOR = "interior"

    #: first halo line
    HALO = "halo_level_1"

    #: 2nd halo line
    HALO_LEVEL_2 = "halo_level_2"

    #: all entries owned on the local grid, that is all entries excluding halo lines
    LOCAL = "local"

    #: lateral boundary (row 1) in LAM model
    LATERAL_BOUNDARY = "lb_level_1"

    #: lateral boundary (row 2) in LAM model
    LATERAL_BOUNDARY_LEVEL_2 = "lb_level_2"

    # ; lateral boundary (row 3) in LAM model
    LATERAL_BOUNDARY_LEVEL_3 = "lb_level_3"

    #: lateral boundary (row 4) in LAM model
    LATERAL_BOUNDARY_LEVEL_4 = "lb_level_4"

    #: lateral boundary (row 5) in LAM model
    LATERAL_BOUNDARY_LEVEL_5 = "lb_level_5"

    #: lateral boundary (row 6) in LAM model
    LATERAL_BOUNDARY_LEVEL_6 = "lb_level_6"

    #: lateral boundary (row 7) in LAM model
    LATERAL_BOUNDARY_LEVEL_7 = "lb_level_7"

    #: lateral boundary (row 8) in LAM model
    LATERAL_BOUNDARY_LEVEL_8 = "lb_level_8"

    #: nudging level in LAM model
    NUDGING = "nudging_level_1"

    #: 2nd nudging level in LAM model
    NUDGING_LEVEL_2 = "nudging_level_2"


def _map_to_index(dim: gtx.Dimension, marker: Zone) -> int:
    match marker:
        case Zone.END:
            return _end(dim)
        case Zone.INTERIOR:
            return _interior(dim)
        case Zone.HALO:
            return _halo(dim, LineNumber.FIRST)
        case Zone.HALO_LEVEL_2:
            return _halo(dim, LineNumber.HALO)
        case Zone.LOCAL:
            return _local(dim)
        case Zone.LATERAL_BOUNDARY:
            return _lateral_boundary(dim, LineNumber.FIRST)
        case Zone.LATERAL_BOUNDARY_LEVEL_2:
            return _lateral_boundary(dim, LineNumber.SECOND)
        case Zone.LATERAL_BOUNDARY_LEVEL_3:
            return _lateral_boundary(dim, LineNumber.THIRD)
        case Zone.LATERAL_BOUNDARY_LEVEL_4:
            return _lateral_boundary(dim, LineNumber.FOURTH)
        case Zone.LATERAL_BOUNDARY_LEVEL_5:
            return _lateral_boundary(dim, LineNumber.FIFTH)
        case Zone.LATERAL_BOUNDARY_LEVEL_6:
            return _lateral_boundary(dim, LineNumber.SIXTH)
        case Zone.LATERAL_BOUNDARY_LEVEL_7:
            return _lateral_boundary(dim, LineNumber.SEVENTH)
        case Zone.LATERAL_BOUNDARY_LEVEL_8:
            return _lateral_boundary(dim, LineNumber.EIGHTH)
        case Zone.NUDGING:
            return _nudging(dim, LineNumber.FIRST)
        case Zone.NUDGING_LEVEL_2:
            return _nudging(dim, LineNumber.SECOND)
        case _:
            raise ValueError(f"Unknown marker {marker}")


class Domain(Protocol):
    """
    Interface for a domain object.

    Used to access horizontal domain zones in the ICON grid.
    """

    _dim: gtx.Dimension
    _marker: Zone
    _index: int

    def __str__(self):
        return f"{self.dim}: {self._marker} /[ {self._index}]"

    @abstractmethod
    def _valid(self, marker: Zone) -> bool:
        ...

    def marker(self, marker: Zone):
        assert self._valid(marker), f" Domain `{marker}` not a valid zone for use with '{self.dim}'"
        self._marker = marker
        self._index = _map_to_index(self.dim, marker)
        return self

    @property
    def dim(self) -> gtx.Dimension:
        return self._dim

    @functools.cached_property
    def local(self) -> bool:
        return self._marker == Zone.LOCAL

    def __call__(self) -> int:
        return self._index


def domain(dim: gtx.Dimension):
    """
    Factory function to create a domain object for a given dimension.

    This is the main entry point to create a domain object for a given dimension. In order to access the start or end index for
    `INTERIOR` (unordered prognostic) cells in ICON one would call:

    >>> import icon4py.model.common.grid.icon as icon_grid
    >>> grid = icon_grid.IconGrid()
    >>> domain = domain(dims.CellDim)(Zone.INTERIOR)
    >>> start_index = grid.start_index(domain)



    """

    def _domain(marker: Zone):
        return _domain_factory(dim, marker)

    assert dim.kind == gtx.DimensionKind.HORIZONTAL, "Only defined for horizontal dimensions"
    return _domain


def _domain_factory(dim: gtx.Dimension, marker: Zone):
    if dim == dims.CellDim:
        return CellDomain().marker(marker)
    elif dim == dims.EdgeDim:
        return EdgeDomain().marker(marker)
    else:
        return VertexDomain().marker(marker)


class EdgeDomain(Domain):
    """Domain object for the Edge dimension."""

    _dim = dims.EdgeDim

    def _valid(self, marker: Zone):
        return True


class VertexDomain(Domain):
    """Domain object for the Vertex dimension."""

    _dim = dims.VertexDim

    def _valid(self, marker: Zone):
        return marker in (
            Zone.END,
            Zone.INTERIOR,
            Zone.HALO,
            Zone.HALO_LEVEL_2,
            Zone.LOCAL,
            Zone.LATERAL_BOUNDARY,
            Zone.LATERAL_BOUNDARY_LEVEL_2,
            Zone.LATERAL_BOUNDARY_LEVEL_3,
            Zone.LATERAL_BOUNDARY_LEVEL_4,
        )


class CellDomain(Domain):
    """Domain object for the Cell dimension."""

    _dim = dims.CellDim

    def _valid(self, marker: Zone):
        return marker in (
            Zone.END,
            Zone.INTERIOR,
            Zone.HALO,
            Zone.HALO_LEVEL_2,
            Zone.LOCAL,
            Zone.LATERAL_BOUNDARY,
            Zone.LATERAL_BOUNDARY_LEVEL_2,
            Zone.LATERAL_BOUNDARY_LEVEL_3,
            Zone.LATERAL_BOUNDARY_LEVEL_4,
            Zone.NUDGING,
        )


# TODO (@ halungge): maybe this should to a separate module
@dataclasses.dataclass(frozen=True)
class HorizontalGridSize:
    num_vertices: int
    num_edges: int
    num_cells: int


# TODO (@ halungge): maybe this should to a separate module
class RefinCtrlLevel:
    _boundary_nudging_start: ClassVar = {
        dims.EdgeDim: _GRF_BOUNDARY_WIDTH_EDGES + 1,
        dims.CellDim: _GRF_BOUNDARY_WIDTH_CELL + 1,
    }

    @classmethod
    def boundary_nudging_start(cls, dim: gtx.Dimension) -> int:
        """Start refin_ctrl levels for boundary nudging (as seen from the child domain)."""
        try:
            return cls._boundary_nudging_start[dim]
        except KeyError as err:
            raise ValueError(
                f"nudging start level only exists for {dims.CellDim} and {dims.EdgeDim}"
            ) from err
