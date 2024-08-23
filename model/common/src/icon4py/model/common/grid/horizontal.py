# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
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


class HorizontalMarkerIndex:
    """
    Handles constants indexing into the start_index and end_index fields.

     ICON uses a double indexing scheme for field indices marking the start and end of special
     grid zone: The constants defined here (from mo_impl_constants.f90 and mo_impl_constants_grf.f90)
     are the indices that are used to index into the start_idx and end_idx arrays
     provided by the grid file where for each dimension the start index of the horizontal
     "zones" are defined:
     f.ex. an inlined access of the field F: Field[[CellDim], double] at the starting point of the lateral boundary zone would be

     F[start_idx_c[_LATERAL_BOUNDARY_CELLS]


     ICON uses a custom index range from [ICON_INDEX_OFFSET... ] such that the index 0 marks the
     internal entities for _all_ dimensions (Cell, Edge, Vertex) that is why we define these
     additional INDEX_OFFSETs here in order to swap back to a 0 base python array.

    """

    _lateral_boundary: ClassVar = {
        dims.CellDim: _LATERAL_BOUNDARY_CELLS,
        dims.EdgeDim: _LATERAL_BOUNDARY_EDGES,
        dims.VertexDim: _LATERAL_BOUNDARY_VERTICES,
    }
    _local: ClassVar = {
        dims.CellDim: _LOCAL_CELLS,
        dims.EdgeDim: _LOCAL_EDGES,
        dims.VertexDim: _LOCAL_VERTICES,
    }
    _halo: ClassVar = {
        dims.CellDim: _HALO_CELLS,
        dims.EdgeDim: _HALO_EDGES,
        dims.VertexDim: _HALO_VERTICES,
    }
    _interior: ClassVar = {
        dims.CellDim: _INTERIOR_CELLS,
        dims.EdgeDim: _INTERIOR_EDGES,
        dims.VertexDim: _INTERIOR_VERTICES,
    }
    _nudging: ClassVar = {
        dims.CellDim: _NUDGING_CELLS,
        dims.EdgeDim: _NUDGING_EDGES,
        # TODO [magdalena] there is no nudging for vertices?
        dims.VertexDim: _NUDGING_VERTICES,
    }
    _end: ClassVar = {
        dims.CellDim: _END_CELLS,
        dims.EdgeDim: _END_EDGES,
        dims.VertexDim: _END_VERTICES,
    }

    _bounds: ClassVar = {
        dims.CellDim: (0, _CELL_GRF - 1),
        dims.EdgeDim: (0, _EDGE_GRF - 1),
        dims.VertexDim: (0, _VERTEX_GRF - 1),
    }

    @classmethod
    def lateral_boundary(cls, dim: gtx.Dimension, offset=0) -> int:
        """Indicate lateral boundary.

        These points correspond to the sorted points in ICON, the marker can be incremented in order
        to access higher order boundary lines
        """
        return cls._domain_index(cls._lateral_boundary, dim, offset)

    @classmethod
    def _domain_index(cls, value_dict, dim, offset):
        index = value_dict[dim] + offset
        assert (
            index <= cls._bounds[dim][1]
        ), f"Index {index} out of bounds for {dim}:  {cls._bounds[dim]}"
        assert (
            index >= cls._bounds[dim][0]
        ), f"Index {index} out of bounds for {dim}: {cls._bounds[dim]}"
        return index

    @classmethod
    def local(cls, dim: gtx.Dimension, offset=0) -> int:
        """Indicate points that are owned by the processing unit, i.e. no halo points."""
        return cls._domain_index(cls._local, dim, offset)

    @classmethod
    def halo(cls, dim: gtx.Dimension, offset=0) -> int:
        return cls._domain_index(cls._halo, dim, offset)

    @classmethod
    def nudging(cls, dim: gtx.Dimension, offset=0) -> int:
        """Indicate the nudging zone."""
        return cls._domain_index(cls._nudging, dim, offset)

    @classmethod
    def nudging_2nd_level(cls, dim: gtx.Dimension) -> int:
        """Indicate the nudging zone for 2nd level."""
        return cls.nudging(dim, 1)

    @classmethod
    def interior(cls, dim: gtx.Dimension, offset=0) -> int:
        """Indicate interior i.e. unordered prognostic cells in ICON."""
        return cls._domain_index(cls._interior, dim, offset)

    @classmethod
    def end(cls, dim: gtx.Dimension) -> int:
        return cls._end[dim]


class LineNumber(enum.IntEnum):
    pass


class HaloLine(LineNumber):
    FIRST = 0
    SECOND = -1


class NudgingLine(LineNumber):
    FIRST = 0
    SECOND = 1


class Level(LineNumber):
    HALO = -1
    FIRST = 0
    SECOND = 1
    THIRD = 2
    FOURTH = 3
    FIFTH = 4
    SIXTH = 5
    SEVENTH = 6


class IndexType(enum.IntEnum):
    START = 0
    END = 1


class Marker(str, enum.Enum):
    """Used to encode the different horizontal domain zones used in ICON.
    The values vaguely corrspond to the constants used in ICON to pass as rl_start or rl_end to the get_indices_[e_c,v] functions.

    The translation table is as follows:



    """

    END = "end"
    INTERIOR = "interior"
    HALO = "halo_level_1"
    HALO_LEVEL_2 = "halo_level_2"
    LOCAL = "local"
    LATERAL_BOUNDARY = "lb_level_1"
    LATERAL_BOUNDARY_LEVEL_2 = "lb_level_2"
    LATERAL_BOUNDARY_LEVEL_3 = "lb_level_3"
    LATERAL_BOUNDARY_LEVEL_4 = "lb_level_4"
    LATERAL_BOUNDARY_LEVEL_5 = "lb_level_5"
    LATERAL_BOUNDARY_LEVEL_6 = "lb_level_6"
    LATERAL_BOUNDARY_LEVEL_7 = "lb_level_7"
    NUDGING = "nudging_level_1"
    NUDGING_LEVEL_2 = "nudging_level_2"


def map_to_index(dim: gtx.Dimension, marker: Marker) -> int:
    if marker == Marker.END:
        return HorizontalMarkerIndex.end(dim)
    elif marker == Marker.INTERIOR:
        return HorizontalMarkerIndex.interior(dim)
    elif marker == Marker.HALO:
        return HorizontalMarkerIndex.halo(dim, Level.FIRST)
    elif marker == Marker.HALO_LEVEL_2:
        return HorizontalMarkerIndex.halo(dim, Level.HALO)
    elif marker == Marker.LOCAL:
        return HorizontalMarkerIndex.local(dim)
    elif marker == Marker.LATERAL_BOUNDARY:
        return HorizontalMarkerIndex.lateral_boundary(dim, Level.FIRST)
    elif marker == Marker.LATERAL_BOUNDARY_LEVEL_2:
        return HorizontalMarkerIndex.lateral_boundary(dim, Level.SECOND)
    elif marker == Marker.LATERAL_BOUNDARY_LEVEL_3:
        return HorizontalMarkerIndex.lateral_boundary(dim, Level.THIRD)
    elif marker == Marker.LATERAL_BOUNDARY_LEVEL_4:
        return HorizontalMarkerIndex.lateral_boundary(dim, Level.FOURTH)
    elif marker == Marker.LATERAL_BOUNDARY_LEVEL_5:
        return HorizontalMarkerIndex.lateral_boundary(dim, Level.FIFTH)
    elif marker == Marker.LATERAL_BOUNDARY_LEVEL_6:
        return HorizontalMarkerIndex.lateral_boundary(dim, Level.SIXTH)
    elif marker == Marker.LATERAL_BOUNDARY_LEVEL_7:
        return HorizontalMarkerIndex.lateral_boundary(dim, Level.SEVENTH)
    elif marker == Marker.NUDGING:
        return HorizontalMarkerIndex.nudging(dim, Level.FIRST)
    elif marker == Marker.NUDGING_LEVEL_2:
        return HorizontalMarkerIndex.nudging(dim, Level.SECOND)
    else:
        raise ValueError(f"Unknown marker {marker}")


class Domain(Protocol):
    _dim: gtx.Dimension
    _marker: Marker
    _index: int

    @abstractmethod
    def _valid(self, marker: Marker) -> bool:
        ...

    def marker(self, marker: Marker):
        assert self._valid(marker), f" Domain `{marker}` not a valid for use with '{self.dim}'"
        self._marker = marker
        self._index = map_to_index(self.dim, marker)
        return self

    @property
    def dim(self) -> gtx.Dimension:
        return self._dim

    @functools.cached_property
    def local(self) -> bool:
        return self._marker == Marker.LOCAL

    def __call__(self) -> int:
        return self._index


def domain(dim: gtx.Dimension):
    def _domain(marker: Marker):
        return _domain_factory(dim, marker)

    assert dim.kind == gtx.DimensionKind.HORIZONTAL, "Only defined for horizontal dimensions"
    if dim == dims.CellDim:
        return _domain
    elif dim == dims.EdgeDim:
        return _domain
    else:
        return _domain


def _domain_factory(dim: gtx.Dimension, marker: Marker):
    if dim == dims.CellDim:
        return CellDomain().marker(marker)
    elif dim == dims.EdgeDim:
        return EdgeDomain().marker(marker)
    else:
        return VertexDomain().marker(marker)


class EdgeDomain(Domain):
    _dim = dims.EdgeDim

    def _valid(self, marker: Marker):
        return True


class VertexDomain(Domain):
    _dim = dims.VertexDim

    def _valid(self, marker: Marker):
        return marker in (
            Marker.END,
            Marker.INTERIOR,
            Marker.HALO,
            Marker.HALO_LEVEL_2,
            Marker.LOCAL,
            Marker.LATERAL_BOUNDARY,
            Marker.LATERAL_BOUNDARY_LEVEL_2,
            Marker.LATERAL_BOUNDARY_LEVEL_3,
            Marker.LATERAL_BOUNDARY_LEVEL_4,
        )


class CellDomain(Domain):
    _dim = dims.CellDim

    def _valid(self, marker: Marker):
        return marker in (
            Marker.END,
            Marker.INTERIOR,
            Marker.HALO,
            Marker.HALO_LEVEL_2,
            Marker.LOCAL,
            Marker.LATERAL_BOUNDARY,
            Marker.LATERAL_BOUNDARY_LEVEL_2,
            Marker.LATERAL_BOUNDARY_LEVEL_3,
            Marker.LATERAL_BOUNDARY_LEVEL_4,
            Marker.NUDGING,
        )


@dataclasses.dataclass(frozen=True)
class HorizontalGridSize:
    num_vertices: int
    num_edges: int
    num_cells: int


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
