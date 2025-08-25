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
from collections.abc import Callable
from typing import Any, Final

import gt4py.next as gtx
import numpy as np

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

_LATERAL_BOUNDARY_VERTICES: Final[int] = 1 + _ICON_INDEX_OFFSET_VERTEX  # 8
_INTERIOR_VERTICES: Final[int] = _ICON_INDEX_OFFSET_VERTEX  # 7
_NUDGING_VERTICES: Final[int] = 0
_HALO_VERTICES: Final[int] = _MIN_RL_VERTEX_INT - 1 + _ICON_INDEX_OFFSET_VERTEX  # 2
_LOCAL_VERTICES: Final[int] = _MIN_RL_VERTEX_INT + _ICON_INDEX_OFFSET_VERTEX  # 3
_END_VERTICES: Final[int] = 0

_EDGE_GRF: Final[int] = 24
_CELL_GRF: Final[int] = 14
_VERTEX_GRF: Final[int] = 13

GRID_REFINEMENT_SIZE: Final[dict[gtx.Dimension, int]] = {
    dims.CellDim: _CELL_GRF,
    dims.EdgeDim: _EDGE_GRF,
    dims.VertexDim: _VERTEX_GRF,
}


_LATERAL_BOUNDARY: Final[dict[gtx.Dimension, int]] = {
    dims.CellDim: _LATERAL_BOUNDARY_CELLS,
    dims.EdgeDim: _LATERAL_BOUNDARY_EDGES,
    dims.VertexDim: _LATERAL_BOUNDARY_VERTICES,
}
_LOCAL: Final[dict[gtx.Dimension, int]] = {
    dims.CellDim: _LOCAL_CELLS,
    dims.EdgeDim: _LOCAL_EDGES,
    dims.VertexDim: _LOCAL_VERTICES,
}
_HALO: Final[dict[gtx.Dimension, int]] = {
    dims.CellDim: _HALO_CELLS,
    dims.EdgeDim: _HALO_EDGES,
    dims.VertexDim: _HALO_VERTICES,
}
_INTERIOR: Final[dict[gtx.Dimension, int]] = {
    dims.CellDim: _INTERIOR_CELLS,
    dims.EdgeDim: _INTERIOR_EDGES,
    dims.VertexDim: _INTERIOR_VERTICES,
}

_NUDGING: Final[dict[gtx.Dimension, int]] = {
    dims.CellDim: _NUDGING_CELLS,
    dims.EdgeDim: _NUDGING_EDGES,
    dims.VertexDim: _NUDGING_VERTICES,
}
_END: Final[dict[gtx.Dimension, int]] = {
    dims.CellDim: _END_CELLS,
    dims.EdgeDim: _END_EDGES,
    dims.VertexDim: _END_VERTICES,
}

_BOUNDS: Final[dict[gtx.Dimension, tuple[int, int]]] = {
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


def _lateral_boundary(dim: gtx.Dimension, offset: LineNumber = LineNumber.FIRST) -> int:
    """Indicate lateral boundary.

    These points correspond to the sorted points in ICON, the marker can be incremented in order
    to access higher order boundary lines
    """
    return _domain_index(_LATERAL_BOUNDARY, dim, offset)


def _domain_index(value_dict: dict, dim: gtx.Dimension, offset: LineNumber) -> int:
    index = value_dict[dim] + offset
    assert index <= _BOUNDS[dim][1], f"Index {index} out of bounds for {dim}:  {_BOUNDS[dim]}"
    assert index >= _BOUNDS[dim][0], f"Index {index} out of bounds for {dim}: {_BOUNDS[dim]}"
    return index


def _local(dim: gtx.Dimension, offset: LineNumber = LineNumber.FIRST) -> int:
    """
    Indicate points that are owned by the processing unit, i.e. non halo points.

    This is true to the exception that it excludes points in the halo lines. For classical ICON ordering
    this zone might include halo points that are part of lateral boundary and are ordered in the lateral boundary zone.
    It is there _not_ identical to the fully correct owner mask in the [DecompositionInfo](../../../../../decomposition/definitions.py)
    """
    return _domain_index(_LOCAL, dim, offset)


def _halo(dim: gtx.Dimension, offset: LineNumber = LineNumber.FIRST) -> int:
    return _domain_index(_HALO, dim, offset)


def _nudging(dim: gtx.Dimension, offset: LineNumber = LineNumber.FIRST) -> int:
    """Indicate the nudging zone."""
    return _domain_index(_NUDGING, dim, offset)


def _interior(dim: gtx.Dimension, offset: LineNumber = LineNumber.FIRST) -> int:
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

    def is_halo(self) -> bool:
        return self in (Zone.HALO, Zone.HALO_LEVEL_2)


VERTEX_ZONES = (
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


CELL_ZONES = (
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

EDGE_ZONES = tuple(Zone)


def _map_to_icon_index(dim: gtx.Dimension, marker: Zone) -> int:
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


@dataclasses.dataclass(frozen=True)
class Domain:
    """
    Domain Description on the horizontal grid
    Used to access domain bounds in concrete the ICON grid.
    """

    _dim: gtx.Dimension
    _zone: Zone

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Domain):
            return self.dim == other.dim and self.zone == other.zone
        return False

    def __hash__(self) -> int:
        return hash((self.dim, self.zone))

    def __str__(self) -> str:
        return f"Domain (dim = {self.dim}: zone = {self._zone} /ICON index[ {_map_to_icon_index(self.dim, self.zone)} ])"

    @property
    def zone(self) -> Zone:
        return self._zone

    @property
    def dim(self) -> gtx.Dimension:
        return self._dim

    @functools.cached_property
    def is_local(self) -> bool:
        return self._zone == Zone.LOCAL


def domain(dim: gtx.Dimension) -> Callable[[Zone], Domain]:
    """
    Factory function to create a domain object for a given dimension.

    This is the main entry point to create a domain object for a given dimension. In order to access the start or end index for
    `INTERIOR` (unordered prognostic) cells in ICON one would call:

    >>> import icon4py.model.common.grid.icon as icon_grid
    >>> grid = icon_grid.IconGrid()
    >>> domain = domain(dims.CellDim)(Zone.INTERIOR)
    >>> start_index = grid.start_index(domain)



    """

    def _domain(marker: Zone) -> Domain:
        return _domain_factory(dim, marker)

    assert dim.kind == gtx.DimensionKind.HORIZONTAL, "Only defined for horizontal dimensions"
    return _domain


def _domain_factory(dim: gtx.Dimension, zone: Zone) -> Domain:
    assert _validate(
        dim, zone
    ), f"Invalid zone {zone} for dimension {dim}. Valid zones are: {get_zones_for_dim(dim)}"
    return Domain(dim, zone)


def _validate(dim: gtx.Dimension, marker: Zone) -> bool:
    match dim:
        case dims.CellDim:
            return marker in CELL_ZONES
        case dims.EdgeDim:
            return marker in EDGE_ZONES
        case dims.VertexDim:
            return marker in VERTEX_ZONES
        case _:
            return False


def get_zones_for_dim(dim: gtx.Dimension) -> tuple[Zone, ...]:
    """
    Get the grid zones valid for a given horizontal dimension in ICON .
    """
    match dim:
        case dims.CellDim:
            return CELL_ZONES
        case dims.EdgeDim:
            return tuple(Zone)
        case dims.VertexDim:
            return VERTEX_ZONES
        case _:
            raise ValueError(
                f"Dimension should be one of {(dims.MAIN_HORIZONTAL_DIMENSIONS.values())} but was {dim}"
            )


def map_icon_domain_bounds(
    dim: gtx.Dimension, pre_computed_bounds: np.ndarray
) -> dict[Domain, gtx.int32]:  # type: ignore [name-defined]
    get_domain = domain(dim)
    domains = (get_domain(zone) for zone in get_zones_for_dim(dim))
    return {
        d: gtx.int32(pre_computed_bounds[_map_to_icon_index(dim, d.zone)].item())
        for d in domains  # type: ignore [attr-defined]
    }
