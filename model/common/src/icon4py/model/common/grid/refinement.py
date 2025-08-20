# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import dataclasses
import enum
import logging
from types import ModuleType
from typing import Final, Mapping

import numpy as np
from gt4py import next as gtx

from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common import dimension as dims
from icon4py.model.common.utils import data_allocation as data_alloc


"""
Refinement control for ICON grid.

Grid refinement is used in the context of
- local area grids to determine the type of a grid point,
- nested horizontal grids to determine the nested overlap regions for feedback
- domain decomposition to order grid points

See Zaengl et al. Grid Refinement in ICON v2.6.4 (Geosci. Model Dev., 15, 7153-7176, 202)

This module only contains functionality related to grid refinement as we use it in ICON4Py.

"""
_log = logging.getLogger(__name__)

#TODO(halunge): from grid file cell_grf, edge_grf, vertex_grf
GRF_DIMENSION: Final[dict[gtx.Dimension,int]] = {
    dims.CellDim: 14,
    dims.EdgeDim: 24,
    dims.VertexDim: 13
},

_MAX_ORDERED: Final[dict[gtx.Dimension, int]] = {
    dims.CellDim: 14,
    dims.EdgeDim: 28,
    dims.VertexDim: 14,
}
"""Lateral boundary points are ordered and have an index indicating the (cell) s distance to the boundary,
generally the number of ordered rows can be defined in the grid generator, but it will never exceed 14 for cells.
"""


_UNORDERED: Final[dict[gtx.Dimension, tuple[int, int]]] = {
    dims.CellDim: (0, -4),
    dims.EdgeDim: (0, -8),
    dims.VertexDim: (0, -4),
}
"""Value indicating a point is in the unordered interior (fully prognostic) region: this is encoded by 0 or -4 in coarser parent grid."""

_MIN_ORDERED: Final[dict[gtx.Dimension, int]] = {
    dim: value[1] + 1 for dim, value in _UNORDERED.items()
}
"""For coarse parent grids the overlapping boundary regions are counted with negative values, from -1 to max -3, (as -4 is used to mark interior points)"""

_NUDGING_START: Final[dict[gtx.Dimension, int]] = {
    dims.CellDim: h_grid._GRF_BOUNDARY_WIDTH_CELL + 1,
    dims.EdgeDim: h_grid._GRF_BOUNDARY_WIDTH_EDGES + 1,
}
"""Start refin_ctrl levels for boundary nudging (as seen from the child domain)."""


@dataclasses.dataclass(frozen=True)
class RefinementValue:
    domain: h_grid.Domain
    _value: int|tuple[int,...]

    def __post_init__(self):
        _log.debug(f"Checking refinement value {self.value} for dimension {self.domain}")
        for v in self.value:
            assert (
                _UNORDERED[self.domain.dim][1] <= v <= _MAX_ORDERED[self.domain.dim]
            ), f"Invalid refinement control constant {self.value}"

    def is_nested(self) -> bool:
        return self.value < 0

    def is_ordered(self) -> bool:
        return self.value not in _UNORDERED[self.domain.dim]

    @property
    def value(self)-> tuple[int,]:
        return self._value if isinstance(self._value, tuple) else (self._value,)


cell_domain = h_grid.domain(dims.CellDim)
edge_domain = h_grid.domain(dims.EdgeDim)
vertex_domain = h_grid.domain(dims.VertexDim)

# TODO(halungge): can the GRF_BOUNDARY_WIDTH be made dynamic
_REFINEMENT_CONTROL:dict[h_grid.Domain, RefinementValue] = {
    cell_domain(h_grid.Zone.LATERAL_BOUNDARY): RefinementValue(cell_domain(h_grid.Zone.LATERAL_BOUNDARY), 1),
    cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2): RefinementValue(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2), 2),
    cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3): RefinementValue(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3), 3 ),
    cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4): RefinementValue(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4), 4),
    cell_domain(h_grid.Zone.NUDGING): RefinementValue(cell_domain(h_grid.Zone.NUDGING), h_grid._GRF_BOUNDARY_WIDTH_CELL + 1),
    cell_domain(h_grid.Zone.LOCAL): RefinementValue(cell_domain(h_grid.Zone.LOCAL), _UNORDERED[dims.CellDim]),
    cell_domain(h_grid.Zone.INTERIOR): RefinementValue(cell_domain(h_grid.Zone.INTERIOR), _UNORDERED[dims.CellDim]),
    cell_domain(h_grid.Zone.END): RefinementValue(cell_domain(h_grid.Zone.END), 0),
    cell_domain(h_grid.Zone.HALO): RefinementValue(cell_domain(h_grid.Zone.HALO), 0), # TODO(halungge)
    cell_domain(h_grid.Zone.HALO_LEVEL_2): RefinementValue(cell_domain(h_grid.Zone.HALO_LEVEL_2), 0), # TODO(halungge)

    vertex_domain(h_grid.Zone.LATERAL_BOUNDARY): RefinementValue(cell_domain(h_grid.Zone.LATERAL_BOUNDARY), 1),
    vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2): RefinementValue(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2), 2),
    vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3): RefinementValue(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3), 3),
    vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4): RefinementValue(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4), 4),
    vertex_domain(h_grid.Zone.LOCAL): RefinementValue(cell_domain(h_grid.Zone.LOCAL), _UNORDERED[dims.VertexDim]),
    vertex_domain(h_grid.Zone.INTERIOR): RefinementValue(cell_domain(h_grid.Zone.INTERIOR), _UNORDERED[dims.VertexDim]),
    vertex_domain(h_grid.Zone.END): RefinementValue(cell_domain(h_grid.Zone.END), 0),
    vertex_domain(h_grid.Zone.HALO): RefinementValue(cell_domain(h_grid.Zone.HALO), 0), # TODO(halungge)
    vertex_domain(h_grid.Zone.HALO_LEVEL_2): RefinementValue(cell_domain(h_grid.Zone.HALO_LEVEL_2), 0), # TODO(halungge)
    edge_domain(h_grid.Zone.LATERAL_BOUNDARY): RefinementValue(edge_domain(h_grid.Zone.LATERAL_BOUNDARY), 1),
    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2): RefinementValue(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2), 2),
    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3): RefinementValue(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3), 3),
    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4): RefinementValue(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4), 4),
    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5): RefinementValue(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5), 5),
    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_6): RefinementValue(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_6), 6),
    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7): RefinementValue(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7), 7),
    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_8): RefinementValue(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_8), 8),
    edge_domain(h_grid.Zone.NUDGING): RefinementValue(edge_domain(h_grid.Zone.NUDGING), h_grid._GRF_BOUNDARY_WIDTH_EDGES),
    edge_domain(h_grid.Zone.NUDGING_LEVEL_2): RefinementValue(edge_domain(h_grid.Zone.NUDGING_LEVEL_2), h_grid._GRF_BOUNDARY_WIDTH_EDGES + 1),
    edge_domain(h_grid.Zone.LOCAL): RefinementValue(edge_domain(h_grid.Zone.LOCAL), _UNORDERED[dims.EdgeDim]), # TODO(halungge) meaning?
    edge_domain(h_grid.Zone.INTERIOR): RefinementValue(edge_domain(h_grid.Zone.INTERIOR), _UNORDERED[dims.EdgeDim]),
    edge_domain(h_grid.Zone.END): RefinementValue(edge_domain(h_grid.Zone.END), 0),  # TODO(halungge) meaning?
    edge_domain(h_grid.Zone.HALO): RefinementValue(edge_domain(h_grid.Zone.HALO), 0), # TODO(halungge)
    edge_domain(h_grid.Zone.HALO_LEVEL_2): RefinementValue(edge_domain(h_grid.Zone.HALO_LEVEL_2), 0), # TODO(halungge)

}



def is_unordered_field(
    field: data_alloc.NDArray, dim: gtx.Dimension, array_ns: ModuleType = np
) -> data_alloc.NDArray:
    assert field.dtype in (gtx.int32, gtx.int64), f"not an integer type {field.dtype}"
    return array_ns.where(
        field == _UNORDERED[dim][0], True, array_ns.where(field == _UNORDERED[dim][1], True, False)
    )


def convert_to_unnested_refinement_values(
    field: data_alloc.NDArray, dim: gtx.Dimension, array_ns: ModuleType = np
) -> data_alloc.NDArray:
    """Convenience function that converts the grid refinement value from a coarser
    parent grid to the canonical values used in an unnested setup.

    The nested values are used for example in the radiation grids.
    """
    assert field.dtype in (gtx.int32, gtx.int64), f"not an integer type {field.dtype}"
    return array_ns.where(field == _UNORDERED[dim][1], 0, np.where(field < 0, -field, field))


def refine_control_value(dim: gtx.Dimension, zone: h_grid.Zone) -> RefinementValue:
    assert (
        dim.kind == gtx.DimensionKind.HORIZONTAL
    ), f"dim = {dim=} refinement control values only exist for horizontal dimensions"
    match zone:
        case zone.NUDGING:
            assert dim in (dims.EdgeDim, dims.CellDim), "no nudging on vertices!"
            return RefinementValue(dim, _NUDGING_START[dim])
        case _:
            raise NotImplementedError


def is_limited_area_grid(
    refinement_field: data_alloc.NDArray,
    array_ns: ModuleType = np,
) -> bool:
    """Check if the grid is a local area grid.

    This is done by checking whether there are Boundary points (Refinement value > 0) in the grid.
    The .item() call is needed to get a scalar return for cupy arrays.
    """
    return array_ns.any(refinement_field > 0).item()


def compute_start_index(
    domain: h_grid.Domain, refinement_fields: dict[gtx.Dimension, gtx.Field], array_ns: ModuleType = np
) -> gtx.int32:
    """
    Compute the start index for the refinement control field for a given dimension.

    Args:
        domain: Dimension to handle, one out of CellDim, EdgeDim, VertexDim
        refinement_fields: refinement control arrays as dictionary mapping dimension to arrays
        array_ns: numpy or cupy module to use for array operations
    Returns:
        start index for the domain
    """

    refinement_ctrl = refinement_fields.get(domain.dim).ndarray
    refinement_value = _REFINEMENT_CONTROL[domain].value
    refinement_mask = array_ns.zeros_like(refinement_ctrl, dtype=bool)
    # Check for any of the refinement values
    for value in refinement_value:
        refinement_mask = refinement_mask | (refinement_ctrl == value)

    #TODO(halungge) should we not rather return refinement_ctrl.size??
    index = array_ns.where(refinement_mask)[0]
    start_index = 0 if index.size == 0 else array_ns.min(index).item()

    return gtx.int32(start_index)


def compute_end_index(
    domain: h_grid.Domain, refinement_fields: dict[gtx.Dimension, gtx.Field], array_ns: ModuleType = np
) -> gtx.int32:
    """
    Compute the end index for the refinement control field for a given dimension.

    Args:
        domain: Dimension to handle, one out of CellDim, EdgeDim, VertexDim
        refinement_fields: refinement control arrays as dictionary mapping dimension to arrays
        array_ns: numpy or cupy module to use for array operations
    Returns:
        last index of this  domain
    """

    refinement_ctrl = refinement_fields.get(domain.dim).ndarray
    refinement_value = _REFINEMENT_CONTROL[domain].value
    refinement_mask = array_ns.zeros_like(refinement_ctrl, dtype=bool)
    # Check for any of the refinement values
    for value in refinement_value:
        refinement_mask = refinement_mask | (refinement_ctrl == value)

    #TODO(halungge) should we not rather return refinement_ctrl.size??
    index = array_ns.where(refinement_mask)[0]
    start_index = 0 if index.size == 0 else array_ns.max(index).item() + 1


    return gtx.int32(start_index)
