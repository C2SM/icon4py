# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import logging
from types import ModuleType
from typing import Final

import numpy as np
from gt4py import next as gtx

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc


"""
Refinement control for ICON grid.

Grid refinement is used in the context of
- local area grids to determine the type of a grid point: its distance from the boundary
- nested horizontal grids to determine the nested overlap regions for feedback
- domain decomposition to order grid points

See Zaengl et al. Grid Refinement in ICON v2.6.4 (Geosci. Model Dev., 15, 7153-7176, 202)

This module only contains functionality related to grid refinement as we use it in ICON4Py.

"""
_log = logging.getLogger(__name__)


_MAX_BOUNDARY_DISTANCE: Final[dict[gtx.Dimension, int]] = {
    dims.CellDim: 14,
    dims.EdgeDim: 28,
    dims.VertexDim: 14,
}
"""
Grid points in the grid refinement fields are labeled with their distance to the lateral boundary.
The distance is counted in "rows" and distances _MAX_BOUNDARY_DISTANCE are computed.

**Rows** can be defined like:
### CELLS:
1. row: describes all cells that share *at least* 1 vertex with the lateral boundary
2. row: shares at least 1 vertex with the 1. row

### VERTEX:
1. row: vertex on the lateral boundary
2. row: shared between cell of 1. row and 2. row cell.
...

## EDGES:
1. row: edge on boundary
2. row: edge only between cells of 1. row
3. row: edge between cells of 1. row and 2. row
4. row: edges only between 2. row cells
...


Grid points can be ordered (moved to the beginning of an field array) up to any distance <= MAX_BOUNDARY_DISTANCE in the grid-generators, the exact number of
ordered rows is a parameter to the grid generation.
"""

_GRID_REFINEMENT_BOUNDARY_WIDTH: Final[dict[gtx.Dimension, int]] = {
    dims.CellDim: 4,
    dims.EdgeDim: 9,
    dims.VertexDim: 4,
}
"""Number of cell rows used in ICON grid refinement: meaning there are for lateral boundary cell-rows.
"""

_UNORDERED: Final[dict[gtx.Dimension, tuple[int, int]]] = {
    dims.CellDim: (0, -4),
    dims.EdgeDim: (0, -8),
    dims.VertexDim: (0, -4),
}
"""Refinement value indicating a point is in the unordered interior (fully prognostic) of the grid: this is encoded by 0 or -4 in coarser parent grid."""

_MIN_ORDERED: Final[dict[gtx.Dimension, int]] = {
    dim: value[1] + 1 for dim, value in _UNORDERED.items()
}
"""For coarse parent grids the refinement control value of overlapping boundary regions are counted with negative values.
"""


DEFAULT_GRF_NUDGEZONE_WIDTH: Final[int] = 8


_LAST_NUDGING: dict[gtx.Dimension, h_grid.Zone] = {
    dims.EdgeDim: h_grid.Zone.NUDGING_LEVEL_2,
    dims.CellDim: h_grid.Zone.NUDGING,
    dims.VertexDim: h_grid.Zone.NUDGING,
}
_LAST_BOUNDARY: dict[gtx.Dimension, h_grid.Zone] = {
    dims.EdgeDim: h_grid.Zone.LATERAL_BOUNDARY_LEVEL_8,
    dims.CellDim: h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4,
    dims.VertexDim: h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4,
}


def compute_domain_bounds(
    dim: gtx.Dimension, refinement_fields: dict[gtx.Dimension, gtx.Field], array_ns: ModuleType = np
) -> tuple[dict[h_grid.Domain, gtx.int32], dict[h_grid.Domain, gtx.int32]]:  # type: ignore   [name-defined]
    refinement_ctrl = refinement_fields[dim].ndarray
    refinement_ctrl = convert_to_unnested_refinement_values(refinement_ctrl, dim, array_ns)

    domains = h_grid.get_domains_for_dim(dim)
    start_indices = {}
    end_indices = {}
    for domain in domains:
        start_index = 0
        end_index = refinement_ctrl.shape[0]
        my_zone = domain.zone
        if (
            my_zone is h_grid.Zone.END or my_zone.is_halo()
        ):  # TODO(halungge): implement for distributed
            start_index = refinement_ctrl.shape[0]
            end_index = refinement_ctrl.shape[0]
        elif my_zone.is_lateral_boundary():
            found = array_ns.where(refinement_ctrl == my_zone.level)[0]
            start_index, end_index = (
                (array_ns.min(found).item(), array_ns.max(found).item() + 1)
                if found.size > 0
                else (0, 0)
            )
        elif my_zone.is_nudging():
            value = _LAST_BOUNDARY[dim].level + my_zone.level
            found = array_ns.where(refinement_ctrl == value)[0]
            start_index, end_index = (
                (array_ns.min(found).item(), array_ns.max(found).item() + 1)
                if found.size > 0
                else (0, 0)
            )
        elif my_zone is h_grid.Zone.INTERIOR:
            # for the Vertex and Edges the level after the nudging zones are not ordered anymore, so
            # we rely on using the end index of the nudging zone for INTERIOR
            value = get_nudging_refinement_value(dim)
            found = array_ns.where(refinement_ctrl == value)[0]
            start_index = array_ns.max(found).item() + 1 if found.size > 0 else 0
            end_index = refinement_ctrl.shape[0]
        start_indices[domain] = gtx.int32(start_index)  # type: ignore [attr-defined]
        end_indices[domain] = gtx.int32(end_index)  # type: ignore [attr-defined]
    return start_indices, end_indices


def get_nudging_refinement_value(dim: gtx.Dimension) -> int:
    return _LAST_BOUNDARY[dim].level + _LAST_NUDGING[dim].level


def is_unordered_field(
    field: data_alloc.NDArray, dim: gtx.Dimension, array_ns: ModuleType = np
) -> data_alloc.NDArray:
    assert field.dtype in (gtx.int32, gtx.int64), f"not an integer type {field.dtype}"  # type: ignore [attr-defined]
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
    assert field.dtype in (gtx.int32, gtx.int64), f"not an integer type {field.dtype}"  # type: ignore [attr-defined]
    return array_ns.where(field == _UNORDERED[dim][1], 0, np.where(field < 0, -field, field))


def is_limited_area_grid(
    refinement_field: data_alloc.NDArray,
    array_ns: ModuleType = np,
) -> bool:
    """Check if the grid is a local area grid.

    This is done by checking whether there are Boundary points (Refinement value > 0) in the grid.
    The .item() call is needed to get a scalar return for cupy arrays.
    """
    return array_ns.any(refinement_field > 0).item()
