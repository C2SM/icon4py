# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import dataclasses
import logging
from typing import Final

import numpy as np
from gt4py import next as gtx

import icon4py.model.common.grid.horizontal as h_grid
from icon4py.model.common import dimension as dims


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

_MAX_ORDERED: Final[dict[dims.Dimension, int]] = {
    dims.CellDim: 14,
    dims.EdgeDim: 24,
    dims.VertexDim: 13,
}
"""Lateral boundary points are ordered and have an index indicating the (cell) s distance to the boundary,
generally the number of ordered rows can be defined in the grid generator, but it will never exceed 14 for cells.
TODO: Are these the x_grf dimension in the netcdf grid file?
"""


_UNORDERED: Final[dict[gtx.Dimension : tuple[int, int]]] = {
    dims.CellDim: (0, -4),
    dims.EdgeDim: (0, -8),
    dims.VertexDim: (0, -4),
}
"""Value indicating a point is in the unordered interior (fully prognostic) region: this is encoded by 0 or -4 in coarser parent grid."""

_MIN_ORDERED: Final[dict[dims.Dimension, int]] = {
    dim: value[1] + 1 for dim, value in _UNORDERED.items()
}
"""For coarse parent grids the overlapping boundary regions are counted with negative values, from -1 to max -3, (as -4 is used to mark interior points)"""

_NUDGING_START: Final[dict[gtx.Dimension : int]] = {
    dims.CellDim: h_grid._GRF_BOUNDARY_WIDTH_CELL + 1,
    dims.EdgeDim: h_grid._GRF_BOUNDARY_WIDTH_EDGES + 1,
}
"""Start refin_ctrl levels for boundary nudging (as seen from the child domain)."""


@dataclasses.dataclass(frozen=True)
class RefinementValue:
    dim: dims.Dimension
    value: int

    def __post_init__(self):
        _log.debug(f"Checking refinement value {self.value} for dimension {self.dim}")
        assert (
            _UNORDERED[self.dim][1] <= self.value <= _MAX_ORDERED[self.dim]
        ), f"Invalid refinement control constant {self.value}"

    def is_nested(self) -> bool:
        return self.value < 0

    def is_ordered(self) -> bool:
        return self.value not in _UNORDERED[self.dim]


def is_unordered_field(field: NDArray, dim: dims.Dimension) -> NDArray:
    assert field.dtype in (gtx.int32, gtx.int64), f"not an integer type {field.dtype}"
    return np.where(
        field == _UNORDERED[dim][0], True, np.where(field == _UNORDERED[dim][1], True, False)
    )


def convert_to_unnested_refinement_values(field: NDArray, dim: dims.Dimension) -> NDArray:
    """Convenience function that converts the grid refinement value from a coarser
    parent grid to the canonical values used in an unnested setup.

    The nested values are used for example in the radiation grids.
    """
    assert field.dtype in (gtx.int32, gtx.int64), f"not an integer type {field.dtype}"
    return np.where(field == _UNORDERED[dim][1], 0, np.where(field < 0, -field, field))


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
