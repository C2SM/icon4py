# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from typing import Final

from icon4py.model.common import dimension as dims
from icon4py.model.common.settings import xp


"""
Refinement control for ICON grid.

Grid refinement is used in the context of 
- local area grids to determine the type of a grid point,
- nested horizontal grids to determine the nested overlap regions for feedback
- domain decomposition to order grid points

See Zaengl et al. Grid Refinement in ICON v2.6.4 (Geosci. Model Dev., 15, 7153-7176, 202)

This module only contains functionality related to grid refinement as we use it in ICON4Py.

"""


_MAX_ORDERED: Final[dict[dims.Dimension, int]] = {
    dims.CellDim: 14,
    dims.EdgeDim: 15,
    dims.VertexDim: 16,
}
"""Lateral boundary points are ordered and have an index indicating the (cell) s distance to the boundary,
generally the number of ordered rows can be defined in the grid generator, but it will never exceed 14.
"""


_UNORDERED: Final[dict[dims.Dimension : tuple[int, int]]] = {
    dims.CellDim: (0, -4),
    dims.EdgeDim: (0, -8),
    dims.VertexDim: (0, -4),
}
"""Value indicating a point is int the unordered interior (fully prognostic) region: this is encoded by 0 or -4 in coarser parent grid."""

_MIN_ORDERED: Final[dict[dims.Dimension, int]] = {
    dim: value[1] + 1 for dim, value in _UNORDERED.items()
}
"""For coarse parent grids the overlapping boundary regions are counted with negative values, from -1 to max -3, (as -4 is used to mark interior points)"""


@dataclasses.dataclass(frozen=True)
class RefinementValue:
    value: int
    dim: dims.Dimension

    def __post_init__(self):
        assert (
            _UNORDERED[self.dim][1] <= self.value <= _MAX_ORDERED[self.dim]
        ), f"Invalid refinement control constant {self.value}"

    def is_nested(self) -> bool:
        return self.value < 0

    def is_ordered(self) -> bool:
        return self.value not in _UNORDERED[self.dim]


def is_unordered(field: xp.ndarray, dim: dims.Dimension) -> xp.ndarray:
    assert field.dtype == xp.int32 or field.dtype == xp.int64, f"not an integer type {field.dtype}"
    return xp.where(
        field == _UNORDERED[dim][0], True, xp.where(field == _UNORDERED[dim][1], True, False)
    )


def to_unnested(field: xp.ndarray, dim: dims.Dimension) -> xp.ndarray:
    assert field.dtype == xp.int32 or field.dtype == xp.int64, f"not an integer type {field.dtype}"
    return xp.where(field == _UNORDERED[dim][1], 0, xp.where(field < 0, -field, field))
