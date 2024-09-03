import dataclasses
from typing import Final


"""
Refinement control for ICON grid.

Grid refinement is used in the context of 
- local area grids to determine the type of a grid point,
- nested horizontal grids to determine the nested overlap regions for feedback
- domain decomposition to order grid points

See Zaengl et al. Grid Refinement in ICON v2.6.4 (Geosci. Model Dev., 15, 7153–7176, 202)


"""


_MAX_ORDERED: Final[int] = 14
"""Lateral boundary points are ordered and have an index indicating the (cell) s distance to the boundary,
generally the number of ordered rows can be defined in the grid generator, but it will never exceed 14.
"""


_UNORDERED: Final[tuple[int, int]] = (0, -4)
"""Value indicating a point is int the unordered interior (fully prognostic) region: this is encoded by 0 or -4 in coarser parent grid."""

_MIN_ORDERED: Final[int] = -3
"""For coarse parent grids the overlapping boundary regions are counted with negative values, from -1 to max -3, (as -4 is used to mark interior points)"""

@dataclasses.dataclass(frozen=True)
class RefinementValue():
    value: int
    
    def __post_init__(self):
        assert _UNORDERED[1] <= self.value <= _MAX_ORDERED, f"Invalid refinement control constant {self.value}"


    def is_nested(self) -> bool:
        return self.value < 0
    
    def is_ordered(self) -> bool:
        return self.value not in _UNORDERED