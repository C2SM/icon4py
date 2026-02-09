# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import dataclasses
import logging
import math
from collections.abc import Callable
from typing import Final, TypeVar

import gt4py.next as gtx
from gt4py.next import allocators as gtx_allocators

from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc


log = logging.getLogger(__name__)

CONNECTIVITIES_ON_BOUNDARIES = (
    dims.C2E2C2EDim,
    dims.E2CDim,
    dims.C2E2CDim,
    dims.C2E2CODim,
    dims.E2C2VDim,
    dims.E2C2EDim,
    dims.E2C2EODim,
    dims.C2E2C2E2CDim,
)
CONNECTIVITIES_ON_PENTAGONS = (dims.V2EDim, dims.V2CDim, dims.V2E2VDim)


@dataclasses.dataclass(frozen=True, kw_only=True)
class GridSubdivision:
    root: int
    level: int


@dataclasses.dataclass(kw_only=True)
class GridShape:
    geometry_type: base.GeometryType
    subdivision: GridSubdivision | None

    def __init__(
        self,
        geometry_type: base.GeometryType | None = None,
        subdivision: GridSubdivision | None = None,
    ) -> None:
        if geometry_type is None and subdivision is None:
            raise ValueError("Either geometry_type or subdivision must be provided")
        match geometry_type:
            case base.GeometryType.ICOSAHEDRON:
                if subdivision is None:
                    raise ValueError("Subdivision must be provided for icosahedron geometry type")
                if subdivision.root < 1 or subdivision.level < 0:
                    raise ValueError(
                        f"For icosahedron geometry type, root must be >= 1 and level must be >= 0, got {subdivision.root=} and {subdivision.level=}"
                    )
            case base.GeometryType.TORUS:
                subdivision = None

        self.geometry_type = geometry_type
        self.subdivision = subdivision


_T = TypeVar("_T")


@dataclasses.dataclass(kw_only=True, frozen=True)
class GlobalGridParams:
    grid_shape: Final[GridShape | None] = None
    radius: float = constants.EARTH_RADIUS
    domain_length: float | None = None
    domain_height: float | None = None
    global_num_cells: int | None = None
    num_cells: int | None = None
    characteristic_length: float | None = None

    def __post_init__(self) -> None:
        if self.geometry_type is not None:
            match self.geometry_type:
                case base.GeometryType.ICOSAHEDRON:
                    object.__setattr__(self, "domain_length", None)
                    object.__setattr__(self, "domain_height", None)
                    if self.radius is None:
                        object.__setattr__(self, "radius", constants.EARTH_RADIUS)
                case base.GeometryType.TORUS:
                    object.__setattr__(self, "radius", None)

        if self.global_num_cells is None and self.geometry_type is base.GeometryType.ICOSAHEDRON:
            object.__setattr__(
                self,
                "global_num_cells",
                compute_icosahedron_num_cells(self.grid_shape.subdivision),
            )

        if self.num_cells is None and self.global_num_cells is not None:
            object.__setattr__(self, "num_cells", self.global_num_cells)

    @property
    def geometry_type(self) -> base.GeometryType | None:
        return self.grid_shape.geometry_type if self.grid_shape else None

    @property
    def subdivision(self) -> GridSubdivision | None:
        return self.grid_shape.subdivision if self.grid_shape else None


def compute_icosahedron_num_cells(subdivision: GridSubdivision) -> int:
    return 20 * subdivision.root**2 * 4**subdivision.level


def compute_mean_cell_area_for_sphere(radius, num_cells) -> float:
    """
    Compute the mean cell area.

    Computes the mean cell area by dividing the sphere by the number of cells in the
    global grid.

    Args:
        radius: average earth radius, might be rescaled by a scaling parameter
        num_cells: number of cells on the global grid
    Returns: mean area of one cell [m^2]
    """
    return 4.0 * math.pi * radius**2.0 / num_cells


@dataclasses.dataclass(frozen=True)
class IconGrid(base.Grid):
    global_properties: GlobalGridParams = dataclasses.field(default=None, kw_only=True)
    refinement_control: dict[gtx.Dimension, gtx.Field] = dataclasses.field(
        default=None, kw_only=True
    )


def _has_skip_values(offset: gtx.FieldOffset, limited_area_or_distributed: bool) -> bool:
    """
    For the icosahedral global grid skip values are only present for the pentagon points.

    In the local area model or a distributed grid there are also skip values at the boundaries or halos when
    accessing neighbouring cells or edges from vertices.
    """
    dimension = offset.target[1]
    assert dimension.kind == gtx.DimensionKind.LOCAL, "only local dimensions can have skip values"
    value = dimension in CONNECTIVITIES_ON_PENTAGONS or (
        limited_area_or_distributed and dimension in CONNECTIVITIES_ON_BOUNDARIES
    )

    return value


def _should_replace_skip_values(
    offset: gtx.FieldOffset, keep_skip_values: bool, limited_area: bool
) -> bool:
    """
    Check if the skip_values in a neighbor table  should be replaced.

    There are various reasons for skip_values in neighbor tables depending on the type of grid:
        - pentagon points (icosahedral grid),
        - boundary layers of limited area grids,
        - halos for distributed grids.

    There is config flag to evaluate whether skip_value replacement should be done at all.
    If so, we replace skip_values for halos and boundary layers of limited area grids.

    Even though by specifying the correct output domain of a stencil, access to
    invalid indices is avoided in the output fields, temporary computations
    inside a stencil do run over the entire data buffer including halos and boundaries
    as the output domain is unknown at that point.

    Args:
        dim: The (local) dimension for which the neighbor table is checked.
    Returns:
        bool: True if the skip values in the neighbor table should be replaced, False otherwise.

    """
    return not keep_skip_values and (limited_area or not _has_skip_values(offset, limited_area))


def icon_grid(
    id_: str,
    allocator: gtx_allocators.FieldBufferAllocationUtil | None,
    config: base.GridConfig,
    neighbor_tables: dict[gtx.FieldOffset, data_alloc.NDArray],
    start_index: Callable[[h_grid.Domain], gtx.int32],
    end_index: Callable[[h_grid.Domain], gtx.int32],
    global_properties: GlobalGridParams,
    refinement_control: dict[gtx.Dimension, gtx.Field] | None = None,
) -> IconGrid:
    # TODO(msimberg): What should we do about this. (The global) num_cells is
    # not guaranteed to be set here when used through fortran. Should we:
    # 1. Ignore distributed?
    # 2. Compute num_cells with a reduction?
    # 3. Use a ProcessProperties to detect it?
    distributed = (
        config.num_cells < global_properties.num_cells
        if global_properties.num_cells is not None
        else False
    )
    limited_area_or_distributed = config.limited_area or distributed
    connectivities = {
        offset.value: base.construct_connectivity(
            offset,
            data_alloc.import_array_ns(allocator).asarray(table),
            skip_value=-1 if _has_skip_values(offset, limited_area_or_distributed) else None,
            allocator=allocator,
            replace_skip_values=_should_replace_skip_values(
                offset, config.keep_skip_values, limited_area_or_distributed
            ),
        )
        for offset, table in neighbor_tables.items()
    }
    return IconGrid(
        id=id_,
        config=config,
        connectivities=connectivities,
        geometry_type=global_properties.geometry_type,
        start_index=start_index,
        end_index=end_index,
        global_properties=global_properties,
        refinement_control=refinement_control or {},
    )


def get_start_and_end_index(
    constructor: Callable[
        [gtx.Dimension], tuple[dict[h_grid.Domain, gtx.int32], dict[h_grid.Domain, gtx.int32]]
    ],
) -> tuple[Callable[[h_grid.Domain], gtx.int32], Callable[[h_grid.Domain], gtx.int32]]:
    """
    Return start_index and end_index functions to be passed to the Grid constructor.

    This function defines a version of `start_index` and `end_index` that looks up the indeces in an internal map from [Domain](horizontal.py::Domain) -> gtx.int32
    It takes the constructor function of this map as input.

    Args:
        constructor: function that takes a dimension as argument and constructs  a lookup table
        dict[Domain, gtx.int32] for all domains for a given dimension

    Returns:
        tuple of functions `start_index` and `end_index` to be passed to the [Grid](./base.py::Grid)

    """
    start_indices = {}
    end_indices = {}
    for dim in dims.MAIN_HORIZONTAL_DIMENSIONS.values():
        start_map, end_map = constructor(dim)
        start_indices.update(start_map)
        end_indices.update(end_map)

    def start_index(domain: h_grid.Domain) -> gtx.int32:
        return start_indices[domain]

    def end_index(domain: h_grid.Domain) -> gtx.int32:
        return end_indices[domain]

    return start_index, end_index
