# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import dataclasses
import functools
import logging
import math
import uuid
from collections.abc import Mapping
from typing import Final

import gt4py.next as gtx
from gt4py.next import allocators as gtx_allocators
from typing_extensions import assert_never

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

        if geometry_type is None:
            geometry_type = base.GeometryType.ICOSAHEDRON

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
            case _:
                assert_never(geometry_type)

        self.geometry_type = geometry_type
        self.subdivision = subdivision


@dataclasses.dataclass
class GlobalGridParams:
    grid_shape: Final[GridShape | None] = None
    _num_cells: int | None = None
    _mean_cell_area: float | None = None
    radius: float = constants.EARTH_RADIUS

    @classmethod
    def from_mean_cell_area(
        cls,
        mean_cell_area: float,
        grid_shape: GridShape | None = None,
        num_cells: int | None = None,
        radius: float = constants.EARTH_RADIUS,
    ):
        return cls(
            grid_shape,
            num_cells,
            mean_cell_area,
            radius,
        )

    @property
    def geometry_type(self) -> base.GeometryType | None:
        return self.grid_shape.geometry_type if self.grid_shape else None

    @functools.cached_property
    def num_cells(self) -> int:
        if self._num_cells is None:
            match self.geometry_type:
                case base.GeometryType.ICOSAHEDRON:
                    assert self.grid_shape.subdivision is not None
                    return compute_icosahedron_num_cells(self.grid_shape.subdivision)
                case base.GeometryType.TORUS:
                    raise NotImplementedError("TODO : lookup torus cell number computation")
                case _:
                    raise ValueError(f"Unknown geometry type {self.geometry_type}")

        return self._num_cells

    @functools.cached_property
    def characteristic_length(self) -> float:
        return math.sqrt(self.mean_cell_area)

    @functools.cached_property
    def mean_cell_area(self) -> float:
        if self._mean_cell_area is None:
            match self.geometry_type:
                case base.GeometryType.ICOSAHEDRON:
                    return compute_mean_cell_area_for_sphere(self.radius, self.num_cells)
                case base.GeometryType.TORUS:
                    raise NotImplementedError(
                        f"mean_cell_area not implemented for {self.geometry_type}"
                    )
                case _:
                    raise NotImplementedError(f"Unknown geometry type {self.geometry_type}")

        return self._mean_cell_area


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


def _has_skip_values(offset: gtx.FieldOffset, limited_area: bool) -> bool:
    """
    For the icosahedral global grid skip values are only present for the pentagon points.

    In the local area model there are also skip values at the boundaries when
    accessing neighbouring cells or edges from vertices.
    """
    dimension = offset.target[1]
    assert dimension.kind == gtx.DimensionKind.LOCAL, "only local dimensions can have skip values"
    value = dimension in CONNECTIVITIES_ON_PENTAGONS or (
        limited_area and dimension in CONNECTIVITIES_ON_BOUNDARIES
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
    id_: uuid.UUID,
    allocator: gtx_allocators.FieldBufferAllocationUtil | None,
    config: base.GridConfig,
    neighbor_tables: dict[gtx.FieldOffset, data_alloc.NDArray],
    start_indices: Mapping[h_grid.Domain, gtx.int32],
    end_indices: Mapping[h_grid.Domain, gtx.int32],
    global_properties: GlobalGridParams,
    refinement_control: dict[gtx.Dimension, gtx.Field] | None = None,
) -> IconGrid:
    connectivities = {
        offset.value: base.construct_connectivity(
            offset,
            data_alloc.import_array_ns(allocator).asarray(table),
            skip_value=-1 if _has_skip_values(offset, config.limited_area) else None,
            allocator=allocator,
            replace_skip_values=_should_replace_skip_values(
                offset, config.keep_skip_values, config.limited_area
            ),
        )
        for offset, table in neighbor_tables.items()
    }
    return IconGrid(
        id=id_,
        config=config,
        connectivities=connectivities,
        geometry_type=global_properties.geometry_type,
        _start_indices=start_indices,
        _end_indices=end_indices,
        global_properties=global_properties,
        refinement_control=refinement_control or {},
    )
