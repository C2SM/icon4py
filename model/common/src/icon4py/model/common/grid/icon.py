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
from typing import Final, Callable

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


@dataclasses.dataclass(frozen=True)
class GlobalGridParams:
    root: int | None = None
    level: int | None = None
    _num_cells: int | None = None
    _mean_cell_area: float | None = None
    geometry_type: Final[base.GeometryType] = base.GeometryType.ICOSAHEDRON
    radius: float = constants.EARTH_RADIUS

    @classmethod
    def from_mean_cell_area(
        cls,
        mean_cell_area: float,
        root: int | None = None,
        level: int | None = None,
        num_cells: int | None = None,
        geometry_type: Final[base.GeometryType] = base.GeometryType.ICOSAHEDRON,
        radius: float = constants.EARTH_RADIUS,
    ):
        return cls(root, level, num_cells, mean_cell_area, geometry_type)

    @functools.cached_property
    def num_cells(self):
        if self._num_cells is None:
            match self.geometry_type:
                case base.GeometryType.ICOSAHEDRON:
                    assert self.root is not None and self.level is not None
                    return compute_icosahedron_num_cells(self.root, self.level)
                case base.GeometryType.TORUS:
                    raise NotImplementedError("TODO : lookup torus cell number computation")
                case _:
                    raise NotImplementedError(f"Unknown geometry type {self.geometry_type}")

        return self._num_cells

    @functools.cached_property
    def characteristic_length(self):
        return math.sqrt(self.mean_cell_area)

    @functools.cached_property
    def mean_cell_area(self):
        if self._mean_cell_area is None:
            match self.geometry_type:
                case base.GeometryType.ICOSAHEDRON:
                    return compute_mean_cell_area_for_sphere(constants.EARTH_RADIUS, self.num_cells)
                case base.GeometryType.TORUS:
                    NotImplementedError(f"mean_cell_area not implemented for {self.geometry_type}")
                case _:
                    NotImplementedError(f"Unknown geometry type {self.geometry_type}")

        return self._mean_cell_area


def compute_icosahedron_num_cells(root: int, level: int):
    return 20.0 * root**2 * 4.0**level


def compute_mean_cell_area_for_sphere(radius, num_cells):
    """
    Compute the mean cell area.

    Computes the mean cell area by dividing the sphere by the number of cells in the
    global grid.

    Args:
        radius: average earth radius, might be rescaled by a scaling parameter
        num_cells: number of cells on the global grid
    Returns: mean area of one cell [m^2]
    """
    return 4.0 * math.pi * radius**2 / num_cells


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
    start_index: Callable[[h_grid.Domain], gtx.int32],
    end_index: Callable[[h_grid.Domain], gtx.int32],
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
    Return start_index and end_index functions to be used in the Gri.

    This function constructs a map from a [Domain](./horizontal.py::Domain) to the start and end indices of this domain in ICON fields.
    It returns then functions that lookup the indices in from that map

    Args:
        constructor: function that takes a dimension as argument and constructs constructs a lookup
        dict[Domain, gtx.int32] for all domains for that dimension

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
