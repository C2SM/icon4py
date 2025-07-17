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
import logging
import math
import uuid
from types import ModuleType
from typing import Callable, Dict, Sequence

import gt4py.next as gtx
import numpy as np
from gt4py.next import allocators as gtx_allocators, common as gtx_common

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.grid.gridfile import GridFile
from icon4py.model.common.utils import data_allocation as data_alloc


_log = logging.getLogger(__name__)


class MissingConnectivity(ValueError):
    pass


class GeometryType(enum.Enum):
    """Define geometries of the horizontal domain supported by the ICON grid.

    Values are the same as mo_grid_geometry_info.f90.
    """

    ICOSAHEDRON = 1
    TORUS = 2


@dataclasses.dataclass(frozen=True)
class HorizontalGridSize:
    num_vertices: int
    num_edges: int
    num_cells: int


@dataclasses.dataclass(frozen=True, kw_only=True)
class GridConfig:
    horizontal_config: HorizontalGridSize
    # TODO (Magdalena): Decouple the vertical from horizontal grid.
    vertical_size: int
    limited_area: bool = True
    n_shift_total: int = 0
    length_rescale_factor: float = 1.0
    lvertnest: bool = False
    on_gpu: bool = False  # TODO can this be removed?
    keep_skip_values: bool = True

    @property
    def num_levels(self):
        return self.vertical_size

    @property
    def num_vertices(self):
        return self.horizontal_config.num_vertices

    @property
    def num_edges(self):
        return self.horizontal_config.num_edges

    @property
    def num_cells(self):
        return self.horizontal_config.num_cells


def _1d_size(connectivity: gtx_common.NeighborTable) -> int:
    return math.prod(connectivity.shape)


def _default_1d_sparse_connectivity_constructor(
    offset: gtx.FieldOffset,
    shape2d: tuple[int, int],
    allocator: gtx_allocators.FieldBufferAllocatorProtocol | None = None,
) -> data_alloc.NDArray:
    return gtx.as_connectivity(
        domain=offset.target,
        codomain=offset.source,
        data=np.arange(shape2d[0] * shape2d[1], dtype=gtx.int32).reshape(shape2d),
        allocator=allocator,
    )


@dataclasses.dataclass(frozen=True)
class BaseGrid:
    """
    Contains core features of a grid.

    The 'BaseGrid' is ICON4Py specific: it expects certain connectivities to be present
    to construct derived (1D sparse) connectivities.

    Note: A 'BaseGrid' can be used in 'StencilTest's, while some components of ICON4Py may
    require an 'IconGrid'.
    """

    id: uuid.UUID
    """
    Unique identifier of the horizontal grid.

    ICON grid files contain a UUID that uniquely identifies the horizontal grid
    described in the file (global attribute `uuidOfHGrid`).
    UUID from icon grid files are UUID v1.
    """
    config: GridConfig
    connectivities: gtx_common.OffsetProvider
    geometry_type: GeometryType
    _start_indices: dict[gtx.Dimension, data_alloc.NDArray]
    _end_indices: dict[gtx.Dimension, data_alloc.NDArray]
    _1d_sparse_connectivity_constructor: Callable[
        [gtx.FieldOffset, tuple[int, int], gtx_allocators.FieldBufferAllocatorProtocol | None],
        gtx_common.NeighborTable,
    ] = _default_1d_sparse_connectivity_constructor
    _allocator: gtx_allocators.FieldBufferAllocatorFactoryProtocol | None = None

    def __post_init__(self):
        self._validate()
        # TODO(havogt): replace `Koff[k]` by `KDim + k` syntax and remove the following line.
        self.connectivities[dims.Koff.value] = dims.KDim
        # 1d sparse connectivities
        self.connectivities[dims.C2CE.value] = self._1d_sparse_connectivity_constructor(
            dims.C2CE, self.get_connectivity(dims.C2E).shape, allocator=self._allocator
        )
        self.connectivities[dims.E2ECV.value] = self._1d_sparse_connectivity_constructor(
            dims.E2ECV, self.get_connectivity(dims.E2C2V).shape, allocator=self._allocator
        )
        self.connectivities[dims.E2EC.value] = self._1d_sparse_connectivity_constructor(
            dims.E2EC, self.get_connectivity(dims.E2C).shape, allocator=self._allocator
        )
        self.connectivities[dims.C2CEC.value] = self._1d_sparse_connectivity_constructor(
            dims.C2CEC, self.get_connectivity(dims.C2E2C).shape, allocator=self._allocator
        )
        if dims.C2E2C2E2C.value in self.connectivities:  # TODO is this optional?
            self.connectivities[dims.C2CECEC.value] = self._1d_sparse_connectivity_constructor(
                dims.C2CECEC, self.get_connectivity(dims.C2E2C2E2C).shape, allocator=self._allocator
            )

    def _validate(self):
        # TODO check all expected connectivities are present

        ...

    @functools.cached_property
    def size(self) -> Dict[gtx.Dimension, int]:
        sizes = {
            dims.KDim: self.config.num_levels,
            dims.CellDim: self.config.num_cells,
            dims.EdgeDim: self.config.num_edges,
            dims.VertexDim: self.config.num_vertices,
            # 1d sparse sizes cannot be deduced from their connectivity
            dims.ECVDim: _1d_size(self.get_connectivity(dims.E2C2V)),
            dims.CEDim: _1d_size(self.get_connectivity(dims.C2E)),
            dims.ECDim: _1d_size(self.get_connectivity(dims.E2C)),
        }

        # extract sizes from connectivities # TODO consider extracting into function
        for offset, connectivity in self.connectivities.items():
            if gtx_common.is_neighbor_table(connectivity):
                for dim, size in zip(connectivity.domain.dims, connectivity.shape, strict=True):
                    if dim in sizes:
                        if sizes[dim] != size:
                            raise ValueError(
                                f"Inconsistent sizes for {dim}: expected {sizes[dim]}, got {size}."
                            )
                    else:
                        sizes[dim] = size
            elif isinstance(connectivity, gtx.Dimension):
                ...
            else:
                raise TypeError(
                    f"Unsupported connectivity type {type(connectivity)} for offset {offset}."
                )
        return sizes

    @property
    def num_cells(self) -> int:
        return self.config.num_cells

    @property
    def num_vertices(self) -> int:
        return self.config.num_vertices

    @property
    def num_edges(self) -> int:
        return self.config.num_edges

    @property
    def num_levels(self) -> int:
        return self.config.num_levels

    @property
    def limited_area(self) -> bool:
        return self.config.limited_area

    def get_connectivity(self, offset: str | gtx.FieldOffset) -> gtx_common.NeighborTable:
        """Get the connectivity by its name."""
        if isinstance(offset, gtx.FieldOffset):
            offset = offset.value
        connectivity = self.connectivities[offset]
        assert gtx_common.is_neighbor_table(connectivity)
        return connectivity

    @functools.cached_property
    def neighbor_tables(self) -> Dict[gtx.Dimension, data_alloc.NDArray]:
        # TODO this should be removed
        return {
            dim: v.ndarray
            for k, v in self.connectivities.items()
            if (dim := dims.DIMENSIONS_BY_OFFSET_NAME.get(k)) is not None
            and gtx_common.is_neighbor_connectivity(v)
        }

    def start_index(self, domain: h_grid.Domain) -> gtx.int32:
        """
        Use to specify lower end of domains of a field for field_operators.

        For a given dimension, returns the start index of the
        horizontal region in a field given by the marker.
        """
        if domain.local:
            # special treatment because this value is not set properly in the underlying data.
            return gtx.int32(0)
        return gtx.int32(self._start_indices[domain.dim][domain])

    def end_index(self, domain: h_grid.Domain) -> gtx.int32:
        """
        Use to specify upper end of domains of a field for field_operators.

        For a given dimension, returns the end index of the
        horizontal region in a field given by the marker.
        """
        if domain.zone == h_grid.Zone.INTERIOR and not self.limited_area:
            # special treatment because this value is not set properly in the underlying data, for a global grid
            return gtx.int32(self.size[domain.dim])
        return gtx.int32(self._end_indices[domain.dim][domain])


def construct_connectivity(
    offset: gtx.FieldOffset,
    table: data_alloc.NDArray,
    skip_value: int | None = None,
    *,
    allocator: gtx_allocators.FieldBufferAllocatorProtocol | None = None,
    replace_skip_values: bool = False,
):
    from_dim, dim = offset.target
    to_dim = offset.source

    # TODO maybe caller should already do the replacement?
    if replace_skip_values:
        _log.debug(f"Replacing skip values in connectivity for {dim} with max valid neighbor.")
        skip_value = None
        table = _replace_skip_values(dim, table, array_ns=np)  # TODO fix numpy/cupy

    return gtx.as_connectivity(
        [from_dim, dim],
        to_dim,
        data=table,
        dtype=gtx.int32,
        skip_value=skip_value,
        allocator=allocator,
    )


def _replace_skip_values(
    domain: Sequence[gtx.Dimension], neighbor_table: data_alloc.NDArray, array_ns: ModuleType = np
) -> data_alloc.NDArray:
    """
    Manipulate a Connectivity's neighbor table to remove invalid indices.

    This is a workaround to account for the lack of a domain inference for unstructured neighbor accesses in GT4Py: when computing into temporaries we do not use the minimal domain (interior + respective halo), but the full domain, i.e. we compute at the outer-most halo/boundary lines where some neighbors do not exist.

    (Remaining) invalid indices in the neighbor tables are replaced by a valid (other) index of the given
    entry (we arbitrarily choose the maximum), for example for a C2E2C table assume that  cell = 16 looks like this:

    16 ->(15, -1, -1, 17)

    it will become
    16 -> (15, 17, 17, 17)

    in the case that there are no valid neighbors around a point (esssentially a diconnected grid point)
    the neighbor indices are set to 0, ie
    16 -> (-1, -1, -1, -1) will become 16 -> (0, 0, 0, 0)

    This might potentially lead to wrong results if computation over such values are effectively used.

    ICON (Fortran) does something similar: They replace INVALID indices with the last valid neighbor and set interpolation coefficients to 0.
    The don't do this for all neighbor tables but only for the ones where the apparently know the loop over, that is why even when

    Hence, when calling from Fortran through py2fgen connectivity tables are passed into the wrapper
    and most of them should already be manipulated only leaving those where invalid neighbors are not accessed in the dycore.

    Args:
        domain: the domain of the Connectivity
        connectivity: NDArray object to be manipulated
        array_ns: numpy or cupy module to use for array operations
    Returns:
        NDArray without skip values
    """
    if _has_skip_values_in_table(neighbor_table, array_ns):
        _log.info(f"Found invalid indices in {domain}. Replacing...")
        max_valid_neighbor = neighbor_table.max(axis=1, keepdims=True)
        if not array_ns.all(max_valid_neighbor >= 0):
            _log.warning(
                f"{domain} contains entries without any valid neighbor, disconnected grid?"
            )
            max_valid_neighbor = array_ns.where(max_valid_neighbor < 0, 0, max_valid_neighbor)
        neighbor_table[:] = array_ns.where(
            neighbor_table == GridFile.INVALID_INDEX, max_valid_neighbor, neighbor_table
        )
    else:
        _log.debug(f"Found no invalid indices in {domain}.")
    return neighbor_table


def _has_skip_values_in_table(data: data_alloc.NDArray, array_ns: ModuleType) -> bool:
    return array_ns.amin(data).item() == GridFile.INVALID_INDEX
