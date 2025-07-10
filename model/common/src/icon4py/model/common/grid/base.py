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
import uuid
import warnings
from abc import ABC, abstractmethod
from types import ModuleType
from typing import Callable, Dict, Sequence, Optional

import gt4py.next as gtx
import numpy as np
from gt4py.next import common as gtx_common

from icon4py.model.common import dimension as dims, utils
from icon4py.model.common.grid import horizontal as h_grid, utils as grid_utils
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
    on_gpu: bool = False
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


class BaseGrid(ABC):
    def __init__(self):
        self.config: Optional[GridConfig] = None
        self._neighbor_tables: Dict[gtx.Dimension, data_alloc.NDArray] = {}
        self.size: Dict[gtx.Dimension, int] = {}
        self._connectivity_mapping: Dict[str, tuple[Callable, gtx.Dimension, ...]] = {}

    @property
    @abstractmethod
    def id(self) -> uuid.UUID:
        """Unique identifier of the horizontal grid.

        ICON grid files contain a UUID that uniquely identifies the horizontal grid described in the file (global attribute `uuidOfHGrid`).
        UUID from icon grid files are UUID v1.
        """
        pass

    @property
    @abstractmethod
    def num_cells(self) -> int:
        pass

    @property
    @abstractmethod
    def num_vertices(self) -> int:
        pass

    @property
    @abstractmethod
    def num_edges(self) -> int:
        pass

    @property
    @abstractmethod
    def num_levels(self) -> int:
        pass

    @property
    @abstractmethod
    def geometry_type(self) -> GeometryType:
        ...

    @functools.cached_property
    def neighbor_tables(self) -> Dict[gtx.Dimension, data_alloc.NDArray]:
        return {
            dim: v.ndarray
            for k, v in self.connectivities.items()
            if (dim := dims.DIMENSIONS_BY_OFFSET_NAME.get(k)) is not None
            and gtx_common.is_neighbor_connectivity(v)
        }

    @functools.cached_property
    def limited_area(self) -> bool:
        return self.config.limited_area

    @abstractmethod
    def _has_skip_values(self, dimension: gtx.Dimension) -> bool:
        """Determine whether a sparse dimension has skip values."""
        ...

    @functools.cached_property
    def connectivities(self) -> Dict[str, gtx.Connectivity]:
        connectivity_map = {}
        for key, value in self._connectivity_mapping.items():
            try:
                method, *args = value
                connectivity_map[key] = method(*args) if args else method()
            except MissingConnectivity:
                warnings.warn(f"{key} connectivity is missing from grid.", stacklevel=2)

        return connectivity_map

    @utils.chainable
    def set_neighbor_tables(self, connectivity: Dict[gtx.Dimension, data_alloc.NDArray]):
        self._neighbor_tables.update({d: k.astype(gtx.int32) for d, k in connectivity.items()})
        self.size.update({d: t.shape[1] for d, t in connectivity.items()})

    @utils.chainable
    def set_config(self, config: GridConfig):
        self.config = config
        self._update_size()

    def _update_size(self):
        self.size[dims.VertexDim] = self.config.num_vertices
        self.size[dims.CellDim] = self.config.num_cells
        self.size[dims.EdgeDim] = self.config.num_edges
        self.size[dims.KDim] = self.config.num_levels

    def _construct_connectivity(self, dim, from_dim, to_dim):
        if dim not in self._neighbor_tables:
            raise MissingConnectivity(f"no neighbor_table for dimension {dim}.")
        assert (
            self._neighbor_tables[dim].dtype == gtx.int32
        ), 'Neighbor table\'s "{}" data type must be gtx.int32. Instead it\'s "{}"'.format(
            dim, self._neighbor_tables[dim].dtype
        )
        skip_value = -1 if self._has_skip_values(dim) else None
        if self._do_replace_skip_values_in_table(dim):
            _log.debug(f"Replacing skip values in connectivity for {dim} with max valid neighbor.")
            skip_value = None
            neighbor_table = replace_skip_values(
                dim, self._neighbor_tables[dim], array_ns=data_alloc.array_ns(self.config.on_gpu)
            )
        else:
            neighbor_table = self._neighbor_tables[dim]
        connectivity = gtx.as_connectivity(
            [from_dim, dim],
            to_dim,
            data=neighbor_table,
            skip_value=skip_value,
        )
        return connectivity

    def _do_replace_skip_values_in_table(self, dim: gtx.Dimension) -> bool:
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
        return not self.config.keep_skip_values and (
            self.limited_area or not self._has_skip_values(dim)
        )

    def _get_connectivity_sparse_fields(self, dim, from_dim, to_dim):
        if dim not in self._neighbor_tables:
            raise MissingConnectivity(f"No neighbor table for dimension {dim}.")
        xp = data_alloc.array_ns(self.config.on_gpu)
        return grid_utils.connectivity_for_1d_sparse_fields(
            dim,
            self._neighbor_tables[dim].shape,
            from_dim,
            to_dim,
            has_skip_values=False,
            array_ns=xp,
        )

    def get_connectivity(self, name: str) -> gtx.Connectivity:
        if name in self._connectivity_mapping:
            method, *args = self._connectivity_mapping[name]
            return method(*args)
        else:
            raise MissingConnectivity(f"Offset provider for {name} not found.")

    def update_size_connectivities(self, new_sizes):
        self.size.update(new_sizes)

    @abstractmethod
    def start_index(self, domain: h_grid.Domain) -> gtx.int32:
        ...

    @abstractmethod
    def end_index(self, domain: h_grid.Domain) -> gtx.int32:
        ...


def replace_skip_values(
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
    # TODO @halungge: neighbour tables are copied, when constructing the Connectivity: should the original be discarded from the grid?
    #   Would that work for the wrapper?

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
