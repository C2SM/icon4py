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
import uuid
import warnings
from abc import ABC, abstractmethod
from typing import Callable, Dict

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, utils
from icon4py.model.common.grid import horizontal as h_grid, utils as grid_utils
from icon4py.model.common.utils import data_allocation as data_alloc


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
        self.config: GridConfig = None
        self._neighbor_tables: Dict[gtx.Dimension, data_alloc.NDArray] = {}
        self.size: Dict[gtx.Dimension, int] = {}
        self.connectivity_mapping: Dict[str, tuple[Callable, gtx.Dimension, ...]] = {}

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

    @property
    def neighbor_tables(self) -> Dict[gtx.Dimension, data_alloc.NDArray]:
        return self._neighbor_tables

    @functools.cached_property
    def limited_area(self) -> bool:
        return self.config.limited_area

    @abstractmethod
    def _has_skip_values(self, dimension: gtx.Dimension) -> bool:
        pass

    def has_skip_values(self):
        """
        Check whether there are skip values on any connectivity in the grid.

        Decision is made base on the following properties:
        - limited_area = True -> True
        - geometry_type: either TORUS or ICOSAHEDRON, ICOSAHEDRON has Pentagon points ->True

        """
        return self.config.limited_area or self.geometry_type == GeometryType.ICOSAHEDRON

    @functools.cached_property
    def connectivities(self) -> Dict[str, gtx.Connectivity]:
        connectivity_map = {}
        for key, value in self.connectivity_mapping.items():
            try:
                method, *args = value
                connectivity_map[key] = method(*args) if args else method()
            except MissingConnectivity:
                warnings.warn(f"{key} connectivity is missing from grid.", stacklevel=2)

        return connectivity_map

    @utils.chainable
    def with_neighbor_tables(self, connectivity: Dict[gtx.Dimension, data_alloc.NDArray]):
        self._neighbor_tables.update({d: k.astype(gtx.int32) for d, k in connectivity.items()})
        self.size.update({d: t.shape[1] for d, t in connectivity.items()})

    @utils.chainable
    def with_config(self, config: GridConfig):
        self.config = config
        self._update_size()

    def _update_size(self):
        self.size[dims.VertexDim] = self.config.num_vertices
        self.size[dims.CellDim] = self.config.num_cells
        self.size[dims.EdgeDim] = self.config.num_edges
        self.size[dims.KDim] = self.config.num_levels

    def _construct_connectivity(self, dim, from_dim, to_dim):
        if dim not in self._neighbor_tables:
            raise MissingConnectivity()
        assert (
            self._neighbor_tables[dim].dtype == gtx.int32
        ), 'Neighbor table\'s "{}" data type must be gtx.int32. Instead it\'s "{}"'.format(
            dim, self._neighbor_tables[dim].dtype
        )
        return gtx.as_connectivity(
            [from_dim, dim],
            to_dim,
            self._neighbor_tables[dim],
            skip_value=-1 if self._has_skip_values(dim) else None,
        )

    def _get_connectivity_sparse_fields(self, dim, from_dim, to_dim):
        if dim not in self._neighbor_tables:
            raise MissingConnectivity()
        xp = data_alloc.array_ns(self.config.on_gpu)
        return grid_utils.connectivity_for_1d_sparse_fields(
            dim,
            self._neighbor_tables[dim].shape,
            from_dim,
            to_dim,
            has_skip_values=self._has_skip_values(dim),
            array_ns=xp,
        )

    def get_connectivity(self, name: str) -> gtx.Connectivity:
        if name in self.connectivity_mapping:
            method, *args = self.connectivity_mapping[name]
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
