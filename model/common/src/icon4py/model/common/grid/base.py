# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
import dataclasses
import functools
import uuid
import warnings
from abc import ABC, abstractmethod
from typing import Callable, Dict

import gt4py.next as gtx
import numpy as np

import icon4py.model.common.utils as common_utils
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim, VertexDim
from icon4py.model.common.grid import utils as grid_utils, vertical as v_grid
from icon4py.model.common.settings import xp


class MissingConnectivity(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class HorizontalGridSize:
    num_vertices: int
    num_edges: int
    num_cells: int


@dataclasses.dataclass(frozen=True, kw_only=True)
class GridConfig:
    horizontal_config: HorizontalGridSize
    vertical_config: v_grid.VerticalGridSize
    limited_area: bool = True
    n_shift_total: int = 0
    length_rescale_factor: float = 1.0
    lvertnest: bool = False
    on_gpu: bool = False

    @property
    def num_levels(self):
        return self.vertical_config.num_lev

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
        self.connectivities: Dict[gtx.Dimension, np.ndarray] = {}
        self.size: Dict[gtx.Dimension, int] = {}
        self.offset_provider_mapping: Dict[str, tuple[Callable, gtx.Dimension, ...]] = {}

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

    @abstractmethod
    def _has_skip_values(self, dimension: gtx.Dimension) -> bool:
        pass

    @functools.cached_property
    def offset_providers(self):
        offset_providers = {}
        for key, value in self.offset_provider_mapping.items():
            try:
                method, *args = value
                offset_providers[key] = method(*args) if args else method()
            except MissingConnectivity:
                warnings.warn(f"{key} connectivity is missing from grid.", stacklevel=2)

        return offset_providers

    @common_utils.builder
    def with_connectivities(self, connectivity: Dict[gtx.Dimension, np.ndarray]):
        self.connectivities.update({d: k.astype(gtx.int32) for d, k in connectivity.items()})
        self.size.update({d: t.shape[1] for d, t in connectivity.items()})

    @common_utils.builder
    def with_config(self, config: GridConfig):
        self.config = config
        self._update_size()

    def _update_size(self):
        self.size[VertexDim] = self.config.num_vertices
        self.size[CellDim] = self.config.num_cells
        self.size[EdgeDim] = self.config.num_edges
        self.size[KDim] = self.config.num_levels

    def _get_offset_provider(self, dim, from_dim, to_dim):
        if dim not in self.connectivities:
            raise MissingConnectivity()
        assert (
            self.connectivities[dim].dtype == gtx.int32
        ), 'Neighbor table\'s "{}" data type must be int32. Instead it\'s "{}"'.format(
            dim, self.connectivities[dim].dtype
        )
        return gtx.NeighborTableOffsetProvider(
            xp.asarray(self.connectivities[dim]),
            from_dim,
            to_dim,
            self.size[dim],
            has_skip_values=self._has_skip_values(dim),
        )

    def _get_offset_provider_for_sparse_fields(self, dim, from_dim, to_dim):
        if dim not in self.connectivities:
            raise MissingConnectivity()
        return grid_utils.neighbortable_offset_provider_for_1d_sparse_fields(
            self.connectivities[dim].shape,
            from_dim,
            to_dim,
            has_skip_values=self._has_skip_values(dim),
        )

    def get_offset_provider(self, name):
        if name in self.offset_provider_mapping:
            method, *args = self.offset_provider_mapping[name]
            return method(*args)
        else:
            raise Exception(f"Offset provider for {name} not found.")

    def update_size_connectivities(self, new_sizes):
        self.size.update(new_sizes)
