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

from typing import Dict, Tuple

import numpy as np
from gt4py.next.common import Dimension, DimensionKind, Field
from gt4py.next.ffront.fbuiltins import int32
from gt4py.next.iterator.embedded import NeighborTableOffsetProvider

from icon4py import common
from icon4py.common.dimension import CellDim, EdgeDim, KDim, VertexDim, ECVDim
from icon4py.diffusion.horizontal import HorizontalMeshSize


class VerticalMeshConfig:
    def __init__(self, num_lev: int):
        self._num_lev = num_lev

    @property
    def num_lev(self) -> int:
        return self._num_lev


class MeshConfig:
    def __init__(
        self,
        horizontal_config: HorizontalMeshSize,
        vertical_config: VerticalMeshConfig,
        limited_area=True,
    ):
        self._vertical = vertical_config
        self._n_shift_total = 0
        self._limited_area = limited_area
        self._horizontal = horizontal_config

    @property
    def limited_area(self):
        return self._limited_area

    @property
    def num_k_levels(self):
        return self._vertical.num_lev

    @property
    def n_shift_total(self):
        return self._n_shift_total

    @property
    def num_vertices(self):
        return self._horizontal.num_vertices

    @property
    def num_edges(self):
        return self._horizontal.num_edges

    @property
    def num_cells(self):
        return self._horizontal.num_cells


def builder(func):
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        return self

    return wrapper


class IconGrid:
    def __init__(self):
        self.config: MeshConfig = None
        self.start_indices = {}
        self.end_indices = {}
        self.connectivities: Dict[str, np.ndarray] = {}
        self.size: Dict[Dimension, int] = {}

    def _update_size(self, config: MeshConfig):
        self.size[VertexDim] = config.num_vertices
        self.size[CellDim] = config.num_cells
        self.size[EdgeDim] = config.num_edges
        self.size[KDim] = config.num_k_levels

    @builder
    def with_config(self, config: MeshConfig):
        self.config = config
        self._update_size(config)

    @builder
    def with_start_end_indices(
        self, dim: Dimension, start_indices: np.ndarray, end_indices: np.ndarray
    ):
        self.start_indices[dim] = start_indices.astype(int)
        self.end_indices[dim] = end_indices.astype(int)

    @builder
    def with_connectivities(self, connectivity: Dict[Dimension, np.ndarray]):
        self.connectivities.update(
            {d.value.lower(): k for d, k in connectivity.items()}
        )
        self.size.update({d: t.shape[1] for d, t in connectivity.items()})

    def limited_area(self):
        # TODO defined in mo_grid_nml.f90
        return self.config.limited_area

    def n_lev(self):
        return self.config.num_k_levels if self.config else 0

    def num_cells(self):
        return self.config.num_cells if self.config else 0

    def num_vertices(self):
        return self.config.num_vertices if self.config else 0

    def num_edges(self):
        return self.config.num_edges

    def get_indices_from_to(
        self, dim: Dimension, start_marker: int, end_marker: int
    ) -> Tuple[int, int]:
        """
        Use to specify domains of a field for field_operator.

        For a given dimension, returns the start and end index if a
        horizontal region in a field given by the markers.

        field operators will then run from start of the region given by the
        start_marker to the end of the region given by the end_marker
        """
        if dim.kind != DimensionKind.HORIZONTAL:
            raise ValueError(
                "only defined for {} dimension kind ", DimensionKind.HORIZONTAL
            )
        return self.start_indices[dim][start_marker], self.end_indices[dim][end_marker]

    def get_c2e_connectivity(self):
        table = self.connectivities["c2e"]
        return NeighborTableOffsetProvider(table, CellDim, EdgeDim, table.shape[1])

    def get_e2c_connectivity(self):
        table = self.connectivities["e2c"]
        return NeighborTableOffsetProvider(table, EdgeDim, CellDim, table.shape[1])

    def get_e2v_connectivity(self):
        table = self.connectivities["e2v"]
        return NeighborTableOffsetProvider(table, EdgeDim, VertexDim, table.shape[1])

    def get_c2e2c_connectivity(self):
        table = self.connectivities["c2e2c"]
        return NeighborTableOffsetProvider(table, CellDim, CellDim, table.shape[1])

    def get_c2e2co_connectivity(self):
        table = self.connectivities["c2e2co"]
        return NeighborTableOffsetProvider(table, CellDim, CellDim, table.shape[1])

    def get_e2c2v_connectivity(self):
        table = self.connectivities["e2c2v"]
        return NeighborTableOffsetProvider(table, EdgeDim, VertexDim, table.shape[1])

    def get_v2e_connectivity(self):
        table = self.connectivities["v2e"]
        return NeighborTableOffsetProvider(table, VertexDim, EdgeDim, table.shape[1])

    def get_e2ecv_connectivity(self):
        old_shape = self.connectivities["e2c2v"].shape
        v2ecv_table = np.arange(old_shape[0] * old_shape[1]).reshape(old_shape)
        return NeighborTableOffsetProvider(v2ecv_table, EdgeDim, ECVDim, v2ecv_table.shape[1])








class VerticalModelParams:
    def __init__(self, vct_a: Field[[KDim], float], rayleigh_damping_height: float):
        """
        Contains vertical physical parameters defined on the grid.

        Args:
            vct_a:  field containing the physical heights of the k level
            rayleigh_damping_height: height of rayleigh damping in [m] mo_nonhydro_nml
        """
        self._rayleigh_damping_height = rayleigh_damping_height
        self._vct_a = vct_a
        self._index_of_damping_height = int32(np.argmax(np.where(np.asarray(self._vct_a) >= self._rayleigh_damping_height)))

    @property
    def index_of_damping_layer(self):
        return self._index_of_damping_height

    @property
    def physical_heights(self) -> Field[[KDim], float]:
        return self._vct_a

    @property
    def rayleigh_damping_height(self):
        return self._rayleigh_damping_height
