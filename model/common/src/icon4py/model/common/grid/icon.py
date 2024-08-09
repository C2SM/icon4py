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

import gt4py.next.common as gt_common
import gt4py.next.ffront.fbuiltins as gt_builtins
import numpy as np

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid.base import BaseGrid
from icon4py.model.common.utils import builder


@dataclasses.dataclass(frozen=True)
class GlobalGridParams:
    root: int
    level: int

    @functools.cached_property
    def num_cells(self):
        return 20.0 * self.root**2 * 4.0**self.level


class IconGrid(BaseGrid):
    def __init__(self, id_: uuid.UUID):
        """Instantiate a grid according to the ICON model."""
        super().__init__()
        self._id = id_
        self.start_indices = {}
        self.end_indices = {}
        self.global_properties = None
        self.offset_provider_mapping = {
            "C2E": (self._get_offset_provider, dims.C2EDim, dims.CellDim, dims.EdgeDim),
            "E2C": (self._get_offset_provider, dims.E2CDim, dims.EdgeDim, dims.CellDim),
            "E2V": (self._get_offset_provider, dims.E2VDim, dims.EdgeDim, dims.VertexDim),
            "C2E2C": (self._get_offset_provider, dims.C2E2CDim, dims.CellDim, dims.CellDim),
            "C2E2C2E": (self._get_offset_provider, dims.C2E2C2EDim, dims.CellDim, dims.EdgeDim),
            "E2EC": (
                self._get_offset_provider_for_sparse_fields,
                dims.E2CDim,
                dims.EdgeDim,
                dims.ECDim,
            ),
            "C2E2CO": (self._get_offset_provider, dims.C2E2CODim, dims.CellDim, dims.CellDim),
            "E2C2V": (self._get_offset_provider, dims.E2C2VDim, dims.EdgeDim, dims.VertexDim),
            "V2E": (self._get_offset_provider, dims.V2EDim, dims.VertexDim, dims.EdgeDim),
            "V2C": (self._get_offset_provider, dims.V2CDim, dims.VertexDim, dims.CellDim),
            "C2V": (self._get_offset_provider, dims.C2VDim, dims.CellDim, dims.VertexDim),
            "E2ECV": (
                self._get_offset_provider_for_sparse_fields,
                dims.E2C2VDim,
                dims.EdgeDim,
                dims.ECVDim,
            ),
            "C2CEC": (
                self._get_offset_provider_for_sparse_fields,
                dims.C2E2CDim,
                dims.CellDim,
                dims.CECDim,
            ),
            "C2CE": (
                self._get_offset_provider_for_sparse_fields,
                dims.C2EDim,
                dims.CellDim,
                dims.CEDim,
            ),
            "E2C2E": (self._get_offset_provider, dims.E2C2EDim, dims.EdgeDim, dims.EdgeDim),
            "E2C2EO": (self._get_offset_provider, dims.E2C2EODim, dims.EdgeDim, dims.EdgeDim),
            "Koff": (lambda: dims.KDim,),  # Koff is a special case
            "C2CECEC ": (
                self._get_offset_provider_for_sparse_fields,
                dims.C2E2C2E2CDim,
                dims.CellDim,
                dims.CECECDim,
            ),
        }

    @builder.builder
    def with_start_end_indices(
        self, dim: gt_common.Dimension, start_indices: np.ndarray, end_indices: np.ndarray
    ):
        self.start_indices[dim] = start_indices.astype(gt_builtins.int32)
        self.end_indices[dim] = end_indices.astype(gt_builtins.int32)

    @builder.builder
    def with_global_params(self, global_params: GlobalGridParams):
        self.global_properties = global_params

    @property
    def num_levels(self):
        return self.config.num_levels if self.config else 0

    @property
    def num_cells(self):
        return self.config.num_cells if self.config else 0

    @property
    def global_num_cells(self):
        """
        Return the number of cells in the global grid.

        If the global grid parameters are not set, it assumes that we are in a one node scenario
        and returns the local number of cells.
        """
        return self.global_properties.num_cells if self.global_properties else self.num_cells

    @property
    def num_vertices(self):
        return self.config.num_vertices if self.config else 0

    @property
    def num_edges(self):
        return self.config.num_edges if self.config else 0

    @property
    def limited_area(self):
        # defined in mo_grid_nml.f90
        return self.config.limited_area

    def _has_skip_values(self, dimension: gt_common.Dimension) -> bool:
        """
        Determine whether a sparse dimension has skip values.

        For the icosahedral global grid skip values are only present for the pentagon points. In the local area model there are also skip values at the boundaries when
        accessing neighbouring cells or edges from vertices.
        """
        assert (
            dimension.kind == gt_common.DimensionKind.LOCAL
        ), "only local dimensions can have skip values"
        if dimension in (dims.V2EDim, dims.V2CDim):
            return True
        elif self.limited_area:
            if dimension in (
                dims.C2E2C2E2CDim,
                dims.C2E2C2EDim,
                dims.E2CDim,
                dims.C2E2CDim,
                dims.C2E2CODim,
                dims.E2C2VDim,
                dims.E2C2EDim,
                dims.E2C2EODim,
            ):
                return True
        else:
            return False

    @property
    def id(self):
        return self._id

    @property
    def n_shift(self):
        return self.config.n_shift_total if self.config else 0

    @property
    def lvert_nest(self):
        return True if self.config.lvertnest else False

    def get_start_index(self, dim: gt_common.Dimension, marker: int) -> gt_builtins.int32:
        """
        Use to specify lower end of domains of a field for field_operators.

        For a given dimension, returns the start index of the
        horizontal region in a field given by the marker.
        """
        return self.start_indices[dim][marker]

    def get_end_index(self, dim: gt_common.Dimension, marker: int) -> gt_builtins.int32:
        """
        Use to specify upper end of domains of a field for field_operators.

        For a given dimension, returns the end index of the
        horizontal region in a field given by the marker.
        """
        return self.end_indices[dim][marker]
