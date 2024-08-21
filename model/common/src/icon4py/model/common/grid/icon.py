# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import functools
import uuid

import gt4py.next as gtx
import numpy as np

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.utils import builder


@dataclasses.dataclass(frozen=True)
class GlobalGridParams:
    root: int
    level: int

    @functools.cached_property
    def num_cells(self):
        return 20.0 * self.root**2 * 4.0**self.level


class IconGrid(base.BaseGrid):
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
        self, dim: gtx.Dimension, start_indices: np.ndarray, end_indices: np.ndarray
    ):
        self.start_indices[dim] = start_indices.astype(gtx.int32)
        self.end_indices[dim] = end_indices.astype(gtx.int32)

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

    def _has_skip_values(self, dimension: gtx.Dimension) -> bool:
        """
        Determine whether a sparse dimension has skip values.

        For the icosahedral global grid skip values are only present for the pentagon points. In the local area model there are also skip values at the boundaries when
        accessing neighbouring cells or edges from vertices.
        """
        assert (
            dimension.kind == gtx.DimensionKind.LOCAL
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

    def get_start_index(self, dim: gtx.Dimension, marker: int) -> gtx.int32:
        """
        Use to specify lower end of domains of a field for field_operators.

        For a given dimension, returns the start index of the
        horizontal region in a field given by the marker.
        """
        return self.start_indices[dim][marker]

    def get_end_index(self, dim: gtx.Dimension, marker: int) -> gtx.int32:
        """
        Use to specify upper end of domains of a field for field_operators.

        For a given dimension, returns the end index of the
        horizontal region in a field given by the marker.
        """
        return self.end_indices[dim][marker]



    def local(self, dim:gtx.Dimension, marker: h_grid.IndexType)-> gtx.int32:
        """Returns the domain bound for a local field of a given dimension: this essentially returns 0 or the local num_edge, num_cell, or num_vertex, depending on the IndexType.
        
        Args:
            dim: The dimension of the array (one of CellDim, VertexDim, EdgeDim).
            marker: The IndexType (on of START or END)
        """
        assert dim in (dims.CellDim, dims.VertexDim, dims.EdgeDim), f"Invalid dimension {dim}"
        if marker == h_grid.IndexType.START:
            return gtx.int32(0)
        else:
            return gtx.int32(self.size.get(dim))
        
        
    def end(self, dim:gtx.Dimension)-> gtx.int32:
        """Returns the domain bound for the end of the local fields: this essentially returns local num_edge, num_cell, or num_vertex.
        
        Args:
            dim: The dimension of the array (one of CellDim, VertexDim, EdgeDim).
            
        """
        assert dim in (dims.CellDim, dims.VertexDim, dims.EdgeDim), f"Invalid dimension {dim}"
        return gtx.int32(self.size.get(dim))
        
    def halo(self, dim:gtx.Dimension, index_type:h_grid.IndexType, line_number: h_grid.HaloLine = h_grid.HaloLine.FIRST )-> gtx.int32:
        """Returns the domain bound for the halo fields: this essentially returns 2 for the halo fields.
        
        Args:
            dim: The dimension of the array (one of CellDim, VertexDim, EdgeDim).
            
        """
        assert dim in (dims.CellDim, dims.VertexDim, dims.EdgeDim), f"Invalid dimension {dim}"
        if index_type == h_grid.IndexType.START:
            return self.start_indices[dim][h_grid.HorizontalMarkerIndex.halo(dim, line_number)]
        else:
            return self.end_indices[dim][h_grid.HorizontalMarkerIndex.halo(dim, line_number)]

    def lateral_boundary(self, dim, index_type:h_grid.IndexType, line:h_grid.BoundaryLine = h_grid.BoundaryLine.FIRST)-> gtx.int32:
        """Returns the domain bound for the lateral boundary fields
        
        Args:
            dim: The dimension of the array (one of CellDim, VertexDim, EdgeDim).
            index_type: The IndexType (on of START or END)
            line: The BoundaryLine (on of FIRST, SECOND, THIRD, FOURTH)
            
        """
        assert dim in (dims.CellDim, dims.VertexDim, dims.EdgeDim), f"Invalid dimension for lateral_boundary '{dim}'"
        if line > h_grid.BoundaryLine.FOURTH and dim !=dims.EdgeDim:
            raise ValueError(f"Invalid line number '{line}' and dimension '{dim}'")
        if index_type == h_grid.IndexType.START:
            return self.start_indices[dim][h_grid.HorizontalMarkerIndex.lateral_boundary(dim, line)]
        else:
            return self.end_indices[dim][h_grid.HorizontalMarkerIndex.lateral_boundary(dim, line)]
        
        
    def nudging(self, dim, index_type:h_grid.IndexType, line:h_grid.NudgingLine = h_grid.NudgingLine.FIRST)-> gtx.int32:
        """Returns the domain bound for the nudging fields.
        The function is only defined on edges and cells as there are no nudging related constants defined in 'mo_impl_constants_grf.f90'
        
        
        Args:
            dim: The dimension of the array (one of CellDim, EdgeDim).
            index_type: The IndexType (on of START or END)
            
        """
        assert dim in (dims.CellDim, dims.EdgeDim), f"Invalid dimension for nudging '{dim}'"
        if line > h_grid.NudgingLine.FIRST and dim !=dims.EdgeDim:
            raise ValueError(f"Invalid line number '{line}' and dimension '{dim}'")
        if index_type == h_grid.IndexType.START:
            return self.start_indices[dim][h_grid.HorizontalMarkerIndex.nudging(dim, line)]
        else:
            return self.end_indices[dim][h_grid.HorizontalMarkerIndex.nudging(dim, line)]
        
        
    def interior(self, dim:gtx.Dimension, index_type:h_grid.IndexType)->gtx.int32:
        """Returns the domain bound for the interior fields
    
        Args:
            dim: The dimension of the array (one of CellDim, VertexDim, EdgeDim).
            index_type: The IndexType (on of START or END)
            
        """
        assert dim in (dims.CellDim, dims.VertexDim, dims.EdgeDim), f"Invalid dimension for interior '{dim}'"
        if index_type == h_grid.IndexType.START:
            return self.start_indices[dim][h_grid.HorizontalMarkerIndex.interior(dim)]
        else:
            return self.end_indices[dim][h_grid.HorizontalMarkerIndex.interior(dim)]