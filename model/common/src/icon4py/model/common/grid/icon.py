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
import uuid
from typing import Final

import gt4py.next as gtx
import numpy as np

from icon4py.model.common import constants, dimension as dims, utils
from icon4py.model.common.grid import base, horizontal as h_grid


log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class GlobalGridParams:
    root: int
    level: int
    geometry_type: Final[base.GeometryType] = base.GeometryType.ICOSAHEDRON
    radius = constants.EARTH_RADIUS

    @functools.cached_property
    def num_cells(self):
        match self.geometry_type:
            case base.GeometryType.ICOSAHEDRON:
                return compute_icosahedron_num_cells(self.root, self.level)
            case base.GeometryType.TORUS:
                return compute_torus_num_cells(1000, 1000)
            case _:
                NotImplementedError(f"Unknown gemoetry type {self.geometry_type}")


def compute_icosahedron_num_cells(root: int, level: int):
    return 20.0 * root**2 * 4.0**level


def compute_torus_num_cells(x: int, y: int):
    # TODO (@halungge) add implementation
    raise NotImplementedError("TODO : lookup torus cell number computation")


class IconGrid(base.BaseGrid):
    def __init__(self, id_: uuid.UUID):
        """Instantiate a grid according to the ICON model."""
        super().__init__()
        self._id = id_
        self._start_indices = {}
        self._end_indices = {}
        self.global_properties: GlobalGridParams = None
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
            "C2E2C2E2C": (self._get_offset_provider, dims.C2E2C2E2CDim, dims.CellDim, dims.CellDim),
            "Koff": (lambda: dims.KDim,),  # Koff is a special case
            "C2CECEC": (
                self._get_offset_provider_for_sparse_fields,
                dims.C2E2C2E2CDim,
                dims.CellDim,
                dims.CECECDim,
            ),
        }

    def __repr__(self):
        return f"{self.__class__.__name__}: id={self._id}, R{self.global_properties.root}B{self.global_properties.level}"

    def __eq__(self, other: "IconGrid"):
        """TODO (@halungge)  this might not be enough at least for the distributed case: we might additional properties like sizes"""
        if isinstance(other, IconGrid):
            return self.id == other.id

        else:
            return False

    @utils.chainable
    def with_start_end_indices(
        self,
        dim: gtx.Dimension,
        start_indices: np.ndarray,
        end_indices: np.ndarray,
    ):
        log.debug(f"Using start_indices {dim} {start_indices}, end_indices {dim} {end_indices}")
        self._start_indices[dim] = start_indices.astype(gtx.int32)
        self._end_indices[dim] = end_indices.astype(gtx.int32)

    @utils.chainable
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
    def geometry_type(self) -> base.GeometryType:
        return self.global_properties.geometry_type

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

    def start_index(self, domain: h_grid.Domain) -> gtx.int32:
        """
        Use to specify lower end of domains of a field for field_operators.

        For a given dimension, returns the start index of the
        horizontal region in a field given by the marker.
        """
        if domain.local:
            # special treatment because this value is not set properly in the underlying data.
            return gtx.int32(0)
        return gtx.int32(self._start_indices[domain.dim][domain()])

    def end_index(self, domain: h_grid.Domain) -> gtx.int32:
        """
        Use to specify upper end of domains of a field for field_operators.

        For a given dimension, returns the end index of the
        horizontal region in a field given by the marker.
        """
        if domain.zone == h_grid.Zone.INTERIOR and not self.limited_area:
            # special treatment because this value is not set properly in the underlying data, for a global grid
            return gtx.int32(self.size[domain.dim])
        return gtx.int32(self._end_indices[domain.dim][domain()])
