# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import dataclasses
import functools
import logging
import math
import uuid
from typing import Final, Optional

import gt4py.next as gtx
from gt4py.next import common as gtx_common

from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.grid.base import data_alloc


log = logging.getLogger(__name__)

CONNECTIVITIES_ON_BOUNDARIES = (
    dims.C2E2C2E,
    dims.E2C,
    dims.C2E2C,
    dims.C2E2CO,
    dims.E2C2V,
    dims.E2C2E,
    dims.E2C2EO,
    dims.C2E2C2E2C,
)
CONNECTIVITIES_ON_PENTAGONS = (dims.V2E, dims.V2C, dims.V2E2V)


@dataclasses.dataclass(frozen=True)
class GlobalGridParams:
    root: Optional[int] = None
    level: Optional[int] = None
    _num_cells: Optional[int] = None
    _mean_cell_area: Optional[float] = None
    geometry_type: Final[base.GeometryType] = base.GeometryType.ICOSAHEDRON
    radius: float = constants.EARTH_RADIUS

    @classmethod
    def from_mean_cell_area(
        cls,
        mean_cell_area: float,
        root: Optional[int] = None,
        level: Optional[int] = None,
        num_cells: Optional[int] = None,
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


class IconGrid(base.BaseGrid):
    def __init__(
        self,
        id_: uuid.UUID,
        config: base.GridConfig,
        start_end_indices: dict[gtx.Dimension, tuple[data_alloc.NDArray, data_alloc.NDArray]],
        mesh: gtx_common.OffsetProvider,
        global_params: GlobalGridParams,
        refinement_control: dict[gtx.Dimension, gtx.Field] | None = None,
        extra_sizes: dict[gtx.Dimension, int] | None = None,
    ):
        """Instantiate a grid according to the ICON model."""
        super().__init__(config=config, mesh=mesh, extra_sizes=extra_sizes)
        self._id = id_
        self._refinement_control = refinement_control
        # TODO maybe store as single dict?
        self._start_indices = {k: v[0] for k, v in start_end_indices.items()}
        self._end_indices = {k: v[1] for k, v in start_end_indices.items()}
        self.global_properties: GlobalGridParams = global_params  # TODO params or properties???

    def __post_init__(self):
        ...
        # TODO use this info to verify we have all connectivities set
        # self._connectivity_mapping = {
        #     "C2E": (self._construct_connectivity, dims.C2EDim, dims.CellDim, dims.EdgeDim),
        #     "E2C": (self._construct_connectivity, dims.E2CDim, dims.EdgeDim, dims.CellDim),
        #     "E2V": (self._construct_connectivity, dims.E2VDim, dims.EdgeDim, dims.VertexDim),
        #     "C2E2C": (self._construct_connectivity, dims.C2E2CDim, dims.CellDim, dims.CellDim),
        #     "C2E2C2E": (self._construct_connectivity, dims.C2E2C2EDim, dims.CellDim, dims.EdgeDim),
        #     "E2EC": (
        #         self._get_connectivity_sparse_fields,
        #         dims.E2CDim,
        #         dims.EdgeDim,
        #         dims.ECDim,
        #     ),
        #     "C2E2CO": (self._construct_connectivity, dims.C2E2CODim, dims.CellDim, dims.CellDim),
        #     "E2C2V": (self._construct_connectivity, dims.E2C2VDim, dims.EdgeDim, dims.VertexDim),
        #     "V2E": (self._construct_connectivity, dims.V2EDim, dims.VertexDim, dims.EdgeDim),
        #     "V2C": (self._construct_connectivity, dims.V2CDim, dims.VertexDim, dims.CellDim),
        #     "C2V": (self._construct_connectivity, dims.C2VDim, dims.CellDim, dims.VertexDim),
        #     "E2ECV": (
        #         self._get_connectivity_sparse_fields,
        #         dims.E2C2VDim,
        #         dims.EdgeDim,
        #         dims.ECVDim,
        #     ),
        #     "C2CEC": (
        #         self._get_connectivity_sparse_fields,
        #         dims.C2E2CDim,
        #         dims.CellDim,
        #         dims.CECDim,
        #     ),
        #     "C2CE": (
        #         self._get_connectivity_sparse_fields,
        #         dims.C2EDim,
        #         dims.CellDim,
        #         dims.CEDim,
        #     ),
        #     "E2C2E": (self._construct_connectivity, dims.E2C2EDim, dims.EdgeDim, dims.EdgeDim),
        #     "E2C2EO": (self._construct_connectivity, dims.E2C2EODim, dims.EdgeDim, dims.EdgeDim),
        #     "C2E2C2E2C": (
        #         self._construct_connectivity,
        #         dims.C2E2C2E2CDim,
        #         dims.CellDim,
        #         dims.CellDim,
        #     ),
        #     "Koff": (lambda: dims.KDim,),  # Koff is a special case
        #     "C2CECEC": (
        #         self._get_connectivity_sparse_fields,
        #         dims.C2E2C2E2CDim,
        #         dims.CellDim,
        #         dims.CECECDim,
        #     ),
        # }

    def __repr__(self):
        return f"{self.__class__.__name__}: id={self._id}, R{self.global_properties.root}B{self.global_properties.level}"

    def __eq__(self, other: IconGrid):
        """TODO (@halungge)  this might not be enough at least for the distributed case: we might additional properties like sizes"""
        if isinstance(other, IconGrid):
            return self.id == other.id

        else:
            return False

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

    @functools.cached_property
    def limited_area(self):
        # defined in mo_grid_nml.f90
        return self.config.limited_area

    def _has_skip_values(self, dimension: gtx.Dimension) -> bool:
        """
        For the icosahedral global grid skip values are only present for the pentagon points.

        In the local area model there are also skip values at the boundaries when
        accessing neighbouring cells or edges from vertices.
        """
        assert (
            dimension.kind == gtx.DimensionKind.LOCAL
        ), "only local dimensions can have skip values"
        value = dimension in CONNECTIVITIES_ON_PENTAGONS or (
            self.limited_area and dimension in CONNECTIVITIES_ON_BOUNDARIES
        )
        return value

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

    # @override
    # TODO add as abstractmethod to base.BaseGrid
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
