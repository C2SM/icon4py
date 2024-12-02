# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools

import gt4py.next as gtx
from gt4py.next import backend as gtx_backend

from common.tests.interpolation_tests.test_interpolation_fields import edge_domain
from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions
from icon4py.model.common.grid import (
    geometry,
    geometry_attributes as geometry_attrs,
    horizontal as h_grid,
    icon,
)
from icon4py.model.common.interpolation import (
    interpolation_attributes as attrs,
    interpolation_fields,
)
from icon4py.model.common.states import factory, model
from icon4py.model.common.utils import gt4py_field_allocation as alloc


cell_domain = h_grid.domain(dims.CellDim)


class InterpolationFieldsFactory(factory.FieldSource, factory.GridProvider):
    def __init__(
        self,
        grid: icon.IconGrid,
        decomposition_info: definitions.DecompositionInfo,
        geometry: geometry.GridGeometry,
        backend: gtx_backend.Backend,
        metadata: dict[str, model.FieldMetaData],
    ):
        self._backend = backend
        self._xp = alloc.import_array_ns(backend)
        self._allocator = gtx.constructors.zeros.partial(allocator=backend)
        self._grid = grid
        self._decomposition_info = decomposition_info
        self._attrs = metadata
        self._composite_source = factory.CompositeSource(self, (geometry,))
        self._providers: dict[str, factory.FieldProvider] = {}
        self._register_computed_fields()

    def __repr__(self):
        return f"{self.__class__.__name__} on (grid={self._grid!r}) providing fields f{self.metadata.keys()}"

    @property
    def _sources(self) -> factory.FieldSource:
        return self._composite_source

    def _register_computed_fields(self):
        geofac_div = factory.FieldOperatorProvider(
            # needs to be computed on fieldview-embedded backend
            func=interpolation_fields.compute_geofac_div.with_backend(None),
            domain=(dims.CellDim, dims.C2EDim),
            fields={attrs.GEOFAC_DIV: attrs.GEOFAC_DIV},
            deps={
                "primal_edge_length": geometry_attrs.EDGE_LENGTH,
                "edge_orientation": geometry_attrs.CELL_NORMAL_ORIENTATION,
                "area": geometry_attrs.CELL_AREA,
            },
        )
        self.register_provider(geofac_div)
        geofac_rot = factory.FieldOperatorProvider(
            # needs to be computed on fieldview-embedded backend
            func=interpolation_fields.compute_geofac_rot.with_backend(None),
            domain=(dims.VertexDim, dims.V2EDim),
            fields={attrs.GEOFAC_ROT: attrs.GEOFAC_ROT},
            deps={
                "dual_edge_length": geometry_attrs.DUAL_EDGE_LENGTH,
                "edge_orientation": geometry_attrs.VERTEX_EDGE_ORIENTATION,
                "dual_area": geometry_attrs.DUAL_AREA,
                "owner_mask": "vertex_owner_mask",
            },
        )
        self.register_provider(geofac_rot)

        c_lin_e = factory.NumpyFieldsProvider(
            func=functools.partial(interpolation_fields.compute_c_lin_e, array_ns=self._xp),
            fields=(attrs.C_LIN_E,),
            domain={
                dims.EdgeDim: (
                    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2),
                    edge_domain(h_grid.Zone.END),
                ),
                dims.E2CDim: (0, 2),
            },
            deps={
                "edge_cell_length": geometry_attrs.EDGE_CELL_DISTANCE,
                "inv_dual_edge_length": f"inverse_of_{geometry_attrs.DUAL_EDGE_LENGTH}",
                "edge_owner_mask": "edge_owner_mask",
            },
            params={
                "horizontal_start": self._grid.start_index(
                    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
                )
            },
        )
        self.register_provider(c_lin_e)

    @property
    def metadata(self) -> dict[str, model.FieldMetaData]:
        return self._attrs

    @property
    def backend(self) -> gtx_backend.Backend:
        return self._backend

    @property
    def grid(self):
        return self._grid

    @property
    def vertical_grid(self):
        return None
