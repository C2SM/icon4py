
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import backend as gtx_backend

from icon4py.model.common.decomposition import definitions
from icon4py.model.common.grid import geometry, icon
from icon4py.model.common.interpolation import interpolation_fields
from icon4py.model.common.states import factory, model


class InterpolationFieldsFactory(factory.FieldSource, factory.GridProvider):
    def __init__(self,
                 grid: icon.IconGrid,
                 decomposition_info: definitions.DecompositionInfo,
                 geometry: geometry.GridGeometry,
                 backend: gtx_backend.Backend,
                 metadata: dict[str, model.FieldMetaData]
                 ):
        self._backend = backend
        self._allocator = gtx.constructors.zeros.partial(allocator=backend)
        self._grid = grid
        self._source: dict[str, factory.FieldSource] = {"geometry": geometry, "self": self}
        self._decomposition_info = decomposition_info
        self._attrs = metadata
        self._providers: dict[str, factory.FieldProvider] = {}
        self._register_computed_fields()


    def _register_computed_fields(self):
        # TODO (@halungge) only works on on fieldview-embedded GT4Py backend, as it writes a
        #      sparse field
        geofac_div = factory.FieldOperatorProvider(
            func=interpolation_fields.compute_geofac_div.with_backend(None),
            
        )



    def __repr__(self):
        return f"{self.__class__.__name__} (grid={self._grid.id!r})"

    @property
    def providers(self) -> dict[str, factory.FieldProvider]:
        return self._providers

    @property
    def metadata(self) -> dict[str, model.FieldMetaData]:
        return self._attrs

    @property
    def backend(self) -> gtx_backend.Backend:
        return self._backend

    @property
    def grid_provider(self):
        return self

    @property
    def grid(self):
        return self._grid

    @property
    def vertical_grid(self):
        return None