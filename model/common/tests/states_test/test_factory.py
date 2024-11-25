# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

import gt4py.next as gtx
import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import horizontal as h_grid, vertical as v_grid
from icon4py.model.common.math import helpers as math_helpers
from icon4py.model.common.metrics import metric_fields as metrics
from icon4py.model.common.states import factory, model, utils as state_utils
from icon4py.model.common.test_utils import helpers as test_helpers


cell_domain = h_grid.domain(dims.CellDim)
k_domain = v_grid.domain(dims.KDim)


class SimpleSource(factory.FieldSource):
    def __init__(
        self,
        data_: dict[str, tuple[state_utils.FieldType, model.FieldMetaData]],
        backend,
        grid,
        vertical_grid=None,
    ):
        self._backend = backend
        self._grid = grid
        self._vertical_grid = vertical_grid
        self._metadata = {}
        for key, value in data_.items():
            self.register_provider(factory.PrecomputedFieldProvider({key: value[0]}))
            self._metadata[key] = value[1]

    @property
    def metadata(self):
        return self._metadata

    @property
    def grid(self):
        return self._grid

    @property
    def vertical_grid(self) -> Optional[v_grid.VerticalGrid]:
        return self._vertical_grid

    @property
    def backend(self):
        return self._backend


@pytest.mark.datatest
def test_field_operator_provider(backend, grid_savepoint):
    on_gpu = test_helpers.is_gpu(backend)
    grid = grid_savepoint.construct_icon_grid(on_gpu)

    field_op = math_helpers.geographical_to_cartesian_on_cells.with_backend(None)
    domain = {dims.CellDim: (cell_domain(h_grid.Zone.LOCAL), cell_domain(h_grid.Zone.LOCAL))}
    deps = {"lat": "lat", "lon": "lon"}
    fields = {"x": "x", "y": "y", "z": "z"}
    lat = grid_savepoint.lat(dims.CellDim)
    lon = grid_savepoint.lon(dims.CellDim)
    data = {
        "lat": (lat, {"standard_name": "lat", "units": ""}),
        "lon": (lon, {"standard_name": "lon", "units": ""}),
    }

    field_source = SimpleSource(data_=data, backend=backend, grid=grid)
    provider = factory.FieldOperatorProvider(field_op, domain, fields, deps)
    provider("x", field_source, backend, field_source.grid_provider)
    x = provider.fields["x"]
    assert isinstance(x, gtx.Field)
    assert dims.CellDim in x.domain.dims


@pytest.mark.datatest
def test_program_provider(backend, grid_savepoint, metrics_savepoint):
    on_gpu = test_helpers.is_gpu(backend)
    grid = grid_savepoint.construct_icon_grid(on_gpu)
    z_ifc = metrics_savepoint.z_ifc()
    program = metrics.compute_z_mc

    domain = {
        dims.CellDim: (cell_domain(h_grid.Zone.LOCAL), cell_domain(h_grid.Zone.LOCAL)),
        dims.KDim: (k_domain(v_grid.Zone.TOP), k_domain(v_grid.Zone.BOTTOM)),
    }
    deps = {
        "z_ifc": "input_f",
    }
    fields = {"z_mc": "output_f"}
    vct_a = grid_savepoint.vct_a()
    vct_b = grid_savepoint.vct_b()
    data = {"input_f": (z_ifc, {"standard_name": "input_f", "units": ""})}
    vertical_grid = v_grid.VerticalGrid(v_grid.VerticalGridConfig(num_levels=10), vct_a, vct_b)
    field_source = SimpleSource(data_=data, backend=backend, grid=grid, vertical_grid=vertical_grid)
    provider = factory.ProgramFieldProvider(program, domain, fields, deps)
    provider("output_f", field_source, backend, field_source.grid_provider)
    x = provider.fields["output_f"]
    assert isinstance(x, gtx.Field)
    assert dims.CellDim in x.domain.dims
