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
from icon4py.model.common.grid import horizontal as h_grid, icon, vertical as v_grid
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
        grid: icon.IconGrid,
        vertical_grid: v_grid.VerticalGrid = None,
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
    def _sources(self) -> factory.FieldSource:
        return self

    @property
    def grid(self):
        return self._grid

    @property
    def vertical_grid(self) -> Optional[v_grid.VerticalGrid]:
        return self._vertical_grid

    @property
    def backend(self):
        return self._backend


@pytest.fixture
def cell_coordinate_source(grid_savepoint, backend):
    on_gpu = test_helpers.is_gpu(backend)
    grid = grid_savepoint.construct_icon_grid(on_gpu)
    lat = grid_savepoint.lat(dims.CellDim)
    lon = grid_savepoint.lon(dims.CellDim)
    data = {
        "lat": (lat, {"standard_name": "lat", "units": ""}),
        "lon": (lon, {"standard_name": "lon", "units": ""}),
    }

    coordinate_source = SimpleSource(data_=data, backend=backend, grid=grid)
    return coordinate_source


@pytest.fixture
def height_coordinate_source(metrics_savepoint, grid_savepoint, backend):
    on_gpu = test_helpers.is_gpu(backend)
    grid = grid_savepoint.construct_icon_grid(on_gpu)
    z_ifc = metrics_savepoint.z_ifc()
    vct_a = grid_savepoint.vct_a()
    vct_b = grid_savepoint.vct_b()
    data = {"height_coordinate": (z_ifc, {"standard_name": "height_coordinate", "units": ""})}
    vertical_grid = v_grid.VerticalGrid(v_grid.VerticalGridConfig(num_levels=10), vct_a, vct_b)
    field_source = SimpleSource(data_=data, backend=backend, grid=grid, vertical_grid=vertical_grid)
    return field_source


@pytest.mark.datatest
def test_field_operator_provider(cell_coordinate_source):
    field_op = math_helpers.geographical_to_cartesian_on_cells.with_backend(None)
    domain = {dims.CellDim: (cell_domain(h_grid.Zone.LOCAL), cell_domain(h_grid.Zone.LOCAL))}
    deps = {"lat": "lat", "lon": "lon"}
    fields = {"x": "x", "y": "y", "z": "z"}

    provider = factory.FieldOperatorProvider(field_op, domain, fields, deps)
    provider("x", cell_coordinate_source, cell_coordinate_source.backend, cell_coordinate_source)
    x = provider.fields["x"]
    assert isinstance(x, gtx.Field)
    assert dims.CellDim in x.domain.dims


@pytest.mark.datatest
def test_program_provider(height_coordinate_source):
    program = metrics.compute_z_mc
    domain = {
        dims.CellDim: (cell_domain(h_grid.Zone.LOCAL), cell_domain(h_grid.Zone.LOCAL)),
        dims.KDim: (k_domain(v_grid.Zone.TOP), k_domain(v_grid.Zone.BOTTOM)),
    }
    deps = {
        "z_ifc": "height_coordinate",
    }
    fields = {"z_mc": "output_f"}
    provider = factory.ProgramFieldProvider(program, domain, fields, deps)
    provider(
        "output_f",
        height_coordinate_source,
        height_coordinate_source.backend,
        height_coordinate_source,
    )
    x = provider.fields["output_f"]
    assert isinstance(x, gtx.Field)
    assert dims.CellDim in x.domain.dims


def test_field_source_raise_error_on_register(cell_coordinate_source):
    program = metrics.compute_z_mc
    domain = {
        dims.CellDim: (cell_domain(h_grid.Zone.LOCAL), cell_domain(h_grid.Zone.LOCAL)),
        dims.KDim: (k_domain(v_grid.Zone.TOP), k_domain(v_grid.Zone.BOTTOM)),
    }
    deps = {
        "z_ifc": "height_coordinate",
    }
    fields = {"z_mc": "output_f"}
    provider = factory.ProgramFieldProvider(program, domain, fields, deps)
    with pytest.raises(ValueError) as err:
        cell_coordinate_source.register_provider(provider)
        assert "not provided by source " in err.value


def test_composite_field_source_contains_all_metadata(
    cell_coordinate_source, height_coordinate_source
):
    backend = cell_coordinate_source.backend
    grid = cell_coordinate_source.grid
    foo = test_helpers.random_field(grid, dims.CellDim, dims.KDim)
    bar = test_helpers.random_field(grid, dims.EdgeDim, dims.KDim)
    data = {
        "foo": (foo, {"standard_name": "foo", "units": ""}),
        "bar": (bar, {"standard_name": "bar", "units": ""}),
    }

    test_source = SimpleSource(data_=data, grid=grid, backend=backend)
    composite = factory.CompositeSource(
        test_source, (cell_coordinate_source, height_coordinate_source)
    )

    assert composite.backend == test_source.backend
    assert composite.grid.id == test_source.grid.id
    assert test_source.metadata.items() <= composite.metadata.items()
    assert height_coordinate_source.metadata.items() <= composite.metadata.items()
    assert cell_coordinate_source.metadata.items() <= composite.metadata.items()


def test_composite_field_source_get_all_fields(cell_coordinate_source, height_coordinate_source):
    backend = cell_coordinate_source.backend
    grid = cell_coordinate_source.grid
    foo = test_helpers.random_field(grid, dims.CellDim, dims.KDim)
    bar = test_helpers.random_field(grid, dims.EdgeDim, dims.KDim)
    data = {
        "foo": (foo, {"standard_name": "foo", "units": ""}),
        "bar": (bar, {"standard_name": "bar", "units": ""}),
    }

    test_source = SimpleSource(data_=data, grid=grid, backend=backend)
    composite = factory.CompositeSource(
        test_source, (cell_coordinate_source, height_coordinate_source)
    )
    x = composite.get("foo")
    assert isinstance(x, gtx.Field)
    assert dims.CellDim in x.domain.dims
    assert dims.KDim in x.domain.dims
    x = composite.get("bar")
    assert len(x.domain.dims) == 2
    assert isinstance(x, gtx.Field)
    assert dims.EdgeDim in x.domain.dims
    assert dims.KDim in x.domain.dims
    assert len(x.domain.dims) == 2

    x = composite.get("lon")
    assert isinstance(x, gtx.Field)
    assert dims.CellDim in x.domain.dims
    assert len(x.domain.dims) == 1

    x = composite.get("height_coordinate")
    assert isinstance(x, gtx.Field)
    assert dims.KDim in x.domain.dims
    assert len(x.domain.dims) == 2


def test_composite_field_source_raises_upon_get_unknown_field(
    cell_coordinate_source, height_coordinate_source
):
    backend = cell_coordinate_source.backend
    grid = cell_coordinate_source.grid
    foo = test_helpers.random_field(grid, dims.CellDim, dims.KDim)
    bar = test_helpers.random_field(grid, dims.EdgeDim, dims.KDim)
    data = {
        "foo": (foo, {"standard_name": "foo", "units": ""}),
        "bar": (bar, {"standard_name": "bar", "units": ""}),
    }

    test_source = SimpleSource(data_=data, grid=grid, backend=backend)
    composite = factory.CompositeSource(
        test_source, (cell_coordinate_source, height_coordinate_source)
    )
    with pytest.raises(ValueError) as err:
        composite.get("alice")
        assert "not provided by source " in err.value
