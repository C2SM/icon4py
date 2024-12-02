# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import icon4py.model.common.states.factory as factory
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.interpolation import (
    interpolation_attributes as attrs,
    interpolation_factory,
)
from icon4py.model.common.test_utils import (
    datatest_utils as dt_utils,
    grid_utils as gridtest_utils,
    helpers as test_helpers,
)


V2E_SIZE = 6

C2E_SIZE = 3
E2C_SIZE = 2


interpolation_factories = {}

vertex_domain = h_grid.domain(dims.VertexDim)


@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
    ],
)
@pytest.mark.datatest
def test_factory_raises_error_on_unknown_field(grid_file, experiment, backend, decomposition_info):
    geometry = gridtest_utils.get_grid_geometry(backend, experiment, grid_file)
    interpolation_source = interpolation_factory.InterpolationFieldsFactory(
        grid=geometry.grid,
        decomposition_info=decomposition_info,
        geometry=geometry,
        backend=backend,
        metadata=attrs.attrs,
    )
    with pytest.raises(ValueError) as error:
        interpolation_source.get("foo", factory.RetrievalType.METADATA)
        assert "unknown field" in error.value


@pytest.mark.parametrize(
    "grid_file, experiment, rtol",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 5e-9),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 1e-11),
    ],
)
@pytest.mark.datatest
def test_get_c_lin_e(interpolation_savepoint, grid_file, experiment, backend, rtol):
    field_ref = interpolation_savepoint.c_lin_e()
    factory = get_interpolation_factory(backend, experiment, grid_file)
    grid = factory.grid
    field = factory.get(attrs.C_LIN_E)
    assert field.shape == (grid.num_edges, E2C_SIZE)
    assert test_helpers.dallclose(field.asnumpy(), field_ref.asnumpy(), rtol=rtol)


def get_interpolation_factory(
    backend, experiment, grid_file
) -> interpolation_factory.InterpolationFieldsFactory:
    name = grid_file.join(backend.name)
    factory = interpolation_factories.get(name)
    if not factory:
        geometry = gridtest_utils.get_grid_geometry(backend, experiment, grid_file)

        factory = interpolation_factory.InterpolationFieldsFactory(
            grid=geometry.grid,
            decomposition_info=geometry._decomposition_info,
            geometry=geometry,
            backend=backend,
            metadata=attrs.attrs,
        )
        interpolation_factories[name] = factory
    return factory


@pytest.mark.parametrize(
    "grid_file, experiment, rtol",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 1e-9),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 1e-12),
    ],
)
@pytest.mark.datatest
def test_get_geofac_div(interpolation_savepoint, grid_file, experiment, backend, rtol):
    field_ref = interpolation_savepoint.geofac_div()
    factory = get_interpolation_factory(backend, experiment, grid_file)
    grid = factory.grid
    field = factory.get(attrs.GEOFAC_DIV)
    assert field.shape == (grid.num_cells, C2E_SIZE)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy(), rtol=rtol)


@pytest.mark.parametrize(
    "grid_file, experiment, rtol",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 5e-9),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 1e-11),
    ],
)
@pytest.mark.datatest
def test_get_geofac_rot(interpolation_savepoint, grid_file, experiment, backend, rtol):
    field_ref = interpolation_savepoint.geofac_rot()
    factory = get_interpolation_factory(backend, experiment, grid_file)
    grid = factory.grid
    field = factory.get(attrs.GEOFAC_ROT)
    horizontal_start = grid.start_index(vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    assert field.shape == (grid.num_vertices, V2E_SIZE)
    assert test_helpers.dallclose(
        field_ref.asnumpy()[horizontal_start:, :], field.asnumpy()[horizontal_start:, :], rtol=rtol
    )
