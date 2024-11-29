# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import icon4py.model.common.states.factory as factory
from icon4py.model.common.interpolation import (
    interpolation_attributes as attrs,
    interpolation_factory,
)
from icon4py.model.common.test_utils import datatest_utils as dt_utils, grid_utils as gridtest_utils


C2E_SIZE = 3
E2C_SIZE = 2


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
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
@pytest.mark.datatest
def test_get_c_lin_e(grid_file, experiment, backend, decomposition_info):
    geometry = gridtest_utils.get_grid_geometry(backend, experiment, grid_file)
    grid = geometry.grid
    factory = interpolation_factory.InterpolationFieldsFactory(
        grid=grid,
        decomposition_info=decomposition_info,
        geometry=geometry,
        backend=backend,
        metadata=attrs.attrs,
    )
    field = factory.get(attrs.C_LIN_E)
    assert field.asnumpy().shape == (grid.num_edges, E2C_SIZE)


@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
@pytest.mark.datatest
def test_get_geofac_div(grid_file, experiment, backend, decomposition_info):
    geometry = gridtest_utils.get_grid_geometry(backend, experiment, grid_file)
    grid = geometry.grid
    factory = interpolation_factory.InterpolationFieldsFactory(
        grid=grid,
        decomposition_info=decomposition_info,
        geometry=geometry,
        backend=backend,
        metadata=attrs.attrs,
    )
    field = factory.get(attrs.GEOFAC_DIV)
    assert field.asnumpy().shape == (grid.num_cells, C2E_SIZE)


@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
@pytest.mark.datatest
def test_get_geofac_rot(grid_file, experiment, backend, decomposition_info):
    geometry = gridtest_utils.get_grid_geometry(backend, experiment, grid_file)
    grid = geometry.grid
    factory = interpolation_factory.InterpolationFieldsFactory(
        grid=grid,
        decomposition_info=decomposition_info,
        geometry=geometry,
        backend=backend,
        metadata=attrs.attrs,
    )
    field = factory.get(attrs.GEOFAC_ROT)
    assert field.asnumpy().shape == (grid.num_vertices, 6)
