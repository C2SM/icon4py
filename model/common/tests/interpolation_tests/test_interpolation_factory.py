# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

import icon4py.model.common.states.factory as factory
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.interpolation import (
    interpolation_attributes as attrs,
    interpolation_factory,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import (
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
        geometry_source=geometry,
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
    registry_key = experiment.join(backend.name)
    factory = interpolation_factories.get(registry_key)
    if not factory:
        geometry = gridtest_utils.get_grid_geometry(backend, experiment, grid_file)

        factory = interpolation_factory.InterpolationFieldsFactory(
            grid=geometry.grid,
            decomposition_info=geometry._decomposition_info,
            geometry_source=geometry,
            backend=backend,
            metadata=attrs.attrs,
        )
        interpolation_factories[registry_key] = factory
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
    # TODO (any): This test does not work on gpu backend because the field operator is run with embedded backend
    if data_alloc.is_cupy_device(backend):
        pytest.skip("skipping: gpu backend is unsupported")
    field_ref = interpolation_savepoint.geofac_div()
    factory = get_interpolation_factory(backend, experiment, grid_file)
    grid = factory.grid
    field = factory.get(attrs.GEOFAC_DIV)
    assert field.shape == (grid.num_cells, C2E_SIZE)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy(), rtol=rtol)


## FIXME: does not validate
#   -> connectivity order between reference from serialbox and computed value is different
@pytest.mark.parametrize(
    "grid_file, experiment, rtol",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 5e-9),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 1e-11),
    ],
)
@pytest.mark.datatest
def test_get_geofac_grdiv(interpolation_savepoint, grid_file, experiment, backend, rtol):
    # TODO (any): This test does not work with gpu backend because geofac_div cannot be computed on gpu backend.
    if data_alloc.is_cupy_device(backend):
        pytest.skip("skipping: gpu backend is unsupported")
    field_ref = interpolation_savepoint.geofac_grdiv()
    factory = get_interpolation_factory(backend, experiment, grid_file)
    grid = factory.grid
    field = factory.get(attrs.GEOFAC_GRDIV)
    assert field.shape == (grid.num_edges, 5)
    # FIXME: e2c2e constructed from grid file has different ordering than the serialized one
    assert_reordered(field.asnumpy(), field_ref.asnumpy(), rtol)


def assert_reordered(val: np.ndarray, ref: np.ndarray, rtol):
    assert val.shape == ref.shape, f"arrays do not have the same shape: {val.shape} vs {ref.shape}"
    s_val = np.argsort(val)
    s_ref = np.argsort(ref)
    for i in range(val.shape[0]):
        assert test_helpers.dallclose(
            val[i, s_val[i, :]], ref[i, s_ref[i, :]], rtol=rtol
        ), f"assertion failed for row {i}"


@pytest.mark.parametrize(
    "grid_file, experiment, rtol",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 5e-9),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 1e-11),
    ],
)
@pytest.mark.datatest
def test_get_geofac_rot(interpolation_savepoint, grid_file, experiment, backend, rtol):
    # TODO (any): This test does not work on gpu backend because the field operator is run with embedded backend
    if data_alloc.is_cupy_device(backend):
        pytest.skip("skipping: gpu backend is unsupported")
    field_ref = interpolation_savepoint.geofac_rot()
    factory = get_interpolation_factory(backend, experiment, grid_file)
    grid = factory.grid
    field = factory.get(attrs.GEOFAC_ROT)
    horizontal_start = grid.start_index(vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    assert field.shape == (grid.num_vertices, V2E_SIZE)
    assert test_helpers.dallclose(
        field_ref.asnumpy()[horizontal_start:, :], field.asnumpy()[horizontal_start:, :], rtol=rtol
    )


@pytest.mark.parametrize(
    "grid_file, experiment, rtol",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 5e-9),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 1e-11),
    ],
)
@pytest.mark.datatest
def test_get_geofac_n2s(interpolation_savepoint, grid_file, experiment, backend, rtol):
    # TODO (any): This test does not work with gpu backend because geofac_div cannot be computed on gpu backend.
    if data_alloc.is_cupy_device(backend):
        pytest.skip("skipping: gpu backend is unsupported")
    field_ref = interpolation_savepoint.geofac_n2s()
    factory = get_interpolation_factory(backend, experiment, grid_file)
    grid = factory.grid
    field = factory.get(attrs.GEOFAC_N2S)
    assert field.shape == (grid.num_cells, 4)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy(), rtol=rtol)


@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
@pytest.mark.datatest
def test_get_geofac_grg(interpolation_savepoint, grid_file, experiment, backend):
    # TODO (any): This test does not work on gpu backend because the field operator is run with embedded backend
    if data_alloc.is_cupy_device(backend):
        pytest.skip("skipping: gpu backend is unsupported")
    field_ref = interpolation_savepoint.geofac_grg()
    factory = get_interpolation_factory(backend, experiment, grid_file)
    grid = factory.grid
    field_x = factory.get(attrs.GEOFAC_GRG_X)
    assert field_x.shape == (grid.num_cells, 4)
    field_y = factory.get(attrs.GEOFAC_GRG_Y)
    assert field_y.shape == (grid.num_cells, 4)
    # TODO (@halungge) tolerances are high, especially in the 0th (central) component, check stencil
    #   this passes due to the atol which is too large for the values
    assert test_helpers.dallclose(
        field_ref[0].asnumpy(),
        field_x.asnumpy(),
        rtol=1e-7,
        atol=1e-6,
    )
    assert test_helpers.dallclose(
        field_ref[1].asnumpy(),
        field_y.asnumpy(),
        rtol=1e-7,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "grid_file, experiment, rtol",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 5e-9),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 1e-11),
    ],
)
@pytest.mark.datatest
def test_get_mass_conserving_cell_average_weight(
    interpolation_savepoint, grid_file, experiment, backend, rtol
):
    field_ref = interpolation_savepoint.c_bln_avg()
    factory = get_interpolation_factory(backend, experiment, grid_file)
    grid = factory.grid
    field = factory.get(attrs.C_BLN_AVG)

    assert field.shape == (grid.num_cells, 4)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy(), rtol=rtol)


@pytest.mark.parametrize(
    "grid_file, experiment, rtol",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 5e-9),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 1e-11),
    ],
)
@pytest.mark.datatest
def test_cells_aw_verts(interpolation_savepoint, grid_file, experiment, backend, rtol):
    if data_alloc.is_cupy_device(backend):
        pytest.skip("skipping: gpu backend is unsupported")
    field_ref = interpolation_savepoint.c_intp()
    factory = get_interpolation_factory(backend, experiment, grid_file)
    grid = factory.grid
    field = factory.get(attrs.CELL_AW_VERTS)

    assert field.shape == (grid.num_vertices, 6)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy(), rtol=rtol)
