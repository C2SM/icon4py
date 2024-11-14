import pytest

import icon4py.model.common.states.factory as factory
import icon4py.model.common.test_utils.datatest_utils as dt_utils
from icon4py.model.common.interpolation import (
    interpolation_attributes as attrs,
    interpolation_factory,
)

from .. import utils


C2E_SIZE = 3

@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
    ],
)
@pytest.mark.datatest
def test_factory_raises_error_on_unknown_field(grid_file, experiment, backend, decomposition_info):
    geometry = utils.get_grid_geometry(backend, grid_file)
    interpolation_source = interpolation_factory.InterpolationFieldsFactory(
        grid = geometry.grid,
        decomposition_info=decomposition_info,
        geometry=geometry,
        backend=backend,
        metadata= attrs.attrs
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
    geometry = utils.get_grid_geometry(backend, grid_file)
    grid = geometry.grid
    factory = interpolation_factory.InterpolationFieldsFactory(
        grid = grid,
        decomposition_info=decomposition_info,
        geometry=geometry,
        backend=backend,
        metadata= attrs.attrs
    )
    field = factory.get(attrs.C_LIN_E)
    assert field.asnumpy().shape == (grid.num_edges, 2)

@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
@pytest.mark.datatest
def test_get_geofac_div(grid_file, experiment, backend, decomposition_info):
    geometry = utils.get_grid_geometry(backend, grid_file)
    grid = geometry.grid
    factory = interpolation_factory.InterpolationFieldsFactory(
        grid = grid,
        decomposition_info=decomposition_info,
        geometry=geometry,
        backend=backend,
        metadata= attrs.attrs
    )
    field = factory.get(attrs.GEOFAC_DIV)
    assert field.asnumpy().shape == (grid.num_cells, C2E_SIZE)
