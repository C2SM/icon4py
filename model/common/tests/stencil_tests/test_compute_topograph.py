import pytest

from icon4py.model.common.external_parameters import compute_topography

from icon4py.model.testing import datatest_utils as dt_utils, grid_utils, helpers
from icon4py.model.common.grid import geometry as grid_geometry
from model.atmosphere.diffusion.tests.diffusion_tests.test_benchmark_diffusion import \
    _construct_dummy_decomposition_info, get_cell_geometry_for_grid_file
from icon4py.model.common.grid import (
    geometry_attributes as geometry_meta,
)

@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file",
    [
        (dt_utils.R02B04_GLOBAL),
    ],
)
def test_compute_topograph(
    grid_file,
    backend,
    topography_savepoint,
):
    num_levels = 65
    grid_manager = grid_utils.get_grid_manager(grid_file=grid_file, num_levels=num_levels, keep_skip_values=True, backend=backend)
    grid = grid_manager.grid
    coordinates = grid_manager.coordinates
    geometry_input_fields = grid_manager.geometry_fields

    geometry_field_source = grid_geometry.GridGeometry(
        grid=grid,
        decomposition_info=_construct_dummy_decomposition_info(grid, backend),
        backend=backend,
        coordinates=coordinates,
        extra_fields=geometry_input_fields,
        metadata=geometry_meta.attrs
    )

    cell_geometry = get_cell_geometry_for_grid_file(grid_file, geometry_field_source, backend)

    topo_c = compute_topography(
        cell_lat=cell_geometry.lat.asnumpy(),
        u0=1.0,
        backend=backend,
    )

    topo_c_ref = topography_savepoint.topo_c().asnumpy()

    assert  helpers.dallclose(
        topo_c,
        topo_c_ref,
    )


