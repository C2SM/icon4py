import gt4py.next as gtx
import numpy as np
import pytest

import icon4py.model.common.constants as constants
import icon4py.model.common.dimension as dims
import icon4py.model.common.grid.geometry as geometry
import icon4py.model.common.math.helpers
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.test_utils import datatest_utils as dt_utils, helpers


@pytest.mark.parametrize("experiment", [dt_utils.GLOBAL_EXPERIMENT])
@pytest.mark.datatest
def test_dual_edge_length(experiment, grid_savepoint, icon_grid):
    expected = grid_savepoint.dual_edge_length()


    cell_center_lat = grid_savepoint.cell_center_lat()
    cell_center_lon = grid_savepoint.cell_center_lon()
    result = helpers.zero_field(icon_grid, dims.EdgeDim)
    buffer = np.vstack((np.ones(icon_grid.num_edges), -1 * np.ones(icon_grid.num_edges))).T
    subtraction_coeff = gtx.as_field((dims.EdgeDim, dims.E2CDim), data = buffer)
    

    
    geometry.dual_edge_length.with_backend(None)(cell_center_lat, cell_center_lon, subtraction_coeff, constants.EARTH_RADIUS,
                                                 offset_provider = {"E2C": icon_grid.get_offset_provider("E2C")}, out = result)

    assert helpers.dallclose(result, expected.asnumpy())
    
@pytest.mark.parametrize("experiment", [dt_utils.GLOBAL_EXPERIMENT])
@pytest.mark.datatest
def test_primal_edge_length(grid_savepoint, icon_grid):
    expected = grid_savepoint.primal_edge_length()
    vertex_lat = grid_savepoint.verts_vertex_lat()
    vertex_lon = grid_savepoint.verts_vertex_lon()
    result = helpers.zero_field(icon_grid, dims.EdgeDim)
    buffer = np.vstack((np.ones(icon_grid.num_edges), -1 * np.ones(icon_grid.num_edges))).T
    subtract_coeff = gtx.as_field((dims.EdgeDim, dims.E2VDim), data = buffer)
    geometry.primal_edge_length.with_backend(None)(vertex_lat, vertex_lon, subtract_coeff, constants.EARTH_RADIUS, offset_provider = {"E2V": icon_grid.get_offset_provider("E2V")}, out = result)
    assert helpers.dallclose(result.asnumpy(), expected.asnumpy())
    
    
@pytest.mark.datatest
def test_vertex_vertex_length(grid_savepoint, icon_grid):
    expected = grid_savepoint.inv_vert_vert_length()

    vertex_lat = grid_savepoint.verts_vertex_lat()
    vertex_lon = grid_savepoint.verts_vertex_lon()
    length = helpers.zero_field(icon_grid, dims.EdgeDim)
    inverse = helpers.zero_field(icon_grid, dims.EdgeDim)
    domain = h_grid.domain(dims.EdgeDim)
    horizontal_start = icon_grid.start_index(domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    horizontal_end = icon_grid.end_index(domain(h_grid.Zone.LOCAL))
    
    geometry.vertex_vertex_length.with_backend(None)(vertex_lat, vertex_lon, radius = 1.0,# constants.EARTH_RADIUS,
                                                     offset_provider = {"E2C2V":icon_grid.get_offset_provider("E2C2V")}, 
                                                     out = length,
                                                     domain={dims.EdgeDim: (horizontal_start, horizontal_end)}
                                                     )
    icon4py.model.common.math.helpers.invert(length, offset_provider = {}, out = inverse)
    assert helpers.dallclose(expected.asnumpy(), inverse.asnumpy())
    
    

@pytest.mark.parametrize("experiment", [dt_utils.GLOBAL_EXPERIMENT])
@pytest.mark.datatest
def test_edge_control_area(grid_savepoint, icon_grid):
    expected = grid_savepoint.edge_areas()
    owner_mask = grid_savepoint.e_owner_mask()
    primal_edge_length = grid_savepoint.primal_edge_length()
    dual_edge_length = grid_savepoint.dual_edge_length()
    result = helpers.zero_field(icon_grid, dims.EdgeDim)
    geometry.edge_control_area(owner_mask,primal_edge_length, dual_edge_length, offset_provider = {}, out=result)
    assert helpers.dallclose(expected.asnumpy(), result.asnumpy())


@pytest.mark.parametrize("experiment", [dt_utils.GLOBAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT])
@pytest.mark.datatest
def test_coriolis_parameter(grid_savepoint, icon_grid):
    expected = grid_savepoint.f_e()
    result = helpers.zero_field(icon_grid, dims.EdgeDim)
    lat = grid_savepoint.edge_center_lat()
    geometry.coriolis_parameter_on_edges(lat, constants.EARTH_ANGULAR_VELOCITY, offset_provider={}, out=result)
    assert helpers.dallclose(expected.asnumpy(), result.asnumpy())