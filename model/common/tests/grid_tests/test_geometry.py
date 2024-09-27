import gt4py.next as gtx
import numpy as np
import pytest

import icon4py.model.common.constants as constants
import icon4py.model.common.dimension as dims
import icon4py.model.common.grid.geometry as geometry
import icon4py.model.common.math.helpers as math_helpers
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.test_utils import datatest_utils as dt_utils, helpers

from . import utils


@pytest.mark.parametrize("experiment", [dt_utils.GLOBAL_EXPERIMENT])
@pytest.mark.datatest
def test_dual_edge_length(experiment, grid_savepoint, icon_grid):
    expected = grid_savepoint.dual_edge_length().asnumpy()


    cell_center_lat = grid_savepoint.cell_center_lat()
    cell_center_lon = grid_savepoint.cell_center_lon()
    result_arc = helpers.zero_field(icon_grid, dims.EdgeDim)
    result_tendon = helpers.zero_field(icon_grid, dims.EdgeDim)
    buffer = np.vstack((np.ones(icon_grid.num_edges), -1 * np.ones(icon_grid.num_edges))).T
    subtraction_coeff = gtx.as_field((dims.EdgeDim, dims.E2CDim), data = buffer)
    
    
    geometry.dual_edge_length.with_backend(None)(cell_center_lat, 
                                                 cell_center_lon, 
                                                 subtraction_coeff, 
                                                 constants.EARTH_RADIUS,
                                                 offset_provider = {"E2C": icon_grid.get_offset_provider("E2C")}, 
                                                 
                                                 out = (result_arc, result_tendon))

    arch_array = result_arc.asnumpy()
    tendon_array = result_tendon.asnumpy()
    rel_error = np.abs(arch_array - expected) / expected
    assert np.max(rel_error < 1e-12)
    assert helpers.dallclose(arch_array, expected, atol=1e-6)
    

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
    

@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
@pytest.mark.datatest
def test_vertex_vertex_length(grid_savepoint, grid_file):
    
    gm = utils.run_grid_manager(grid_file)
    grid = gm.grid
    expected = grid_savepoint.inv_vert_vert_length()
    result = helpers.zero_field(grid, dims.EdgeDim)

    lat = gtx.as_field((dims.VertexDim, ), gm.coordinates(dims.VertexDim)["lat"], dtype=float)
    lon = gtx.as_field((dims.VertexDim, ), gm.coordinates(dims.VertexDim)["lon"], dtype = float)
    edge_domain = h_grid.domain(dims.EdgeDim)
    start = grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    end = grid.end_index(edge_domain(h_grid.Zone.END))
    geometry.vertex_vertex_length(lat, 
                                  lon, constants.EARTH_RADIUS, out=result, 
                                  offset_provider={"E2C2V":grid.get_offset_provider("E2C2V")}, 
                                  domain={dims.EdgeDim:(start, end)})
    math_helpers.invert(result, offset_provider={}, out=result)

    assert helpers.dallclose(expected.asnumpy(), result.asnumpy())
        
    
    