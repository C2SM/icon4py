import os

import numpy as np
from _pytest.fixtures import fixture

from icon4py.atm_dyn_iconam.horizontal import HorizontalMarkerIndex, HorizontalMeshConfig, HorizontalMeshParams
from icon4py.atm_dyn_iconam.icon_grid import IconGrid, MeshConfig
from icon4py.common.dimension import VertexDim, EdgeDim, CellDim
from icon4py.testutils.serialbox_utils import IconSerialDataProvider


@fixture
def with_grid():
    data_path = os.path.join(os.path.dirname(__file__), "ser_icondata")
    sp = IconSerialDataProvider("icon_diffusion_init", data_path).from_savepoint(linit=True,
                                                                                 date="2021-06-20T12:00:10.000")
    nproma, nlev, num_v, num_c, num_e = sp.get_metadata("nproma", "nlev", "num_vert", "num_cells",
                                                        "num_edges")
    cell_starts = sp.cells_start_index()
    cell_ends = sp.cells_end_index()
    vertex_starts = sp.vertex_start_index()
    vertex_ends = sp.vertex_end_index()
    edge_starts = sp.edge_start_index()
    edge_ends = sp.edge_end_index()

    config = MeshConfig(HorizontalMeshConfig(num_vertices=num_v, num_cells=num_c, num_edges=num_e))
    c2e2c = np.squeeze(sp.c2e2c(), axis=1)
    c2e2c0 = np.column_stack((c2e2c, (np.asarray(range(c2e2c.shape[0])))))
    grid = IconGrid().with_config(config) \
        .with_start_end_indices(VertexDim, vertex_starts, vertex_ends) \
        .with_start_end_indices(EdgeDim, edge_starts, edge_ends)\
        .with_start_end_indices(CellDim, cell_starts, cell_ends)\
        .with_connectivity(c2e=sp.c2e()).with_connectivity(e2c=sp.e2c())\
        .with_connectivity(c2e2c=c2e2c) \
        .with_connectivity(e2v=sp.e2v())\
        .with_connectivity(c2e2c0=c2e2c0)\

    return grid

def test_horizontal_grid_cell_indices(with_grid):
    assert with_grid.get_indices_from_to(CellDim, 3, 3) == (20897, 20896)  # halo +1
    #assert grid.get_indices_from_to(CellDim, 4, 4) == (1, 20896) #halo instead is (0,20896) why
    assert with_grid.get_indices_from_to(CellDim, 8, 8) == (4105, 20896) # interior
    assert with_grid.get_indices_from_to(CellDim, 9, 9) == (1, 850) #lb+1
    assert with_grid.get_indices_from_to(CellDim, 10, 10) == (851, 1688)
    assert with_grid.get_indices_from_to(CellDim, 11, 11) == (1689, 2511) #lb+2
    assert with_grid.get_indices_from_to(CellDim, 12,12) ==(2512, 3316) #lb+3
    assert with_grid.get_indices_from_to(CellDim, HorizontalMarkerIndex.START_PROG_CELL.value , HorizontalMarkerIndex.START_PROG_CELL.value) == (3317, 4104) #nudging


def test_horizontal_edge_indices(with_grid):
    assert with_grid.get_indices_from_to(EdgeDim, 0, 0) == (31559, 31558)
    assert with_grid.get_indices_from_to(EdgeDim, 3, 3) == (31559, 31558)
    assert with_grid.get_indices_from_to(EdgeDim, 4, 4) == (31559, 31558)
   # assert with_grid.get_indices_from_to(EdgeDim, 5, 5) == (1, 31558) #halo
    assert with_grid.get_indices_from_to(EdgeDim, 23, 23) == (5388, 6176) # nudging +1
    assert with_grid.get_indices_from_to(EdgeDim, 22,22) == (4990, 5387) #nudging
    assert with_grid.get_indices_from_to(EdgeDim, 21, 21) == (4185, 4989) #lb +7
    assert with_grid.get_indices_from_to(EdgeDim, 20, 20) == (3778, 4184)  # lb +6
    assert with_grid.get_indices_from_to(EdgeDim, 19, 19) == (2955, 3777)  # lb +5
    assert with_grid.get_indices_from_to(EdgeDim, 18, 18) == (2539, 2954)  # lb +4
    assert with_grid.get_indices_from_to(EdgeDim, 17, 17) == (1701, 2538)  # lb +3
    assert with_grid.get_indices_from_to(EdgeDim, 16, 16) == (1279, 1700)  # lb +2
    assert with_grid.get_indices_from_to(EdgeDim, 15, 15) == (429, 1278)  # lb +1
    assert with_grid.get_indices_from_to(EdgeDim, 14, 14) == (1, 428)  # lb +0


def test_horizontal_vertex_indices(with_grid):
    assert with_grid.get_indices_from_to(VertexDim, 0, 0) == (10664, 10664)



