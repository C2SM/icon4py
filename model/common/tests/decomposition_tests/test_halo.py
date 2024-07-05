import mpi4py
import mpi4py.MPI
import pytest
import xugrid as xu

import icon4py.model.common.dimension as dims
from icon4py.model.common.decomposition import definitions as defs
from icon4py.model.common.decomposition.halo import HaloGenerator
from icon4py.model.common.grid import simple
from icon4py.model.common.settings import xp
from icon4py.model.common.test_utils.parallel_helpers import (  # noqa: F401  # import fixtures from test_utils package
    check_comm_size,
    processor_props,
)


simple_distribution = xp.asarray([0, # 0c 
                                  1, # 1c
                                  1, # 2c
                                  0, # 3c
                                  0, # 4c
                                  1, # 5c
                                  0, # 6c
                                  0, # 7c
                                  2, # 8c 
                                  2, # 9c 
                                  0, # 10c
                                  2, # 11c
                                  3, # 12c
                                  3, # 13c 
                                  1, # 14c
                                  3, # 15c
                                  3, # 16c
                                  1, #17c
                                  ])  
cell_own = {0:[0,3,4,6,7,10], 
            1:[1,2,5,14,17], 
            2:[8,9,11], 
            3:[12,13,15,16]}
cell_halos = {0:[2,15, 1, 11, 13, 9,17, 5, 12, 14, 8,16], 
              1:[4, 16, 3, 8, 15, 11, 13, 0,7,6,9,10,12],
              2:[5, 6, 12,14,7, 2,1,4,3, 10, 15, 16, 17],
              3:[9, 10, 17, 14, 0, 1,6, 7,8,2, 3, 4, 5,11]}


edge_own = {0:[1,5,12,13,14,9], 
            1:[8,7,6,25,4, 2], 
            2:[16,11,15,17,10,24], 
            3:[19,23,22,26,0,3,20,18,21]}

edge_halos = {0:[0,4,21,10,2, 3,8,6,7,19,20,17,16,11,18,26,25,15,23,24,22], 
              1:[5,12,22,23,3,1,9,15,16,11,19,20,0,17,24,21,26,13,10,14,18,],
              2:[7,6,9,8,14,18,19,23,25,20,12,13,2,3,4,5,1,21,22,0,26], 
              3:[10,11,13,14,25,6,24,1,5,4,8,9,17,12,15,16,2,7]}
 
vertex_own = {0:[4],
              1:[],
              2:[3,5],
              3:[0,1,2,6,7,8,]}
vertex_halo = {0:[0,1,2,3,5,6,7,8],
               1:[1,2,0,5,3,8,6, 7, 4],
               2:[8,6,7,4, 0,2,1 ],
               3:[3,4,5]}

owned ={dims.CellDim:cell_own, dims.EdgeDim:edge_own, dims.VertexDim:vertex_own}
halos = {dims.CellDim:cell_halos, dims.EdgeDim:edge_halos, dims.VertexDim:vertex_halo}
def test_halo_constructor_owned_cells(processor_props, simple_ugrid): # noqa: F811  # fixture
    
    grid = simple_ugrid
    halo_generator = HaloGenerator(ugrid=grid,
                                   rank_info=processor_props,
                                   rank_mapping=simple_distribution,
                                   num_lev=1, 
                                   face_face_connectivity=simple.SimpleGridData.c2e2c_table)
    my_owned_cells = halo_generator.owned_cells()
    
    print(f"rank {processor_props.rank} owns {my_owned_cells} ")
    assert my_owned_cells.size == len(cell_own[processor_props.rank])
    assert xp.setdiff1d(my_owned_cells, cell_own[processor_props.rank]).size == 0

@pytest.mark.skip(reason="mpi.GATHER ???")
@pytest.mark.mpi(min_size=4)
def test_cell_ownership_is_unique(processor_props, simple_ugrid):
    grid = simple_ugrid
    num_cells = simple.SimpleGrid._CELLS
    halo_generator = HaloGenerator(ugrid=grid,
                                   rank_info=processor_props,
                                   rank_mapping=simple_distribution,
                                   num_lev=1, face_face_connectivity=simple.SimpleGridData.c2e2c_table)
    my_owned_cells = halo_generator.owned_cells()
    print(f"rank {processor_props.rank} owns {my_owned_cells} ")
    # assert that each cell is only owned by one rank
    if not mpi4py.MPI.Is_initialized():
        mpi4py.MPI.Init()
    if processor_props.rank == 0:
        gathered = -1 *xp.ones([processor_props.comm_size, num_cells], dtype=xp.int64)
    else:
        gathered = None
    processor_props.comm.Gather(my_owned_cells, gathered,  root=0)
    if processor_props.rank == 0:
        print(gathered.shape)
        print(gathered)
    #     gathered = xp.where(gathered.reshape(processor_props.comm_size * 18) > 0)
    #    assert gathered == simple_distribution.size - 1
    #    assert gathered.size == len(xp.unique(gathered))
@pytest.mark.parametrize("dim", [dims.CellDim, dims.EdgeDim, dims.VertexDim])
def test_halo_constructor_decomposition_info(processor_props, simple_ugrid, dim): # noqa: F811  # fixture
    grid = simple_ugrid
    halo_generator = HaloGenerator(ugrid=grid,
                                   rank_info=processor_props,
                                   rank_mapping=simple_distribution,
                                   num_lev=1, face_face_connectivity=simple.SimpleGridData.c2e2c_table, 
                                   node_face_connectivity=simple.SimpleGridData.v2c_table)

    decomp_info = halo_generator.construct_decomposition_info()
    my_halo = decomp_info.global_index(dim, defs.DecompositionInfo.EntryType.HALO) 
    print(f"rank {processor_props.rank} has halo {dim} : {my_halo}")
    assert my_halo.size == len(halos[dim][processor_props.rank])
    assert xp.setdiff1d(my_halo, halos[dim][processor_props.rank], assume_unique=True).size == 0
    my_owned = decomp_info.global_index(dim, defs.DecompositionInfo.EntryType.OWNED)
    print(f"rank {processor_props.rank} owns {dim} : {my_owned} ")
    assert my_owned.size == len(owned[dim][processor_props.rank])
    assert xp.setdiff1d(my_owned, owned[dim][processor_props.rank], assume_unique=True).size == 0
    




@pytest.fixture
def simple_ugrid()->xu.Ugrid2d:
    """
    Programmatically construct a xugrid.ugrid.ugrid2d.Ugrid2d object
    
    Returns: a Ugrid2d object base on the SimpleGrid

    """
    simple_mesh = simple.SimpleGrid()
    fill_value = -1
    node_x = xp.arange(simple_mesh.num_vertices, dtype=xp.float64)
    node_y = xp.arange(simple_mesh.num_vertices, dtype=xp.float64)
    grid = xu.Ugrid2d(node_x, node_y, fill_value,
                      projected=True,
                      face_node_connectivity=simple_mesh.connectivities[dims.C2VDim],
                      edge_node_connectivity=simple_mesh.connectivities[dims.E2VDim], )

    return grid
    

    