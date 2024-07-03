import enum

import scipy as sp
import xugrid.ugrid.ugrid2d as ux

import icon4py.model.common.decomposition.definitions as defs
from icon4py.model.common import dimension as dims
from icon4py.model.common.settings import xp


#TODO do we need three of those? 
class DecompositionFlag(enum.IntEnum):
    #: cell is owned by this rank
    OWNED = 0,
    #: cell is in the first halo line: that is cells that share and edge with an owned cell
    FIRST_HALO_LINE = 1,
    #: cell is in the second halo line: that is cells that share only a vertex with an owned cell (and at least an edge with a FIRST_HALO_LINE cell)
    SECOND_HALO_LINE = 2



class HaloGenerator:
    """Creates necessary halo information for a given rank."""
    def __init__(self, rank_info:defs.ProcessorProperties, rank_mapping: xp.ndarray, ugrid:ux.Ugrid2d, num_lev:int):
        """
        
        Args:
            rank_info: contains information on the communicator and local compute node.
            rank_mapping: array with shape (global_num_cells) mapping of global cell indices to their rank in the distribution
        """
        self._props = rank_info
        self._mapping = rank_mapping
        self._global_grid = ugrid
        self._num_lev = num_lev
       
        
        
    def _validate(self):
        assert self._mapping.ndim == 1
        # the decomposition should match the communicator size
        assert xp.max(self._mapping) == self._props.comm_size - 1
       
  
        
    def _post_init(self):
        self._validate()


    def next_halo_line(self, cell_line: xp.ndarray, depot=None):
        """Returns the global indices of the next halo line.
        
        Args:
            cell_line: global indices of cells we want to find the neighbors of
            depot: global indices that have already been collected
        Returns:
            next_halo_cells: global indices of the next halo line
        """
        cell_neighbors = self._face_face_connectivity(cell_line)

        if depot is not None:
            cells_so_far = xp.hstack((depot, cell_line))
        else:
            cells_so_far = cell_line

        next_halo_cells = xp.setdiff1d(xp.unique(cell_neighbors), cells_so_far, assume_unique=True)
        # assert next_halo_cells.shape[0] == next_halo_size
        return next_halo_cells


    def _face_face_connectivity(self, cells: xp.ndarray):
        """ In xugrid face-face connectivity is a scipy spars matrix, so we reduce it to the our regular sparse matrix format: (n_cells, 3)  
        """
        conn = self._global_grid.face_face_connectivity
        _, c2e2c, _ = sp.sparse.find(conn[cells, :])
        return c2e2c


    def _find_neighbors(self, cell_line:xp.ndarray, connectivity: xp.ndarray)->xp.ndarray:
        """ Get a flattened list of all (unique) neighbors to a given global index list"""
        neighbors = connectivity[cell_line, :]
        shp = neighbors.shape
        unique_neighbors = xp.unique(neighbors.reshape(shp[0] * shp[1]))
        return unique_neighbors

 
    def find_edge_neighbors_for_cells(self, cell_line: xp.ndarray) -> xp.ndarray:
        return self._find_neighbors(self, cell_line, connectivity=self._global_grid.face_edge_connectivity)
    def find_vertex_neighbors_for_cells(self, cell_line:xp.ndarray)->xp.ndarray:
        return self._find_neighbors(self, cell_line, connectivity=self._global_grid.face_node_connectivity)
    
    
    def owned_cells(self)->xp.ndarray:
        """Returns the global indices of the cells owned by this rank"""
        owned_cells = self._mapping == self._props.rank
        return xp.asarray(owned_cells).nonzero()[0]
        
    def construct_decomposition_info(self):
        """Constructs the decomposition info for the current rank"""
        
        #: cells
        owned_cells = self.owned_cells() # global indices of owned cells
        first_halo_cells = self.next_halo_line(owned_cells)
        second_halo_cells = self.next_halo_line(first_halo_cells, owned_cells)
        
        total_halo_cells = xp.hstack((first_halo_cells, second_halo_cells))
        global_cell_index = xp.hstack((owned_cells, total_halo_cells))

        c_owner_mask = xp.isin(global_cell_index, owned_cells)
        
        decomp_info = defs.DecompositionInfo(klevels=self._num_lev).with_dimension(dims.CellDim,
                                                                          global_cell_index,
                                                                          c_owner_mask)
        
        #: edges
        edges_on_owned_cells = self.find_edge_neighbors_for_cells(owned_cells)
        edges_on_first_halo_line = self.find_edge_neighbors_for_cells(first_halo_cells)
        edges_on_second_halo_line = self.find_edge_neighbors_for_cells(second_halo_cells)
        # reduce overlap
        
        all_edges = xp.hstack((edges_on_owned_cells, xp.setdiff1d(edges_on_owned_cells, edges_on_first_halo_line), xp.setdiff1d(edges_on_first_halo_line, edges_on_second_halo_line)))
        """
        We need to reduce the overlap:
       
        `edges_on_owned_cells` and `edges_on_first_halo_line` contain the edges on the cutting line. 
        In order to have unique ownership of edges (and vertices) among nodes there needs to be a convention as to where
        those elements on the cutting line go: according to a remark in `mo_decomposition_tools.f90` ICON puts them to the node with the higher rank.
        """
        edge_owner_mask = xp.isin(all_edges, edges_on_owned_cells)
        intersect_owned_first_line = xp.intersect1d(edges_on_owned_cells, edges_on_first_halo_line)

        for edge in intersect_owned_first_line:
            local_index = xp.where(all_edges == edge)[0][0]
            owning_ranks = self._mapping[self._global_grid.edge_face_connectivity[edge]] 
            assert owning_ranks.shape[0] == 2
            assert owning_ranks[0] != owning_ranks[1], "both neighboring cells are owned by the same rank"
            assert self._props.rank in owning_ranks, "neither of the neighboring cells is owned by this rank"           
            # assign the edge to the rank with the higher rank
            if max(owning_ranks) > self._props.rank:
                edge_owner_mask[local_index] = False
            else:
                edge_owner_mask[local_index] = True
                
                
        decomp_info.with_dimension(dims.EdgeDim, all_edges, edge_owner_mask)
        
        # vertices
        vertices_on_owned_cells = self.find_vertex_neighbors_for_cells(owned_cells)
        vertices_on_first_halo_line = self.find_vertex_neighbors_for_cells(first_halo_cells)
        vertices_on_second_halo_line = self.find_vertex_neighbors_for_cells(second_halo_cells) #TODO: do we need that?
        unique_vertices_on_halo_cells = xp.setdiff1d(vertices_on_first_halo_line, vertices_on_owned_cells)
        vertices_on_owned_edges = xp.unique(self._global_grid.edge_node_connectivity[all_edges[edge_owner_mask]])

        # create decomposition_info
        all_vertices = xp.hstack((vertices_on_owned_cells, unique_vertices_on_halo_cells))
        v_owner_mask = xp.isin(all_vertices, vertices_on_owned_edges)
        decomp_info.with_dimension(dims.VertexDim, all_vertices, v_owner_mask)
        return decomp_info

        