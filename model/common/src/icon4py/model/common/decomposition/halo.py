import enum
import functools
from typing import Union

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
    def __init__(self, rank_info:defs.ProcessProperties, rank_mapping: xp.ndarray, ugrid:ux.Ugrid2d, num_lev:int, face_face_connectivity:xp.ndarray = None, node_face_connectivity = None):
        """
        
        Args:
            rank_info: contains information on the communicator and local compute node.
            rank_mapping: array with shape (global_num_cells) mapping of global cell indices to their rank in the distribution
            ugrid: the global grid
            num_lev: number of vertical levels
            face_face_connectivity: face-face connectivity matrix: (n_cells, 3) xugrid uses a 
                    scipy.sparse matrix for this which causes problems with zero based indices so we 
                    allow to pass it directly as a workaround 
            node_face_connectivity: node-face connectivity matrix: (n_vertex, 6) xugrid uses a  
                sparse matrix for this which causes problems with zero based indices so wes      
        """
        self._props = rank_info
        self._mapping = rank_mapping
        self._global_grid = ugrid
        self._num_lev = num_lev
        self._c2e2c = face_face_connectivity
        self._v2c = node_face_connectivity
        
        
        
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
        cell_neighbors = self._cell_neighbors(cell_line)

        if depot is not None:
            cells_so_far = xp.hstack((depot, cell_line))
        else:
            cells_so_far = cell_line

        next_halo_cells = xp.setdiff1d(xp.unique(cell_neighbors), cells_so_far, assume_unique=True)
        # assert next_halo_cells.shape[0] == next_halo_size
        return next_halo_cells


    def _cell_neighbors(self, cells: xp.ndarray):
        if self._c2e2c is not None:
            return xp.unique(self._c2e2c[cells, :])
        else:
            return self._cell_neighbors_from_sparse(cells)
    @functools.cached_property
    def _node_face_connectivity(self)->Union[xp.ndarray, sp.sparse.csr_matrix]:
        if self._v2c is not None:
            return self._v2c
        else:
            return self._global_grid._node_face_connectivity
    
    def _cell_neighbors_from_sparse(self, cells: xp.ndarray):
        """ In xugrid face-face connectivity is a scipy spars matrix, so we reduce it to the regular sparse matrix format: (n_cells, 3)  
        """
        conn = self._c2e2c.face_face_connectivity

        neighbors = conn[cells, :]
        # There is an issue with explicit 0 (for zero based indices) since sparse.find projects them out...
        i, j, vals = sp.sparse.find(neighbors)

        return j


    def _find_neighbors(self, cell_line:xp.ndarray, connectivity: xp.ndarray)->xp.ndarray:
        """ Get a flattened list of all (unique) neighbors to a given global index list"""
        neighbors = connectivity[cell_line, :]
        shp = neighbors.shape
        unique_neighbors = xp.unique(neighbors.reshape(shp[0] * shp[1]))
        return unique_neighbors

 
    def find_edge_neighbors_for_cells(self, cell_line: xp.ndarray) -> xp.ndarray:
        return self._find_neighbors(cell_line, connectivity=self._global_grid.face_edge_connectivity)
    def find_vertex_neighbors_for_cells(self, cell_line:xp.ndarray)->xp.ndarray:
        return self._find_neighbors(cell_line, connectivity=self._global_grid.face_node_connectivity)
    
    
    def owned_cells(self)->xp.ndarray:
        """Returns the global indices of the cells owned by this rank"""
        owned_cells = self._mapping == self._props.rank
        return xp.asarray(owned_cells).nonzero()[0]
        
    def construct_decomposition_info(self)->defs.DecompositionInfo:
        """
        Constructs the DecompositionInfo for the current rank.
        
        The DecompositionInfo object is constructed for all horizontal dimension starting from the 
        cell distribution. Edges and vertices are then handled through their connectivity to the distributed cells.
        """
         
        #: cells
        owned_cells = self.owned_cells() # global indices of owned cells
        first_halo_cells = self.next_halo_line(owned_cells)
        second_halo_cells = self.next_halo_line(first_halo_cells, owned_cells)
        
        total_halo_cells = xp.hstack((first_halo_cells, second_halo_cells))
        all_cells = xp.hstack((owned_cells, total_halo_cells))

        c_owner_mask = xp.isin(all_cells, owned_cells)
        
        decomp_info = defs.DecompositionInfo(klevels=self._num_lev).with_dimension(dims.CellDim,
                                                                          all_cells,
                                                                          c_owner_mask)
        
        #: edges
        edges_on_owned_cells = self.find_edge_neighbors_for_cells(owned_cells)
        edges_on_first_halo_line = self.find_edge_neighbors_for_cells(first_halo_cells)
        edges_on_second_halo_line = self.find_edge_neighbors_for_cells(second_halo_cells)
        
        all_edges = xp.hstack((edges_on_owned_cells, xp.setdiff1d(edges_on_first_halo_line, edges_on_owned_cells), xp.setdiff1d(edges_on_second_halo_line, edges_on_first_halo_line)))
        all_edges = xp.unique(all_edges)
        # We need to reduce the overlap:
        # `edges_on_owned_cells` and `edges_on_first_halo_line` both contain the edges on the cutting line. 
        intersect_owned_first_line = xp.intersect1d(edges_on_owned_cells, edges_on_first_halo_line)
    
        def _update_owner_mask_by_max_rank_convention(owner_mask, all_indices, indices_on_cutting_line, target_connectivity):
            """
            In order to have unique ownership of edges (and vertices) among nodes there needs to be 
            a convention as to where those elements on the cutting line go: 
            according to a remark in `mo_decomposition_tools.f90` ICON puts them to the node 
            with the higher rank.
            
            # TODO (@halungge): can we add an assert for the target dimension of the connectivity being cells. 
            Args:   
                owner_mask: owner mask for the dimension
                all_indices: (global) indices of the dimension
                indices_on_cutting_line: global indices of the elements on the cutting line
                target_connectivity: connectivity matrix mapping the dimension d to faces
            Returns:
                updated owner mask
            """
            for index in indices_on_cutting_line:
                local_index = xp.nonzero(all_indices == index)[0][0]
                owning_ranks = self._mapping[target_connectivity[index]]
                assert xp.unique(owning_ranks).size > 1, f"rank {self._props.rank}: all neighboring cells are owned by the same rank"
                assert self._props.rank in owning_ranks, f"rank {self._props.rank}: neither of the neighboring cells: {owning_ranks} is owned by me"           
                # assign the index to the rank with the higher rank
                if max(owning_ranks) > self._props.rank:
                    owner_mask[local_index] = False
                else:
                    owner_mask[local_index] = True
            return owner_mask
        

        # construct the owner mask
        edge_owner_mask = xp.isin(all_edges, edges_on_owned_cells)
        edge_owner_mask = _update_owner_mask_by_max_rank_convention(edge_owner_mask, all_edges, intersect_owned_first_line, self._global_grid.edge_face_connectivity)
        decomp_info.with_dimension(dims.EdgeDim, all_edges, edge_owner_mask)
        
        # vertices
        vertices_on_owned_cells = self.find_vertex_neighbors_for_cells(owned_cells)
        vertices_on_first_halo_line = self.find_vertex_neighbors_for_cells(first_halo_cells)
        vertices_on_second_halo_line = self.find_vertex_neighbors_for_cells(second_halo_cells) #TODO (@halungge): do we need that?
        intersect_owned_first_line = xp.intersect1d(vertices_on_owned_cells, vertices_on_first_halo_line)
        
        # create decomposition_info for vertices
        all_vertices = xp.unique(xp.hstack((vertices_on_owned_cells, vertices_on_first_halo_line)))
        v_owner_mask = xp.isin(all_vertices, vertices_on_owned_cells)
        v_owner_mask = _update_owner_mask_by_max_rank_convention(v_owner_mask, all_vertices, intersect_owned_first_line, self._node_face_connectivity)
        decomp_info.with_dimension(dims.VertexDim, all_vertices, v_owner_mask)
        return decomp_info

        