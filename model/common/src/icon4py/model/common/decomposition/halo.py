# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import enum
import logging
import uuid

import gt4py.next as gtx
import scipy as sp
import xugrid.ugrid.ugrid2d as ux

import icon4py.model.common.decomposition.definitions as defs
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base as base_grid, icon as icon_grid
from icon4py.model.common.settings import xp


log = logging.getLogger(__name__)

# TODO do we need three of those?
class DecompositionFlag(enum.IntEnum):
    #: cell is owned by this rank
    OWNED = (0,)
    #: cell is in the first halo line: that is cells that share and edge with an owned cell
    FIRST_HALO_LINE = (1,)
    #: cell is in the second halo line: that is cells that share only a vertex with an owned cell (and at least an edge with a FIRST_HALO_LINE cell)
    SECOND_HALO_LINE = 2


class HaloGenerator:
    """Creates necessary halo information for a given rank."""

    def __init__(
        self,
        rank_info: defs.ProcessProperties,
        rank_mapping: xp.ndarray,
        ugrid: ux.Ugrid2d,
        num_lev: int,
        face_face_connectivity: xp.ndarray = None,
        node_face_connectivity=None,
        node_edge_connectivity=None,
    ):
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
        self._connectivities = {
            dims.C2E2CDim: face_face_connectivity
            if face_face_connectivity is not None
            else self._global_grid.face_face_connectivity,
            dims.C2EDim: self._global_grid.face_edge_connectivity,
            dims.C2VDim: self._global_grid.face_node_connectivity,
            dims.E2CDim: self._global_grid.edge_face_connectivity,
            dims.E2VDim: self._global_grid.edge_node_connectivity,
            dims.V2CDim: node_face_connectivity
            if node_face_connectivity is not None
            else self._global_grid.node_face_connectivity,
            dims.V2EDim: node_edge_connectivity
            if node_edge_connectivity is not None
            else self._global_grid.node_edge_connectivity,
        }

    def _validate(self):
        assert self._mapping.ndim == 1
        # the decomposition should match the communicator size
        assert xp.max(self._mapping) == self._props.comm_size - 1

    def _post_init(self):
        self._validate()
        # TODO handle sparse neibhbors tables?

    def connectivity(self, dim) -> xp.ndarray:
        try:
            conn_table = self._connectivities[dim]
            if sp.sparse.issparse(conn_table):
                raise NotImplementedError("Scipy sparse for connectivity tables are not supported")
                # TODO this is not tested/working. It returns a (N x N) matrix not a (N x sparse_dim) matrix
                # _and_ we cannot make the difference between a valid "0" and a missing-value 0
                # return conn_table.toarray(order="C")
            return conn_table
        except KeyError as err:
            raise (f"Connectivity for dimension {dim} is not available") from err

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
        return next_halo_cells

    def _cell_neighbors(self, cells: xp.ndarray):
        return xp.unique(self.connectivity(dims.C2E2CDim)[cells, :])

    def _cell_neighbors_from_sparse(self, cells: xp.ndarray):
        """In xugrid face-face connectivity is a scipy spars matrix, so we reduce it to the regular sparse matrix format: (n_cells, 3)"""
        conn = self.connectivity(dims.C2E2CDim)

        neighbors = conn[cells, :]
        # There is an issue with explicit 0 (for zero based indices) since sparse.find projects them out...
        i, j, vals = sp.sparse.find(neighbors)

        return j

    def _find_neighbors(self, cell_line: xp.ndarray, connectivity: xp.ndarray) -> xp.ndarray:
        """Get a flattened list of all (unique) neighbors to a given global index list"""
        neighbors = connectivity[cell_line, :]
        shp = neighbors.shape
        unique_neighbors = xp.unique(neighbors.reshape(shp[0] * shp[1]))
        return unique_neighbors

    def find_edge_neighbors_for_cells(self, cell_line: xp.ndarray) -> xp.ndarray:
        return self._find_neighbors(cell_line, connectivity=self.connectivity(dims.C2EDim))

    def find_vertex_neighbors_for_cells(self, cell_line: xp.ndarray) -> xp.ndarray:
        return self._find_neighbors(cell_line, connectivity=self.connectivity(dims.C2VDim))

    def owned_cells(self) -> xp.ndarray:
        """Returns the global indices of the cells owned by this rank"""
        owned_cells = self._mapping == self._props.rank
        return xp.asarray(owned_cells).nonzero()[0]

    # TODO (@halungge): move out of halo generator
    def construct_decomposition_info(self) -> defs.DecompositionInfo:
        """
        Constructs the DecompositionInfo for the current rank.

        The DecompositionInfo object is constructed for all horizontal dimension starting from the
        cell distribution. Edges and vertices are then handled through their connectivity to the distributed cells.
        """

        #: cells
        owned_cells = self.owned_cells()  # global indices of owned cells
        first_halo_cells = self.next_halo_line(owned_cells)
        second_halo_cells = self.next_halo_line(first_halo_cells, owned_cells)

        total_halo_cells = xp.hstack((first_halo_cells, second_halo_cells))
        all_cells = xp.hstack((owned_cells, total_halo_cells))

        c_owner_mask = xp.isin(all_cells, owned_cells)

        decomp_info = defs.DecompositionInfo(klevels=self._num_lev).with_dimension(
            dims.CellDim, all_cells, c_owner_mask
        )

        #: edges
        edges_on_owned_cells = self.find_edge_neighbors_for_cells(owned_cells)
        edges_on_first_halo_line = self.find_edge_neighbors_for_cells(first_halo_cells)
        edges_on_second_halo_line = self.find_edge_neighbors_for_cells(second_halo_cells)

        all_edges = xp.hstack(
            (
                edges_on_owned_cells,
                xp.setdiff1d(edges_on_first_halo_line, edges_on_owned_cells),
                xp.setdiff1d(edges_on_second_halo_line, edges_on_first_halo_line),
            )
        )
        all_edges = xp.unique(all_edges)
        # We need to reduce the overlap:
        # `edges_on_owned_cells` and `edges_on_first_halo_line` both contain the edges on the cutting line.
        intersect_owned_first_line = xp.intersect1d(edges_on_owned_cells, edges_on_first_halo_line)

        def _update_owner_mask_by_max_rank_convention(
            owner_mask, all_indices, indices_on_cutting_line, target_connectivity
        ):
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
                assert (
                    xp.unique(owning_ranks).size > 1
                ), f"rank {self._props.rank}: all neighboring cells are owned by the same rank"
                assert (
                    self._props.rank in owning_ranks
                ), f"rank {self._props.rank}: neither of the neighboring cells: {owning_ranks} is owned by me"
                # assign the index to the rank with the higher rank
                if max(owning_ranks) > self._props.rank:
                    owner_mask[local_index] = False
                else:
                    owner_mask[local_index] = True
            return owner_mask

        # construct the owner mask
        edge_owner_mask = xp.isin(all_edges, edges_on_owned_cells)
        edge_owner_mask = _update_owner_mask_by_max_rank_convention(
            edge_owner_mask,
            all_edges,
            intersect_owned_first_line,
            self._global_grid.edge_face_connectivity,
        )
        decomp_info.with_dimension(dims.EdgeDim, all_edges, edge_owner_mask)

        # vertices
        vertices_on_owned_cells = self.find_vertex_neighbors_for_cells(owned_cells)
        vertices_on_first_halo_line = self.find_vertex_neighbors_for_cells(first_halo_cells)
        vertices_on_second_halo_line = self.find_vertex_neighbors_for_cells(
            second_halo_cells
        )  # TODO (@halungge): do we need that?
        intersect_owned_first_line = xp.intersect1d(
            vertices_on_owned_cells, vertices_on_first_halo_line
        )

        # create decomposition_info for vertices
        all_vertices = xp.unique(xp.hstack((vertices_on_owned_cells, vertices_on_first_halo_line)))
        v_owner_mask = xp.isin(all_vertices, vertices_on_owned_cells)
        v_owner_mask = _update_owner_mask_by_max_rank_convention(
            v_owner_mask, all_vertices, intersect_owned_first_line, self.connectivity(dims.V2CDim)
        )
        decomp_info.with_dimension(dims.VertexDim, all_vertices, v_owner_mask)
        return decomp_info

    def construct_local_connectivity(self, field_offset: gtx.FieldOffset,
                                     decom_info: defs.DecompositionInfo) -> xp.ndarray:
        """
        Construct a connectivity table for use on a given rank: it maps from source to target dimension in local indices.
        
        Args:
            field_offset: FieldOffset for which we want to construct the offset table 
            decom_info: DecompositionInfo for the current rank

        Returns: array, containt the connectivity table for the field_offset with rank-local indices

        """
        source_dim = field_offset.source
        target_dim = field_offset.target[0]
        local_dim = field_offset.target[1]
        connectivity = self.connectivity(local_dim)
        global_node_idx = decom_info.global_index(source_dim, defs.DecompositionInfo.EntryType.ALL)
        global_node_sorted = xp.argsort(global_node_idx)
        sliced_connectivity = connectivity[
            decom_info.global_index(target_dim, defs.DecompositionInfo.EntryType.ALL)
        ]
        log.debug(f"rank {self._props.rank} has local connectivity f: {sliced_connectivity}")
        for i in xp.arange(sliced_connectivity.shape[0]):
            positions = xp.searchsorted(
                global_node_idx[global_node_sorted], sliced_connectivity[i, :]
            )
            indices = global_node_sorted[positions]
            sliced_connectivity[i, :] = indices
        log.debug(f"rank {self._props.rank} has local connectivity f: {sliced_connectivity}")
        return sliced_connectivity

# should be done in grid manager!
def local_grid(props: defs.ProcessProperties, 
               decomp_info:defs.DecompositionInfo, 
               global_params:icon_grid.GlobalGridParams,
               num_lev:int, 
               limited_area:bool = False, on_gpu:bool=False, )->base_grid.BaseGrid:
    """
    Constructs a local grid for this rank based on the decomposition info.
    TODO (@halungge): for now only returning BaseGrid as we have not start/end indices implementation yet
    TODO (@halungge): make sure the INVALID_INDEX is set correctly: - when set in the original (global index) connectivity it should remain
    TODO (@halungge): how to handle the (source) indices of last halo line: their (global) neighbors are not all present on the local grid, set INVALID_INDEX (that is what xugrid does)
                                                                           check what ICON does, (they probably duplicate the valid indices...)
    Args:
        decomp_info: the decomposition info for this rank
    Returns:
        local_grid: the local grid
    """
    num_vertices = decomp_info.global_index(dims.VertexDim, defs.DecompositionInfo.EntryType.ALL).size
    num_edges = decomp_info.global_index(dims.EdgeDim, defs.DecompositionInfo.EntryType.ALL).size
    num_cells = decomp_info.global_index(dims.CellDim, defs.DecompositionInfo.EntryType.ALL).size
    grid_size = base_grid.HorizontalGridSize(
        num_vertices=num_vertices, num_edges=num_edges, num_cells=num_cells
    )
    config = base_grid.GridConfig(horizontal_config = grid_size, 
                                  vertical_size = num_lev, 
                                  on_gpu=on_gpu, 
                                  limited_area=limited_area)
    
    local_grid = icon_grid.IconGrid(uuid.uuid4()).with_config(config).with_global_params(global_params)
    # add connectivities

    return local_grid