# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
import logging
from types import ModuleType
from typing import Optional, Protocol, runtime_checkable

import gt4py.next as gtx
import gt4py.next.backend as gtx_backend
import numpy as np

from icon4py.model.common import dimension as dims, exceptions
from icon4py.model.common.decomposition import definitions as defs
from icon4py.model.common.grid import base
from icon4py.model.common.utils import data_allocation as data_alloc


log = logging.getLogger(__name__)


@runtime_checkable
class HaloConstructor(Protocol):
    """Callable that takes a mapping from faces (aka cells) to ranks"""

    def __call__(self, face_to_rank: data_alloc.NDArray) -> defs.DecompositionInfo:
        ...


class NoHalos(HaloConstructor):
    def __init__(
        self,
        horizontal_size: base.HorizontalGridSize,
        num_levels: int,
        backend: Optional[gtx_backend.Backend] = None,
    ):
        self._size = horizontal_size
        self._num_levels = num_levels
        self._backend = backend

    def __call__(self, face_to_rank: data_alloc.NDArray) -> defs.DecompositionInfo:
        xp = data_alloc.import_array_ns(self._backend)
        create_arrays = functools.partial(_create_dummy_decomposition_arrays, array_ns=xp)
        decomposition_info = defs.DecompositionInfo(klevels=self._num_levels)

        decomposition_info.with_dimension(dims.EdgeDim, *create_arrays(self._size.num_edges))
        decomposition_info.with_dimension(dims.CellDim, *create_arrays(self._size.num_cells))
        decomposition_info.with_dimension(dims.VertexDim, *create_arrays(self._size.num_vertices))
        return decomposition_info


def _create_dummy_decomposition_arrays(size: int, array_ns: ModuleType = np):
    indices = array_ns.arange(size, dtype=gtx.int32)
    owner_mask = array_ns.ones((size,), dtype=bool)
    halo_levels = array_ns.ones((size,), dtype=gtx.int32) * defs.DecompositionFlag.OWNED
    return indices, owner_mask, halo_levels


class IconLikeHaloConstructor(HaloConstructor):
    """Creates necessary halo information for a given rank."""

    def __init__(
        self,
        run_properties: defs.ProcessProperties,
        connectivities: dict[gtx.Dimension, data_alloc.NDArray],
        num_levels,  # TODO is currently needed for ghex, pass via a different struct that the decomposition info and remove
        backend: Optional[gtx_backend.Backend] = None,
    ):
        """

        Args:
            run_properties: contains information on the communicator and local compute node.
            connectivities: connectivity arrays needed to construct the halos
            backend: GT4Py (used to determine the array ns import)
        """
        self._xp = data_alloc.import_array_ns(backend)
        self._num_levels = num_levels
        self._props = run_properties
        self._connectivities = connectivities
        self._assert_all_neighbor_tables()

    @property
    def face_face_connectivity(self):
        return self._connectivity(dims.C2E2CDim)

    @property
    def edge_face_connectivity(self):
        return self._connectivity(dims.E2CDim)

    @property
    def face_edge_connectivity(self):
        return self._connectivity(dims.C2EDim)

    @property
    def node_edge_connectivity(self):
        return self._connectivity(dims.V2EDim)

    @property
    def node_face_connectivity(self):
        return self._connectivity(dims.V2CDim)

    @property
    def face_node_connectivity(self):
        return self._connectivity(dims.C2VDim)

    def _validate_mapping(self, face_to_rank_mapping: data_alloc.NDArray):
        # validate the distribution mapping:
        num_cells = self.face_face_connectivity.shape[0]
        expected_shape = (num_cells,)
        if not face_to_rank_mapping.shape == expected_shape:
            raise exceptions.ValidationError(
                "rank_mapping",
                f"should have shape {expected_shape} but is {face_to_rank_mapping.shape}",
            )

        # the decomposition should match the communicator size
        if self._xp.max(face_to_rank_mapping) > self._props.comm_size - 1:
            raise exceptions.ValidationError(
                "rank_mapping",
                f"The distribution assumes more nodes than the current run is scheduled on  {self._props} ",
            )

    def _assert_all_neighbor_tables(self):
        # make sure we have all connectivity arrays used in the halo construction
        relevant_dimension = [
            dims.C2E2CDim,
            dims.E2CDim,
            dims.C2EDim,
            dims.C2VDim,
            dims.V2CDim,
            dims.V2EDim,
        ]
        for d in relevant_dimension:
            assert (
                d in self._connectivities.keys()
            ), f"Table for {d} is missing from the neighbor table array."

    def _connectivity(self, dim: gtx.Dimension) -> data_alloc.NDArray:
        try:
            conn_table = self._connectivities[dim]
            return conn_table
            return conn_table
        except KeyError as err:
            raise exceptions.MissingConnectivity(
                f"Connectivity for offset {dim} is not available"
            ) from err

    def next_halo_line(self, cell_line: data_alloc.NDArray, depot=None):
        """Returns the global indices of the next halo line.

        Args:
            cell_line: global indices of cells we want to find the neighbors of
            depot: global indices that have already been collected
        Returns:
            next_halo_cells: global indices of the next halo line
        """
        cell_neighbors = self._cell_neighbors(cell_line)

        if depot is not None:
            cells_so_far = self._xp.hstack((depot, cell_line))
        else:
            cells_so_far = cell_line

        next_halo_cells = self._xp.setdiff1d(
            self._xp.unique(cell_neighbors), cells_so_far, assume_unique=True
        )
        return next_halo_cells

    def _cell_neighbors(self, cells: data_alloc.NDArray):
        return self._xp.unique(self.face_face_connectivity[cells, :])

    def _find_neighbors(
        self, source_indices: data_alloc.NDArray, connectivity: data_alloc.NDArray
    ) -> data_alloc.NDArray:
        """Get a flattened list of all (unique) neighbors to a given global index list"""
        neighbors = connectivity[source_indices, :]
        shp = neighbors.shape
        unique_neighbors = self._xp.unique(neighbors.reshape(shp[0] * shp[1]))
        return unique_neighbors

    def find_edge_neighbors_for_cells(self, cell_line: data_alloc.NDArray) -> data_alloc.NDArray:
        return self._find_neighbors(cell_line, connectivity=self.face_edge_connectivity)

    def find_edge_neighbors_for_vertices(
        self, vertex_line: data_alloc.NDArray
    ) -> data_alloc.NDArray:
        return self._find_neighbors(vertex_line, connectivity=self.node_edge_connectivity)

    def find_vertex_neighbors_for_cells(self, cell_line: data_alloc.NDArray) -> data_alloc.NDArray:
        return self._find_neighbors(cell_line, connectivity=self.face_node_connectivity)

    def owned_cells(self, face_to_rank: data_alloc.NDArray) -> data_alloc.NDArray:
        """Returns the global indices of the cells owned by this rank"""
        owned_cells = face_to_rank == self._props.rank
        return self._xp.asarray(owned_cells).nonzero()[0]

    def _update_owner_mask_by_max_rank_convention(
        self, face_to_rank, owner_mask, all_indices, indices_on_cutting_line, target_connectivity
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
            local_index = self._xp.nonzero(all_indices == index)[0][0]
            owning_ranks = face_to_rank[target_connectivity[index]]
            assert (
                self._xp.unique(owning_ranks).size > 1
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

    # TODO (@halungge): move out of halo generator?
    def __call__(self, face_to_rank: data_alloc.NDArray) -> defs.DecompositionInfo:
        """
        Constructs the DecompositionInfo for the current rank.

        The DecompositionInfo object is constructed for all horizontal dimension starting from the
        cell distribution. Edges and vertices are then handled through their connectivity to the distributed cells.
        """
        self._validate_mapping(face_to_rank)
        #: cells
        owned_cells = self.owned_cells(face_to_rank)  # global indices of owned cells
        first_halo_cells = self.next_halo_line(owned_cells)
        second_halo_cells = self.next_halo_line(first_halo_cells, owned_cells)

        total_halo_cells = self._xp.hstack((first_halo_cells, second_halo_cells))
        all_cells = self._xp.hstack((owned_cells, total_halo_cells))

        cell_owner_mask = self._xp.isin(all_cells, owned_cells)
        cell_halo_levels = defs.DecompositionFlag.UNDEFINED * self._xp.ones(
            all_cells.size, dtype=int
        )
        cell_halo_levels[cell_owner_mask] = defs.DecompositionFlag.OWNED
        cell_halo_levels[
            self._xp.isin(all_cells, first_halo_cells)
        ] = defs.DecompositionFlag.FIRST_HALO_LINE
        cell_halo_levels[
            self._xp.isin(all_cells, second_halo_cells)
        ] = defs.DecompositionFlag.SECOND_HALO_LINE
        decomp_info = defs.DecompositionInfo(klevels=self._num_levels).with_dimension(
            dims.CellDim, all_cells, cell_owner_mask, cell_halo_levels
        )

        #: vertices
        vertex_on_owned_cells = self.find_vertex_neighbors_for_cells(owned_cells)
        vertex_on_first_halo_line = self.find_vertex_neighbors_for_cells(first_halo_cells)
        vertex_on_second_halo_line = self.find_vertex_neighbors_for_cells(
            second_halo_cells
        )  # TODO (@halungge): do we need that at all?

        vertex_on_cutting_line = self._xp.intersect1d(
            vertex_on_owned_cells, vertex_on_first_halo_line
        )

        # create decomposition_info for vertices
        all_vertices = self._xp.unique(
            self._xp.hstack((vertex_on_owned_cells, vertex_on_first_halo_line))
        )
        vertex_owner_mask = self._xp.isin(all_vertices, vertex_on_owned_cells)
        vertex_owner_mask = self._update_owner_mask_by_max_rank_convention(
            face_to_rank,
            vertex_owner_mask,
            all_vertices,
            vertex_on_cutting_line,
            self.node_face_connectivity,
        )
        vertex_second_level = self._xp.setdiff1d(vertex_on_first_halo_line, vertex_on_owned_cells)
        vertex_halo_levels = defs.DecompositionFlag.UNDEFINED * self._xp.ones(
            all_vertices.size, dtype=int
        )
        vertex_halo_levels[vertex_owner_mask] = defs.DecompositionFlag.OWNED
        vertex_halo_levels[
            self._xp.logical_and(
                self._xp.logical_not(vertex_owner_mask),
                self._xp.isin(all_vertices, vertex_on_cutting_line),
            )
        ] = defs.DecompositionFlag.FIRST_HALO_LINE
        vertex_halo_levels[
            self._xp.isin(all_vertices, vertex_second_level)
        ] = defs.DecompositionFlag.SECOND_HALO_LINE
        decomp_info.with_dimension(
            dims.VertexDim, all_vertices, vertex_owner_mask, vertex_halo_levels
        )

        # edges
        edges_on_owned_cells = self.find_edge_neighbors_for_cells(owned_cells)
        edges_on_first_halo_line = self.find_edge_neighbors_for_cells(first_halo_cells)
        edges_on_second_halo_line = self.find_edge_neighbors_for_cells(second_halo_cells)

        level_two_edges = self._xp.setdiff1d(
            self.find_edge_neighbors_for_vertices(vertex_on_cutting_line), edges_on_owned_cells
        )

        # level_two_edges = xp.setdiff1d(edges_on_first_halo_line, edges_on_owned_cells)
        all_edges = self._xp.hstack(
            (
                edges_on_owned_cells,
                level_two_edges,
                self._xp.setdiff1d(edges_on_second_halo_line, edges_on_first_halo_line),
            )
        )
        all_edges = self._xp.unique(all_edges)
        # We need to reduce the overlap:
        # `edges_on_owned_cells` and `edges_on_first_halo_line` both contain the edges on the cutting line.
        edge_intersect_owned_first_line = self._xp.intersect1d(
            edges_on_owned_cells, edges_on_first_halo_line
        )

        # construct the owner mask
        edge_owner_mask = self._xp.isin(all_edges, edges_on_owned_cells)
        edge_owner_mask = self._update_owner_mask_by_max_rank_convention(
            face_to_rank,
            edge_owner_mask,
            all_edges,
            edge_intersect_owned_first_line,
            self.edge_face_connectivity,
        )
        edge_halo_levels = defs.DecompositionFlag.UNDEFINED * self._xp.ones(
            all_edges.shape, dtype=int
        )
        edge_halo_levels[edge_owner_mask] = defs.DecompositionFlag.OWNED
        edge_halo_levels[
            self._xp.logical_and(
                self._xp.logical_not(edge_owner_mask),
                self._xp.isin(all_edges, edge_intersect_owned_first_line),
            )
        ] = defs.DecompositionFlag.FIRST_HALO_LINE
        edge_halo_levels[
            self._xp.isin(all_edges, level_two_edges)
        ] = defs.DecompositionFlag.SECOND_HALO_LINE
        decomp_info.with_dimension(dims.EdgeDim, all_edges, edge_owner_mask, edge_halo_levels)

        return decomp_info


# TODO (@halungge): refine type hints: adjacency_matrix should be a connectivity matrix of C2E2C and
#  the return value an array of shape (n_cells,)


@runtime_checkable
class Decomposer(Protocol):
    def __call__(self, adjacency_matrix: data_alloc.NDArray, n_part: int) -> data_alloc.NDArray:
        """
        Call the decomposition.

        Args:
            adjacency_matrix: face-to-face connectivity matrix on the global (undecomposed) grid. In the Icon4py context this C2E2C
            n_part: number of nodes
        """
        ...


class SimpleMetisDecomposer(Decomposer):
    """
    A simple decomposer using METIS for partitioning a grid topology.

    We use the simple pythonic interface to pymetis: just passing the adjacency matrix
    if more control is needed (for example by using weights we need to switch to the C like interface)
    https://documen.tician.de/pymetis/functionality.html
    """

    def __call__(self, adjacency_matrix: data_alloc.NDArray, n_part: int) -> data_alloc.NDArray:
        """
        Generate partition labesl for this grid topology using METIS:
        https://github.com/KarypisLab/METIS

        This method utilizes the pymetis Python bindings:
        https://github.com/inducer/pymetis

        Args:
            n_part: int, number of partitions to create
        Returns: np.ndarray: array with partition label (int, rank number) for each cell
        """

        import pymetis

        cut_count, partition_index = pymetis.part_graph(nparts=n_part, adjacency=adjacency_matrix)
        return np.array(partition_index)


class SingleNodeDecomposer(Decomposer):
    def __call__(self, adjacency_matrix: data_alloc.NDArray, n_part: int) -> data_alloc.NDArray:
        """Dummy decomposer for single node: assigns all cells to rank = 0"""
        return np.zeros(adjacency_matrix.shape[0], dtype=gtx.int32)
