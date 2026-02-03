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
from typing import Protocol, runtime_checkable

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import numpy as np

from icon4py.model.common import dimension as dims, exceptions
from icon4py.model.common.decomposition import definitions as defs
from icon4py.model.common.grid import base, gridfile
from icon4py.model.common.utils import data_allocation as data_alloc


log = logging.getLogger(__name__)


def _value(k: gtx.FieldOffset | str) -> str:
    return str(k.value) if isinstance(k, gtx.FieldOffset) else k


@runtime_checkable
class HaloConstructor(Protocol):
    """Callable that takes a mapping from faces (aka cells) to ranks"""

    def __call__(self, face_to_rank: data_alloc.NDArray) -> defs.DecompositionInfo: ...


class NoHalos(HaloConstructor):
    def __init__(
        self,
        horizontal_size: base.HorizontalGridSize,
        allocator: gtx_typing.FieldBufferAllocationUtil | None = None,
    ):
        self._size = horizontal_size
        self._allocator = allocator

    def __call__(self, face_to_rank: data_alloc.NDArray) -> defs.DecompositionInfo:
        xp = data_alloc.import_array_ns(self._allocator)
        create_arrays = functools.partial(_create_dummy_decomposition_arrays, array_ns=xp)
        decomposition_info = defs.DecompositionInfo()

        decomposition_info.set_dimension(dims.EdgeDim, *create_arrays(self._size.num_edges))
        decomposition_info.set_dimension(dims.CellDim, *create_arrays(self._size.num_cells))
        decomposition_info.set_dimension(dims.VertexDim, *create_arrays(self._size.num_vertices))
        return decomposition_info


def _create_dummy_decomposition_arrays(
    size: int, array_ns: ModuleType = np
) -> tuple[data_alloc.NDArray, data_alloc.NDArray, data_alloc.NDArray]:
    indices = array_ns.arange(size, dtype=gtx.int32)  # type: ignore  [attr-defined]
    owner_mask = array_ns.full((size,), True, dtype=bool)
    halo_levels = array_ns.full((size,), defs.DecompositionFlag.OWNED.value, dtype=gtx.int32)  # type: ignore  [attr-defined]
    return indices, owner_mask, halo_levels


class IconLikeHaloConstructor(HaloConstructor):
    """Creates necessary halo information for a given rank."""

    def __init__(
        self,
        run_properties: defs.ProcessProperties,
        connectivities: dict[gtx.FieldOffset | str, data_alloc.NDArray],
        allocator: gtx_typing.FieldBufferAllocationUtil | None = None,
    ):
        """

        Args:
            run_properties: contains information on the communicator and local compute node.
            connectivities: connectivity arrays needed to construct the halos
            allocator: GT4Py buffer allocator
        """
        self._xp = data_alloc.import_array_ns(allocator)
        self._props = run_properties
        self._connectivities = {_value(k): v for k, v in connectivities.items()}
        self._assert_all_neighbor_tables()

    @property
    def face_face_connectivity(self) -> data_alloc.NDArray:
        return self._connectivity(dims.C2E2C)

    @property
    def edge_face_connectivity(self) -> data_alloc.NDArray:
        return self._connectivity(dims.E2C)

    @property
    def face_edge_connectivity(self) -> data_alloc.NDArray:
        return self._connectivity(dims.C2E)

    @property
    def node_edge_connectivity(self) -> data_alloc.NDArray:
        return self._connectivity(dims.V2E)

    @property
    def node_face_connectivity(self) -> data_alloc.NDArray:
        return self._connectivity(dims.V2C)

    @property
    def face_node_connectivity(self) -> data_alloc.NDArray:
        return self._connectivity(dims.C2V)

    def _validate_mapping(self, face_to_rank_mapping: data_alloc.NDArray) -> None:
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

    def _assert_all_neighbor_tables(self) -> None:
        # make sure we have all connectivity arrays used in the halo construction
        relevant_dimension = [
            dims.C2E2C,
            dims.E2C,
            dims.C2E,
            dims.C2V,
            dims.V2C,
            dims.V2E,
        ]
        for d in relevant_dimension:
            assert (
                d.value in self._connectivities
            ), f"Table for {d} is missing from the neighbor table array."

    def _connectivity(self, offset: gtx.FieldOffset | str) -> data_alloc.NDArray:
        try:
            conn_table = self._connectivities.get(_value(offset))
            return conn_table
        except KeyError as err:
            raise exceptions.MissingConnectivityError(
                f"Connectivity for offset {offset} is not available"
            ) from err

    def next_halo_line(
        self, cells: data_alloc.NDArray, depot: data_alloc.NDArray | None = None
    ) -> data_alloc.NDArray:
        """Returns the full-grid indices of the next halo line.

        If a depot is given the function only return indices that are not in the depot

        Args:
            cells: 1d array, full-grid indices of cells we want to find the neighbors of
            depot: full-grid indices that have already been collected
        Returns:
            next_halo_cells: full-grid indices of the next halo line
        """
        assert cells.ndim == 1, "input should be 1d array"
        cell_neighbors = self._find_cell_neighbors(cells)

        cells_so_far = self._xp.hstack((depot, cells)) if depot is not None else cells

        next_halo_cells = self._xp.setdiff1d(
            self._xp.unique(cell_neighbors), cells_so_far, assume_unique=True
        )
        return next_halo_cells

    def _find_neighbors(
        self, source_indices: data_alloc.NDArray, connectivity: data_alloc.NDArray
    ) -> data_alloc.NDArray:
        """Get a flattened list of all (unique) neighbors to a given global index list"""
        neighbors = connectivity[source_indices, :]
        shp = neighbors.shape
        unique_neighbors = self._xp.unique(neighbors.reshape(shp[0] * shp[1]))
        return unique_neighbors

    def _find_cell_neighbors(self, cells: data_alloc.NDArray) -> data_alloc.NDArray:
        """Find all neighboring cells of a list of cells."""
        return self._find_neighbors(cells, connectivity=self.face_face_connectivity)

    def find_edge_neighbors_for_cells(self, cell_line: data_alloc.NDArray) -> data_alloc.NDArray:
        return self._find_neighbors(cell_line, connectivity=self.face_edge_connectivity)

    def find_edge_neighbors_for_vertices(
        self, vertex_line: data_alloc.NDArray
    ) -> data_alloc.NDArray:
        return self._find_neighbors(vertex_line, connectivity=self.node_edge_connectivity)

    def find_vertex_neighbors_for_cells(self, cell_line: data_alloc.NDArray) -> data_alloc.NDArray:
        return self._find_neighbors(cell_line, connectivity=self.face_node_connectivity)

    def find_cell_neighbors_for_vertices(
        self, vertex_line: data_alloc.NDArray
    ) -> data_alloc.NDArray:
        return self._find_neighbors(vertex_line, connectivity=self.node_face_connectivity)

    def owned_cells(self, face_to_rank: data_alloc.NDArray) -> data_alloc.NDArray:
        """Returns the full-grid indices of the cells owned by this rank"""
        owned_cells = face_to_rank == self._props.rank
        return self._xp.asarray(owned_cells).nonzero()[0]

    def _update_owner_mask_by_max_rank_convention(
        self,
        face_to_rank: data_alloc.NDArray,
        owner_mask: data_alloc.NDArray,
        all_indices: data_alloc.NDArray,
        indices_on_cutting_line: data_alloc.NDArray,
        target_connectivity: data_alloc.NDArray,
    ) -> data_alloc.NDArray:
        """
        In order to have unique ownership of edges (and vertices) among nodes there needs to be
        a convention as to where those elements on the cutting line go:
        according to a remark in `mo_decomposition_tools.f90` ICON puts them to the node
        with the higher rank.

        # TODO(halungge): can we add an assert for the target dimension of the connectivity being cells?
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

    def __call__(self, face_to_rank: data_alloc.NDArray) -> defs.DecompositionInfo:
        """
             Constructs the DecompositionInfo for the current rank.

             Args:
                 face_to_rank: a mapping of cells to a rank

             The DecompositionInfo object is constructed for all horizontal dimension starting from the
             cell distribution.
             Edges and vertices are then handled through their connectivity to the distributed cells.

             This constructs a halo similar to ICON which consists of **exactly** 2 cell-halo lines

             |         /|         /|         /|
             |        / |        / |        / |
             |      /   |      /   |  cy  /   |
             |    /     |    / cx  |    /     |
             |  /       |  /       |  /   cz  |          rank 0
             |/         |/         |/         |
        -----v0---e0----v1---e1----v2---e2----v3---e3--  cutting line
             |         /|         /|         /|
             |  c0    / |   c1   / |   c2   / |
             |      /   |      /   |      /   |          rank 1
             e4   e5    e6   e7    e8    e9   e10
             |   /      |   /      |   /      |  /
             | /   c3   | /   c4   | /   c5   | /
             |/         |/         |/         |/
            -v4---e11---v5---e12---v6----e13--v7---------
             |         /|         /|         /|
             |        / |        / |        / |
             |      /   |      /   |      /   |
             |    /     |    /     |    /     |
             |  /       |  /       |  /       |  /
             |/         |/         |/         |/


             Cells:
             The "numbered" cells and edges are relevant for the halo construction from the point of view of rank 0
             Cells (c0, c1, c2) are the 1. HALO LEVEL: these are cells that are neighbors of an owned cell
             Cells (c3, c4. c5) are the 2. HALO LEVEL: cells that "close" the hexagon of a vertex on the cutting line and do not share an edge with an owned cell (that is are not LEVEL 1 cells)
                                                       this is _not_ the same as the neighboring cells of LEVEL 1 cells, and the definition might be different from ICON.
                                                       In the above picture if the cut was along a corner (e0 -> e1-> e8) then cy is part of the 2. LEVEL because it is part of the hexagon on v2, but it is not
                                                       in c2e2c(c2e2c(c4))

             Note that this definition of 1. and 2. line differs from the definition of boundary line counting used in [grid refinement](grid_refinement.py), in terms
             of "distance" to the cutting line all halo cells have a distance of 1.


             Vertices:
             - 1. HALO LEVEL: are vertices on the cutting line that are not owned, or in a different wording: all vertices of owned cells that are not
             owned.
             In ICON every element in an array needs **exactly one owner** (otherwise there would be duplicates and double-counting).
             For elements on the cutting line (vertices and edges) there is no clear indication which rank should own it,
             ICON uses the rank with the higher rank value (see (_update_owner_mask_by_max_rank_convention))
             In the example above (v0, v1, v2, v3) are in the 1. HALO LEVEL of rank 0 and owned by rank 1.
             As a consequence, there are ranks that have no 1. HALO LEVEL vertices.
             - 2. HALO LEVEL: are vertices that are on HALO LEVEL cells, but not on owned. For rank 0 these are (v4, v5, v6, v7)


             Edges:
             For edges a similar pattern is used as for the vertices.
             - 1. HALO LEVEL: edges that are on owned cells but not owned themselves (these are edges that share 2 vertices with a owned cell).
             In terms of ownership the same convention is applied as for the vertices: (e0, e1, e2, e3) are in the HALO LEVEL 1 of rank 0, and are owned by rank 1
             - 2. HALO LEVEL: edges that share exactly one vertex with an owned cell. The definition via vertices is important: TODO (halungge): EXAMPLE???
             For rank 0 these are the edges (e4, e5, e6, e7, e8, e9, e10) in the example above.
             - 3. HALO LEVEL:
             In **ICON4Py ONLY**, edges that "close" the halo cells and share exactly 2 vertices with a HALO LEVEL 2 cell, but none with
             an owned cell. These edges are **not** included in the halo in ICON. These are (e11, e12, e13) for rank 0 in the example above.
             This is the HALO LINE which makes the C2E connectivity complete (= without skip value) for a distributed setup.
        """

        self._validate_mapping(face_to_rank)

        #: cells
        owned_cells = self.owned_cells(face_to_rank)  # global indices of owned cells
        first_halo_cells = self.next_halo_line(owned_cells)
        #: vertices
        vertex_on_owned_cells = self.find_vertex_neighbors_for_cells(owned_cells)
        vertex_on_halo_cells = self.find_vertex_neighbors_for_cells(
            self._xp.hstack(
                (first_halo_cells, (self.next_halo_line(first_halo_cells, owned_cells)))
            )
        )
        vertex_on_cutting_line = self._xp.intersect1d(vertex_on_owned_cells, vertex_on_halo_cells)
        vertex_second_halo = self._xp.setdiff1d(vertex_on_halo_cells, vertex_on_cutting_line)
        all_vertices = self._xp.hstack((vertex_on_owned_cells, vertex_second_halo))

        #: update cells to include all cells of the "dual cell" (hexagon) for nodes on the cutting line
        dual_cells = self.find_cell_neighbors_for_vertices(vertex_on_cutting_line)
        total_halo_cells = self._xp.setdiff1d(dual_cells, owned_cells)
        second_halo_cells = self._xp.setdiff1d(total_halo_cells, first_halo_cells)
        all_cells = self._xp.hstack((owned_cells, first_halo_cells, second_halo_cells))

        #: edges
        edges_on_owned_cells = self.find_edge_neighbors_for_cells(owned_cells)
        edges_on_any_halo_line = self.find_edge_neighbors_for_cells(total_halo_cells)

        edges_on_cutting_line = self._xp.intersect1d(edges_on_owned_cells, edges_on_any_halo_line)

        # needs to be defined as vertex neighbor due to "corners" in the cut.
        edge_second_level = self._xp.setdiff1d(
            self.find_edge_neighbors_for_vertices(vertex_on_cutting_line), edges_on_owned_cells
        )
        edge_third_level = self._xp.setdiff1d(edges_on_any_halo_line, edge_second_level)
        edge_third_level = self._xp.setdiff1d(edge_third_level, edges_on_cutting_line)

        all_edges = self._xp.hstack((edges_on_owned_cells, edge_second_level, edge_third_level))
        #: construct decomposition info
        decomp_info = defs.DecompositionInfo()
        cell_owner_mask = self._xp.isin(all_cells, owned_cells)
        cell_halo_levels = self._xp.full(
            all_cells.size,
            defs.DecompositionFlag.UNDEFINED.value,
            dtype=gtx.int32,  # type: ignore  [attr-defined]
        )
        cell_halo_levels[cell_owner_mask] = defs.DecompositionFlag.OWNED
        cell_halo_levels[self._xp.isin(all_cells, first_halo_cells)] = (
            defs.DecompositionFlag.FIRST_HALO_LEVEL
        )
        cell_halo_levels[self._xp.isin(all_cells, second_halo_cells)] = (
            defs.DecompositionFlag.SECOND_HALO_LEVEL
        )
        decomp_info.set_dimension(dims.CellDim, all_cells, cell_owner_mask, cell_halo_levels)
        vertex_owner_mask = self._xp.isin(all_vertices, vertex_on_owned_cells)
        vertex_owner_mask = self._update_owner_mask_by_max_rank_convention(
            face_to_rank,
            vertex_owner_mask,
            all_vertices,
            vertex_on_cutting_line,
            self.node_face_connectivity,
        )
        vertex_second_level = self._xp.setdiff1d(vertex_on_halo_cells, vertex_on_owned_cells)
        vertex_halo_levels = self._xp.full(
            all_vertices.size,
            defs.DecompositionFlag.UNDEFINED.value,
            dtype=gtx.int32,  # type: ignore  [attr-defined]
        )
        vertex_halo_levels[vertex_owner_mask] = defs.DecompositionFlag.OWNED
        vertex_halo_levels[
            self._xp.logical_not(vertex_owner_mask)
            & self._xp.isin(all_vertices, vertex_on_cutting_line)
        ] = defs.DecompositionFlag.FIRST_HALO_LEVEL

        vertex_halo_levels[self._xp.isin(all_vertices, vertex_second_level)] = (
            defs.DecompositionFlag.SECOND_HALO_LEVEL
        )
        decomp_info.set_dimension(
            dims.VertexDim, all_vertices, vertex_owner_mask, vertex_halo_levels
        )

        edge_owner_mask = self._xp.isin(all_edges, edges_on_owned_cells)
        edge_owner_mask = self._update_owner_mask_by_max_rank_convention(
            face_to_rank,
            edge_owner_mask,
            all_edges,
            edges_on_cutting_line,
            self.edge_face_connectivity,
        )

        edge_halo_levels = self._xp.full(
            all_edges.shape,
            defs.DecompositionFlag.UNDEFINED.value,
            dtype=gtx.int32,  # type: ignore  [attr-defined]
        )
        edge_halo_levels[edge_owner_mask] = defs.DecompositionFlag.OWNED
        # LEVEL_ONE edges are on an owned cell but are not owned: these are all edges on the cutting line that are not owned (by the convention)

        edge_halo_levels[
            self._xp.logical_not(edge_owner_mask) & self._xp.isin(all_edges, edges_on_cutting_line)
        ] = defs.DecompositionFlag.FIRST_HALO_LEVEL

        # LEVEL_TWO edges share exactly one vertex with an owned cell, they are on the first halo-line cells, but not on the cutting line
        edge_halo_levels[self._xp.isin(all_edges, edge_second_level)] = (
            defs.DecompositionFlag.SECOND_HALO_LEVEL
        )

        # LEVEL_THREE edges
        # LEVEL_TWO edges share exactly one vertex with an owned cell, they are on the first halo-line cells, but not on the cutting line
        edge_halo_levels[self._xp.isin(all_edges, edge_third_level)] = (
            defs.DecompositionFlag.THIRD_HALO_LEVEL
        )
        decomp_info.set_dimension(dims.EdgeDim, all_edges, edge_owner_mask, edge_halo_levels)
        return decomp_info


@runtime_checkable
class Decomposer(Protocol):
    def __call__(
        self, adjacency_matrix: data_alloc.NDArray, num_partitions: int
    ) -> data_alloc.NDArray:
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

    We use the simple pythonic interface to pymetis: just passing the adjacency matrix, which for ICON is
    the full grid C2E2C neigbhor table.
    if more control is needed (for example by using weights we need to switch to the C like interface)
    https://documen.tician.de/pymetis/functionality.html
    """

    def __call__(
        self, adjacency_matrix: data_alloc.NDArray, num_partitions: int
    ) -> data_alloc.NDArray:
        """
        Generate partition labels for this grid topology using METIS:
        https://github.com/KarypisLab/METIS

        This method utilizes the pymetis Python bindings:
        https://github.com/inducer/pymetis

        Args:
            n_part: int, number of partitions to create
            adjacency_matrix: nd array: neighbor table describing of the main dimension object to be distributed: for example cell -> cell neighbors
        Returns: data_alloc.NDArray: array with partition label (int, rank number) for each cell
        """

        import pymetis  # type: ignore [import-untyped]

        _, partition_index = pymetis.part_graph(nparts=num_partitions, adjacency=adjacency_matrix)
        return data_alloc.array_ns_from_array(adjacency_matrix).array(partition_index)


class SingleNodeDecomposer(Decomposer):
    def __call__(
        self, adjacency_matrix: data_alloc.NDArray, num_partitions: int = 1
    ) -> data_alloc.NDArray:
        """Dummy decomposer for single node: assigns all cells to rank = 0"""
        return data_alloc.array_ns_from_array(adjacency_matrix).zeros(
            adjacency_matrix.shape[0],
            dtype=gtx.int32,  # type: ignore  [attr-defined]
        )


def get_halo_constructor(
    run_properties: defs.ProcessProperties,
    full_grid_size: base.HorizontalGridSize,
    connectivities: dict[gtx.FieldOffset | str, data_alloc.NDArray],
    allocator: gtx_typing.FieldBufferAllocationUtil | None,
) -> HaloConstructor:
    """
    Factory method to create the halo constructor.

    Currently there is only one halo type (except for single node dummy).
    If in the future we want to experiment with different halo types we should add an extra selection
    parameter
    Args:
        processor_props:
        full_grid_size
        allocator:
        connectivities:

    Returns: a HaloConstructor suitable for the run_properties

    """
    if run_properties.is_single_rank():
        return NoHalos(
            horizontal_size=full_grid_size,
            allocator=allocator,
        )
    else:
        return IconLikeHaloConstructor(
            run_properties=run_properties,
            connectivities=connectivities,
            allocator=allocator,
        )


def global_to_local(
    global_indices: data_alloc.NDArray,
    indices_to_translate: data_alloc.NDArray,
    array_ns: ModuleType = np,
) -> data_alloc.NDArray:
    """Translate an array of global indices into rank-local ones.

    Args:
        global_indices: global indices owned on the rank: this is the implicit mapping encoding the local to global
        indices_to_translate: the array to map to local indices

    """
    sorter = array_ns.argsort(global_indices)

    mask = array_ns.isin(indices_to_translate, global_indices)
    positions = array_ns.searchsorted(global_indices, indices_to_translate, sorter=sorter)
    local_neighbors = array_ns.full_like(indices_to_translate, gridfile.GridFile.INVALID_INDEX)
    local_neighbors[mask] = sorter[positions[mask]]
    return local_neighbors
