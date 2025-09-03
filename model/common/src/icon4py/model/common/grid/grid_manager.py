# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import logging
import pathlib
from types import ModuleType
from typing import Literal, TypeAlias

import gt4py.next as gtx
import gt4py.next.backend as gtx_backend
import numpy as np

from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.decomposition import definitions as decomposition, halo
from icon4py.model.common.decomposition.halo import SingleNodeDecomposer
from icon4py.model.common.exceptions import InvalidConfigError
from icon4py.model.common.grid import (
    base,
    gridfile,
    horizontal as h_grid,
    icon,
    refinement,
    vertical as v_grid,
)
from icon4py.model.common.utils import data_allocation as data_alloc


_log = logging.getLogger(__name__)
_single_node_decomposer = halo.SingleNodeDecomposer()
_single_process_props = decomposition.SingleNodeProcessProperties()
_fortan_to_python_transformer = gridfile.ToZeroBasedIndexTransformation()


class IconGridError(RuntimeError):
    pass


CoordinateDict: TypeAlias = dict[gtx.Dimension, dict[Literal["lat", "lon"], gtx.Field]]
GeometryDict: TypeAlias = dict[gridfile.GeometryName, gtx.Field]


class GridManager:
    """
    Read ICON grid file and set up grid topology, refinement information and geometry fields.

    It handles the reading of the ICON grid file and extracts information such as:
    - topology (connectivity arrays)
    - refinement information: association of field positions to specific zones in the horizontal grid like boundaries, inner prognostic cells, etc.
    - geometry fields present in the grid file


    """

    def __init__(
        self,
        grid_file: pathlib.Path | str,
        config: v_grid.VerticalGridConfig,
        transformation: gridfile.IndexTransformation = _fortan_to_python_transformer,
    ):
        self._transformation = transformation
        self._file_name = str(grid_file)
        self._vertical_config = config
        self._halo_constructor: halo.HaloConstructor | None = None
        # Output
        self._grid: icon.IconGrid | None = None
        self._decomposition_info: decomposition.DecompositionInfo | None = None
        self._geometry: GeometryDict = {}
        self._coordinates: CoordinateDict = {}
        self._reader = None

    def open(self):
        """Open the gridfile resource for reading."""
        self._reader = gridfile.GridFile(self._file_name, self._transformation)
        self._reader.open()

    def close(self):
        """close the gridfile resource."""
        self._reader.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        if exc_type is not None:
            _log.debug(
                f"Exception '{exc_type}: {exc_val}' while reading the grid file {self._file_name}"
            )
        if exc_type is FileNotFoundError:
            raise FileNotFoundError(f"gridfile {self._file_name} not found, aborting")

    def __call__(
        self,
        backend: gtx_backend.Backend | None,
        keep_skip_values: bool,
        decomposer: halo.Decomposer = _single_node_decomposer,
        run_properties=_single_process_props,
    ):
        if not run_properties.single_node() and isinstance(decomposer, SingleNodeDecomposer):
            raise InvalidConfigError("Need a Decomposer for multi node run")

        if not self._reader:
            self.open()

        self._construct_decomposed_grid(
            backend=backend,
            with_skip_values=keep_skip_values,
            decomposer=decomposer,
            run_properties=run_properties,
        )
        self._coordinates = self._read_coordinates(backend)
        self._geometry = self._read_geometry_fields(backend)

        self.close()

    def _read_coordinates(self, backend: gtx_backend.Backend | None) -> CoordinateDict:
        my_cell_indices = self._decomposition_info.global_index(dims.CellDim)
        my_edge_indices = self._decomposition_info.global_index(dims.EdgeDim)
        my_vertex_indices = self._decomposition_info.global_index(dims.VertexDim)
        return {
            dims.CellDim: {
                "lat": gtx.as_field(
                    (dims.CellDim,),
                    self._reader.variable(
                        gridfile.CoordinateName.CELL_LATITUDE, indices=my_cell_indices
                    ),
                    dtype=ta.wpfloat,
                    allocator=backend,
                ),
                "lon": gtx.as_field(
                    (dims.CellDim,),
                    self._reader.variable(
                        gridfile.CoordinateName.CELL_LONGITUDE, indices=my_cell_indices
                    ),
                    dtype=ta.wpfloat,
                    allocator=backend,
                ),
            },
            dims.EdgeDim: {
                "lat": gtx.as_field(
                    (dims.EdgeDim,),
                    self._reader.variable(
                        gridfile.CoordinateName.EDGE_LATITUDE, indices=my_edge_indices
                    ),
                    dtype=ta.wpfloat,
                    allocator=backend,
                ),
                "lon": gtx.as_field(
                    (dims.EdgeDim,),
                    self._reader.variable(
                        gridfile.CoordinateName.EDGE_LONGITUDE, indices=my_edge_indices
                    ),
                    dtype=ta.wpfloat,
                    allocator=backend,
                ),
            },
            dims.VertexDim: {
                "lat": gtx.as_field(
                    (dims.VertexDim,),
                    self._reader.variable(
                        gridfile.CoordinateName.VERTEX_LATITUDE, indices=my_vertex_indices
                    ),
                    allocator=backend,
                    dtype=ta.wpfloat,
                ),
                "lon": gtx.as_field(
                    (dims.VertexDim,),
                    self._reader.variable(
                        gridfile.CoordinateName.VERTEX_LONGITUDE, indices=my_vertex_indices
                    ),
                    allocator=backend,
                    dtype=ta.wpfloat,
                ),
            },
        }

    def _read_geometry_fields(self, backend: gtx_backend.Backend | None):
        my_cell_indices = self._decomposition_info.global_index(dims.CellDim)
        my_edge_indices = self._decomposition_info.global_index(dims.EdgeDim)
        my_vertex_indices = self._decomposition_info.global_index(dims.VertexDim)
        return {
            # TODO(halungge): still needs to ported, values from "our" grid files contains (wrong) values:
            #   based on bug in generator fixed with this [PR40](https://gitlab.dkrz.de/dwd-sw/dwd_icon_tools/-/merge_requests/40) .
            gridfile.GeometryName.CELL_AREA.value: gtx.as_field(
                (dims.CellDim,),
                self._reader.variable(gridfile.GeometryName.CELL_AREA, indices=my_cell_indices),
                allocator=backend,
            ),
            # TODO(halungge): easily computed from a neighbor_sum V2C over the cell areas?
            gridfile.GeometryName.DUAL_AREA.value: gtx.as_field(
                (dims.VertexDim,),
                self._reader.variable(gridfile.GeometryName.DUAL_AREA),
                allocator=backend,
            ),
            gridfile.GeometryName.EDGE_CELL_DISTANCE.value: gtx.as_field(
                (dims.EdgeDim, dims.E2CDim),
                self._reader.variable(
                    gridfile.GeometryName.EDGE_CELL_DISTANCE,
                    transpose=True,
                    indices=my_edge_indices,
                ),
                allocator=backend,
            ),
            gridfile.GeometryName.EDGE_VERTEX_DISTANCE.value: gtx.as_field(
                (dims.EdgeDim, dims.E2VDim),
                self._reader.variable(
                    gridfile.GeometryName.EDGE_VERTEX_DISTANCE,
                    transpose=True,
                    indices=my_edge_indices,
                ),
                allocator=backend,
            ),
            # TODO(halungge): recompute from coordinates? field in gridfile contains NaN on boundary edges
            gridfile.GeometryName.TANGENT_ORIENTATION.value: gtx.as_field(
                (dims.EdgeDim,),
                self._reader.variable(
                    gridfile.GeometryName.TANGENT_ORIENTATION, indices=my_edge_indices
                ),
                allocator=backend,
            ),
            gridfile.GeometryName.CELL_NORMAL_ORIENTATION.value: gtx.as_field(
                (dims.CellDim, dims.C2EDim),
                self._reader.int_variable(
                    gridfile.GeometryName.CELL_NORMAL_ORIENTATION,
                    transpose=True,
                    apply_transformation=False,
                    indices=my_cell_indices,
                ),
                allocator=backend,
            ),
            gridfile.GeometryName.EDGE_ORIENTATION_ON_VERTEX.value: gtx.as_field(
                (dims.VertexDim, dims.V2EDim),
                self._reader.int_variable(
                    gridfile.GeometryName.EDGE_ORIENTATION_ON_VERTEX,
                    transpose=True,
                    apply_transformation=False,
                    indices=my_vertex_indices,
                ),
                allocator=backend,
            ),
        }

    def _read_grid_refinement_fields(
        self,
        backend: gtx_backend.Backend | None = None,
    ) -> dict[gtx.Dimension, gtx.Field]:
        """
        Reads the refinement control fields from the grid file.

        Refinement control contains the classification of each entry in a field to predefined horizontal grid zones as for example the distance to the boundaries,
        see [refinement.py](refinement.py)

        Args:
            backend: Optional backend to use for reading the fields, if not provided the default backend is used.
        Returns:
            dict[gtx.Dimension, gtx.Field]: A dictionary containing the refinement control fields for each dimension.
        """
        refinement_control_names = {
            dims.CellDim: gridfile.GridRefinementName.CONTROL_CELLS,
            dims.EdgeDim: gridfile.GridRefinementName.CONTROL_EDGES,
            dims.VertexDim: gridfile.GridRefinementName.CONTROL_VERTICES,
        }
        refinement_control_fields = {
            dim: gtx.as_field(
                (dim,),
                self._reader.int_variable(
                    name,
                    indices=self._decomposition_info.global_index(dim),
                    transpose=False,
                    apply_transformation=False,
                ),
                allocator=backend,
            )
            for dim, name in refinement_control_names.items()
        }
        return refinement_control_fields

    def _read_start_end_indices(
        self,
    ) -> tuple[
        dict[gtx.Dimension, data_alloc.NDArray],
        dict[gtx.Dimension, data_alloc.NDArray],
    ]:
        """ "
        Read the start/end indices from the grid file.

        This should be used for a single node run. In the case of a multi node distributed run the  start and end indices need to be reconstructed from the decomposed grid.
        """
        _child_dom = 0
        grid_refinement_dimensions = {
            dims.CellDim: gridfile.DimensionName.CELL_GRF,
            dims.EdgeDim: gridfile.DimensionName.EDGE_GRF,
            dims.VertexDim: gridfile.DimensionName.VERTEX_GRF,
        }
        max_refinement_control_values = {
            dim: self._reader.dimension(name) for dim, name in grid_refinement_dimensions.items()
        }
        start_index_names = {
            dims.CellDim: gridfile.GridRefinementName.START_INDEX_CELLS,
            dims.EdgeDim: gridfile.GridRefinementName.START_INDEX_EDGES,
            dims.VertexDim: gridfile.GridRefinementName.START_INDEX_VERTICES,
        }

        start_indices = {
            dim: self._get_index_field(name, transpose=False, apply_offset=True)[_child_dom]
            for dim, name in start_index_names.items()
        }
        for dim in grid_refinement_dimensions:
            assert start_indices[dim].shape == (
                max_refinement_control_values[dim],
            ), f"start index array for {dim} has wrong shape"

        end_index_names = {
            dims.CellDim: gridfile.GridRefinementName.END_INDEX_CELLS,
            dims.EdgeDim: gridfile.GridRefinementName.END_INDEX_EDGES,
            dims.VertexDim: gridfile.GridRefinementName.END_INDEX_VERTICES,
        }
        end_indices = {
            dim: self._get_index_field(name, transpose=False, apply_offset=False)[_child_dom]
            for dim, name in end_index_names.items()
        }
        for dim in grid_refinement_dimensions:
            assert start_indices[dim].shape == (
                max_refinement_control_values[dim],
            ), f"start index array for {dim} has wrong shape"
            assert end_indices[dim].shape == (
                max_refinement_control_values[dim],
            ), f"start index array for {dim} has wrong shape"

        return start_indices, end_indices

    @property
    def grid(self) -> icon.IconGrid:
        return self._grid

    @property
    def geometry(self) -> GeometryDict:
        return self._geometry

    @property
    def coordinates(self) -> CoordinateDict:
        return self._coordinates

    @property
    def decomposition_info(self) -> decomposition.DecompositionInfo:
        return self._decomposition_info

    def _construct_decomposed_grid(
        self,
        backend: gtx_backend.Backend | None,
        with_skip_values: bool,
        decomposer: halo.Decomposer,
        run_properties: decomposition.ProcessProperties,
    ) -> None:
        """Construct the grid topology from the icon grid file.

        Reads connectivity fields from the grid file and constructs derived connectivities needed in
        Icon4py from them. Adds constructed start/end index information to the grid.

        """
        xp = data_alloc.import_array_ns(backend)
        ## FULL GRID PROPERTIES
        global_params, grid_config = self._construct_config(with_skip_values, xp)

        cell_to_cell_neighbors = self._get_index_field(gridfile.ConnectivityName.C2E2C)
        neighbor_tables = {
            dims.C2E2C: cell_to_cell_neighbors,
            dims.C2E: self._get_index_field(gridfile.ConnectivityName.C2E),
            dims.E2C: self._get_index_field(gridfile.ConnectivityName.E2C),
            dims.V2E: self._get_index_field(gridfile.ConnectivityName.V2E),
            dims.V2C: self._get_index_field(gridfile.ConnectivityName.V2C),
            dims.C2V: self._get_index_field(gridfile.ConnectivityName.C2V),
            dims.V2E2V: self._get_index_field(gridfile.ConnectivityName.V2E2V),
            dims.E2V: self._get_index_field(gridfile.ConnectivityName.E2V),
        }

        cells_to_rank_mapping = decomposer(cell_to_cell_neighbors, run_properties.comm_size)
        # HALO CONSTRUCTION
        # TODO(halungge): reduce the set of neighbor tables used in the halo construction
        # TODO(halungge): figure out where to do the host to device copies (xp.asarray...)
        neighbor_tables_for_halo_construction = neighbor_tables
        halo_constructor = halo.halo_constructor(
            run_properties=run_properties,
            grid_config=grid_config,
            connectivities=neighbor_tables_for_halo_construction,
            backend=backend,
        )

        self._decomposition_info = halo_constructor(cells_to_rank_mapping)

        # TODO(halungge): run this only for distrbuted grids otherwise to nothing internally
        neighbor_tables = {
            k: self._decomposition_info.global_to_local(
                k.source, v[self._decomposition_info.global_index(k.target[0])]
            )
            for k, v in neighbor_tables_for_halo_construction.items()
        }

        # JOINT functionality
        # COMPUTE remaining derived connectivities

        neighbor_tables.update(_get_derived_connectivities(neighbor_tables, array_ns=xp))

        start, end = self._read_start_end_indices()
        start_indices = {
            k: v
            for dim in dims.MAIN_HORIZONTAL_DIMENSIONS.values()
            for k, v in h_grid.map_icon_domain_bounds(dim, start[dim]).items()
        }
        end_indices = {
            k: v
            for dim in dims.MAIN_HORIZONTAL_DIMENSIONS.values()
            for k, v in h_grid.map_icon_domain_bounds(dim, end[dim]).items()
        }
        refinement_fields = self._read_grid_refinement_fields(backend)

        grid = icon.icon_grid(
            self._reader.attribute(gridfile.MandatoryPropertyName.GRID_UUID),
            allocator=backend,
            config=grid_config,
            neighbor_tables=neighbor_tables,
            start_indices=start_indices,
            end_indices=end_indices,
            global_properties=global_params,
            refinement_control=refinement_fields,
        )
        self._grid = grid

    def _construct_config(self, with_skip_values, xp):
        grid_root = self._reader.attribute(gridfile.MandatoryPropertyName.ROOT)
        grid_level = self._reader.attribute(gridfile.MandatoryPropertyName.LEVEL)
        global_params = icon.GlobalGridParams(root=grid_root, level=grid_level)
        cell_refinement = self._reader.variable(gridfile.GridRefinementName.CONTROL_CELLS)
        limited_area = refinement.is_limited_area_grid(cell_refinement, array_ns=xp)
        grid_config = base.GridConfig(
            horizontal_size=self._read_full_grid_size(),
            vertical_size=self._vertical_config.num_levels,
            limited_area=limited_area,
            keep_skip_values=with_skip_values,
        )
        return global_params, grid_config

    def _get_index_field(
        self,
        field: gridfile.GridFileName,
        indices: np.ndarray | None = None,
        transpose=True,
        apply_offset=True,
    ):
        return self._reader.int_variable(
            field, indices=indices, transpose=transpose, apply_transformation=apply_offset
        )

    def _read_full_grid_size(self) -> base.HorizontalGridSize:
        """
        Read the grid size propertes (cells, edges, vertices) from the grid file.

        As the grid file contains the _full_ (non-distributed) grid, these are the sizes of prior to distribution.

        """
        num_cells = self._reader.dimension(gridfile.DimensionName.CELL_NAME)
        num_edges = self._reader.dimension(gridfile.DimensionName.EDGE_NAME)
        num_vertices = self._reader.dimension(gridfile.DimensionName.VERTEX_NAME)
        full_grid_size = base.HorizontalGridSize(
            num_vertices=num_vertices, num_edges=num_edges, num_cells=num_cells
        )
        return full_grid_size


def _get_derived_connectivities(
    neighbor_tables: dict[gtx.FieldOffset, data_alloc.NDArray], array_ns: ModuleType = np
) -> dict[gtx.FieldOffset, data_alloc.NDArray]:
    e2v_table = neighbor_tables[dims.E2V]
    c2v_table = neighbor_tables[dims.C2V]
    e2c_table = neighbor_tables[dims.E2C]
    c2e_table = neighbor_tables[dims.C2E]
    c2e2c_table = neighbor_tables[dims.C2E2C]
    e2c2v = _construct_diamond_vertices(
        e2v_table,
        c2v_table,
        e2c_table,
        array_ns=array_ns,
    )
    e2c2e = _construct_diamond_edges(e2c_table, c2e_table, array_ns=array_ns)
    e2c2e0 = array_ns.column_stack((array_ns.asarray(range(e2c2e.shape[0])), e2c2e))

    c2e2c2e = _construct_triangle_edges(c2e2c_table, c2e_table, array_ns=array_ns)
    c2e2c0 = array_ns.column_stack(
        (
            array_ns.asarray(range(c2e2c_table.shape[0])),
            (c2e2c_table),
        )
    )
    c2e2c2e2c = _construct_butterfly_cells(c2e2c_table, array_ns=array_ns)

    return {
        dims.C2E2CO: c2e2c0,
        dims.C2E2C2E: c2e2c2e,
        dims.C2E2C2E2C: c2e2c2e2c,
        dims.E2C2V: e2c2v,
        dims.E2C2E: e2c2e,
        dims.E2C2EO: e2c2e0,
    }


def _construct_diamond_vertices(
    e2v: data_alloc.NDArray,
    c2v: data_alloc.NDArray,
    e2c: data_alloc.NDArray,
    array_ns: ModuleType = np,
) -> data_alloc.NDArray:
    r"""
    Construct the connectivity table for the vertices of a diamond in the ICON triangular grid.

    Starting from the e2v and c2v connectivity the connectivity table for e2c2v is built up.

             v0
            /  \
           /    \
          /      \
         /        \
        v1---e0---v3
         \        /
          \      /
           \    /
            \  /
             v2

    For example for this diamond: e0 -> (v0, v1, v2, v3)
    Ordering is the same as ICON uses.

    Args:
        e2v: ndarray containing the connectivity table for edge-to-vertex
        c2v: ndarray containing the connectivity table for cell-to-vertex
        e2c: ndarray containing the connectivity table for edge-to-cell

    Returns: ndarray containing the connectivity table for edge-to-vertex on the diamond
    """
    dummy_c2v = _patch_with_dummy_lastline(c2v, array_ns=array_ns)
    expanded = dummy_c2v[e2c, :]
    sh = expanded.shape
    flat = expanded.reshape(sh[0], sh[1] * sh[2])
    far_indices = array_ns.zeros_like(e2v)
    # TODO(halungge): vectorize speed this up?
    for i in range(sh[0]):
        far_indices[i, :] = flat[i, ~array_ns.isin(flat[i, :], e2v[i, :])][:2]
    return array_ns.hstack((e2v, far_indices))


def _construct_diamond_edges(
    e2c: data_alloc.NDArray, c2e: data_alloc.NDArray, array_ns: ModuleType = np
) -> data_alloc.NDArray:
    r"""
    Construct the connectivity table for the edges of a diamond in the ICON triangular grid.

    Starting from the e2c and c2e connectivity the connectivity table for e2c2e is built up.

            /  \
           /    \
          e2 c0 e1
         /        \
         ----e0----
         \        /
          e3 c1 e4
           \    /
            \  /

    For example, for this diamond for e0 -> (e1, e2, e3, e4)


    Args:
        e2c: ndarray containing the connectivity table for edge-to-cell
        c2e: ndarray containing the connectivity table for cell-to-edge

    Returns: ndarray containing the connectivity table for central edge-to- boundary edges
             on the diamond
    """
    dummy_c2e = _patch_with_dummy_lastline(c2e, array_ns=array_ns)
    expanded = dummy_c2e[e2c[:, :], :]
    sh = expanded.shape
    flattened = expanded.reshape(sh[0], sh[1] * sh[2])

    diamond_sides = 4
    e2c2e = gridfile.GridFile.INVALID_INDEX * array_ns.ones((sh[0], diamond_sides), dtype=gtx.int32)
    for i in range(sh[0]):
        var = flattened[
            i,
            (
                ~array_ns.isin(
                    flattened[i, :], array_ns.asarray([i, gridfile.GridFile.INVALID_INDEX])
                )
            ),
        ]
        e2c2e[i, : var.shape[0]] = var
    return e2c2e


def _construct_triangle_edges(
    c2e2c: data_alloc.NDArray, c2e: data_alloc.NDArray, array_ns: ModuleType = np
) -> data_alloc.NDArray:
    r"""Compute the connectivity from a central cell to all neighboring edges of its cell neighbors.

         ----e3----  ----e7----
         \        /  \        /
          e4 c1  /    \  c3 e8
           \    e2 c0 e1    /
            \  /        \  /
               ----e0----
               \        /
                e5 c2 e6
                 \    /
                  \  /


    For example, for the triangular shape above, c0 -> (e3, e4, e2, e0, e5, e6, e7, e1, e8).

    Args:
        c2e2c: shape (n_cells, 3) connectivity table from a central cell to its cell neighbors
        c2e: shape (n_cells, 3), connectivity table from a cell to its neighboring edges
    Returns:
        ndarray: shape(n_cells, 9) connectivity table from a central cell to all neighboring
            edges of its cell neighbors
    """
    dummy_c2e = _patch_with_dummy_lastline(c2e, array_ns=array_ns)
    table = array_ns.reshape(dummy_c2e[c2e2c, :], (c2e2c.shape[0], 9))
    return table


def _construct_butterfly_cells(
    c2e2c: data_alloc.NDArray, array_ns: ModuleType = np
) -> data_alloc.NDArray:
    r"""Compute the connectivity from a central cell to all neighboring cells of its cell neighbors.

                  /  \        /  \
                 /    \      /    \
                /  c4  \    /  c5  \
               /        \  /        \
               ----e3----  ----e7----
            /  \        /  \        /  \
           /    e4 c1  /    \  c3 e8    \
          /  c9  \    e2 c0 e1    /  c6  \
         /        \  /        \  /        \
         ----------  ----e0----  ----------
                  /  \        /  \
                 /    e5 c2 e6    \
                /  c8  \    /  c7  \
               /        \  /        \
               ----------  ----------

    For example, for the shape above, c0 -> (c1, c4, c9, c2, c7, c8, c3, c5, c6).

    Args:
        c2e2c: shape (n_cells, 3) connectivity table from a central cell to its cell neighbors
    Returns:
        ndarray: shape(n_cells, 9) connectivity table from a central cell to all neighboring cells of its cell neighbors
    """
    dummy_c2e2c = _patch_with_dummy_lastline(c2e2c, array_ns=array_ns)
    c2e2c2e2c = array_ns.reshape(dummy_c2e2c[c2e2c, :], (c2e2c.shape[0], 9))
    return c2e2c2e2c


def _patch_with_dummy_lastline(ar, array_ns: ModuleType = np):
    """
    Patch an array for easy access with another offset containing invalid indices (-1).

    Enlarges this table to contain a fake last line to account for numpy wrap around when
    encountering a -1 = GridFile.INVALID_INDEX value

    Args:
        ar: ndarray connectivity array to be patched

    Returns: same array with an additional line containing only GridFile.INVALID_INDEX

    """
    patched_ar = array_ns.append(
        ar,
        gridfile.GridFile.INVALID_INDEX * array_ns.ones((1, ar.shape[1]), dtype=gtx.int32),
        axis=0,
    )
    return patched_ar


def construct_local_connectivity(
    field_offset: gtx.FieldOffset,
    decomposition_info: decomposition.DecompositionInfo,
    connectivity: np.ndarray,
) -> np.ndarray:
    """
    Construct a connectivity table for use on a given rank: it maps from source to target dimension in _local_ indices.

    Starting from the connectivity table on the global grid
    - we reduce it to the lines for the locally present entries of the the target dimension
    - the reduced connectivity then still maps to global source dimension indices:
        we replace those source dimension indices not present on the node to SKIP_VALUE and replace the rest with the local indices

    Args:
        field_offset: FieldOffset for which we want to construct the local connectivity table
        decomposition_info: DecompositionInfo for the current rank.
        connectivity:

    Returns:
        connectivity are for the same FieldOffset but mapping from local target dimension indices to local source dimension indices.
    """
    source_dim = field_offset.source
    target_dim = field_offset.target[0]
    sliced_connectivity = connectivity[
        decomposition_info.global_index(target_dim, decomposition.DecompositionInfo.EntryType.ALL)
    ]

    global_idx = decomposition_info.global_index(
        source_dim, decomposition.DecompositionInfo.EntryType.ALL
    )

    # replace indices in the connectivity that do not exist on the local node by the SKIP_VALUE (those are for example neighbors of the outermost halo points)
    local_connectivity = np.where(
        np.isin(sliced_connectivity, global_idx),
        sliced_connectivity,
        gridfile.GridFile.INVALID_INDEX,
    )

    # map to local source indices
    sorted_index_of_global_idx = np.argsort(global_idx)
    global_idx_sorted = global_idx[sorted_index_of_global_idx]
    for i in np.arange(local_connectivity.shape[0]):
        valid_neighbor_mask = local_connectivity[i, :] != gridfile.GridFile.INVALID_INDEX
        positions = np.searchsorted(global_idx_sorted, local_connectivity[i, valid_neighbor_mask])
        indices = sorted_index_of_global_idx[positions]
        local_connectivity[i, valid_neighbor_mask] = indices
    return local_connectivity


def _single_node_decomposition_info(
    grid_config: base.GridConfig, xp: ModuleType
) -> decomposition.DecompositionInfo:
    cell_size = (grid_config.num_cells,)
    edge_size = (grid_config.num_edges,)
    vertex_size = (grid_config.num_vertices,)
    info = (
        decomposition.DecompositionInfo(grid_config.num_levels)
        .set_dimension(
            dims.CellDim,
            xp.arange(cell_size[0], dtype=gtx.int32),
            xp.ones(cell_size, dtype=bool),
            xp.zeros(cell_size),
        )
        .set_dimension(
            dims.EdgeDim,
            xp.arange(edge_size[0], dtype=gtx.int32),
            xp.ones(edge_size, dtype=bool),
            xp.zeros(edge_size),
        )
        .set_dimension(
            dims.VertexDim,
            xp.arange(vertex_size[0], dtype=gtx.int32),
            xp.ones(vertex_size, dtype=bool),
            xp.zeros(vertex_size),
        )
    )
    return info
