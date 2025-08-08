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
from typing import Literal, Optional, Protocol, TypeAlias, Union

import gt4py.next as gtx
import gt4py.next.backend as gtx_backend
import numpy as np

from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import base, gridfile, icon, refinement, vertical as v_grid
from icon4py.model.common.utils import data_allocation as data_alloc


_log = logging.getLogger(__name__)


class IconGridError(RuntimeError):
    pass


class IndexTransformation(Protocol):
    """Return a transformation field to be applied to index fields"""

    def __call__(
        self,
        array: data_alloc.NDArray,
    ) -> data_alloc.NDArray: ...


class NoTransformation(IndexTransformation):
    """Empty implementation of the Protocol. Just return zeros."""

    def __call__(self, array: data_alloc.NDArray):
        return np.zeros_like(array)


class ToZeroBasedIndexTransformation(IndexTransformation):
    def __call__(self, array: data_alloc.NDArray):
        """
        Calculate the index offset needed for usage with python.

        Fortran indices are 1-based, hence the offset is -1 for 0-based ness of python except for
        INVALID values which are marked with -1 in the grid file and are kept such.
        """
        return np.asarray(
            np.where(array == gridfile.GridFile.INVALID_INDEX, 0, -1), dtype=gtx.int32
        )


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
        transformation: IndexTransformation,
        grid_file: Union[pathlib.Path, str],
        config: v_grid.VerticalGridConfig,  # TODO (@halungge) remove to separate vertical and horizontal grid
    ):
        self._transformation = transformation
        self._file_name = str(grid_file)
        self._vertical_config = config
        self._grid: Optional[icon.IconGrid] = None
        self._decomposition_info: Optional[decomposition.DecompositionInfo] = None
        self._geometry: GeometryDict = {}
        self._reader = None
        self._coordinates: CoordinateDict = {}

    def open(self):
        """Open the gridfile resource for reading."""
        self._reader = gridfile.GridFile(self._file_name)
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

    def __call__(self, backend: Optional[gtx_backend.Backend], keep_skip_values: bool):
        if not self._reader:
            self.open()
        self._grid = self._construct_grid(backend=backend, with_skip_values=keep_skip_values)
        self._coordinates = self._read_coordinates(backend)
        self._geometry = self._read_geometry_fields(backend)
        self.close()

    def _read_coordinates(self, backend: Optional[gtx_backend.Backend]) -> CoordinateDict:
        return {
            dims.CellDim: {
                "lat": gtx.as_field(
                    (dims.CellDim,),
                    self._reader.variable(gridfile.CoordinateName.CELL_LATITUDE),
                    dtype=ta.wpfloat,
                    allocator=backend,
                ),
                "lon": gtx.as_field(
                    (dims.CellDim,),
                    self._reader.variable(gridfile.CoordinateName.CELL_LONGITUDE),
                    dtype=ta.wpfloat,
                    allocator=backend,
                ),
            },
            dims.EdgeDim: {
                "lat": gtx.as_field(
                    (dims.EdgeDim,),
                    self._reader.variable(gridfile.CoordinateName.EDGE_LATITUDE),
                    dtype=ta.wpfloat,
                    allocator=backend,
                ),
                "lon": gtx.as_field(
                    (dims.EdgeDim,),
                    self._reader.variable(gridfile.CoordinateName.EDGE_LONGITUDE),
                    dtype=ta.wpfloat,
                    allocator=backend,
                ),
            },
            dims.VertexDim: {
                "lat": gtx.as_field(
                    (dims.VertexDim,),
                    self._reader.variable(gridfile.CoordinateName.VERTEX_LATITUDE),
                    allocator=backend,
                    dtype=ta.wpfloat,
                ),
                "lon": gtx.as_field(
                    (dims.VertexDim,),
                    self._reader.variable(gridfile.CoordinateName.VERTEX_LONGITUDE),
                    allocator=backend,
                    dtype=ta.wpfloat,
                ),
            },
        }

    def _read_geometry_fields(self, backend: Optional[gtx_backend.Backend]):
        return {
            # TODO (@halungge) still needs to ported, values from "our" grid files contains (wrong) values:
            #   based on bug in generator fixed with this [PR40](https://gitlab.dkrz.de/dwd-sw/dwd_icon_tools/-/merge_requests/40) .
            gridfile.GeometryName.CELL_AREA.value: gtx.as_field(
                (dims.CellDim,),
                self._reader.variable(gridfile.GeometryName.CELL_AREA),
                allocator=backend,
            ),
            # TODO (@halungge) easily computed from a neighbor_sum V2C over the cell areas?
            gridfile.GeometryName.DUAL_AREA.value: gtx.as_field(
                (dims.VertexDim,),
                self._reader.variable(gridfile.GeometryName.DUAL_AREA),
                allocator=backend,
            ),
            gridfile.GeometryName.EDGE_CELL_DISTANCE.value: gtx.as_field(
                (dims.EdgeDim, dims.E2CDim),
                self._reader.variable(gridfile.GeometryName.EDGE_CELL_DISTANCE, transpose=True),
                allocator=backend,
            ),
            gridfile.GeometryName.EDGE_VERTEX_DISTANCE.value: gtx.as_field(
                (dims.EdgeDim, dims.E2VDim),
                self._reader.variable(gridfile.GeometryName.EDGE_VERTEX_DISTANCE, transpose=True),
                allocator=backend,
            ),
            # TODO (@halungge) recompute from coordinates? field in gridfile contains NaN on boundary edges
            gridfile.GeometryName.TANGENT_ORIENTATION.value: gtx.as_field(
                (dims.EdgeDim,),
                self._reader.variable(gridfile.GeometryName.TANGENT_ORIENTATION),
                allocator=backend,
            ),
            gridfile.GeometryName.CELL_NORMAL_ORIENTATION.value: gtx.as_field(
                (dims.CellDim, dims.C2EDim),
                self._reader.int_variable(
                    gridfile.GeometryName.CELL_NORMAL_ORIENTATION, transpose=True
                ),
                allocator=backend,
            ),
            gridfile.GeometryName.EDGE_ORIENTATION_ON_VERTEX.value: gtx.as_field(
                (dims.VertexDim, dims.V2EDim),
                self._reader.int_variable(
                    gridfile.GeometryName.EDGE_ORIENTATION_ON_VERTEX, transpose=True
                ),
                allocator=backend,
            ),
        }

    def _read_grid_refinement_fields(
        self,
        decomposition_info: Optional[decomposition.DecompositionInfo] = None,
        backend: Optional[gtx_backend.Backend] = None,
    ) -> dict[gtx.Dimension, gtx.Field]:
        """
        Reads the refinement control fields from the grid file.

        Refinement control contains the classification of each entry in a field to predefined horizontal grid zones as for example the distance to the boundaries,
        see [refinement.py](refinement.py)

        Args:
            decomposition_info: Optional decomposition information, if not provided the grid is assumed to be a single node run.
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
                self._reader.int_variable(name, decomposition_info, transpose=False),
                allocator=backend,
            )
            for dim, name in refinement_control_names.items()
        }
        return refinement_control_fields

    def _read_start_end_indices(
        self,
    ) -> tuple[
        dict[gtx.Dimension : data_alloc.NDArray],
        dict[gtx.Dimension : data_alloc.NDArray],
        dict[gtx.Dimension : gtx.int32],
    ]:
        """ "
        Read the start/end indices from the grid file.

        This should be used for a single node run. In the case of a multi node distributed run the  start and end indices need to be reconstructed from the decomposed grid.
        """
        _CHILD_DOM = 0
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
            dim: self._get_index_field(name, transpose=False, apply_offset=True)[_CHILD_DOM]
            for dim, name in start_index_names.items()
        }
        for dim in grid_refinement_dimensions.keys():
            assert start_indices[dim].shape == (
                max_refinement_control_values[dim],
            ), f"start index array for {dim} has wrong shape"

        end_index_names = {
            dims.CellDim: gridfile.GridRefinementName.END_INDEX_CELLS,
            dims.EdgeDim: gridfile.GridRefinementName.END_INDEX_EDGES,
            dims.VertexDim: gridfile.GridRefinementName.END_INDEX_VERTICES,
        }
        end_indices = {
            dim: self._get_index_field(name, transpose=False, apply_offset=False)[_CHILD_DOM]
            for dim, name in end_index_names.items()
        }
        for dim in grid_refinement_dimensions.keys():
            assert start_indices[dim].shape == (
                max_refinement_control_values[dim],
            ), f"start index array for {dim} has wrong shape"
            assert end_indices[dim].shape == (
                max_refinement_control_values[dim],
            ), f"start index array for {dim} has wrong shape"

        return start_indices, end_indices, grid_refinement_dimensions

    @property
    def grid(self) -> icon.IconGrid:
        return self._grid

    @property
    def geometry(self) -> GeometryDict:
        return self._geometry

    @property
    def coordinates(self) -> CoordinateDict:
        return self._coordinates

    def _construct_grid(
        self, backend: Optional[gtx_backend.Backend], with_skip_values: bool
    ) -> icon.IconGrid:
        """Construct the grid topology from the icon grid file.

        Reads connectivity fields from the grid file and constructs derived connectivities needed in
        Icon4py from them. Adds constructed start/end index information to the grid.

        """
        xp = data_alloc.import_array_ns(backend)
        refinement_fields = self._read_grid_refinement_fields(backend=backend)
        limited_area = refinement.is_limited_area_grid(
            refinement_fields[dims.CellDim].ndarray, array_ns=xp
        )

        num_cells = self._reader.dimension(gridfile.DimensionName.CELL_NAME)
        num_edges = self._reader.dimension(gridfile.DimensionName.EDGE_NAME)
        num_vertices = self._reader.dimension(gridfile.DimensionName.VERTEX_NAME)
        uuid_ = self._reader.attribute(gridfile.MandatoryPropertyName.GRID_UUID)
        grid_root = self._reader.attribute(gridfile.MandatoryPropertyName.ROOT)
        grid_level = self._reader.attribute(gridfile.MandatoryPropertyName.LEVEL)
        geometry_type = self._reader.try_attribute(gridfile.MPIMPropertyName.GEOMETRY)
        if geometry_type:
            geometry_type = base.GeometryType(geometry_type)
        global_params = icon.GlobalGridParams(
            icon.GridType(
                geometry_type=geometry_type,
                subdivision=icon.GridSubdivision(root=grid_root, level=grid_level),
            )
        )
        grid_size = base.HorizontalGridSize(
            num_vertices=num_vertices, num_edges=num_edges, num_cells=num_cells
        )
        config = base.GridConfig(
            horizontal_config=grid_size,
            vertical_size=self._vertical_config.num_levels,
            limited_area=limited_area,
            keep_skip_values=with_skip_values,
        )

        neighbor_tables = {
            dims.C2E2C: xp.asarray(self._get_index_field(gridfile.ConnectivityName.C2E2C)),
            dims.C2E: xp.asarray(self._get_index_field(gridfile.ConnectivityName.C2E)),
            dims.E2C: xp.asarray(self._get_index_field(gridfile.ConnectivityName.E2C)),
            dims.V2E: xp.asarray(self._get_index_field(gridfile.ConnectivityName.V2E)),
            dims.E2V: xp.asarray(self._get_index_field(gridfile.ConnectivityName.E2V)),
            dims.V2C: xp.asarray(self._get_index_field(gridfile.ConnectivityName.V2C)),
            dims.C2V: xp.asarray(self._get_index_field(gridfile.ConnectivityName.C2V)),
            dims.V2E2V: xp.asarray(self._get_index_field(gridfile.ConnectivityName.V2E2V)),
        }
        neighbor_tables.update(_get_derived_connectivities(neighbor_tables, array_ns=xp))

        start, end, _ = self._read_start_end_indices()
        start_indices = {dim: start[dim] for dim in dims.MAIN_HORIZONTAL_DIMENSIONS.values()}
        end_indices = {dim: end[dim] for dim in dims.MAIN_HORIZONTAL_DIMENSIONS.values()}

        return icon.icon_grid(
            id_=uuid_,
            allocator=backend,
            config=config,
            neighbor_tables=neighbor_tables,
            start_indices=start_indices,
            end_indices=end_indices,
            global_properties=global_params,
            refinement_control=refinement_fields,
        )

    def _get_index_field(self, field: gridfile.GridFileName, transpose=True, apply_offset=True):
        field = self._reader.int_variable(field, transpose=transpose)
        if apply_offset:
            field = field + self._transformation(field)
        return field


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
    # TODO (magdalena) vectorize speed this up?
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
