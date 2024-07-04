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
import dataclasses
import enum
import logging
from typing import Optional

import gt4py.next as gtx
import numpy as np


try:
    from netCDF4 import Dataset
except ImportError:

    class Dataset:
        """Dummy class to make import run when (optional) netcdf dependency is not installed."""

        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("NetCDF4 is not installed.")


from icon4py.model.common.dimension import (
    C2E2C2EDim,
    C2E2CDim,
    C2E2CODim,
    C2EDim,
    C2VDim,
    CEDim,
    CellDim,
    E2C2EDim,
    E2C2EODim,
    E2C2VDim,
    E2CDim,
    E2VDim,
    ECDim,
    ECVDim,
    EdgeDim,
    V2CDim,
    V2E2VDim,
    V2EDim,
    VertexDim,
)
from icon4py.model.common.grid import (
    base as grid_def,
    icon as icon_grid,
    vertical as v_grid,
)


class GridFileName(str, enum.Enum):
    pass


@dataclasses.dataclass
class GridFileField:
    name: GridFileName
    shape: tuple[int, ...]


def _validate_shape(data: np.array, field_definition: GridFileField):
    if data.shape != field_definition.shape:
        raise IconGridError(
            f"invalid grid file field {field_definition.name} does not have dimension {field_definition.shape}"
        )


class GridFile:
    """Represent and ICON netcdf grid file."""

    INVALID_INDEX = -1

    class PropertyName(GridFileName):
        GRID_ID = "uuidOfHGrid"
        PARENT_GRID_ID = "uuidOfParHGrid"
        LEVEL = "grid_level"
        ROOT = "grid_root"

    class OffsetName(GridFileName):
        """Names for connectivities used in the grid file."""

        # e2c2e/e2c2eO: diamond edges (including origin) not present in grid file-> construct
        #               from e2c and c2e
        # e2c2v: diamond vertices: not present in grid file -> constructed from e2c and c2v

        #: name of C2E2C connectivity in grid file: dims(nv=3, cell)
        C2E2C = "neighbor_cell_index"

        #: name of V2E2V connectivity in gridfile: dims(ne=6, vertex),
        #: all vertices of a pentagon/hexagon, same as V2C2V
        V2E2V = "vertices_of_vertex"  # does not exist in simple.py

        #: name of V2E dimension in grid file: dims(ne=6, vertex)
        V2E = "edges_of_vertex"

        #: name fo V2C connectivity in grid file: dims(ne=6, vertex)
        V2C = "cells_of_vertex"

        #: name of E2V connectivity in grid file: dims(nc=2, edge)
        E2V = "edge_vertices"

        #: name of C2V connectivity in grid file: dims(nv=3, cell)
        C2V = "vertex_of_cell"  # does not exist in grid.simple.py

        #: name of E2C connectivity in grid file: dims(nc=2, edge)
        E2C = "adjacent_cell_of_edge"

        #: name of C2E connectivity in grid file: dims(nv=3, cell)
        C2E = "edge_of_cell"

    class DimensionName(GridFileName):
        """Dimension values (sizes) used in grid file."""

        #: number of vertices
        VERTEX_NAME = "vertex"

        #: number of edges
        EDGE_NAME = "edge"

        #: number of cells
        CELL_NAME = "cell"

        #: number of edges in a diamond: 4
        DIAMOND_EDGE_SIZE = "no"

        #: number of edges/cells neighboring one vertex: 6 (for regular, non pentagons)
        NEIGHBORS_TO_VERTEX_SIZE = "ne"

        #: number of cells edges, vertices and cells neighboring a cell: 3
        NEIGHBORS_TO_CELL_SIZE = "nv"

        #: number of vertices/cells neighboring an edge: 2
        NEIGHBORS_TO_EDGE_SIZE = "nc"

        #: number of child domains (for nesting)
        MAX_CHILD_DOMAINS = "max_chdom"

        #: Grid refinement: maximal number in grid-refinement (refin_ctl) array for each dimension
        CELL_GRF = "cell_grf"
        EDGE_GRF = "edge_grf"
        VERTEX_GRF = "vert_grf"

    class GridRefinementName(GridFileName):
        """Names of arrays in grid file defining the grid control, definition of boundaries layers, start and end indices of horizontal zones."""

        #: refine control value of cell indices
        CONTROL_CELLS = "refin_c_ctrl"

        #: refine control value of edge indices
        CONTROL_EDGES = "refin_e_ctrl"

        #: refine control value of vertex indices
        CONTROL_VERTICES = "refin_v_ctrl"

        #: start indices of horizontal grid zones for cell fields
        START_INDEX_CELLS = "start_idx_c"

        #: start indices of horizontal grid zones for edge fields
        START_INDEX_EDGES = "start_idx_e"

        #: start indices of horizontal grid zones for vertex fields
        START_INDEX_VERTICES = "start_idx_v"

        #: end indices of horizontal grid zones for cell fields
        END_INDEX_CELLS = "end_idx_c"

        #: end indices of horizontal grid zones for edge fields
        END_INDEX_EDGES = "end_idx_e"

        #: end indices of horizontal grid zones for vertex fields
        END_INDEX_VERTICES = "end_idx_v"

    def __init__(self, dataset: Dataset):
        self._dataset = dataset
        self._log = logging.getLogger(__name__)

    def dimension(self, name: GridFileName) -> int:
        return self._dataset.dimensions[name].size

    def int_field(self, name: GridFileName, transpose=True, dtype=gtx.int32) -> np.ndarray:
        try:
            nc_variable = self._dataset.variables[name]

            self._log.debug(f"reading {name}: {nc_variable}")
            data = nc_variable[:]
            data = np.array(data, dtype=dtype)
            return np.transpose(data) if transpose else data
        except KeyError as err:
            msg = f"{name} does not exist in dataset"
            self._log.warning(msg)
            raise IconGridError(msg) from err


class IconGridError(RuntimeError):
    pass


class IndexTransformation:
    def get_offset_for_index_field(
        self,
        array: np.ndarray,
    ):
        return np.zeros(array.shape, dtype=gtx.int32)


class ToGt4PyTransformation(IndexTransformation):
    def get_offset_for_index_field(self, array: np.ndarray):
        """
        Calculate the index offset needed for usage with python.

        Fortran indices are 1-based, hence the offset is -1 for 0-based ness of python except for
        INVALID values which are marked with -1 in the grid file and are kept such.
        """
        return np.asarray(np.where(array == GridFile.INVALID_INDEX, 0, -1), dtype=gtx.int32)


class GridManager:
    """
    Read ICON grid file and set up  IconGrid.

    Reads an ICON grid file and extracts connectivity arrays and start-, end-indices for horizontal
    domain boundaries. Provides an IconGrid instance for further usage.
    """

    def __init__(
        self,
        transformation: IndexTransformation,
        grid_file: str,
        config: v_grid.VerticalGridSize,
    ):
        self._log = logging.getLogger(__name__)
        self._transformation = transformation
        self._config = config
        self._grid: Optional[icon_grid.IconGrid] = None
        self._file_name = grid_file

    def __call__(self, on_gpu: bool = False, limited_area=True):
        dataset = self._read_gridfile(self._file_name)
        grid = self._construct_grid(dataset, on_gpu=on_gpu, limited_area=limited_area)
        self._grid = grid

    def _read_gridfile(self, fname: str) -> Dataset:
        try:
            dataset = Dataset(self._file_name, "r", format="NETCDF4")
            self._log.debug(dataset)
            return dataset
        except FileNotFoundError:
            self._log.error(f"gridfile {fname} not found, aborting")
            exit(1)

    def _read_grid_refinement_information(self, dataset):
        _CHILD_DOM = 0
        reader = GridFile(dataset)

        refin_ctrl = {
            CellDim: reader.int_field(GridFile.GridRefinementName.CONTROL_CELLS),
            EdgeDim: reader.int_field(GridFile.GridRefinementName.CONTROL_EDGES),
            VertexDim: reader.int_field(GridFile.GridRefinementName.CONTROL_VERTICES),
        }
        refin_ctrl_max = {
            CellDim: reader.dimension(GridFile.DimensionName.CELL_GRF),
            EdgeDim: reader.dimension(GridFile.DimensionName.EDGE_GRF),
            VertexDim: reader.dimension(GridFile.DimensionName.VERTEX_GRF),
        }
        start_indices = {
            CellDim: self._get_index_field(
                reader, GridFile.GridRefinementName.START_INDEX_CELLS, transpose=False
            )[_CHILD_DOM],
            EdgeDim: self._get_index_field(
                reader,
                GridFile.GridRefinementName.START_INDEX_EDGES,
                transpose=False,
                dtype=gtx.int32,
            )[_CHILD_DOM],
            VertexDim: self._get_index_field(
                reader,
                GridFile.GridRefinementName.START_INDEX_VERTICES,
                transpose=False,
                dtype=gtx.int32,
            )[_CHILD_DOM],
        }
        end_indices = {
            CellDim: self._get_index_field(
                reader,
                GridFile.GridRefinementName.END_INDEX_CELLS,
                transpose=False,
                apply_offset=False,
                dtype=gtx.int32,
            )[_CHILD_DOM],
            EdgeDim: self._get_index_field(
                reader,
                GridFile.GridRefinementName.END_INDEX_EDGES,
                transpose=False,
                apply_offset=False,
                dtype=gtx.int32,
            )[_CHILD_DOM],
            VertexDim: self._get_index_field(
                reader,
                GridFile.GridRefinementName.END_INDEX_VERTICES,
                transpose=False,
                apply_offset=False,
                dtype=gtx.int32,
            )[_CHILD_DOM],
        }

        return start_indices, end_indices, refin_ctrl, refin_ctrl_max

    @property
    def grid(self):
        return self._grid

    def _get_index(self, dim: gtx.Dimension, start_marker: int, index_dict):
        if dim.kind != gtx.DimensionKind.HORIZONTAL:
            msg = f"getting start index in horizontal domain with non - horizontal dimension {dim}"
            self._log.warning(msg)
            raise IconGridError(msg)
        try:
            return index_dict[dim][start_marker]
        except KeyError as err:
            msg = f"start, end indices for dimension {dim} not present"
            self._log.error(msg)
            raise IconGridError(msg) from err

    def _construct_grid(
        self, dataset: Dataset, on_gpu: bool, limited_area: bool
    ) -> icon_grid.IconGrid:
        return self._from_grid_dataset(dataset, on_gpu=on_gpu, limited_area=limited_area)

    def get_size(self, dim: gtx.Dimension):
        if dim == VertexDim:
            return self._grid.config.num_vertices
        elif dim == CellDim:
            return self._grid.config.num_cells
        elif dim == EdgeDim:
            return self._grid.config.num_edges
        else:
            self._log.warning(f"cannot determine size of unknown dimension {dim}")
            raise IconGridError(f"Unknown dimension {dim}")

    def _get_index_field(
        self,
        reader,
        field: GridFileName,
        transpose=True,
        apply_offset=True,
        dtype=gtx.int32,
    ):
        field = reader.int_field(field, transpose=transpose, dtype=dtype)
        if apply_offset:
            field = field + self._transformation.get_offset_for_index_field(field)
        return field

    def _from_grid_dataset(
        self, dataset: Dataset, on_gpu: bool, limited_area=True
    ) -> icon_grid.IconGrid:
        reader = GridFile(dataset)
        num_cells = reader.dimension(GridFile.DimensionName.CELL_NAME)
        num_edges = reader.dimension(GridFile.DimensionName.EDGE_NAME)
        num_vertices = reader.dimension(GridFile.DimensionName.VERTEX_NAME)
        uuid = dataset.getncattr(GridFile.PropertyName.GRID_ID)
        grid_level = dataset.getncattr(GridFile.PropertyName.LEVEL)
        grid_root = dataset.getncattr(GridFile.PropertyName.ROOT)
        global_params = icon_grid.GlobalGridParams(level=grid_level, root=grid_root)

        grid_size = grid_def.HorizontalGridSize(
            num_vertices=num_vertices, num_edges=num_edges, num_cells=num_cells
        )
        c2e = self._get_index_field(reader, GridFile.OffsetName.C2E)

        e2c = self._get_index_field(reader, GridFile.OffsetName.E2C)
        c2v = self._get_index_field(reader, GridFile.OffsetName.C2V)
        e2v = self._get_index_field(reader, GridFile.OffsetName.E2V)

        e2c2v = self._construct_diamond_vertices(e2v, c2v, e2c)
        e2c2e = self._construct_diamond_edges(e2c, c2e)
        e2c2e0 = np.column_stack((np.asarray(range(e2c2e.shape[0])), e2c2e))

        v2c = self._get_index_field(reader, GridFile.OffsetName.V2C)
        v2e = self._get_index_field(reader, GridFile.OffsetName.V2E)
        v2e2v = self._get_index_field(reader, GridFile.OffsetName.V2E2V)
        c2e2c = self._get_index_field(reader, GridFile.OffsetName.C2E2C)
        c2e2c2e = self._construct_triangle_edges(c2e2c, c2e)
        c2e2c0 = np.column_stack((np.asarray(range(c2e2c.shape[0])), c2e2c))
        (
            start_indices,
            end_indices,
            refine_ctrl,
            refine_ctrl_max,
        ) = self._read_grid_refinement_information(dataset)

        config = grid_def.GridConfig(
            horizontal_config=grid_size,
            vertical_config=self._config,
            on_gpu=on_gpu,
            limited_area=limited_area,
        )
        grid = (
            icon_grid.IconGrid(uuid)
            .with_config(config)
            .with_global_params(global_params)
            .with_connectivities(
                {
                    C2EDim: c2e,
                    E2CDim: e2c,
                    E2VDim: e2v,
                    V2EDim: v2e,
                    V2CDim: v2c,
                    C2VDim: c2v,
                    C2E2CDim: c2e2c,
                    C2E2CODim: c2e2c0,
                    C2E2C2EDim: c2e2c2e,
                    E2C2VDim: e2c2v,
                    V2E2VDim: v2e2v,
                    E2C2EDim: e2c2e,
                    E2C2EODim: e2c2e0,
                }
            )
            .with_start_end_indices(CellDim, start_indices[CellDim], end_indices[CellDim])
            .with_start_end_indices(EdgeDim, start_indices[EdgeDim], end_indices[EdgeDim])
            .with_start_end_indices(VertexDim, start_indices[VertexDim], end_indices[VertexDim])
        )
        grid.update_size_connectivities(
            {
                ECVDim: grid.size[EdgeDim] * grid.size[E2C2VDim],
                CEDim: grid.size[CellDim] * grid.size[C2EDim],
                ECDim: grid.size[EdgeDim] * grid.size[E2CDim],
            }
        )

        return grid

    @staticmethod
    def _construct_diamond_vertices(
        e2v: np.ndarray, c2v: np.ndarray, e2c: np.ndarray
    ) -> np.ndarray:
        r"""
        Construct the connectivity table for the vertices of a diamond in the ICON triangular grid.

        Starting from the e2v and c2v connectivity the connectivity table for e2c2v is built up.

                     v0
                    / \
                  /    \
                 /      \
                /        \
               v1---e0---v3
                \       /
                 \     /
                  \   /
                   \ /
                    v2
        For example for this diamond: e0 -> (v0, v1, v2, v3)
        Ordering is the same as ICON uses.

        Args:
            e2v: np.ndarray containing the connectivity table for edge-to-vertex
            c2v: np.ndarray containing the connectivity table for cell-to-vertex
            e2c: np.ndarray containing the connectivity table for edge-to-cell

        Returns: np.ndarray containing the connectivity table for edge-to-vertex on the diamond
        """
        dummy_c2v = _patch_with_dummy_lastline(c2v)
        expanded = dummy_c2v[e2c[:, :], :]
        sh = expanded.shape
        flat = expanded.reshape(sh[0], sh[1] * sh[2])
        far_indices = np.zeros_like(e2v)
        # TODO (magdalena) vectorize speed this up?
        for i in range(sh[0]):
            far_indices[i, :] = flat[i, ~np.in1d(flat[i, :], e2v[i, :])][:2]
        return np.hstack((e2v, far_indices))

    @staticmethod
    def _construct_diamond_edges(e2c: np.ndarray, c2e: np.ndarray) -> np.ndarray:
        r"""
        Construct the connectivity table for the edges of a diamond in the ICON triangular grid.

        Starting from the e2c and c2e connectivity the connectivity table for e2c2e is built up.

            / \
          /    \
         e2    e1
        /    c0  \
        ----e0----
        \   c1   /
         e3    e4
          \   /
           \ /

        For example, for this diamond for e0 -> (e1, e2, e3, e4)


        Args:
            e2c: np.ndarray containing the connectivity table for edge-to-cell
            c2e: np.ndarray containing the connectivity table for cell-to-edge

        Returns: np.ndarray containing the connectivity table for central edge-to- boundary edges
                 on the diamond
        """
        dummy_c2e = _patch_with_dummy_lastline(c2e)
        expanded = dummy_c2e[e2c[:, :], :]
        sh = expanded.shape
        flattened = expanded.reshape(sh[0], sh[1] * sh[2])

        diamond_sides = 4
        e2c2e = GridFile.INVALID_INDEX * np.ones((sh[0], diamond_sides), dtype=gtx.int32)
        for i in range(sh[0]):
            var = flattened[i, (~np.in1d(flattened[i, :], np.asarray([i, GridFile.INVALID_INDEX])))]
            e2c2e[i, : var.shape[0]] = var
        return e2c2e

    def _construct_triangle_edges(self, c2e2c, c2e):
        """Compute the connectivity from a central cell to all neighboring edges of its cell neighbors.

           ____e3________e7____
           \   c1  / \   c3  /
            \     /   \     /
            e4   e2    e1  e8
              \ /   c0  \ /
                ----e0----
                \   c2  /
                 e5    e6
                  \   /
                   \ /

        For example, for the triangular shape above, c0 -> (e3, e4, e2, e0, e5, e6, e7, e1, e8).

        Args:
            c2e2c: shape (n_cell, 3) connectivity table from a central cell to its cell neighbors
            c2e: shape (n_cell, 3), connectivity table from a cell to its neighboring edges
        Returns:
            np.ndarray: shape(n_cells, 9) connectivity table from a central cell to all neighboring
                edges of its cell neighbors
        """
        dummy_c2e = _patch_with_dummy_lastline(c2e)
        table = np.reshape(dummy_c2e[c2e2c[:, :], :], (c2e2c.shape[0], 9))
        return table


def _patch_with_dummy_lastline(ar):
    """
    Patch an array for easy access with another offset containing invalid indices (-1).

    Enlarges this table to contain a fake last line to account for numpy wrap around when
    encountering a -1 = GridFile.INVALID_INDEX value

    Args:
        ar: np.ndarray connectivity array to be patched

    Returns: same array with an additional line containing only GridFile.INVALID_INDEX

    """
    patched_ar = np.append(
        ar,
        GridFile.INVALID_INDEX * np.ones((1, ar.shape[1]), dtype=gtx.int32),
        axis=0,
    )
    return patched_ar
