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
import logging
from enum import Enum
from typing import Optional
from uuid import UUID

import numpy as np
from gt4py.next.common import Dimension, DimensionKind


try:
    from netCDF4 import Dataset
except ImportError:

    class Dataset:
        """Dummy class to make import run when (optional) netcdf dependency is not installed."""

        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("NetCDF4 is not installed.")


from icon4py.model.common.dimension import (
    C2E2CDim,
    C2E2CODim,
    C2EDim,
    C2VDim,
    CellDim,
    E2C2VDim,
    E2CDim,
    E2VDim,
    EdgeDim,
    V2CDim,
    V2E2VDim,
    V2EDim,
    VertexDim,
    E2C2EDim,
)
from icon4py.model.common.grid.base import GridConfig, VerticalGridSize
from icon4py.model.common.grid.horizontal import HorizontalGridSize
from icon4py.model.common.grid.icon import IconGrid


class GridFileName(str, Enum):
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

    class OffsetName(GridFileName):
        """Names for connectivities used in the grid file."""

        # e2c2e/e2c2eO: diamond edges (including origin) not present in grid file-> calculate?
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

    def int_field(self, name: GridFileName, transpose=True, dtype=np.int32) -> np.ndarray:
        try:
            nc_variable = self._dataset.variables[name]

            self._log.debug(f"reading {name}: {nc_variable}")
            data = nc_variable[:]
            data = np.array(data, dtype=dtype)
            return np.transpose(data) if transpose else data
        except KeyError:
            msg = f"{name} does not exist in dataset"
            self._log.warning(msg)
            raise IconGridError(msg)


class IconGridError(RuntimeError):
    pass


class IndexTransformation:
    def get_offset_for_index_field(
        self,
        array: np.ndarray,
    ):
        return np.zeros(array.shape, dtype=np.int32)


class ToGt4PyTransformation(IndexTransformation):
    def get_offset_for_index_field(self, array: np.ndarray):
        """
        Calculate the index offset needed for usage with python.

        Fortran indices are 1-based, hence the offset is -1 for 0-based ness of python except for
        INVALID values which are marked with -1 in the grid file and are kept such.
        """
        return np.where(array == GridFile.INVALID_INDEX, 0, -1)


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
        config: VerticalGridSize,
    ):
        self._log = logging.getLogger(__name__)
        self._transformation = transformation
        self._config = config
        self._grid: Optional[IconGrid] = None
        self._file_name = grid_file

    def __call__(self):
        dataset = self._read_gridfile(self._file_name)
        _, grid = self._constuct_grid(dataset)
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
                dtype=np.int64,
            )[_CHILD_DOM],
            VertexDim: self._get_index_field(
                reader,
                GridFile.GridRefinementName.START_INDEX_VERTICES,
                transpose=False,
                dtype=np.int64,
            )[_CHILD_DOM],
        }
        end_indices = {
            CellDim: self._get_index_field(
                reader,
                GridFile.GridRefinementName.END_INDEX_CELLS,
                transpose=False,
                apply_offset=False,
                dtype=np.int64,
            )[_CHILD_DOM],
            EdgeDim: self._get_index_field(
                reader,
                GridFile.GridRefinementName.END_INDEX_EDGES,
                transpose=False,
                apply_offset=False,
                dtype=np.int64,
            )[_CHILD_DOM],
            VertexDim: self._get_index_field(
                reader,
                GridFile.GridRefinementName.END_INDEX_VERTICES,
                transpose=False,
                apply_offset=False,
                dtype=np.int64,
            )[_CHILD_DOM],
        }

        return start_indices, end_indices, refin_ctrl, refin_ctrl_max

    def get_grid(self):
        return self._grid

    def _get_index(self, dim: Dimension, start_marker: int, index_dict):
        if dim.kind != DimensionKind.HORIZONTAL:
            msg = f"getting start index in horizontal domain with non - horizontal dimension {dim}"
            self._log.warning(msg)
            raise IconGridError(msg)
        try:
            return index_dict[dim][start_marker]
        except KeyError:
            msg = f"start, end indices for dimension {dim} not present"
            self._log.error(msg)
            raise IconGridError(msg)

    def _constuct_grid(self, dataset: Dataset) -> tuple[UUID, IconGrid]:
        grid_id = UUID(dataset.getncattr(GridFile.PropertyName.GRID_ID))
        return grid_id, self._from_grid_dataset(dataset)

    def get_size(self, dim: Dimension):
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
        dtype=np.int32,
    ):
        field = reader.int_field(field, transpose=transpose, dtype=dtype)
        if apply_offset:
            field = field + self._transformation.get_offset_for_index_field(field)
        return field

    def _from_grid_dataset(self, dataset: Dataset) -> IconGrid:
        reader = GridFile(dataset)
        num_cells = reader.dimension(GridFile.DimensionName.CELL_NAME)
        num_edges = reader.dimension(GridFile.DimensionName.EDGE_NAME)
        num_vertices = reader.dimension(GridFile.DimensionName.VERTEX_NAME)

        grid_size = HorizontalGridSize(
            num_vertices=num_vertices, num_edges=num_edges, num_cells=num_cells
        )
        c2e = self._get_index_field(reader, GridFile.OffsetName.C2E)

        e2c = self._get_index_field(reader, GridFile.OffsetName.E2C)
        c2v = self._get_index_field(reader, GridFile.OffsetName.C2V)
        e2v = self._get_index_field(reader, GridFile.OffsetName.E2V)

        e2c2v = self._construct_diamond_vertices(e2v, c2v, e2c)
        e2c2e = self._construct_diamond_edges(e2c, c2e)

        v2c = self._get_index_field(reader, GridFile.OffsetName.V2C)
        v2e = self._get_index_field(reader, GridFile.OffsetName.V2E)
        v2e2v = self._get_index_field(reader, GridFile.OffsetName.V2E2V)
        c2e2c = self._get_index_field(reader, GridFile.OffsetName.C2E2C)
        c2e2c0 = np.column_stack((c2e2c, (np.asarray(range(c2e2c.shape[0])))))
        (
            start_indices,
            end_indices,
            refine_ctrl,
            refine_ctrl_max,
        ) = self._read_grid_refinement_information(dataset)

        config = GridConfig(
            horizontal_config=grid_size,
            vertical_config=self._config,
        )
        icon_grid = (
            IconGrid()
            .with_config(config)
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
                    E2C2VDim: e2c2v,
                    V2E2VDim: v2e2v,
                    E2C2EDim: e2c2e,
                }
            )
            .with_start_end_indices(CellDim, start_indices[CellDim], end_indices[CellDim])
            .with_start_end_indices(EdgeDim, start_indices[EdgeDim], end_indices[EdgeDim])
            .with_start_end_indices(VertexDim, start_indices[VertexDim], end_indices[VertexDim])
        )

        return icon_grid

    def _construct_diamond_vertices(self, e2v: np.ndarray, c2v: np.ndarray, e2c: np.ndarray):
        n_edges = e2v.shape[0]
        e2c2v = -55 * np.ones((n_edges, 4), dtype=np.int32)
        e2c2v[:, 0:2] = e2v
        # enlarge this table to contain an entry at the end for INVALID_VALUES
        dummy_c2v = np.append(
            c2v,
            GridFile.INVALID_INDEX * np.ones((1, c2v.shape[1]), dtype=np.int32),
            axis=0,
        )
        expanded = dummy_c2v[e2c[:, :], :]
        sh = expanded.shape
        flat = expanded.reshape(sh[0], sh[1] * sh[2])
        # TODO (magdalena) vectorize speed this up!
        for x in range(sh[0]):
            e2c2v[x, 2:] = flat[x, ~np.in1d(flat[x, :], e2c2v[x, :])][:2]

        return e2c2v

    def _construct_diamond_edges(self, e2c: np.ndarray, c2e: np.ndarray):
        dummy_c2e = np.append(
            c2e,
            GridFile.INVALID_INDEX * np.ones((1, c2e.shape[1]), dtype=np.int32),
            axis=0,
        )
        expanded = dummy_c2e[e2c, :]
        sh = expanded.shape
        flattened = expanded.reshape(sh[0], sh[1] * sh[2])
        e2c2e = GridFile.INVALID_INDEX * np.ones((sh[0], 4), dtype=np.int32)
        for x in range(sh[0]):
            var = flattened[x, (~np.in1d(flattened[x, :], np.asarray([x, GridFile.INVALID_INDEX])))]
            e2c2e[x, : var.shape[0]] = var
        return e2c2e
