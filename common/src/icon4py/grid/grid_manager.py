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
from abc import ABC
from enum import Enum
from typing import Optional
from uuid import UUID

import numpy as np
from gt4py.next.common import Dimension, DimensionKind
from netCDF4 import Dataset

from icon4py.common.dimension import (
    C2E2CDim,
    C2EDim,
    C2VDim,
    CellDim,
    E2CDim,
    E2VDim,
    EdgeDim,
    V2CDim,
    V2EDim,
    VertexDim, C2E2CODim, E2C2VDim,
)
from icon4py.grid.horizontal import HorizontalGridSize
from icon4py.grid.icon_grid import GridConfig, IconGrid
from icon4py.grid.vertical import VerticalGridConfig


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
        # e2c2e/e2c2eO: diamond edges (including origin) -> calculate?
        # e2c2v: diamond vertices: constructed from e2c and c2v

        C2E2C = "neighbor_cell_index"  # dims(nv=3, cell)
        V2E2V = "vertices_of_vertex"  # dims(ne=6, vertex) not in simple mesh, = v2c2v vertices in hexa/pentagon
        V2E = "edges_of_vertex"  # dims(ne=6, vertex)
        V2C = "cells_of_vertex"  # dims(ne=6, vertex)
        E2V = "edge_vertices"  # dims(nc=2, edge)
        C2V = "vertex_of_cell"  # dims(nv=3, cell) # not in simple mesh
        E2C = "adjacent_cell_of_edge"  # dims(nc=2, edge)
        C2E = "edge_of_cell"  # dims(nv=3, cell)

    class DimensionName(GridFileName):
        VERTEX_NAME = "vertex"
        EDGE_NAME = "edge"
        CELL_NAME = "cell"
        DIAMOND_EDGE_SIZE = "no"  # 4
        NEIGHBORS_TO_VERTEX_SIZE = "ne"  # 6
        NEIGHBORS_TO_CELL_SIZE = "nv"  # 3
        NEIGHBORS_TO_EDGE_SIZE = "nc"  # 2
        MAX_CHILD_DOMAINS = "max_chdom"

        # TODO: @magdalena what does the grf abbrev. stand for 'grid_refinement'?
        CELL_GRF = "cell_grf"
        EDGE_GRF = "edge_grf"
        VERTEX_GRF = "vert_grf"

    class GridRefinementName(GridFileName):
        CONTROL_CELLS = "refin_c_ctrl"
        CONTROL_EDGES = "refin_e_ctrl"
        CONTROL_VERTICES = "refin_v_ctrl"
        START_INDEX_CELLS = "start_idx_c"
        START_INDEX_EDGES = "start_idx_e"
        START_INDEX_VERTICES = "start_idx_v"
        END_INDEX_CELLS = "end_idx_c"
        END_INDEX_EDGES = "end_idx_e"
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


class IndexTransformation(ABC):
    def get_offset_for_index_field(self, array: np.ndarray,):
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
    def __init__(
        self,
        transformation: IndexTransformation,
        grid_file: str,
        config: VerticalGridConfig,
    ):
        self._log = logging.getLogger(__name__)
        self._transformation = transformation
        self._config = config
        self._grid: Optional[IconGrid] = None
        self._file_names = grid_file

    def init(self):
        dataset = self._read_gridfile(self._file_names)
        _, grid = self._read_grid(dataset)

        self._grid = grid

    def _read_gridfile(self, fname: str) -> Dataset:
        try:
            dataset = Dataset(self._file_names, "r", format="NETCDF4")
            self._log.debug(dataset)
            return dataset
        except FileNotFoundError:
            self._log.error(f"gridfile {fname} not found, aborting")
            exit(1)

    def _read_grid_refinement_information(self, dataset):
        _CHILD_DOM = 0
        reader = GridFile(dataset)

        grf_vertices = reader.dimension(GridFile.DimensionName.VERTEX_GRF)
        grf_edges = reader.dimension(GridFile.DimensionName.EDGE_GRF)
        grf_cells = reader.dimension(GridFile.DimensionName.CELL_GRF)
        refin_c_ctl = reader.int_field(GridFile.GridRefinementName.CONTROL_CELLS)
        refin_v_ctl = reader.int_field(GridFile.GridRefinementName.CONTROL_VERTICES)
        refin_e_ctl = reader.int_field(GridFile.GridRefinementName.CONTROL_EDGES)
        start_indices = {}
        end_indices = {}
        start_indices[CellDim] = self._get_index_field(
            reader, GridFile.GridRefinementName.START_INDEX_CELLS, transpose=False
        )[_CHILD_DOM]
        end_indices[CellDim] = self._get_index_field(
            reader,
            GridFile.GridRefinementName.END_INDEX_CELLS,
            transpose=False,
            apply_offset=False, dtype=np.int64
        )[_CHILD_DOM]
        start_indices[EdgeDim] = self._get_index_field(
            reader, GridFile.GridRefinementName.START_INDEX_EDGES, transpose=False, dtype=np.int64
        )[_CHILD_DOM]
        end_indices[EdgeDim] = self._get_index_field(
            reader,
            GridFile.GridRefinementName.END_INDEX_EDGES,
            transpose=False,
            apply_offset=False, dtype=np.int64
        )[_CHILD_DOM]
        start_indices[VertexDim] = self._get_index_field(
            reader, GridFile.GridRefinementName.START_INDEX_VERTICES, transpose=False, dtype=np.int64
        )[_CHILD_DOM]
        end_indices[VertexDim] = self._get_index_field(
            reader,
            GridFile.GridRefinementName.END_INDEX_VERTICES,
            transpose=False,
            apply_offset=False, dtype=np.int64
        )[_CHILD_DOM]

        return start_indices, end_indices

    # TODO @magdalena make HorizontalMarkerIndex a type that behaves and is compatible with an int

    def get_start_index(self, dim: Dimension, start_marker: int):
        return self._get_index(dim, start_marker, self._grid.start_indices)

    def get_end_index(self, dim: Dimension, start_marker: int):
        return self._get_index(dim, start_marker, self._grid.end_indices)

    def get_grid(self):
        return self._grid

    def _get_index(self, dim: Dimension, start_marker: int, index_dict):
        if dim.kind != DimensionKind.HORIZONTAL:
            msg = f"getting start index in horizontal domain with non - horizontal dimension {dim}"
            self._log.warning(msg)
            raise IconGridError(msg)
        try:
            return index_dict[dim][start_marker]
        except KeyError as error:
            msg = f"start, end indices for dimension {dim} not present"
            self._log.error(msg)
            raise IconGridError(msg)

    def _read_grid(self, dataset: Dataset) -> tuple[UUID, IconGrid]:
        grid_id = UUID(dataset.getncattr(GridFile.PropertyName.GRID_ID))
        return grid_id, self.from_grid_dataset(dataset)

    def get_c2e_connectivity(self):
        return self._grid.get_c2e_connectivity()

    def get_e2v_connectivity(self):
        return self._grid.get_e2v_connectivity()

    def get_e2c_connectivity(self):
        return self._grid.get_e2c_connectivity()

    def get_c2e2c_connectivity(self):
        return self._grid.get_c2e2c_connectivity()

    def get_v2c_connectivity(self):
        return self._grid.get_v2c_connectivity()

    def get_c2v_connectivity(self):
        return self._grid.get_c2v_connectivity()

    def get_c2e2co_connectivity(self):
        return self._grid.get_c2e2co_connectivity()

    def get_e2c2v_connectivity(self):
        return self._grid.get_e2c2v_connectivity()

    def get_v2e_connectivity(self):
        return self._grid.get_v2e_connectivity()

    def get_e2ecv_connectivity(self):
        return self._grid.get_e2ecv_connectivity()

    def get_e2c2e_connectivity(self):
        return self._grid.get_e2c2e_connectivity()

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
        self, reader, field: GridFileName, transpose=True, apply_offset=True, dtype=np.int32
    ):
        field = reader.int_field(field, transpose=transpose, dtype=dtype)
        if apply_offset:
            field = field + self._transformation.get_offset_for_index_field(field)
        return field

    def from_grid_dataset(self, dataset: Dataset) -> IconGrid:
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

        e2c2v = self.construct_diamond_array(c2v, e2c)

        v2c = self._get_index_field(reader, GridFile.OffsetName.V2C)
        v2e = self._get_index_field(reader, GridFile.OffsetName.V2E)
        v2e2v = self._get_index_field(reader, GridFile.OffsetName.V2E2V)
        c2e2c = self._get_index_field(reader, GridFile.OffsetName.C2E2C)
        c2e2c0 = np.column_stack((c2e2c, (np.asarray(range(c2e2c.shape[0])))))
        start_indices, end_indices = self._read_grid_refinement_information(dataset)

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
                }
            )
            .with_start_end_indices(
                CellDim, start_indices[CellDim], end_indices[CellDim]
            )
            .with_start_end_indices(
                EdgeDim, start_indices[EdgeDim], end_indices[EdgeDim]
            )
            .with_start_end_indices(
                VertexDim, start_indices[VertexDim], end_indices[VertexDim]
            )
        )

        return icon_grid

    def construct_diamond_array(self, c2v:np.ndarray, e2c:np.ndarray):
        dummy_c2v = np.append(c2v, GridFile.INVALID_INDEX * np.ones((1, c2v.shape[1]), dtype=np.int32), axis=0)
        expanded = dummy_c2v[e2c[:, :], :]
        sh = expanded.shape
        flattened = expanded.reshape(sh[0], sh[1] * sh[2])
        return np.apply_along_axis(np.unique, 1, flattened)

    def get_c2v_connectivity(self):
        return self._grid.get_c2v_connectivity()

