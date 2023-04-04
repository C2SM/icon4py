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

import logging
from abc import ABC
from enum import Enum
from typing import Optional
from uuid import UUID

import numpy as np
from gt4py.next.common import Dimension
from netCDF4 import Dataset

from icon4py.common.dimension import (
    C2E2CDim,
    C2EDim,
    CellDim,
    E2CDim,
    E2VDim,
    EdgeDim,
    V2CDim,
    V2EDim,
    VertexDim,
    C2VDim,
)
from icon4py.diffusion.horizontal import HorizontalMeshSize, HorizontalMarkerIndex
from icon4py.diffusion.icon_grid import IconGrid, MeshConfig, VerticalMeshConfig


class GridFileName(str, Enum):
    pass


class GridFile:
    """
    Represents and ICON netcdf grid file.
    """

    INVALID_INDEX = -1

    class Property(GridFileName):
        GRID_ID = "uuidOfHGrid"
        PARENT_GRID_ID = "uuidOfParHGrid"

    class Offsets(GridFileName):
        C2E2C = "neighbor_cell_index"
        V2E2V = "vertices_of_vertex"
        V2E = "edges_of_vertex"
        V2C = "cells_of_vertex"
        E2V = "edge_vertices"
        E2C = "adjacent_cell_of_edge"
        C2V = "vertex_of_cell"
        C2E = "edge_of_cell"

    class Dimension(GridFileName):
        VERTEX_NAME = "vertex"
        EDGE_NAME = "edge"
        CELL_NAME = "cell"
        V2E_SIZE = "ne"  # 6
        DIAMOND_EDGE_SIZE = "no"  # 4
        NEIGHBORING_EDGES_TO_CELL_SIZE = "nv"  # 3
        E2V_SIZE = "nc"  # 2

        MAX_CHILD_DOMAINS = "max_chdom"

        # TODO: @magdalena what does the grf stand for 'grid_refinement'?
        CELL_GRF = "cell_grf"
        EDGE_GRF = "edge_grf"
        VERTEX_GRF = "vert_grf"

    class GridRefinement(GridFileName):
        CONTROL_CELLS = "refin_c_ctrl"
        CONTROL_EDGES = "refin_e_ctrl"
        CONTROL_VERTICES = "refin_v_ctrl"
        START_INDEX_CELLS = "start_idx_c"
        START_INDEX_EDGES = "start_idx_e"
        START_INDEX_VERTICES = "start_idx_v"
        END_INDEX_CELLS ="end_idx_c"
        END_INDEX_EDGES = "end_idx_e"
        END_INDEX_VERTICES = "end_idx_v"

    def __init__(self, dataset: Dataset):
        self._dataset = dataset
        self._log = logging.getLogger(__name__)

    def dimension(self, name: GridFileName) -> int:
        return self._dataset.dimensions[name].size

    def int_field(self, name: GridFileName, transpose=True) -> np.ndarray:
        try:
            nc_variable = self._dataset.variables[name]
            self._log.debug(f"{name}: {nc_variable}")
            data = nc_variable[:]

            data = np.array(data, dtype=np.int32)
            return np.transpose(data) if transpose else data
        except KeyError as error:
            msg = f"{name} does not exist in dataset"
            self._log.warning(msg)
            raise IconGridError(msg)


class IconGridError(RuntimeError):
    pass

class IconDomainZone(str,Enum):
    HALO = "halo"
    LATERAL_BOUNDARY = "lb"
    NUDGING = "nudging"
    INTERIOR = "interior" # TODO @magdalena is this the same as "prognostic" if yes then change the naming
    END = "end"


class IndexTransformation(ABC):
    def get_offset_for_index_field(self, array: np.ndarray):
        return np.zeros(array.shape)




class ToGt4PyTransformation(IndexTransformation):
    def get_offset_for_index_field(self, array: np.ndarray):
        """
        Calculate the index offset needed for usage with python.

        Fortran indices are 1-based, hence the offset is -1 for 0-based ness of python except for
        INVALID values which are marked with -1 in the grid file and are kept such.
        """
        return np.where(array < 0, 0, -1)


class GridDomainBoundaryTransformation(ABC):
    def __init__(self):
        pass

class SingleNodeBoundaryTransform(GridDomainBoundaryTransformation):
    pass


class GridManager:
    def __init__(self, transformation: IndexTransformation, grid_file: str):
        self._log = logging.getLogger(__name__)
        self._transformation = transformation
        self._grid: Optional[IconGrid] = None
        self._file_names = grid_file
        self._start_indices = {}
        self._end_indices = {}

    def init(self):
        dataset = self._read_gridfile(self._file_names)
        start_indices, end_indices = self._read_boundaries(dataset)
        self._start_indices.update(start_indices)
        self._end_indices.update(end_indices)
        _, grid= self._read_grid(dataset)
        self._grid = grid

    def _read_gridfile(self, fname:str)->Dataset:
        try:
            dataset = Dataset(self._file_names, "r", format="NETCDF4")
            self._log.debug(dataset)
            return dataset
        except FileNotFoundError:
            self._log.error(f"gridfile {fname} not found, aborting")
            exit(1)

    def _read_boundaries(self, dataset):
        reader = GridFile(dataset)
        grf_vertices = reader.dimension(GridFile.Dimension.VERTEX_GRF)
        grf_edges = reader.dimension(GridFile.Dimension.EDGE_GRF)
        grf_cells = reader.dimension(GridFile.Dimension.CELL_GRF)
        refin_c_ctl = reader.int_field(GridFile.GridRefinement.CONTROL_CELLS)
        refin_v_ctl = reader.int_field(GridFile.GridRefinement.CONTROL_VERTICES)
        refin_e_ctl = reader.int_field(GridFile.GridRefinement.CONTROL_EDGES)
        start_indices = {}
        end_indices = {}
        start_indices[CellDim] = self._get_index_field(reader, GridFile.GridRefinement.START_INDEX_CELLS, transpose=False)
        end_indices[CellDim] = self._get_index_field(reader, GridFile.GridRefinement.END_INDEX_CELLS, transpose=False, apply_offset=False)
        start_indices[EdgeDim] = self._get_index_field(reader, GridFile.GridRefinement.START_INDEX_EDGES, transpose=False)
        end_indices[EdgeDim] = self._get_index_field(reader, GridFile.GridRefinement.END_INDEX_EDGES, transpose=False, apply_offset=False)
        start_indices[VertexDim] = self._get_index_field(reader, GridFile.GridRefinement.START_INDEX_VERTICES, transpose=False)
        end_indices[VertexDim] = self._get_index_field(reader, GridFile.GridRefinement.END_INDEX_VERTICES, transpose=False, apply_offset=False)

        return start_indices, end_indices



    def get_domain_boundaries(self, dim: Dimension, zone: IconDomainZone) -> tuple[int, int]:
        match zone:
            case IconDomainZone.LATERAL_BOUNDARY:
                index = HorizontalMarkerIndex.lateral_boundary(dim)
                return self._get_start_end_indices(dim, index)
            case IconDomainZone.INTERIOR:
                index = HorizontalMarkerIndex.interior(dim)
                return self._get_start_end_indices(dim, index)
            case IconDomainZone.NUDGING:
                index = HorizontalMarkerIndex.nudging(dim)
                return self._get_start_end_indices(dim, index)
            case IconDomainZone.HALO:
                index = HorizontalMarkerIndex.local(dim)
                return self._get_start_end_indices(dim, index)
            case IconDomainZone.END:
                index = HorizontalMarkerIndex.end(dim)
                return self._get_start_end_indices(dim, index)

    def _get_start_end_indices(self, dim, index):
        try:
            return self._start_indices[dim][0][index], self._end_indices[dim][0][index]
        except KeyError as error:
            msg = f"start, end indices for dimension {dim} not present"
            self._log.error(msg)
            raise IconGridError(msg)
    def _read_grid(self, dataset:Dataset) -> tuple[UUID, IconGrid]:

            grid_id = UUID(dataset.getncattr(GridFile.Property.GRID_ID))
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



    def _get_index_field(self, reader, name: GridFileName, transpose=True, apply_offset = True):
        field = reader.int_field(name, transpose=transpose)
        if apply_offset:
            field = field + self._transformation.get_offset_for_index_field(field)
        return field

    def from_grid_dataset(self, dataset: Dataset) -> IconGrid:
        reader = GridFile(dataset)
        num_cells = reader.dimension(GridFile.Dimension.CELL_NAME)
        num_edges = reader.dimension(GridFile.Dimension.EDGE_NAME)
        num_vertices = reader.dimension(GridFile.Dimension.VERTEX_NAME)


        grid_size = HorizontalMeshSize(
            num_vertices=num_vertices, num_edges=num_edges, num_cells=num_cells
        )
        c2e = self._get_index_field(reader, GridFile.Offsets.C2E)
        c2v = self._get_index_field(reader, GridFile.Offsets.C2V)

        e2c = self._get_index_field(reader, GridFile.Offsets.E2C)

        e2v = self._get_index_field(reader, GridFile.Offsets.E2V)
        v2c = self._get_index_field(reader, GridFile.Offsets.V2C)
        v2e = self._get_index_field(reader, GridFile.Offsets.V2E)
        v2e2v = self._get_index_field(reader, GridFile.Offsets.V2E2V)
        c2e2c = self._get_index_field(reader, GridFile.Offsets.C2E2C)
        config = MeshConfig(
            horizontal_config=grid_size, vertical_config=VerticalMeshConfig(0)
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
                    C2E2CDim: c2e2c,
                    C2VDim: c2v,
                }
            )
        )
        return icon_grid


