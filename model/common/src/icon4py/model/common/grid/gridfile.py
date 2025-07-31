# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import enum
import logging
from typing import Union

import numpy as np
from gt4py import next as gtx

from icon4py.model.common import exceptions


_log = logging.getLogger(__name__)


try:
    from netCDF4 import Dataset
except ImportError:

    class Dataset:
        """Dummy class to make import run when (optional) netcdf dependency is not installed."""

        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("NetCDF4 is not installed.")


class GridFileName(str, enum.Enum):
    pass


class OptionalPropertyName(GridFileName):
    """Global grid file attributes hat are not present in all files."""

    HISTORY = "history"
    GRID_ID = "grid_ID"
    PARENT_GRID_ID = "parent_grid_ID"
    MAX_CHILD_DOMAINS = "max_child_dom"


class PropertyName(GridFileName): ...


class LAMPropertyName(PropertyName):
    """
    Properties only present in the LAM file from MCH that we use in mch_ch_r04_b09_dsl.
    The source of this file is currently unknown.
    """

    GLOBAL_GRID = "global_grid"


class MPIMPropertyName(PropertyName):
    """
    Properties only present in the [MPI-M generated](https://gitlab.dkrz.de/mpim-sw/grid-generator)
    [grid files](http://icon-downloads.mpimet.mpg.de/mpim_grids.xml)
    """

    REVISION = "revision"
    DATE = "date"
    USER = "user_name"
    OS = "os_name"
    NUMBER_OF_SUBGRIDS = "number_of_subgrids"
    START_SUBGRID = "start_subgrid_id"
    BOUNDARY_DEPTH = "boundary_depth_index"
    ROTATION = "rotation_vector"
    GEOMETRY = "grid_geometry"
    CELL_TYPE = "grid_cell_type"
    MEAN_EDGE_LENGTH = "mean_edge_length"
    MEAN_DUAL_EDGE_LENGTH = "mean_dual_edge_length"
    MEAN_CELL_AREA = "mean_cell_area"
    MEAN_DUAL_CELL_AREA = "mean_dual_cell_area"
    DOMAIN_LENGTH = "domain_length"
    DOMAIN_HEIGHT = "domain_height"
    SPHERE_RADIUS = "sphere_radius"
    CARTESIAN_CENTER = "domain_cartesian_center"


class MandatoryPropertyName(PropertyName):
    """
    File attributes present in all files.
    DWD generated (from [icon-tools](https://gitlab.dkrz.de/dwd-sw/dwd_icon_tools)
    [grid files](http://icon-downloads.mpimet.mpg.de/dwd_grids.xml) contain only those properties.
    """

    TITLE = "title"
    INSTITUTION = "institution"
    SOURCE = "source"
    GRID_UUID = "uuidOfHGrid"
    PARENT_GRID_ID = "uuidOfParHGrid"
    NUMBER_OF_GRID = "number_of_grid_used"
    URI = "ICON_grid_file_uri"
    CENTER = "center"
    SUB_CENTER = "subcenter"
    CRS_ID = "crs_id"
    CRS_NAME = "crs_name"
    GRID_MAPPING = "grid_mapping_name"
    ELLIPSOID = "ellipsoid_name"
    SEMI_MAJOR_AXIS = "semi_major_axis"
    INVERSE_FLATTENING = "inverse_flattening"
    LEVEL = "grid_level"
    ROOT = "grid_root"


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


class FieldName(GridFileName): ...


class ConnectivityName(FieldName):
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


class GeometryName(FieldName):
    # TODO (@halungge) compute from coordinates
    CELL_AREA = "cell_area"
    # TODO (@halungge) compute from coordinates
    DUAL_AREA = "dual_area"
    CELL_NORMAL_ORIENTATION = "orientation_of_normal"
    TANGENT_ORIENTATION = "edge_system_orientation"
    EDGE_ORIENTATION_ON_VERTEX = "edge_orientation"
    # TODO (@halungge) compute from coordinates
    EDGE_CELL_DISTANCE = "edge_cell_distance"
    EDGE_VERTEX_DISTANCE = "edge_vert_distance"


class CoordinateName(FieldName):
    """
    Coordinates of cell centers, edge midpoints and vertices.
    Units: radianfor both MPI-M and DWD
    """

    CELL_LONGITUDE = "clon"
    CELL_LATITUDE = "clat"
    EDGE_LONGITUDE = "elon"
    EDGE_LATITUDE = "elat"
    VERTEX_LONGITUDE = "vlon"
    VERTEX_LATITUDE = "vlat"


class GridRefinementName(FieldName):
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


class GridFile:
    """Represent and ICON netcdf grid file."""

    INVALID_INDEX = -1

    def __init__(self, file_name: str):
        self._filename = file_name
        self._dataset = None

    def dimension(self, name: DimensionName) -> int:
        """Read a dimension with name 'name' from the grid file."""
        return self._dataset.dimensions[name].size

    def attribute(self, name: PropertyName) -> Union[str, int, float]:
        "Read a global attribute with name 'name' from the grid file."
        return self._dataset.getncattr(name)

    def int_variable(
        self, name: FieldName, indices: np.ndarray = None, transpose: bool = True
    ) -> np.ndarray:
        """Read a integer field from the grid file.

        Reads as gtx.int32.

        Args:
            name: name of the field to read
            transpose: flag to indicate whether the file should be transposed (for 2d fields)
        Returns:
            NDArray: field data

        """
        _log.debug(f"reading {name}: transposing = {transpose}")
        return self.variable(name, indices, transpose=transpose, dtype=gtx.int32)

    def variable(
        self,
        name: FieldName,
        indices: np.ndarray = None,
        transpose=False,
        dtype: np.dtype = gtx.float64,
    ) -> np.ndarray:
        """Read a  field from the grid file.

        If a index array is given it only reads the values at those positions.
        Args:
            name: name of the field to read
            indices: indices to read
            transpose: flag indicateing whether the array needs to be transposed
                to match icon4py dimension ordering, defaults to False
            dtype: datatype of the field
        """
        try:
            variable = self._dataset.variables[name]
            _log.debug(f"reading {name}: transposing = {transpose}")
            data = variable[:] if indices is None else variable[indices]
            data = np.array(data, dtype=dtype)
            return np.transpose(data) if transpose else data
        except KeyError as err:
            msg = f"{name} does not exist in dataset"
            _log.warning(msg)
            _log.debug(f"Error: {err}")
            raise exceptions.IconGridError(msg) from err

    def close(self):
        self._dataset.close()

    def open(self):
        self._dataset = Dataset(self._filename, "r", format="NETCDF4")
        _log.debug(f"opened data set: {self._dataset}")
