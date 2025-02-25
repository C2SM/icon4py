# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import enum
import logging
import pathlib
from types import ModuleType
from typing import Literal, Optional, Protocol, TypeAlias, Union

import gt4py.next as gtx
import gt4py.next.backend as gtx_backend
import numpy as np

from icon4py.model.common import dimension as dims, exceptions, type_alias as ta
from icon4py.model.common.decomposition import (
    definitions as decomposition,
)
from icon4py.model.common.grid import base, icon, vertical as v_grid
from icon4py.model.common.utils import data_allocation as data_alloc


try:
    from netCDF4 import Dataset
except ImportError:

    class Dataset:
        """Dummy class to make import run when (optional) netcdf dependency is not installed."""

        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("NetCDF4 is not installed.")


_log = logging.getLogger(__name__)


class GridFileName(str, enum.Enum):
    pass


class OptionalPropertyName(GridFileName):
    """Global grid file attributes hat are not present in all files."""

    HISTORY = "history"
    GRID_ID = "grid_ID"
    PARENT_GRID_ID = "parent_grid_ID"
    MAX_CHILD_DOMAINS = "max_child_dom"


class PropertyName(GridFileName):
    ...


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


class FieldName(GridFileName):
    ...


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


class IconGridError(RuntimeError):
    pass


class IndexTransformation(Protocol):
    """Return a transformation field to be applied to index fields"""

    def __call__(
        self,
        array: data_alloc.NDArray,
    ) -> data_alloc.NDArray:
        ...


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
        return np.asarray(np.where(array == GridFile.INVALID_INDEX, 0, -1), dtype=gtx.int32)


CoordinateDict: TypeAlias = dict[dims.Dimension, dict[Literal["lat", "lon"], gtx.Field]]
GeometryDict: TypeAlias = dict[GeometryName, gtx.Field]


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
        self._reader = GridFile(self._file_name)
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

    def __call__(self, backend: Optional[gtx_backend.Backend], limited_area=True):
        if not self._reader:
            self.open()
        on_gpu = data_alloc.is_cupy_device(backend)
        self._grid = self._construct_grid(on_gpu=on_gpu, limited_area=limited_area)
        self._refinement = self._read_grid_refinement_fields(backend)
        self._coordinates = self._read_coordinates(backend)
        self._geometry = self._read_geometry_fields(backend)

    def _read_coordinates(self, backend: Optional[gtx_backend.Backend]) -> CoordinateDict:
        return {
            dims.CellDim: {
                "lat": gtx.as_field(
                    (dims.CellDim,),
                    self._reader.variable(CoordinateName.CELL_LATITUDE),
                    dtype=ta.wpfloat,
                    allocator=backend,
                ),
                "lon": gtx.as_field(
                    (dims.CellDim,),
                    self._reader.variable(CoordinateName.CELL_LONGITUDE),
                    dtype=ta.wpfloat,
                    allocator=backend,
                ),
            },
            dims.EdgeDim: {
                "lat": gtx.as_field(
                    (dims.EdgeDim,),
                    self._reader.variable(CoordinateName.EDGE_LATITUDE),
                    dtype=ta.wpfloat,
                    allocator=backend,
                ),
                "lon": gtx.as_field(
                    (dims.EdgeDim,),
                    self._reader.variable(CoordinateName.EDGE_LONGITUDE),
                    dtype=ta.wpfloat,
                    allocator=backend,
                ),
            },
            dims.VertexDim: {
                "lat": gtx.as_field(
                    (dims.VertexDim,),
                    self._reader.variable(CoordinateName.VERTEX_LATITUDE),
                    allocator=backend,
                    dtype=ta.wpfloat,
                ),
                "lon": gtx.as_field(
                    (dims.VertexDim,),
                    self._reader.variable(CoordinateName.VERTEX_LONGITUDE),
                    allocator=backend,
                    dtype=ta.wpfloat,
                ),
            },
        }

    def _read_geometry_fields(self, backend: Optional[gtx_backend.Backend]):
        return {
            # TODO (@halungge) still needs to ported, values from "our" grid files contains (wrong) values:
            #   based on bug in generator fixed with this [PR40](https://gitlab.dkrz.de/dwd-sw/dwd_icon_tools/-/merge_requests/40) .
            GeometryName.CELL_AREA.value: gtx.as_field(
                (dims.CellDim,), self._reader.variable(GeometryName.CELL_AREA), allocator=backend
            ),
            # TODO (@halungge) easily computed from a neighbor_sum V2C over the cell areas?
            GeometryName.DUAL_AREA.value: gtx.as_field(
                (dims.VertexDim,), self._reader.variable(GeometryName.DUAL_AREA), allocator=backend
            ),
            GeometryName.EDGE_CELL_DISTANCE.value: gtx.as_field(
                (dims.EdgeDim, dims.E2CDim),
                self._reader.variable(GeometryName.EDGE_CELL_DISTANCE, transpose=True),
                allocator=backend,
            ),
            GeometryName.EDGE_VERTEX_DISTANCE.value: gtx.as_field(
                (dims.EdgeDim, dims.E2VDim),
                self._reader.variable(GeometryName.EDGE_VERTEX_DISTANCE, transpose=True),
            ),
            # TODO (@halungge) recompute from coordinates? field in gridfile contains NaN on boundary edges
            GeometryName.TANGENT_ORIENTATION.value: gtx.as_field(
                (dims.EdgeDim,),
                self._reader.variable(GeometryName.TANGENT_ORIENTATION),
                allocator=backend,
            ),
            GeometryName.CELL_NORMAL_ORIENTATION.value: gtx.as_field(
                (dims.CellDim, dims.C2EDim),
                self._reader.int_variable(GeometryName.CELL_NORMAL_ORIENTATION, transpose=True),
                allocator=backend,
            ),
            GeometryName.EDGE_ORIENTATION_ON_VERTEX.value: gtx.as_field(
                (dims.VertexDim, dims.V2EDim),
                self._reader.int_variable(GeometryName.EDGE_ORIENTATION_ON_VERTEX, transpose=True),
                allocator=backend,
            ),
        }

    def _read_start_end_indices(
        self,
    ) -> tuple[
        dict[dims.Dimension : data_alloc.NDArray],
        dict[dims.Dimension : data_alloc.NDArray],
        dict[dims.Dimension : gtx.int32],
    ]:
        """ "
        Read the start/end indices from the grid file.

        This should be used for a single node run. In the case of a multi node distributed run the  start and end indices need to be reconstructed from the decomposed grid.
        """
        _CHILD_DOM = 0
        grid_refinement_dimensions = {
            dims.CellDim: DimensionName.CELL_GRF,
            dims.EdgeDim: DimensionName.EDGE_GRF,
            dims.VertexDim: DimensionName.VERTEX_GRF,
        }
        max_refinement_control_values = {
            dim: self._reader.dimension(name) for dim, name in grid_refinement_dimensions.items()
        }
        start_index_names = {
            dims.CellDim: GridRefinementName.START_INDEX_CELLS,
            dims.EdgeDim: GridRefinementName.START_INDEX_EDGES,
            dims.VertexDim: GridRefinementName.START_INDEX_VERTICES,
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
            dims.CellDim: GridRefinementName.END_INDEX_CELLS,
            dims.EdgeDim: GridRefinementName.END_INDEX_EDGES,
            dims.VertexDim: GridRefinementName.END_INDEX_VERTICES,
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

    def _read_grid_refinement_fields(
        self,
        backend: Optional[gtx_backend.Backend],
        decomposition_info: Optional[decomposition.DecompositionInfo] = None,
    ) -> tuple[dict[dims.Dimension : data_alloc.NDArray]]:
        """
        Reads the refinement control fields from the grid file.

        Refinement control contains the classification of each entry in a field to predefined horizontal grid zones as for example the distance to the boundaries,
        see [refinement.py](refinement.py)
        """
        xp = data_alloc.import_array_ns(backend)
        refinement_control_names = {
            dims.CellDim: GridRefinementName.CONTROL_CELLS,
            dims.EdgeDim: GridRefinementName.CONTROL_EDGES,
            dims.VertexDim: GridRefinementName.CONTROL_VERTICES,
        }
        refinement_control_fields = {
            dim: xp.asarray(self._reader.int_variable(name, decomposition_info, transpose=False))
            for dim, name in refinement_control_names.items()
        }
        return refinement_control_fields

    @property
    def grid(self) -> icon.IconGrid:
        return self._grid

    @property
    def refinement(self):
        """
        Refinement control fields.

        TODO (@halungge) should those be added to the IconGrid?
        """
        return self._refinement

    @property
    def geometry(self) -> GeometryDict:
        return self._geometry

    @property
    def coordinates(self) -> CoordinateDict:
        return self._coordinates

    def _construct_grid(self, on_gpu: bool, limited_area: bool) -> icon.IconGrid:
        """Construct the grid topology from the icon grid file.

        Reads connectivity fields from the grid file and constructs derived connectivities needed in
        Icon4py from them. Adds constructed start/end index information to the grid.

        """
        grid = self._initialize_global(limited_area, on_gpu)

        global_connectivities = {
            dims.C2E2C: self._get_index_field(ConnectivityName.C2E2C),
            dims.C2E: self._get_index_field(ConnectivityName.C2E),
            dims.E2C: self._get_index_field(ConnectivityName.E2C),
            dims.V2E: self._get_index_field(ConnectivityName.V2E),
            dims.E2V: self._get_index_field(ConnectivityName.E2V),
            dims.V2C: self._get_index_field(ConnectivityName.V2C),
            dims.C2V: self._get_index_field(ConnectivityName.C2V),
            dims.V2E2V: self._get_index_field(ConnectivityName.V2E2V),
            dims.E2V: self._get_index_field(ConnectivityName.E2V),
            dims.C2V: self._get_index_field(ConnectivityName.C2V),
        }
        xp = data_alloc.array_ns(on_gpu)
        grid.with_connectivities(
            {o.target[1]: xp.asarray(c) for o, c in global_connectivities.items()}
        )
        _add_derived_connectivities(grid, array_ns=xp)
        _update_size_for_1d_sparse_dims(grid)
        start, end, _ = self._read_start_end_indices()
        for dim in dims.global_dimensions.values():
            grid.with_start_end_indices(dim, start[dim], end[dim])

        return grid

    def _get_index_field(self, field: GridFileName, transpose=True, apply_offset=True):
        field = self._reader.int_variable(field, transpose=transpose)
        if apply_offset:
            field = field + self._transformation(field)
        return field

    def _initialize_global(self, limited_area: bool, on_gpu: bool) -> icon.IconGrid:
        """
        Read basic information from the grid file:
        Mostly reads global grid file parameters and dimensions.

        Args:
            limited_area: bool whether or not the produced grid is a limited area grid.
            # TODO (@halungge) this is not directly encoded in the grid, which is why we passed it in. It could be determined from the refinement fields though.

            on_gpu: bool, whether or not we run on GPU. # TODO (@halungge) can this be removed and defined differently.

        Returns:
            IconGrid: basic grid, setup only with id and config information.

        """
        num_cells = self._reader.dimension(DimensionName.CELL_NAME)
        num_edges = self._reader.dimension(DimensionName.EDGE_NAME)
        num_vertices = self._reader.dimension(DimensionName.VERTEX_NAME)
        uuid = self._reader.attribute(MandatoryPropertyName.GRID_UUID)
        grid_level = self._reader.attribute(MandatoryPropertyName.LEVEL)
        grid_root = self._reader.attribute(MandatoryPropertyName.ROOT)
        global_params = icon.GlobalGridParams(level=grid_level, root=grid_root)
        grid_size = base.HorizontalGridSize(
            num_vertices=num_vertices, num_edges=num_edges, num_cells=num_cells
        )
        config = base.GridConfig(
            horizontal_config=grid_size,
            vertical_size=self._vertical_config.num_levels,
            on_gpu=on_gpu,
            limited_area=limited_area,
        )
        grid = icon.IconGrid(uuid).with_config(config).with_global_params(global_params)
        return grid


def _add_derived_connectivities(grid: icon.IconGrid, array_ns: ModuleType = np) -> icon.IconGrid:
    e2c2v = _construct_diamond_vertices(
        grid.connectivities[dims.E2VDim],
        grid.connectivities[dims.C2VDim],
        grid.connectivities[dims.E2CDim],
        array_ns=array_ns,
    )
    e2c2e = _construct_diamond_edges(
        grid.connectivities[dims.E2CDim], grid.connectivities[dims.C2EDim], array_ns=array_ns
    )
    e2c2e0 = array_ns.column_stack((array_ns.asarray(range(e2c2e.shape[0])), e2c2e))

    c2e2c2e = _construct_triangle_edges(
        grid.connectivities[dims.C2E2CDim], grid.connectivities[dims.C2EDim], array_ns=array_ns
    )
    c2e2c0 = array_ns.column_stack(
        (
            array_ns.asarray(range(grid.connectivities[dims.C2E2CDim].shape[0])),
            (grid.connectivities[dims.C2E2CDim]),
        )
    )
    c2e2c2e2c = _construct_butterfly_cells(grid.connectivities[dims.C2E2CDim], array_ns=array_ns)

    grid.with_connectivities(
        {
            dims.C2E2CODim: c2e2c0,
            dims.C2E2C2EDim: c2e2c2e,
            dims.C2E2C2E2CDim: c2e2c2e2c,
            dims.E2C2VDim: e2c2v,
            dims.E2C2EDim: e2c2e,
            dims.E2C2EODim: e2c2e0,
        }
    )

    return grid


def _update_size_for_1d_sparse_dims(grid):
    grid.update_size_connectivities(
        {
            dims.ECVDim: grid.size[dims.EdgeDim] * grid.size[dims.E2C2VDim],
            dims.CEDim: grid.size[dims.CellDim] * grid.size[dims.C2EDim],
            dims.ECDim: grid.size[dims.EdgeDim] * grid.size[dims.E2CDim],
        }
    )


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
    e2c2e = GridFile.INVALID_INDEX * array_ns.ones((sh[0], diamond_sides), dtype=gtx.int32)
    for i in range(sh[0]):
        var = flattened[
            i, (~array_ns.isin(flattened[i, :], array_ns.asarray([i, GridFile.INVALID_INDEX])))
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
        GridFile.INVALID_INDEX * array_ns.ones((1, ar.shape[1]), dtype=gtx.int32),
        axis=0,
    )
    return patched_ar
