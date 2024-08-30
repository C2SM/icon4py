# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import enum
import logging
import pathlib
from typing import Callable, Optional, Sequence, Union

import gt4py.next as gtx
import numpy as np

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import (
    definitions as decomposition,
    halo,
)
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.settings import xp


try:
    from netCDF4 import Dataset
except ImportError:

    class Dataset:
        """Dummy class to make import run when (optional) netcdf dependency is not installed."""

        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("NetCDF4 is not installed.")


from icon4py.model.common.grid import (
    base as base_grid,
    icon as icon_grid,
    vertical as v_grid,
)


class ReadType(enum.IntEnum):
    FLOAT = 0
    INT = 1

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

    class ConnectivityName(GridFileName):
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
        CHILD_DOMAINS = "max_chdom"

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

    class GeometryName(GridFileName):
        CELL_AREA = "cell_area"
        EDGE_LENGTH = "edge_length"

    class CoordinateName(GridFileName):
        CELL_LONGITUDE = "clon"
        CELL_LATITUDE = "clat"
        EDGE_LONGITUDE = "elon"
        EDGE_LATITUDE = "elat"
        VERTEX_LONGITUDE = "vlon"
        VERTEX_LATITUDE = "vlat"

    def __init__(self, dataset: Dataset):
        self._dataset = dataset
        self._log = logging.getLogger(__name__)

    def dimension(self, name: GridFileName) -> int:
        return self._dataset.dimensions[name].size

    def int_field(self, name: GridFileName, transpose=True, dtype=gtx.int32) -> np.ndarray:
        try:
            nc_variable = self._dataset.variables[name]

            self._log.debug(f"reading {name}: {nc_variable}: transposing = {transpose}")

            data = nc_variable[:]
            data = np.array(data, dtype=dtype)
            return np.transpose(data) if transpose else data
        except KeyError as err:
            msg = f"{name} does not exist in dataset"
            self._log.warning(msg)
            raise IconGridError(msg) from err

    def array_1d(
        self, name: GridFileName, indices: np.ndarray = None, dtype=gtx.float64
    ) -> np.ndarray:
        """Read a  field from the grid file.

        If a index array is given it only reads the values at those positions.
        Args:
            name: name of the field to read
            indices: indices to read
            dtype: datatype of the field  
        """
        try:
            # use python slice? 2D fields
            #           (sparse, horizontal)
            variable = self._dataset.variables[name]
            self._log.debug(f"reading {name}: {variable}")
            data = variable[:] if indices is None else variable[indices]
            data = np.array(data, dtype=dtype)
            return data
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


class ToZeroBasedIndexTransformation(IndexTransformation):
    def get_offset_for_index_field(self, array: np.ndarray):
        """
        Calculate the index offset needed for usage with python.

        Fortran indices are 1-based, hence the offset is -1 for 0-based ness of python except for
        INVALID values which are marked with -1 in the grid file and are kept such.
        """
        return np.asarray(np.where(array == GridFile.INVALID_INDEX, 0, -1), dtype=gtx.int32)

# TODO make this a context manager
class GridManager:
    """
    Read ICON grid file and set up  IconGrid.

    Reads an ICON grid file and extracts connectivity arrays and start-, end-indices for horizontal
    domain boundaries. Provides an IconGrid instance for further usage.
    """

    def __init__(
        self,
        transformation: IndexTransformation,
        grid_file: Union[pathlib.Path, str],
        config: v_grid.VerticalGridConfig,  # TODO (@halungge) remove
        decomposer: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
        run_properties: decomposition.ProcessProperties = decomposition.SingleNodeProcessProperties(),
         
    ):
        self._log = logging.getLogger(__name__)
        self._run_properties = run_properties
        self._transformation = transformation
        self._file_name = str(grid_file)

        if decomposer is None:
            self._decompose = halo.SingleNodeDecomposer() if run_properties.single_node() else halo.SimpleMetisDecomposer()
        else: 
            self._decompose = decomposer
        self._config = config
        self._grid: Optional[icon_grid.IconGrid] = None
        self._decomposition_info: Optional[decomposition.DecompositionInfo] = None
        self._dataset = None
        self._reader = None

    def __call__(self, on_gpu: bool = False, limited_area=True):
        self._read_gridfile(self._file_name)

        grid = self._construct_grid(on_gpu=on_gpu, limited_area=limited_area)
        self._grid = grid

    def _read_gridfile(self, fname: str) -> None:
        try:
            dataset = Dataset(self._file_name, "r", format="NETCDF4")
            self._log.debug(dataset)
            self._dataset = dataset
            self._reader = GridFile(dataset)
        except FileNotFoundError:
            self._log.error(f"gridfile {fname} not found, aborting")
            exit(1)

    def _read_grid_refinement_information(self, dataset):
        _CHILD_DOM = 0
        reader = GridFile(dataset)

        control_dims = [
            GridFile.GridRefinementName.CONTROL_CELLS,
            GridFile.GridRefinementName.CONTROL_EDGES,
            GridFile.GridRefinementName.CONTROL_VERTICES,
        ]
        refin_ctrl = {
            dim: reader.int_field(control_dims[i])
            for i, dim in enumerate(dims.horizontal_dims.values())
        }

        grf_dims = [
            GridFile.DimensionName.CELL_GRF,
            GridFile.DimensionName.EDGE_GRF,
            GridFile.DimensionName.VERTEX_GRF,
        ]
        refin_ctrl_max = {
            dim: reader.dimension(grf_dims[i])
            for i, dim in enumerate(dims.horizontal_dims.values())
        }

        start_index_dims = [
            GridFile.GridRefinementName.START_INDEX_CELLS,
            GridFile.GridRefinementName.START_INDEX_EDGES,
            GridFile.GridRefinementName.START_INDEX_VERTICES,
        ]
        start_indices = {
            dim: self._get_index_field(
                start_index_dims[i], transpose=False, dtype=gtx.int32
            )[_CHILD_DOM]
            for i, dim in enumerate(dims.horizontal_dims.values())
        }

        end_index_dims = [
            GridFile.GridRefinementName.END_INDEX_CELLS,
            GridFile.GridRefinementName.END_INDEX_EDGES,
            GridFile.GridRefinementName.END_INDEX_VERTICES,
        ]
        end_indices = {
            dim: self._get_index_field(
                end_index_dims[i], transpose=False, apply_offset=False, dtype=gtx.int32
            )[_CHILD_DOM]
            for i, dim in enumerate(dims.horizontal_dims.values())
        }

        return start_indices, end_indices, refin_ctrl, refin_ctrl_max

    def _read(
        self,
        decomposition_info: decomposition.DecompositionInfo,
        fields: dict[dims.Dimension, Sequence[GridFileName]],
    ):
        reader = self._reader.array_1d
        (cells_on_node, edges_on_node, vertices_on_node) = (
            (
                decomposition_info.global_index(
                    dims.CellDim, decomposition.DecompositionInfo.EntryType.ALL
                ),
                decomposition_info.global_index(
                    dims.EdgeDim, decomposition.DecompositionInfo.EntryType.ALL
                ),
                decomposition_info.global_index(
                    dims.VertexDim, decomposition.DecompositionInfo.EntryType.ALL
                ),
            )
            if decomposition_info is not None
            else (None, None, None)
        )

        def _read_local(fields: dict[dims.Dimension, Sequence[GridFileName]]):
            cell_fields = fields.get(dims.CellDim, [])
            edge_fields = fields.get(dims.EdgeDim, [])
            vertex_fields = fields.get(dims.VertexDim, [])
            vals = (
                {name: reader(name, cells_on_node) for name in cell_fields}
                | {name:reader(name, edges_on_node) for name in edge_fields}
                | {name: reader(name, vertices_on_node) for name in vertex_fields}
            )

            return vals

        return _read_local(fields)

    def read_geometry(self, decomposition_info: Optional[decomposition.DecompositionInfo] = None):
        return self._read(
            decomposition_info,
            {
                dims.CellDim: [GridFile.GeometryName.CELL_AREA],
                dims.EdgeDim: [GridFile.GeometryName.EDGE_LENGTH],
            },
        )

    def read_coordinates(
        self, decomposition_info: Optional[decomposition.DecompositionInfo] = None
    ):
        return self._read(
            decomposition_info,
            {
                dims.CellDim: [
                    GridFile.CoordinateName.CELL_LONGITUDE,
                    GridFile.CoordinateName.CELL_LATITUDE,
                ],
                dims.EdgeDim: [
                    GridFile.CoordinateName.EDGE_LONGITUDE,
                    GridFile.CoordinateName.EDGE_LATITUDE,
                ],
                dims.VertexDim: [
                    GridFile.CoordinateName.VERTEX_LONGITUDE,
                    GridFile.CoordinateName.VERTEX_LATITUDE,
                ],
            },
        )

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
        except KeyError:
            msg = f"start, end indices for dimension {dim} not present"
            self._log.error(msg)

    def _from_grid_dataset(self, grid, on_gpu: bool, limited_area=True) -> icon_grid.IconGrid:
        e2c2v = _construct_diamond_vertices(
            grid.connectivities[dims.E2VDim],
            grid.connectivities[dims.C2VDim],
            grid.connectivities[dims.E2CDim],
        )
        e2c2e = _construct_diamond_edges(
            grid.connectivities[dims.E2CDim], grid.connectivities[dims.C2EDim]
        )
        e2c2e0 = np.column_stack((np.asarray(range(e2c2e.shape[0])), e2c2e))

        c2e2c2e = _construct_triangle_edges(
            grid.connectivities[dims.C2E2CDim], grid.connectivities[dims.C2EDim]
        )
        c2e2c0 = np.column_stack(
            (
                np.asarray(range(grid.connectivities[dims.C2E2CDim].shape[0])),
                (grid.connectivities[dims.C2E2CDim]),
            )
        )

        grid.with_connectivities(
            {
                dims.C2E2CODim: c2e2c0,
                dims.C2E2C2EDim: c2e2c2e,
                dims.E2C2VDim: e2c2v,
                dims.E2C2EDim: e2c2e,
                dims.E2C2EODim: e2c2e0,
            }
        )
        _update_size_for_1d_sparse_dims(grid)

        return grid

    def _add_start_end_indices(self, grid):
        (
            start_indices,
            end_indices,
            refine_ctrl,
            refine_ctrl_max,
        ) = self._read_grid_refinement_information(self._dataset)
        grid.with_start_end_indices(
            dims.CellDim, start_indices[dims.CellDim], end_indices[dims.CellDim]
        ).with_start_end_indices(
            dims.EdgeDim, start_indices[dims.EdgeDim], end_indices[dims.EdgeDim]
        ).with_start_end_indices(
            dims.VertexDim, start_indices[dims.VertexDim], end_indices[dims.VertexDim]
        )
      

    # TODO (@halungge)
    #  - remove duplication,
    # - only read fields globally that are used for halo construction
    # - make halo constructor transparent

    def _construct_grid(self, on_gpu: bool, limited_area: bool) -> icon_grid.IconGrid:
        grid = self._initialize_global(limited_area, on_gpu)

        global_connectivities = {
            dims.C2E2C: self._get_index_field(GridFile.ConnectivityName.C2E2C),
            dims.C2E: self._get_index_field(GridFile.ConnectivityName.C2E),
            dims.E2C: self._get_index_field(GridFile.ConnectivityName.E2C),
            dims.V2E: self._get_index_field(GridFile.ConnectivityName.V2E),
            dims.E2V: self._get_index_field(GridFile.ConnectivityName.E2V),
            dims.V2C: self._get_index_field(GridFile.ConnectivityName.V2C),
            dims.C2V: self._get_index_field(GridFile.ConnectivityName.C2V),
            dims.V2E2V: self._get_index_field(GridFile.ConnectivityName.V2E2V),
        }
        if self._run_properties.single_node():
            global_connectivities.update(
                {
                    dims.E2V: self._get_index_field(GridFile.ConnectivityName.E2V),
                    dims.C2V: self._get_index_field(GridFile.ConnectivityName.C2V),
                }
            )
            grid.with_connectivities({o.target[1]: c for o, c in global_connectivities.items()})
            _add_derived_connectivities(grid)
            self._add_start_end_indices(grid)
            return grid
        else:
            cells_to_rank_mapping = self._decompose(
                global_connectivities[dims.C2E2CDim], self._run_properties.comm_size
            )
            construct_decomposition_info = halo.HaloGenerator(
                self._run_properties,
                cells_to_rank_mapping,
                global_connectivities,
                self._config.num_levels,
            )
            decomposition_info = construct_decomposition_info()
            global_connectivities.update(
                {
                    dims.E2V: self._get_index_field(GridFile.ConnectivityName.E2V),
                    dims.C2V: self._get_index_field(GridFile.ConnectivityName.C2V),
                }
            )
            local_connectivities = {
                offset.target[1]: construct_local_connectivity(
                    offset, decomposition_info, global_connectivities[offset]
                )
                for offset, conn in global_connectivities.keys()
            }
            grid.with_connectivities(local_connectivities)
            _add_derived_connectivities(grid)
            # it has a fixed size that is a dimension in the grid file
            
            

        _update_size_for_1d_sparse_dims(grid)
        return grid
    
    #TODO (@halungge) is this used?
    def get_size(self, dim: gtx.Dimension):
        if dim == dims.VertexDim:
            return self._grid.config.num_vertices
        elif dim == dims.CellDim:
            return self._grid.config.num_cells
        elif dim == dims.EdgeDim:
            return self._grid.config.num_edges
        else:
            self._log.warning(f"cannot determine size of unknown dimension {dim}")
            raise IconGridError(f"Unknown dimension {dim}")

    def _get_index_field(
        self, field: GridFileName, transpose=True, apply_offset=True, dtype=gtx.int32
    ):
        field = self._reader.int_field(field, dtype=dtype, transpose=transpose)
        if apply_offset:
            field = field + self._transformation.get_offset_for_index_field(field)
        return field
   
    def _construct_start_indices(self, decomposition_info: decomposition.DecompositionInfo):
        num_child_domains = self._reader.dimension(GridFile.DimensionName.MAX_CHILD_DOMAINS)
        grid_refinement_dimensions = {dim: self._reader.dimension(name) for dim, name in {dims.CellDim: GridFile.DimensionName.CELL_GRF,
                                      dims.EdgeDim: GridFile.DimensionName.EDGE_GRF,
                                      dims.VertexDim: GridFile.DimensionName.VERTEX_GRF}.items()}
        
        start_index = {dim: np.zeros((num_child_domains, size), dtype=gtx.int32) for dim, size in grid_refinement_dimensions.items()}
        end_index = {dim: np.zeros((num_child_domains, size), dtype=gtx.int32) for dim, size in
                       grid_refinement_dimensions.items()}
        field_names = {dims.CellDim: [GridFile.GridRefinementName.CONTROL_CELLS],
                       dims.EdgeDim: [GridFile.GridRefinementName.CONTROL_EDGES],
                       dims.VertexDim: [GridFile.GridRefinementName.CONTROL_EDGES]}
   
        refine_control_fields = self._read(decomposition_info, fields=field_names)
        # do we need to modify the refinement control for "halo"
        dim = dims.CellDim
        local_size = decomposition_info.global_index(dim, decomposition.DecompositionInfo.EntryType.ALL).shape[0]
        start_index[dim][h_grid.HorizontalMarkerIndex.end(dim)] = local_size
        start_index[dim][h_grid.HorizontalMarkerIndex.local(dim)] = 0
        # find first local halo index which is not on lateral boundary
        local_halo_index = decomposition_info.local_index(dim, decomposition.DecompositionInfo.EntryType.HALO)
        ref_ctrl = refine_control_fields[dim]
        minimial_local_halo_index = np.min(np.where(ref_ctrl == local_halo_index)
        start_index[dim][h_grid.HorizontalMarkerIndex.halo(dim)] = minimial_local_halo_index
        
        
        # for global 
        
        
        
    def _initialize_global(self, limited_area, on_gpu):
        num_cells = self._reader.dimension(GridFile.DimensionName.CELL_NAME)
        num_edges = self._reader.dimension(GridFile.DimensionName.EDGE_NAME)
        num_vertices = self._reader.dimension(GridFile.DimensionName.VERTEX_NAME)
        uuid = self._dataset.getncattr(GridFile.PropertyName.GRID_ID)
        grid_level = self._dataset.getncattr(GridFile.PropertyName.LEVEL)
        grid_root = self._dataset.getncattr(GridFile.PropertyName.ROOT)
        global_params = icon_grid.GlobalGridParams(level=grid_level, root=grid_root)
        grid_size = base_grid.HorizontalGridSize(
            num_vertices=num_vertices, num_edges=num_edges, num_cells=num_cells
        )
        config = base_grid.GridConfig(
            horizontal_config=grid_size,
            vertical_size=self._config.num_levels,
            on_gpu=on_gpu,
            limited_area=limited_area,
        )
        grid = icon_grid.IconGrid(uuid).with_config(config).with_global_params(global_params)
        return grid

        
        
    


###########################

def _add_derived_connectivities(grid) -> icon_grid.IconGrid:
    e2c2v = _construct_diamond_vertices(
        grid.connectivities[dims.E2VDim],
        grid.connectivities[dims.C2VDim],
        grid.connectivities[dims.E2CDim],
    )
    e2c2e = _construct_diamond_edges(
        grid.connectivities[dims.E2CDim], grid.connectivities[dims.C2EDim]
    )
    e2c2e0 = np.column_stack((np.asarray(range(e2c2e.shape[0])), e2c2e))

    c2e2c2e = _construct_triangle_edges(
        grid.connectivities[dims.C2E2CDim], grid.connectivities[dims.C2EDim]
    )
    c2e2c0 = np.column_stack(
        (
            np.asarray(range(grid.connectivities[dims.C2E2CDim].shape[0])),
            (grid.connectivities[dims.C2E2CDim]),
        )
    )

    grid.with_connectivities(
        {
            dims.C2E2CODim: c2e2c0,
            dims.C2E2C2EDim: c2e2c2e,
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
        far_indices[i, :] = flat[i, ~np.isin(flat[i, :], e2v[i, :])][:2]
    return np.hstack((e2v, far_indices))


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
        var = flattened[i, (~np.isin(flattened[i, :], np.asarray([i, GridFile.INVALID_INDEX])))]
        e2c2e[i, : var.shape[0]] = var
    return e2c2e

def _construct_triangle_edges(c2e2c, c2e):
    r"""Compute the connectivity from a central cell to all neighboring edges of its cell neighbors.

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


def construct_local_connectivity(
    field_offset: gtx.FieldOffset,
    decomposition_info: decomposition.DecompositionInfo,
    connectivity: xp.ndarray,
) -> xp.ndarray:
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
    # log.debug(f"rank {self._props.rank} has local connectivity f: {sliced_connectivity}")

    global_idx = decomposition_info.global_index(source_dim, decomposition.DecompositionInfo.EntryType.ALL)

    # replace indices in the connectivity that do not exist on the local node by the SKIP_VALUE (those are for example neighbors of the outermost halo points)
    local_connectivity = xp.where(
        xp.isin(sliced_connectivity, global_idx), sliced_connectivity, GridFile.INVALID_INDEX
    )

    # map to local source indices
    sorted_index_of_global_idx = xp.argsort(global_idx)
    global_idx_sorted = global_idx[sorted_index_of_global_idx]
    for i in xp.arange(local_connectivity.shape[0]):
        valid_neighbor_mask = local_connectivity[i, :] != GridFile.INVALID_INDEX
        positions = xp.searchsorted(global_idx_sorted, local_connectivity[i, valid_neighbor_mask])
        indices = sorted_index_of_global_idx[positions]
        local_connectivity[i, valid_neighbor_mask] = indices
    # log.debug(f"rank {self._props.rank} has local connectivity f: {local_connectivity}")
    return local_connectivity


