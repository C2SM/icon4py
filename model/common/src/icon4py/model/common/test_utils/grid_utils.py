import pytest
from uuid import uuid4

import netCDF4
import numpy as np

from icon4py.model.common.decomposition.definitions import SingleNodeRun
from icon4py.model.common.dimension import E2VDim, E2C2EDim, C2EDim, V2CDim, E2CDim, C2VDim, V2EDim, C2E2CDim
from icon4py.model.common.grid.grid_manager import GridFile, GridFileName
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.datatest_helpers import get_processor_properties_for_run, get_ranked_data_path, \
    get_datapath_for_ranked_data, create_icon_serial_data_provider, SER_DATA_BASEPATH
from icon4py.model.common.test_utils.serialbox_utils import IconSerialDataProvider


def _add_to_dataset(
    dataset: netCDF4.Dataset,
    data: np.ndarray,
    var_name: str,
    dims: tuple[GridFileName, GridFileName],
):
    var = dataset.createVariable(var_name, np.int32, dims)
    var[:] = np.transpose(data)[:]


SIMPLE_GRID_NC = "simple_grid.nc"


@pytest.fixture
def simple_grid_gridfile(tmp_path):
    path = tmp_path.joinpath(SIMPLE_GRID_NC).absolute()
    grid = SimpleGrid()
    dataset = netCDF4.Dataset(path, "w", format="NETCDF4")
    dataset.setncattr(GridFile.PropertyName.GRID_ID, str(uuid4()))
    dataset.createDimension(GridFile.DimensionName.VERTEX_NAME, size=grid.num_vertices)

    dataset.createDimension(GridFile.DimensionName.EDGE_NAME, size=grid.num_edges)
    dataset.createDimension(GridFile.DimensionName.CELL_NAME, size=grid.num_cells)
    dataset.createDimension(GridFile.DimensionName.NEIGHBORS_TO_EDGE_SIZE, size=grid.size[E2VDim])
    dataset.createDimension(GridFile.DimensionName.DIAMOND_EDGE_SIZE, size=grid.size[E2C2EDim])
    dataset.createDimension(GridFile.DimensionName.MAX_CHILD_DOMAINS, size=1)
    # add dummy values for the grf dimensions
    dataset.createDimension(GridFile.DimensionName.CELL_GRF, size=14)
    dataset.createDimension(GridFile.DimensionName.EDGE_GRF, size=24)
    dataset.createDimension(GridFile.DimensionName.VERTEX_GRF, size=13)
    _add_to_dataset(
        dataset,
        np.zeros(grid.num_edges),
        GridFile.GridRefinementName.CONTROL_EDGES,
        (GridFile.DimensionName.EDGE_NAME,),
    )

    _add_to_dataset(
        dataset,
        np.zeros(grid.num_cells),
        GridFile.GridRefinementName.CONTROL_CELLS,
        (GridFile.DimensionName.CELL_NAME,),
    )
    _add_to_dataset(
        dataset,
        np.zeros(grid.num_vertices),
        GridFile.GridRefinementName.CONTROL_VERTICES,
        (GridFile.DimensionName.VERTEX_NAME,),
    )

    dataset.createDimension(GridFile.DimensionName.NEIGHBORS_TO_CELL_SIZE, size=grid.size[C2EDim])
    dataset.createDimension(GridFile.DimensionName.NEIGHBORS_TO_VERTEX_SIZE, size=grid.size[V2CDim])

    _add_to_dataset(
        dataset,
        grid.connectivities[C2EDim],
        GridFile.OffsetName.C2E,
        (
            GridFile.DimensionName.NEIGHBORS_TO_CELL_SIZE,
            GridFile.DimensionName.CELL_NAME,
        ),
    )

    _add_to_dataset(
        dataset,
        grid.connectivities[E2CDim],
        GridFile.OffsetName.E2C,
        (
            GridFile.DimensionName.NEIGHBORS_TO_EDGE_SIZE,
            GridFile.DimensionName.EDGE_NAME,
        ),
    )
    _add_to_dataset(
        dataset,
        grid.connectivities[E2VDim],
        GridFile.OffsetName.E2V,
        (
            GridFile.DimensionName.NEIGHBORS_TO_EDGE_SIZE,
            GridFile.DimensionName.EDGE_NAME,
        ),
    )

    _add_to_dataset(
        dataset,
        grid.connectivities[V2CDim],
        GridFile.OffsetName.V2C,
        (
            GridFile.DimensionName.NEIGHBORS_TO_VERTEX_SIZE,
            GridFile.DimensionName.VERTEX_NAME,
        ),
    )

    _add_to_dataset(
        dataset,
        grid.connectivities[C2VDim],
        GridFile.OffsetName.C2V,
        (
            GridFile.DimensionName.NEIGHBORS_TO_CELL_SIZE,
            GridFile.DimensionName.CELL_NAME,
        ),
    )
    _add_to_dataset(
        dataset,
        np.zeros((grid.num_vertices, 4), dtype=np.int32),
        GridFile.OffsetName.V2E2V,
        (GridFile.DimensionName.DIAMOND_EDGE_SIZE, GridFile.DimensionName.VERTEX_NAME),
    )
    _add_to_dataset(
        dataset,
        grid.connectivities[V2EDim],
        GridFile.OffsetName.V2E,
        (
            GridFile.DimensionName.NEIGHBORS_TO_VERTEX_SIZE,
            GridFile.DimensionName.VERTEX_NAME,
        ),
    )
    _add_to_dataset(
        dataset,
        grid.connectivities[C2E2CDim],
        GridFile.OffsetName.C2E2C,
        (
            GridFile.DimensionName.NEIGHBORS_TO_CELL_SIZE,
            GridFile.DimensionName.CELL_NAME,
        ),
    )

    _add_to_dataset(
        dataset,
        np.ones((1, 24), dtype=np.int32),
        GridFile.GridRefinementName.START_INDEX_EDGES,
        (GridFile.DimensionName.MAX_CHILD_DOMAINS, GridFile.DimensionName.EDGE_GRF),
    )
    _add_to_dataset(
        dataset,
        np.ones((1, 14), dtype=np.int32),
        GridFile.GridRefinementName.START_INDEX_CELLS,
        (GridFile.DimensionName.MAX_CHILD_DOMAINS, GridFile.DimensionName.CELL_GRF),
    )
    _add_to_dataset(
        dataset,
        np.ones((1, 13), dtype=np.int32),
        GridFile.GridRefinementName.START_INDEX_VERTICES,
        (GridFile.DimensionName.MAX_CHILD_DOMAINS, GridFile.DimensionName.VERTEX_GRF),
    )
    _add_to_dataset(
        dataset,
        np.ones((1, 24), dtype=np.int32),
        GridFile.GridRefinementName.END_INDEX_EDGES,
        (GridFile.DimensionName.MAX_CHILD_DOMAINS, GridFile.DimensionName.EDGE_GRF),
    )
    _add_to_dataset(
        dataset,
        np.ones((1, 14), dtype=np.int32),
        GridFile.GridRefinementName.END_INDEX_CELLS,
        (GridFile.DimensionName.MAX_CHILD_DOMAINS, GridFile.DimensionName.CELL_GRF),
    )
    _add_to_dataset(
        dataset,
        np.ones((1, 13), dtype=np.int32),
        GridFile.GridRefinementName.END_INDEX_VERTICES,
        (GridFile.DimensionName.MAX_CHILD_DOMAINS, GridFile.DimensionName.VERTEX_GRF),
    )
    dataset.close()
    yield path
    path.unlink()


def get_icon_grid():
    processor_properties = get_processor_properties_for_run(SingleNodeRun())
    ranked_path = get_ranked_data_path(SER_DATA_BASEPATH, processor_properties)
    data_path = get_datapath_for_ranked_data(ranked_path)
    icon_data_provider = create_icon_serial_data_provider(data_path, processor_properties)
    grid_savepoint = icon_data_provider.from_savepoint_grid()
    return grid_savepoint.construct_icon_grid()


def get_grid_by_type(grid_type):
    if grid_type == "simple_grid":
        return SimpleGrid()
    elif grid_type == "icon_grid":
        return get_icon_grid()
    else:
        raise ValueError(f"Unknown grid type: {grid_type}")


@pytest.fixture
def grid(request):
    return request.param
