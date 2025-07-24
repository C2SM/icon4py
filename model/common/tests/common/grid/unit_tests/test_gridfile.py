# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import gridfile
from icon4py.model.testing import datatest_utils as dt_utils, grid_utils as gridtest_utils

from icon4py.model.testing.fixtures import grid_savepoint
from .. import utils


@pytest.mark.with_netcdf
def test_grid_file_dimension():
    global_grid_file = str(gridtest_utils.resolve_full_grid_file_name(dt_utils.R02B04_GLOBAL))
    parser = gridfile.GridFile(global_grid_file)
    try:
        parser.open()
        assert parser.dimension(gridfile.DimensionName.CELL_NAME) == utils.R02B04_GLOBAL_NUM_CELLS
        assert (
            parser.dimension(gridfile.DimensionName.VERTEX_NAME) == utils.R02B04_GLOBAL_NUM_VERTEX
        )
        assert parser.dimension(gridfile.DimensionName.EDGE_NAME) == utils.R02B04_GLOBAL_NUM_EDGES
    except Exception:
        pytest.fail()
    finally:
        parser.close()


@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
def test_grid_file_vertex_cell_edge_dimensions(grid_savepoint, grid_file):
    file = gridtest_utils.resolve_full_grid_file_name(grid_file)
    parser = gridfile.GridFile(str(file))
    try:
        parser.open()
        assert parser.dimension(gridfile.DimensionName.CELL_NAME) == grid_savepoint.num(
            dims.CellDim
        )
        assert parser.dimension(gridfile.DimensionName.VERTEX_NAME) == grid_savepoint.num(
            dims.VertexDim
        )
        assert parser.dimension(gridfile.DimensionName.EDGE_NAME) == grid_savepoint.num(
            dims.EdgeDim
        )
    except Exception as error:
        pytest.fail(f"reading of dimension from netcdf failed: {error}")
    finally:
        parser.close()
