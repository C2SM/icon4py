# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from typing import TYPE_CHECKING

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import gridfile
from icon4py.model.testing import (
    grid_utils as gridtest_utils,
    definitions,
)
from icon4py.model.testing.fixtures import (
    backend,
    data_provider,
    download_ser_data,
    grid_savepoint,
    processor_props,
    ranked_data_path,
)

if TYPE_CHECKING:
    from icon4py.model.testing import serialbox


@pytest.mark.with_netcdf
def test_grid_file_dimension():
    grid_descriptor = definitions.Grids.R02B04_GLOBAL
    global_grid_file = str(gridtest_utils.resolve_full_grid_file_name(grid_descriptor))
    parser = gridfile.GridFile(global_grid_file)
    try:
        parser.open()
        assert parser.dimension(gridfile.DimensionName.CELL_NAME) == grid_descriptor.sizes["cell"]
        assert (
            parser.dimension(gridfile.DimensionName.VERTEX_NAME) == grid_descriptor.sizes["vertex"]
        )
        assert parser.dimension(gridfile.DimensionName.EDGE_NAME) == grid_descriptor.sizes["edge"]
    except Exception:
        pytest.fail()
    finally:
        parser.close()


@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "experiment",
    [
        definitions.Experiments.MCH_CH_R04B09,
        definitions.Experiments.EXCLAIM_APE,
    ],
)
def test_grid_file_vertex_cell_edge_dimensions(
    experiment: definitions.Experiment, grid_savepoint: serialbox.IconGridSavepoint
):
    file = gridtest_utils.resolve_full_grid_file_name(experiment.grid)
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
