# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import icon4py.model.common.grid.horizontal as h_grid
from icon4py.model.common import dimension as dims
from icon4py.model.common.metrics.compute_coeff_gradekin import compute_coeff_gradekin
from icon4py.model.testing import datatest_utils as dt_utils, helpers
from icon4py.model.testing.fixtures.datatest import (
    backend,
    data_provider,
    download_ser_data,
    grid_savepoint,
    icon_grid,
    metrics_savepoint,
    processor_props,
    ranked_data_path,
)


@pytest.mark.level("unit")
@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_coeff_gradekin(icon_grid, grid_savepoint, experiment, metrics_savepoint):
    edge_cell_length = grid_savepoint.edge_cell_length().asnumpy()
    inv_dual_edge_length = grid_savepoint.inv_dual_edge_length().asnumpy()
    coeff_gradekin_ref = metrics_savepoint.coeff_gradekin()
    horizontal_start = icon_grid.start_index(
        h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    horizontal_end = icon_grid.num_edges

    coeff_gradekin_full = compute_coeff_gradekin(
        edge_cell_length, inv_dual_edge_length, horizontal_start, horizontal_end
    )
    assert helpers.dallclose(coeff_gradekin_ref.asnumpy(), coeff_gradekin_full)
