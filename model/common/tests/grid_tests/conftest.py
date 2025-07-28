# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.testing import cases, grid_utils
from icon4py.model.testing.datatest_fixtures import (  # noqa: F401
    damping_height,
    data_provider,
    decomposition_info,
    experiment_data_files,
    flat_height,
    htop_moist_proc,
    icon_grid,
    icon_grid_savepoint,
    interpolation_savepoint,
    lowest_layer_thickness,
    maximal_layer_thickness,
    metrics_savepoint,
    model_top_height,
    processor_props,
    ranked_data_path,
    stretch_factor,
    top_height_limit_for_maximal_layer_thickness,
    topography_savepoint,
)


@pytest.fixture
def grid_file():
    grid_file_path = cases.SerializedExperiment.MCH_CH_R04B09.grid.file_name
    assert grid_file_path, "Grid file name must be defined in the experiment."
    return grid_utils.get_grid_file_path(grid_file_path)
