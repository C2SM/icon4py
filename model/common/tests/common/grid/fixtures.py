# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.testing.datatest_utils import REGIONAL_EXPERIMENT
from icon4py.model.testing.fixtures.datatest import (
    damping_height,
    data_provider,
    decomposition_info,
    download_ser_data,
    experiment,
    flat_height,
    grid_savepoint,
    htop_moist_proc,
    icon_grid,
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
    return REGIONAL_EXPERIMENT
